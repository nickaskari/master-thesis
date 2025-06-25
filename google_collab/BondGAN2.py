import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from dotenv.main import load_dotenv
from torch import autograd, device 
from numba import jit
from scipy import stats
from tqdm.auto import tqdm as progress_bar
import matplotlib.pyplot as plt

import tqdm

load_dotenv(override=True)

'''
gan = BondGAN(asset_returns, asset_name, n_epochs=1000, lambda_gp = 7, cond_scale=1)
gan.g_lr = 10e-6
gan.d_lr = 50e-8
'''

class BondGAN:

    def __init__(self, returns_df, asset_name, latent_dim=150, window_size=252, 
                 batch_size=1000, n_epochs=3000, lambda_gp=5, lambda_tail=1, 
                 lag_periods=None, cond_scale = 2, lambda_skew=50, lambda_kurtosis=0):

        torch.manual_seed(4)
        np.random.seed(1)


        self.returns_df = returns_df
        self.asset_name = asset_name
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_gp = lambda_gp
        self.lambda_tail = lambda_tail

        self.lambda_skew = lambda_skew
        self.lambda_kurtosis = lambda_kurtosis

        self.d_losses = []
        self.g_losses = []
        self.gp_losses = []
        self.tail_losses = []
        self.generator_lr = 0.0001
        self.discriminator_lr = 0.0002

        self.cond_scale = cond_scale

        self.lag_periods = lag_periods if lag_periods is not None else [14, 90, 180]
        
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]
        else:
            self.returns_series = returns_df

        raw_conditions = self.create_multi_lag_conditions(self.returns_series, self.window_size)

        max_hist_lag = max(self.lag_periods)
        
        aligned_conditions_list = []
        aligned_rolling_returns_list = []
        
        self.scaler = StandardScaler() #LambertW(self.returns_series)

        returns_np = self.returns_series.values
        #returns_np = self.scaler.fit_transform(self.returns_series.values.reshape(-1, 1))
        
        num_raw_conditions = raw_conditions.shape[0]

        for k in range(num_raw_conditions):
            idx_condition_ends = max_hist_lag + k - 1
            idx_target_window_starts = idx_condition_ends + 1
            idx_target_window_ends = idx_target_window_starts + self.window_size

            if idx_target_window_ends <= len(returns_np):
                target_window = returns_np[idx_target_window_starts : idx_target_window_ends]
                aligned_rolling_returns_list.append(target_window)
                aligned_conditions_list.append(raw_conditions[k])
        

        self.rolling_returns = np.array(aligned_rolling_returns_list)

        self.rolling_returns = self.scaler.fit_transform(self.rolling_returns)

        self.conditions = np.array(aligned_conditions_list)

        self.input_shape = (self.window_size,)
        self.cond_dim = self.conditions.shape[1]
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(self.latent_dim, self.cond_dim, self.input_shape)
        self.discriminator = Discriminator(self.input_shape, self.cond_dim)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None

    def positive_negative_ratio(self, window):
      positives = window[window > 0].sum()
      negatives = np.abs(window[window < 0].sum())
      if negatives == 0:
          return 10.0
      return positives / negatives

    def create_multi_lag_conditions(self, returns_series, window_size):
      if not isinstance(returns_series, pd.Series):
          returns_series = pd.Series(returns_series)
      
      @jit(nopython=True)
      def calculate_crash_duration(window):
          count = 0
          max_count = 0
          for r in window:
              if r < 0:
                  count += 1
                  max_count = max(max_count, count)
              else:
                  count = 0
          return max_count
      
      def calculate_max_drawdown(window_cum):
          n = len(window_cum)
          running_max = np.zeros_like(window_cum)
          if n == 0: return 0.0 # Handle empty window
          running_max[0] = window_cum[0]
          for i in range(1, n):
              running_max[i] = max(running_max[i-1], window_cum[i])
          drawdowns = (window_cum - running_max) / running_max
          drawdowns[running_max == 0] = 0 
          return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
      
      n = len(returns_series)
      max_l = max(self.lag_periods) 
      conditions = []
      
      returns_arr = returns_series.values
      
      for i in range(max_l, n + 1):
          condition_vector = []
          for lag in self.lag_periods: 
              window = returns_arr[i-lag:i]
              if len(window) == 0: continue 

              cum_return = np.prod(1 + window) - 1
              volatility = np.std(window, ddof=1) if len(window) > 1 else 0.0
              mean_return = np.mean(window) if len(window) > 0 else 0.0
              
              kurtosis_val = pd.Series(window).kurt() if len(window) > 0 else 0.0
              threshold = 10.0
              if kurtosis_val <= threshold:
                  kurtosis = kurtosis_val
              else:
                  excess = kurtosis_val - threshold
                  kurtosis = threshold + np.sqrt(excess)
              
              window_cum = np.cumprod(1 + window)
              max_drawdown = calculate_max_drawdown(window_cum)

              pos_neg_ratio = self.positive_negative_ratio(window) * 50
              
              condition_vector.extend([
                  float(cum_return), float(volatility), float(kurtosis_val),
                  float(max_drawdown),
              ])
          
          if condition_vector: # Only append if features were extracted
            conditions.append(condition_vector)


      conditions_np = np.array(conditions, dtype=float)
      self.condition_scaler = StandardScaler()
      normalized_conditions = self.condition_scaler.fit_transform(conditions_np)*self.cond_scale
      
      return normalized_conditions

    def update_latest_condition(self):
      def calculate_max_drawdown(window_cum):
          if len(window_cum) == 0: return 0.0
          running_max = np.zeros_like(window_cum)
          running_max[0] = window_cum[0]
          for i in range(1, len(window_cum)):
              running_max[i] = max(running_max[i-1], window_cum[i])
          drawdowns = (window_cum - running_max) / running_max
          drawdowns[running_max == 0] = 0
          return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
      
      returns_arr = self.returns_series.values
      n_returns = len(self.returns_series)
      latest_condition = []
      
      for lag in self.lag_periods: # Use self.lag_periods
          if n_returns < lag: continue # Not enough data for this lag
          window = returns_arr[n_returns-lag:n_returns]
          if len(window) == 0: continue

          cum_return = np.prod(1 + window) - 1
          volatility = np.std(window, ddof=1) if len(window) > 1 else 0.0
          mean_return = np.mean(window) if len(window) > 0 else 0.0

          window_cum = np.cumprod(1 + window)
          max_drawdown = calculate_max_drawdown(window_cum)
          pos_neg_ratio = self.positive_negative_ratio(window)*50

          kurtosis_val = pd.Series(window).kurt() if len(window) > 0 else 0.0
          threshold = 10.0
          if kurtosis_val <= threshold:
              kurtosis = kurtosis_val
          else:
              excess = kurtosis_val - threshold
              kurtosis = threshold + np.sqrt(excess)
          
          latest_condition.extend([
            float(cum_return), float(volatility), float(kurtosis_val),
            float(max_drawdown),
            ])
      
      if not latest_condition: # If no features could be extracted
          return self.conditions # Return existing conditions unchanged

      latest_condition_array = np.array([latest_condition], dtype=float)
      latest_condition_normalized = self.condition_scaler.transform(latest_condition_array)*self.cond_scale
      self.conditions = np.vstack([self.conditions, latest_condition_normalized])

      return self.conditions

    def setup(self):
        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
        conditions_tensor = torch.tensor(self.conditions, dtype=torch.float32)


        self.dataloader = DataLoader(
            TensorDataset(rolling_returns_tensor, conditions_tensor),
            batch_size=self.batch_size,
            drop_last=True 
        )

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999))


    def compute_gradient_penalty(self, real_samples, fake_samples, cond):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, device=real_samples.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates, cond)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty

    def compute_tail_penalty(self, real_returns, gen_returns):
        real_flat = real_returns.view(-1)
        gen_flat = gen_returns.view(-1)
        
        k = int(0.1 * real_flat.size(0))
        k = max(k, 1)
        
        real_sorted, _ = torch.sort(real_flat)
        gen_sorted, _ = torch.sort(gen_flat)
        
        real_lower_tail = torch.mean(real_sorted[:k])
        gen_lower_tail = torch.mean(gen_sorted[:k])
        lower_penalty = (gen_lower_tail - real_lower_tail) ** 2
        
        real_upper_tail = torch.mean(real_sorted[-k:])
        gen_upper_tail = torch.mean(gen_sorted[-k:])
        upper_penalty = (gen_upper_tail - real_upper_tail) ** 2
        
        return lower_penalty + upper_penalty

    def compute_skewness_penalty(self, real_returns, gen_returns):
      """Penalize positive skew in generated returns"""
      def skewness(x):
          x_flat = x.view(-1)
          mean = torch.mean(x_flat)
          std = torch.std(x_flat)
          if std == 0:
              return torch.tensor(0.0)
          skew = torch.mean(((x_flat - mean) / std) ** 3)
          return skew
    
      real_skew = skewness(real_returns)
      gen_skew = skewness(gen_returns)
      
      # Penalize if generated skew > real skew (more positive = bad)
      skew_penalty = torch.clamp(gen_skew - real_skew, min=0) ** 2
      
      # Extra penalty for any positive skew
      positive_skew_penalty = torch.clamp(gen_skew, min=0) ** 2
      
      return skew_penalty + 0.5 * positive_skew_penalty
  
    def compute_kurtosis_penalty(self, real_returns, gen_returns):
      """Penalize difference in kurtosis between real and generated returns"""
      def kurtosis_torch(x):
          x_flat = x.view(-1)
          mean = torch.mean(x_flat)
          std = torch.std(x_flat, unbiased=True)
          if std == 0:
              return torch.tensor(0.0, device=x.device)
          
          # Calculate excess kurtosis (subtract 3 for normal distribution baseline)
          centered = (x_flat - mean) / std
          kurt = torch.mean(centered ** 4) - 3.0
          return kurt
      
      real_kurt = kurtosis_torch(real_returns)
      gen_kurt = kurtosis_torch(gen_returns)
      
      # Penalize deviation from real data kurtosis
      kurtosis_penalty = (gen_kurt - real_kurt) ** 2
      
      return kurtosis_penalty
      

    def train(self):
      self.setup()
      
      for epoch in range(self.n_epochs):
          for i, (real_returns, cond) in enumerate(self.dataloader):
              if real_returns.size(0) == 0: continue
              
              batch_size = real_returns.size(0)
              current_device = next(self.discriminator.parameters()).device
              real_returns = real_returns.to(current_device)
              cond = cond.to(current_device)

              self.optimizer_D.zero_grad()
              z = torch.randn(batch_size, self.latent_dim, device=current_device)
              gen_returns = self.generator(z, cond)

              d_real = self.discriminator(real_returns, cond)
              d_fake = self.discriminator(gen_returns.detach(), cond)
              d_loss = -torch.mean(d_real) + torch.mean(d_fake)
              
              gp = self.compute_gradient_penalty(real_returns, gen_returns.detach(), cond)
              d_loss += gp

              d_loss.backward()
              self.optimizer_D.step()
              
              g_loss_val = torch.tensor(0.0)
              tail_penalty_val = torch.tensor(0.0)

              self.d_losses.append(d_loss.item())
              self.gp_losses.append(gp.item())

              if i % 3 == 0:
                  self.optimizer_G.zero_grad()
                  z_g = torch.randn(batch_size, self.latent_dim, device=current_device)
                  gen_returns_g = self.generator(z_g, cond)
                  
                  tail_penalty = self.compute_tail_penalty(real_returns, gen_returns_g)
                  skew_penalty = self.compute_skewness_penalty(real_returns, gen_returns_g)
                  kurtosis_penalty = self.compute_kurtosis_penalty(real_returns, gen_returns_g)
                  
                  g_loss = -torch.mean(self.discriminator(gen_returns_g, cond)) + \
                          self.lambda_tail * tail_penalty + \
                          self.lambda_skew * skew_penalty + \
                          self.lambda_kurtosis * kurtosis_penalty
                  
                  g_loss.backward()
                  self.optimizer_G.step()
                  
                  g_loss_val = g_loss.item()
                  tail_penalty_val = tail_penalty.item()
                  
                  self.g_losses.append(g_loss_val)
                  self.tail_losses.append(tail_penalty_val)
              else:
                  # Pad with last values for consistent tracking
                  if self.g_losses:
                      self.g_losses.append(self.g_losses[-1])
                      self.tail_losses.append(self.tail_losses[-1])

              
              total_iterations = self.n_epochs * len(self.dataloader)
              current_iteration = epoch * len(self.dataloader) + i + 1
              progress_percent = 100 * current_iteration / total_iterations
              bar_length = 20
              filled_length = int(bar_length * current_iteration // total_iterations)
              bar = '█' * filled_length + '-' * (bar_length - filled_length)
              
              output = f"\rTraining: [{bar}] {progress_percent:.1f}% | " \
                      f"Epoch {epoch+1}/{self.n_epochs} | " \
                      f"Batch {i+1}/{len(self.dataloader)} | " \
                      f"D: {d_loss.item():.4f} | " \
                      f"G: {g_loss_val:.4f} | " \
                      f"Tail: {tail_penalty_val:.4f}"
              
              print(output, end='', flush=True)
      
      print() 
      self.plot_losses()

    def plot_losses(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Discriminator loss
        ax1.plot(self.d_losses, label='D Loss', color='red')
        ax1.set_title('Discriminator Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Generator loss
        ax2.plot(self.g_losses, label='G Loss', color='blue')
        ax2.set_title('Generator Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Gradient penalty
        ax3.plot(self.gp_losses, label='GP', color='green')
        ax3.set_title('Gradient Penalty')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Penalty')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Tail penalty
        ax4.plot(self.tail_losses, label='Tail Penalty', color='orange')
        ax4.set_title('Tail Penalty')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Penalty')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'loss_curves_{self.asset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_scenarios(self, save=True, num_scenarios=50000):
        self.generator.eval()
        device = 'cuda' if self.cuda else 'cpu'
        
        # Calculate min and max from historical returns
        historical_min = self.returns_series.min()
        historical_max = self.returns_series.max()
        
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            new_cond = self.conditions[:]
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim, device=device)
                indices = np.random.choice(len(new_cond), size=batch_size, replace=True)
                cond = torch.tensor(new_cond[indices], dtype=torch.float32, device=device)
                gen_returns = self.generator(z, cond)
                gen_returns = self.scaler.inverse_transform(gen_returns.cpu().numpy())
                
                # Clip to historical min/max
                #gen_returns = np.clip(gen_returns, historical_min, historical_max)
                
                all_generated_returns.append(gen_returns)
                
        all_generated_returns = np.vstack(all_generated_returns)
        
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_final_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated scenarios saved to: {save_path}")
        
        return all_generated_returns
    
    def generate_new_scenarios_from_return(self, new_return, date, look_back, next_day, save=False, num_scenarios=10000):
        
        if isinstance(self.returns_series, pd.Series):
            new_row = pd.Series([new_return], index=[date])
            self.returns_series = pd.concat([self.returns_series, new_row])
        else:
            self.returns_series = np.append(self.returns_series, new_return)

        self.conditions = self.update_latest_condition()

        # Calculate min and max from historical returns
        historical_min = self.returns_series.min()
        historical_max = self.returns_series.max()

        device = 'cuda' if self.cuda else 'cpu'

        all_generated_returns = []
        batch_size = 10000
        
        with torch.no_grad():
            rel = self.conditions[-look_back:]
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim, device=device)
                condition_indices = np.random.choice(len(rel), batch_size, replace=True)
                sampled_conditions = rel[condition_indices]
                cond = torch.tensor(sampled_conditions, dtype=torch.float32, device=device)
                
                gen_returns = self.generator(z, cond)
                
                gen_returns_np = gen_returns.cpu().numpy()
                gen_returns_original = self.scaler.inverse_transform(gen_returns_np)
                
                gen_returns_original = np.clip(gen_returns_original, historical_min, historical_max)
                
                all_generated_returns.append(gen_returns_original)
        
        all_generated_returns = np.vstack(all_generated_returns)
        
        if save:
            save_dir = f"test_{next_day}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            #print(f"Generated scenarios saved to: {save_path}")
        
        return all_generated_returns

    def reset_returns(self):
      if isinstance(self.returns_df, pd.DataFrame):
          self.returns_series = self.returns_df[self.asset_name].copy() # Use copy
      else: # Assuming self.returns_df is already a Series or array-like
          self.returns_series = pd.Series(self.returns_df).copy() # Ensure it's a Series and a copy

    def prep_retrain(self):
        self.d_losses = []
        self.g_losses = []
        self.gp_losses = []
        self.tail_losses = []

        raw_conditions = self.create_multi_lag_conditions(self.returns_series, self.window_size)

        max_hist_lag = max(self.lag_periods)
        
        aligned_conditions_list = []
        aligned_rolling_returns_list = []
        
        self.scaler = StandardScaler()

        returns_np = self.returns_series.values
        returns_np = self.scaler.fit_transform(self.returns_series.values.reshape(-1, 1))
        
        num_raw_conditions = raw_conditions.shape[0]

        for k in range(num_raw_conditions):
            idx_condition_ends = max_hist_lag + k - 1
            idx_target_window_starts = idx_condition_ends + 1
            idx_target_window_ends = idx_target_window_starts + self.window_size

            if idx_target_window_ends <= len(returns_np):
                target_window = returns_np[idx_target_window_starts : idx_target_window_ends]
                aligned_rolling_returns_list.append(target_window)
                aligned_conditions_list.append(raw_conditions[k])
        

        self.rolling_returns = np.array(aligned_rolling_returns_list)

        #self.rolling_returns = self.scaler.fit_transform(self.rolling_returns)

        self.conditions = np.array(aligned_conditions_list)

        self.input_shape = (self.window_size,)
        self.cond_dim = self.conditions.shape[1]
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(self.latent_dim, self.cond_dim, self.input_shape)
        self.discriminator = Discriminator(self.input_shape, self.cond_dim)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None

class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.output_shape = output_shape 
        input_dim = latent_dim + cond_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, int(np.prod(output_shape)))
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        out = self.model(x)
        #out = torch.tanh(out)  # Force output to [-1, 1]
        return out.view(out.size(0), *self.output_shape)

class Discriminator(nn.Module):
    def __init__(self, input_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape 
        self.cond_dim = cond_dim
        input_dim = int(np.prod(input_shape)) + cond_dim 
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, returns, condition):
        x = returns.view(returns.size(0), -1)
        x = torch.cat((x, condition), dim=1)
        validity = self.model(x)
        return validity



from scipy.optimize import fmin
from scipy.special import lambertw
from scipy.stats import kurtosis, norm


class LambertW:
    def __init__(self, returns):
      self.returns = returns.values.reshape(-1, 1)
      self.returns_mean = np.mean(self.returns)
      self.returns_norm = returns - self.returns_mean
      self.params = None
      self.returns_max = None
        

    def delta_init(self, z):
      k = kurtosis(z, fisher=False, bias=False)
      if k < 166. / 62.:
          return 0.01
      return np.clip(1. / 66 * (np.sqrt(66 * k - 162.) - 6.), 0.01, 0.48)

    def delta_gmm(self, z):
        delta = self.delta_init(z)
        
        def iter(q):
            u = self.W_delta(z, np.exp(q))
            if not np.all(np.isfinite(u)):
                return 0.
            k = kurtosis(u, fisher=True, bias=False)**2
            if not np.isfinite(k) or k > 1e10:
                return 1e10
            return k
        
        res = fmin(iter, np.log(delta), disp=0)
        return np.around(np.exp(res[-1]), 6)

    def W_delta(self, z, delta):
        return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)

    def W_params(self, z, params):
        return params[0] + params[1] * self.W_delta((z - params[0]) / params[1], params[2])

    def inverse(self, z, params):
        return params[0] + params[1] * (z * np.exp(z * z * (params[2] * 0.5)))

    def igmm(self, z, eps=1e-6, max_iter=100):
        delta = self.delta_init(z)
        params = [np.median(z), np.std(z) * (1. - 2. * delta) ** 0.75, delta]
        for k in range(max_iter):
            params_old = params
            u = (z - params[0]) / params[1]
            params[2] = self.delta_gmm(u)
            x = self.W_params(z, params)
            params[0], params[1] = np.mean(x), np.std(x)
            
            if np.linalg.norm(np.array(params) - np.array(params_old)) < eps:
                break
            if k == max_iter - 1:
                raise RuntimeError("Solution not found")
                
        return params
    
    def fit_transform(self, dummy):
      params = self.igmm(self.returns_norm)
      processed_returns = self.W_delta((self.returns_norm - params[0]) / params[1], params[2])
      returns_max = np.max(np.abs(processed_returns))
      processed_returns /= returns_max

      self.returns_max = returns_max
      self.params = params

      return processed_returns
    
    def fit_transform_new(self, data):
      data = data.values
      new_returns_norm = data - self.returns_mean
      transformed = self.W_delta(
        (new_returns_norm - self.params[0]) / self.params[1], 
        self.params[2]
      )
      transformed /= self.returns_max
      
      return transformed


    def inverse_transform(self, data):
        denorm = data * self.returns_max
        original = self.inverse(denorm, self.params) + self.returns_mean
        return original
