import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from dotenv.main import load_dotenv
from torch import autograd  # for computing gradients

load_dotenv(override=True)

class AuroraGAN:
    # INCREASE LATENT SPACE
    def __init__(self, returns_df, asset_name, latent_dim=200, window_size=252, quarter_length=200, 
                 batch_size=120, n_epochs=600, lambda_gp=60, lambda_tail=15):
        """
        CGAN1: Conditional GAN for equities that conditions on a lagged quarter's cumulative return.
        
        Parameters:
          - returns_df: Empirical returns DataFrame (or Series) for the asset.
          - asset_name: Name of the asset.
          - latent_dim: Dimensionality of the noise vector.
          - window_size: Size of the rolling window in days.
          - quarter_length: Number of days in the quarter used to compute the condition.
          - batch_size: Batch size for training.
          - n_epochs: Number of epochs for training.
          - lambda_gp: Coefficient for the gradient penalty term.
          - lambda_tail: Coefficient for the tail penalty term.
        """
        self.returns_df = returns_df
        self.asset_name = asset_name
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.quarter_length = quarter_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_gp = lambda_gp
        self.lambda_tail = lambda_tail

        # If returns_df is a DataFrame, select the column for the asset.
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]
        else:
            self.returns_series = returns_df

        # Create rolling windows of returns and scale them.
        self.rolling_returns, self.scaler = self.create_rolling_returns(self.returns_series)
        # Automatically create condition data based on the lagged quarter cumulative returns.
        self.conditions = self.create_lagged_quarter_conditions(self.returns_series, window_size, quarter_length)

        self.conditions = self.create_multi_lag_conditions(self.returns_series, window_size, lag_periods=[110, 95, 70])
        # Ensure conditions align with rolling returns: discard the first few windows if necessary.
        min_length = min(len(self.rolling_returns), len(self.conditions))
        self.rolling_returns = self.rolling_returns[-min_length:]
        self.conditions = self.conditions[-min_length:]
        
        self.input_shape = (window_size,)
        self.cond_dim = self.conditions.shape[1]
        self.cuda = torch.cuda.is_available()

        # Instantiate Generator and Discriminator with condition inputs.
        self.generator = Generator(self.latent_dim, self.cond_dim, self.input_shape)
        self.discriminator = Discriminator(self.input_shape, self.cond_dim)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None

    def create_rolling_returns(self, returns_series):
        """
        Create rolling windows of scaled returns.
        """
        window_size = self.window_size
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_series.values.reshape(-1, 1))
        
        rolling_returns = []
        for i in range(len(scaled_returns) - window_size + 1):
            window = scaled_returns[i:i + window_size]
            rolling_returns.append(window.flatten())
        return np.array(rolling_returns), scaler

    def create_lagged_quarter_conditions(self, returns_series, window_size, quarter_length=63):
        """
        Create conditions using the cumulative return over the quarter preceding each rolling window.
        """
        conditions = []
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        
        n = len(returns_series)
        for i in range(quarter_length, n - window_size + 1):
            window = returns_series.iloc[i - quarter_length:i]
            cum_return = np.prod(1 + window) - 1
            volatility = window.std()
            kurtosis = pd.Series(window).kurt()
            conditions.append([cum_return, volatility, kurtosis])       
        return np.array(conditions)
    
    def create_multi_lag_conditions(self, returns_series, window_size, lag_periods=[252, 150, 63]):
        conditions = []
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        
        n = len(returns_series)
        max_lag = max(lag_periods)
        
        for i in range(max_lag, n - window_size + 1):
            condition_vector = []
            for lag in lag_periods:
                window = returns_series.iloc[i - lag:i].to_numpy()
                cum_return = float(np.prod(1 + window) - 1)
                volatility = float(window.std(ddof=1))
                kurtosis = float(pd.Series(window).kurt())
                threshold = np.percentile(window, 10)
                tail_losses = window[window <= threshold]
                if len(tail_losses) > 0:
                    cvar = float(tail_losses.mean())
                else:
                    cvar = float(threshold)
                window_cum = (1 + window).cumprod()
                running_max = np.maximum.accumulate(window_cum)
                drawdowns = (window_cum - running_max) / running_max
                max_drawdown = float(drawdowns.min())
                count = 0
                max_count = 0
                for r in window:
                    if r < 0:
                        count += 1
                        max_count = max(max_count, count)
                    else:
                        count = 0
                crash_duration = float(max_count)
                
                condition_vector.extend([
                    cum_return, 
                    volatility, 
                    max_drawdown, 
                    crash_duration
                ])
            conditions.append(condition_vector)
            
        return np.array(conditions, dtype=float)

    def setup(self):
        dataset = TensorDataset(torch.tensor(self.rolling_returns, dtype=torch.float32),
                                 torch.tensor(self.conditions, dtype=torch.float32))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def compute_gradient_penalty(self, real_samples, fake_samples, cond):
        """Calculates the gradient penalty loss for WGAN-GP"""
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
        """Penalize differences in both lower and upper tails"""
        real_flat = real_returns.view(-1)
        gen_flat = gen_returns.view(-1)
        
        k = int(0.1 * real_flat.size(0))
        k = max(k, 1)
        
        # Sort for lower and upper tails
        real_sorted, _ = torch.sort(real_flat)
        gen_sorted, _ = torch.sort(gen_flat)
        
        # Lower tail penalty
        real_lower_tail = torch.mean(real_sorted[:k])
        gen_lower_tail = torch.mean(gen_sorted[:k])
        lower_penalty = (gen_lower_tail - real_lower_tail) ** 2
        
        # Upper tail penalty
        real_upper_tail = torch.mean(real_sorted[-k:])
        gen_upper_tail = torch.mean(gen_sorted[-k:])
        upper_penalty = (gen_upper_tail - real_upper_tail) ** 2
        
        return lower_penalty + upper_penalty
    
    def compute_structure_penalty(self, real_returns, gen_returns):
        """Penalty to preserve the ring structure in PCA space"""
        # For simplicity, use the first 2 PCs as a proxy
        pca = PCA(n_components=2)
        real_flat = real_returns.view(real_returns.size(0), -1).cpu().numpy()
        gen_flat = gen_returns.view(gen_returns.size(0), -1).cpu().numpy()
        
        # Fit PCA on real data
        pca.fit(real_flat)
        
        # Transform both datasets
        real_pca = torch.tensor(pca.transform(real_flat), device=real_returns.device)
        gen_pca = torch.tensor(pca.transform(gen_flat), device=gen_returns.device)
        
        # Calculate distance from origin for each point
        real_dist = torch.sqrt(torch.sum(real_pca**2, dim=1))
        gen_dist = torch.sqrt(torch.sum(gen_pca**2, dim=1))
        
        # Compare distributions of distances (this preserves the ring structure)
        real_dist_sorted, _ = torch.sort(real_dist)
        gen_dist_sorted, _ = torch.sort(gen_dist)
        
        # Calculate MSE between sorted distances to match the ring structure
        dist_penalty = F.mse_loss(gen_dist_sorted, real_dist_sorted)
        
        return dist_penalty

    def train(self):
        self.setup()
        for epoch in range(self.n_epochs):
            for i, (real_returns, cond) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns = real_returns.to(next(self.discriminator.parameters()).device)
                cond = cond.to(real_returns.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=real_returns.device)
                gen_returns = self.generator(z, cond)

                d_real = self.discriminator(real_returns, cond)
                d_fake = self.discriminator(gen_returns.detach(), cond)
                d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                
                gp = self.compute_gradient_penalty(real_returns, gen_returns.detach(), cond)
                d_loss += gp

                d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z, cond)
                    tail_penalty = self.compute_tail_penalty(real_returns, gen_returns)
                    g_loss = -torch.mean(self.discriminator(gen_returns, cond)) + self.lambda_tail * tail_penalty
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                          f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    def generate_scenarios(self, save=True, num_scenarios=50000):
        self.generator.eval()
        device = 'cuda' if self.cuda else 'cpu'
        
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim, device=device)
                indices = np.random.choice(len(self.conditions), size=batch_size, replace=True)
                cond = torch.tensor(self.conditions[indices], dtype=torch.float32, device=device)
                gen_returns = self.generator(z, cond)
                gen_returns = self.scaler.inverse_transform(gen_returns.cpu().numpy())
                all_generated_returns.append(gen_returns)
                
        all_generated_returns = np.vstack(all_generated_returns)
        
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_final_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated scenarios saved to: {save_path}")
        
        return all_generated_returns


# Conditional Generator: accepts noise and condition input.
class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.output_shape = output_shape  # e.g., (252,)
        input_dim = latent_dim + cond_dim  # Concatenate noise and condition.

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(output_shape)))
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        out = self.model(x)
        return out.view(out.size(0), *self.output_shape)


# Conditional Discriminator: accepts a return sequence and the condition.
class Discriminator(nn.Module):
    def __init__(self, input_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape  # e.g., (252,)
        self.cond_dim = cond_dim
        input_dim = int(np.prod(input_shape)) + cond_dim  # Concatenate flattened returns and condition.
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1)
        )

    def forward(self, returns, condition):
        x = returns.view(returns.size(0), -1)
        x = torch.cat((x, condition), dim=1)
        validity = self.model(x)
        return validity
