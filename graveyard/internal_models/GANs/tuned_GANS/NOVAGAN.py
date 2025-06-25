import argparse
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dotenv.main import load_dotenv
load_dotenv(override=True)

# ------------------------------
# Adapter Network Definition
# (Kept for training and other methods, but not used in generate_new_scenarios_from_return)
# ------------------------------
class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        A lightweight adapter network to map new condition inputs 
        to the condition space expected by the generator.
        """
        super(Adapter, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ------------------------------
# NovaGAN Definition (No Scaler Version)
# ------------------------------
class NovaGAN:
    def __init__(self, returns_df, asset_name, latent_dim=250, window_size=252, quarter_length=200, batch_size=80, n_epochs=5000):
        """
        CGAN1: Conditional GAN for equities that conditions on a lagged quarter's cumulative return.
        This version uses raw returns without any scaling.
        
        Parameters:
          - returns_df: Empirical returns DataFrame (or Series) for the asset.
          - asset_name: Name of the asset (if returns_df is a DataFrame, this selects the column).
          - latent_dim: Dimensionality of the noise vector.
          - window_size: Size of the rolling window (default 252, one year).
          - quarter_length: Days in the quarter used to compute the condition.
          - batch_size: Batch size for training.
          - n_epochs: Number of epochs for training.
        """
        self.returns_df = returns_df
        self.asset_name = asset_name
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.quarter_length = quarter_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Use raw returns from the provided DataFrame/Series.
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]  # raw returns
        else:
            self.returns_series = returns_df  # raw returns

        # Create rolling windows from raw returns (no scaling).
        self.rolling_returns = self.create_rolling_returns(self.returns_series)
        # Create condition data using raw returns.
        #self.conditions = self.create_lagged_quarter_conditions(self.returns_series, window_size, quarter_length)

        # Use multi-lag conditions.
        self.lag_periods = [110, 95, 70]
        self.conditions = self.create_multi_lag_conditions(self.returns_series, window_size, lag_periods=self.lag_periods)
        
        # Ensure that rolling_returns and conditions have the same length.
        min_length = min(len(self.rolling_returns), len(self.conditions))
        self.rolling_returns = self.rolling_returns[-min_length:]
        self.conditions = self.conditions[-min_length:]
        
        self.input_shape = (window_size,)
        self.cond_dim = self.conditions.shape[1]
        self.cuda = torch.cuda.is_available()

        # Instantiate Generator and Discriminator.
        self.generator = Generator(self.latent_dim, self.cond_dim, self.input_shape)
        self.discriminator = Discriminator(self.input_shape, self.cond_dim)
        
        # Initialize the adapter (if used in other methods).
        self.adapter = Adapter(self.cond_dim, self.cond_dim)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.optimizer_adapter = None

        # The full raw returns series is kept in self.returns_series,
        # which is updated when new returns are added.
    
    def create_rolling_returns(self, returns_series):
        """
        Create rolling windows from raw returns without scaling.
        """
        window_size = self.window_size
        returns = returns_series.values.reshape(-1, 1)
        rolling_returns = []
        for i in range(len(returns) - window_size + 1):
            window = returns[i:i + window_size]
            rolling_returns.append(window.flatten())
        return np.array(rolling_returns)

    def create_lagged_quarter_conditions(self, returns_series, window_size, quarter_length=63):
        """
        Compute conditions from raw returns.
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
        """
        Compute multi-lag conditions from raw returns.
        """
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
                window_cum = np.cumprod(1 + window)
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
                condition_vector.extend([cum_return, volatility, max_drawdown, crash_duration])
            conditions.append(condition_vector)
        return np.array(conditions, dtype=float)

    def setup(self):
        dataset = TensorDataset(torch.tensor(self.rolling_returns, dtype=torch.float32),
                                 torch.tensor(self.conditions, dtype=torch.float32))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        device = 'cuda' if self.cuda else 'cpu'
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adapter.cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.optimizer_adapter = torch.optim.Adam(self.adapter.parameters(), lr=0.0001)

    def train(self):
        self.setup()
        device = 'cuda' if self.cuda else 'cpu'
        for epoch in range(self.n_epochs):
            for i, (real_returns, cond) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns = real_returns.to(device)
                cond = cond.to(device)
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_returns = self.generator(z, cond)
                d_loss = -torch.mean(self.discriminator(real_returns, cond)) + torch.mean(self.discriminator(gen_returns.detach(), cond))
                d_loss.backward()
                self.optimizer_D.step()
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, self.latent_dim).to(device)
                    gen_returns = self.generator(z, cond)
                    g_loss = -torch.mean(self.discriminator(gen_returns, cond))
                    g_loss.backward()
                    self.optimizer_G.step()
            print(f"[Epoch {epoch}/{self.n_epochs}] Completed.")

    def generate_scenarios(self, save=True, num_scenarios=50000):
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000
        device = 'cuda' if self.cuda else 'cpu'
        # Use the mean condition from training data.
        cond_value = torch.tensor(self.conditions.mean(axis=0), dtype=torch.float32, device=device)
        cond_value = cond_value.unsqueeze(0).repeat(batch_size, 1)
        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_returns = self.generator(z, cond_value).cpu().numpy()
                # No inverse transform is needed now.
                all_generated_returns.append(gen_returns)
        all_generated_returns = np.vstack(all_generated_returns)
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_final_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated scenarios saved to: {save_path}")
        return all_generated_returns

    def compute_condition_from_new_window(self, new_window):
        """
        Compute the condition vector from a new window of raw returns.
        new_window must have at least max(self.lag_periods) data points.
        This version uses raw returns since we're not scaling.
        """
        new_window = np.array(new_window)
        if len(new_window) < max(self.lag_periods):
            raise ValueError(f"New window must have at least {max(self.lag_periods)} returns.")
        condition_vector = []
        for lag in self.lag_periods:
            window_data = new_window[-lag:]
            cum_return = float(np.prod(1 + window_data) - 1)
            volatility = float(np.std(window_data, ddof=1))
            window_cum = np.cumprod(1 + window_data)
            running_max = np.maximum.accumulate(window_cum)
            drawdowns = (window_cum - running_max) / running_max
            max_drawdown = float(drawdowns.min())
            count = 0
            max_count = 0
            for r in window_data:
                if r < 0:
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 0
            crash_duration = float(max_count)
            condition_vector.extend([cum_return, volatility, max_drawdown, crash_duration])
        return np.array(condition_vector, dtype=float)

    def generate_new_scenarios_from_window(self, new_window, save=True, num_scenarios=50000):
        """
        Generate scenarios based on a new window of raw returns.
        The condition is computed from the new window.
        """
        new_condition = self.compute_condition_from_new_window(new_window)
        self.generator.eval()
        self.adapter.eval()
        all_generated_returns = []
        batch_size = 1000
        device = 'cuda' if self.cuda else 'cpu'
        new_cond_tensor = torch.tensor(new_condition, dtype=torch.float32, device=device)
        new_cond_tensor = new_cond_tensor.unsqueeze(0).repeat(batch_size, 1)
        adapted_condition = self.adapter(new_cond_tensor)
        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_returns = self.generator(z, adapted_condition).cpu().numpy()
                all_generated_returns.append(gen_returns)
        all_generated_returns = np.vstack(all_generated_returns)
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_new_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated new scenarios saved to: {save_path}")
        return all_generated_returns

    def generate_new_scenarios_from_return(self, new_return, new_date=None, save=True, num_scenarios=50000):
        """
        Generate new scenarios based on a new return (or returns) using raw returns.
        
        Process:
          1. If new_return is scalar, convert to a list.
          2. Append the new return(s) to the raw returns series (self.returns_series).
          3. Extract the last window (of length window_size) from self.returns_series.
          4. Compute the condition vector from this window.
          5. Generate scenarios using the generator without using the adapter.
        
        Note: new_return is assumed to be raw (unscaled), consistent with training.
        """
        if np.isscalar(new_return):
            new_return = [new_return]
        if new_date is None:
            new_date = pd.Timestamp.now()
        # Create a Series for the new return(s) with the given date.
        new_series = pd.Series(new_return, index=[new_date] * len(new_return))
        # Append to the stored raw returns.
        self.returns_series = pd.concat([self.returns_series, new_series])
        # Extract the last window of raw returns.
        if len(self.returns_series) < self.window_size:
            raise ValueError("Not enough data to form a full window.")
        window = self.returns_series.tail(self.window_size).values
        # Compute the condition vector from the window.
        new_condition = self.compute_condition_from_new_window(window)
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000
        device = 'cuda' if self.cuda else 'cpu'
        cond_tensor = torch.tensor(new_condition, dtype=torch.float32, device=device)
        cond_tensor = cond_tensor.unsqueeze(0).repeat(batch_size, 1)
        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_returns = self.generator(z, cond_tensor).cpu().numpy()
                all_generated_returns.append(gen_returns)
        all_generated_returns = np.vstack(all_generated_returns)
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_updated_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated new scenarios saved to: {save_path}")
        return all_generated_returns

# ------------------------------
# Generator and Discriminator Definitions (Unchanged)
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.output_shape = output_shape  # e.g. (252,)
        input_dim = latent_dim + cond_dim
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

class Discriminator(nn.Module):
    def __init__(self, input_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape  # e.g. (252,)
        self.cond_dim = cond_dim
        input_dim = int(np.prod(input_shape)) + cond_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.9),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.9),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.9),
            nn.Linear(1000, 1)
        )
    def forward(self, returns, condition):
        x = returns.view(returns.size(0), -1)
        x = torch.cat((x, condition), dim=1)
        validity = self.model(x)
        return validity
