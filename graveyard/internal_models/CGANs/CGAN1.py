import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from dotenv.main import load_dotenv
load_dotenv(override=True)

class CGAN1:
    def __init__(self, returns_df, asset_name, latent_dim=200, window_size=252, quarter_length=63, batch_size=200, n_epochs=3000):
        """
        CGAN1: Conditional GAN for equities that conditions on a lagged quarter's cumulative return.
        
        Parameters:
          - returns_df: Empirical returns DataFrame (or Series) for the asset.
          - asset_name: Name of the asset (used to select a column if returns_df is a DataFrame).
          - latent_dim: Dimensionality of the noise vector.
          - window_size: Size of the rolling window in days (default 252, one year).
          - quarter_length: Number of days in the quarter used to compute the condition (default 63).
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

        # If returns_df is a DataFrame, select the column for the asset.
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]
        else:
            self.returns_series = returns_df

        # Create rolling windows of returns and scale them.
        self.rolling_returns, self.scaler = self.create_rolling_returns(self.returns_series)
        # Automatically create condition data based on the lagged quarter cumulative returns.
        self.conditions = self.create_lagged_quarter_conditions(self.returns_series, window_size, quarter_length)
        # Ensure conditions align with rolling returns: discard the first few windows if necessary.
        min_length = min(len(self.rolling_returns), len(self.conditions))
        self.rolling_returns = self.rolling_returns[-min_length:]
        self.conditions = self.conditions[-min_length:].reshape(-1, 1)
        
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
        
        Parameters:
          returns_series: pandas Series of daily returns.
          window_size: number of days per window.
        
        Returns:
          rolling_returns: numpy array of shape (n_windows, window_size)
          scaler: fitted StandardScaler (applied to individual returns).
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
        Automatically create conditions using the cumulative return over the quarter immediately preceding each rolling window.
        
        For a window starting at index i, condition = cumulative return over returns_series[i - quarter_length : i].
        Only windows where i >= quarter_length are considered.
        
        Returns:
          conditions: numpy array of shape (n_conditions, 1)
        """
        conditions = []
        # Ensure returns_series is a pandas Series.
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        
        # Only start from index = quarter_length so that there's a full quarter of data before.
        n = len(returns_series)
        for i in range(quarter_length, n - window_size + 1):
            window = returns_series.iloc[i - quarter_length:i]
            cum_return = np.prod(1 + window) - 1
            conditions.append(cum_return)
        return np.array(conditions)

    def setup(self):
        # Combine rolling returns with their corresponding condition data.
        dataset = TensorDataset(torch.tensor(self.rolling_returns, dtype=torch.float32),
                                 torch.tensor(self.conditions, dtype=torch.float32))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    def train(self):
        self.setup()
        for epoch in range(self.n_epochs):
            for i, (real_returns, cond) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                valid = torch.ones(batch_size, 1).to(real_returns.device)
                fake = torch.zeros(batch_size, 1).to(real_returns.device)
                real_returns = real_returns.to(real_returns.device)
                cond = cond.to(real_returns.device)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z, cond)
                d_loss = -torch.mean(self.discriminator(real_returns, cond)) + torch.mean(self.discriminator(gen_returns.detach(), cond))
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator every 3 steps.
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z, cond)
                    g_loss = -torch.mean(self.discriminator(gen_returns, cond))
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    def generate_scenarios(self, save=True, num_scenarios=50000):
        """
        Generate scenarios using the trained generator.
        For a static VaR, we use the mean of the conditions.
        
        Returns:
          all_generated_returns: NumPy array of generated 252-day return sequences.
        """
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000
        device = 'cuda' if self.cuda else 'cpu'
        # Use the mean condition from training data.
        cond_value = torch.tensor(self.conditions.mean(axis=0), dtype=torch.float32, device=device)
        cond_value = cond_value.unsqueeze(0).repeat(batch_size, 1)  # shape: (batch_size, cond_dim)
        
        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_returns = self.generator(z, cond_value).cpu().numpy()
                gen_returns = self.scaler.inverse_transform(gen_returns)
                all_generated_returns.append(gen_returns)
        all_generated_returns = np.vstack(all_generated_returns)
        
        save_dir = "generated_CGAN_output_test"

        if save:
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
            nn.Dropout(0.9),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.9),
            nn.Linear(1000, 1)
        )

    def forward(self, returns, condition):
        x = returns.view(returns.size(0), -1)
        x = torch.cat((x, condition), dim=1)
        validity = self.model(x)
        return validity