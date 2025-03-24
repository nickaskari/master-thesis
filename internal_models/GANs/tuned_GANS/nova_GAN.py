import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import pandas as pd

class NovaGAN:
    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        
        dir_path = os.path.join('generated_GAN_output', f"generated_returns_{self.asset_name}")
        os.makedirs(dir_path, exist_ok=True)

        # otherwise assume it's already a pandas Series.
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]
        else:
            self.returns_series = returns_df

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate for generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate for discriminator")
        parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
        parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval between sampling generated return sequences")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt
        self.rolling_returns, self.scaler = self.create_rolling_returns(returns_df)
        self.conditions = self.create_lagged_quarter_conditions(returns_df, self.opt.window_size, quarter_length=63)
        # Align the lengths if necessary.
        min_length = min(len(self.rolling_returns), len(self.conditions))
        self.rolling_returns = self.rolling_returns[-min_length:]
        self.conditions = self.conditions[-min_length:].reshape(-1, 1)
        self.cond_dim = self.conditions.shape[1]
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(opt, self.input_shape, self.cond_dim)
        self.discriminator = Discriminator(self.input_shape, self.cond_dim)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None

        # For online training: initialize current window and an accumulator for new returns.
        self.current_window = self.rolling_returns[-1].copy()  # shape: (window_size, 1)
        self.accumulated_online_returns = []  # will store scaled new returns

    def create_rolling_returns(self, returns_df):
        window_size = self.opt.window_size
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_df.values.reshape(-1, 1))
        
        rolling_returns = []
        for i in range(len(scaled_returns) - window_size):
            window = scaled_returns[i:i + window_size]
            rolling_returns.append(window)
        return np.array(rolling_returns), scaler
    
    def create_lagged_quarter_conditions(self, returns_df, window_size, quarter_length=63):
        conditions = []
        # Make sure returns_df is a pandas Series.
        if not isinstance(returns_df, pd.Series):
            returns_df = pd.Series(returns_df)
        
        # Only compute conditions where there is enough historical data.
        n = len(returns_df)
        for i in range(quarter_length, n - window_size + 1):
            window = returns_df.iloc[i - quarter_length:i]
            cum_return = np.prod(1 + window) - 1
            conditions.append(cum_return)
        return np.array(conditions)


    def setup(self):
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        # Use the rolling_returns array from training.
        rolling_returns = self.rolling_returns  # shape: (N, window_size, 1)
        N = rolling_returns.shape[0]
        
        # Build weights for full retraining: a linear ramp from 0.5 (oldest) to 1.5 (newest)
        weights_array = np.linspace(0.5, 1.5, N)
        weights_tensor = torch.tensor(weights_array, dtype=torch.float32)

        # If there are fewer than batch_size samples, repeat them.
        if N < self.opt.batch_size:
            factor = self.opt.batch_size // N + 1
            rolling_returns = np.repeat(rolling_returns, factor, axis=0)
            N = rolling_returns.shape[0]
            weights_array = np.tile(weights_array, factor)[:N]
            weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
            
        rolling_returns_tensor = torch.tensor(rolling_returns, dtype=torch.float32)
        
        # Create WeightedRandomSampler for full retraining.
        sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=N, replacement=True)
        conditions_tensor = torch.tensor(self.conditions, dtype=torch.float32)
        dataset = TensorDataset(rolling_returns_tensor, conditions_tensor)
        self.dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, sampler=sampler, drop_last=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999))

    def train(self):
        self.setup()
        for epoch in range(self.opt.n_epochs):
            for i, (real_returns, cond) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns = real_returns.to(real_returns.device)
                cond = cond.to(real_returns.device)
                
                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z, cond)
                d_loss = -torch.mean(self.discriminator(real_returns, cond)) + torch.mean(self.discriminator(gen_returns.detach(), cond))
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator every 3 batches.
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                    gen_returns = self.generator(z, cond)
                    g_loss = -torch.mean(self.discriminator(gen_returns, cond))
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def compute_condition_for_window(self, window):
        """
        Compute the condition for the current window.
        For example, use the cumulative return over the last quarter (63 days).
        Assumes window is a 2D array of shape (window_size, 1).
        Returns a 1D NumPy array of length equal to the condition dimension.
        """
        quarter_length = 63
        if window.shape[0] < quarter_length:
            # Not enough data; return a default condition (e.g. zero).
            return np.array([0.0])
        else:
            # Use the last quarter of returns.
            recent_returns = window[-quarter_length:]
            cum_return = np.prod(1 + recent_returns) - 1
            return np.array([cum_return])
            
    def get_updated_returns_series(self):
        """
        Update the returns series by appending raw new returns from online training.
        Assumes self.returns_series is a pandas Series containing the historical (raw) returns.
        Each accumulated online return in self.accumulated_online_returns is stored in scaled form,
        so we first convert it back to raw values using the scaler.
        """
        # Convert each accumulated scaled return back to a raw return.
        raw_new_returns = [self.scaler.inverse_transform(x)[0, 0] for x in self.accumulated_online_returns]
        # Create a new pandas Series for these new returns.
        new_returns_series = pd.Series(raw_new_returns)
        # Append the new returns to the historical series.
        updated_series = pd.concat([self.returns_series, new_returns_series], ignore_index=True)
        return updated_series



    def online_training(self, new_return):
        """
        Perform an online update with the new daily return in a conditional setting.
        This method updates the current window, computes the corresponding condition,
        and then performs online fine-tuning (or full retraining once enough new returns are accumulated).
        """
        # Scale the new return using the existing scaler.
        new_return_scaled = self.scaler.transform(np.array([[new_return]]))
        self.accumulated_online_returns.append(new_return_scaled)
        
        # Update current window: drop the oldest value and append the new return.
        self.current_window = np.concatenate(
            (self.current_window[1:], new_return_scaled.reshape(1, 1)), axis=0
        )
        
        # Compute the condition for the current window.
        current_condition = self.compute_condition_for_window(self.current_window)
        
        # Determine device.
        device = 'cuda' if self.cuda else 'cpu'
        
        # Fine-tuning phase: fewer than window_size new returns.
        if len(self.accumulated_online_returns) < self.opt.window_size:
            self.generator.train()
            self.discriminator.train()
            
            # Replicate the current window to create a dataset of length window_size.
            # current_window is assumed to have shape (window_size, 1).
            replicated_window = np.repeat(self.current_window[np.newaxis, :, :], self.opt.window_size, axis=0)
            # Squeeze last dimension to get shape: (window_size, window_size)
            window_tensor = torch.tensor(replicated_window.squeeze(-1), dtype=torch.float32).to(device)
            
            # Replicate the current condition to match the number of samples.
            # current_condition is assumed to be 1D (e.g., shape: (cond_dim,))
            cond_tensor = (
                torch.tensor(current_condition, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(self.opt.window_size, 1)
                .to(device)
            )
            
            # Build weights for the window: a linear ramp from 1 (oldest) to 2 (newest).
            weights_array = np.linspace(1, 2, self.opt.window_size)
            weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
            
            # Create a dataset that returns (window, condition) pairs.
            from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader
            dataset = TensorDataset(window_tensor, cond_tensor)
            sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=self.opt.window_size, replacement=True)
            fine_tune_loader = DataLoader(dataset, batch_size=self.opt.batch_size, sampler=sampler)
            
            online_epochs = 2  # Adjust epochs as needed.
            for epoch in range(online_epochs):
                for batch_window, batch_cond in fine_tune_loader:
                    batch_size = batch_window.size(0)
                    self.optimizer_D.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(device)
                    # Now pass both noise and condition to the generator.
                    gen_returns = self.generator(z, batch_cond)
                    d_loss = -torch.mean(self.discriminator(batch_window, batch_cond)) \
                            + torch.mean(self.discriminator(gen_returns.detach(), batch_cond))
                    d_loss.backward()
                    self.optimizer_D.step()
                    
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(device)
                    gen_returns = self.generator(z, batch_cond)
                    g_loss = -torch.mean(self.discriminator(gen_returns, batch_cond))
                    g_loss.backward()
                    self.optimizer_G.step()
            print(f"{self.asset_name}: Weighted online fine-tuning update completed for new return {new_return}.")
        
        else:
            updated_series = self.get_updated_returns_series()  
            
            # Recompute rolling returns and update the scaler.
            self.rolling_returns, self.scaler = self.create_rolling_returns(updated_series)
            
            # Recompute conditions based on the updated returns series.
            self.conditions = self.create_lagged_quarter_conditions(updated_series, self.opt.window_size, quarter_length=63)
            
            # Align the lengths of rolling_returns and conditions.
            min_length = min(len(self.rolling_returns), len(self.conditions))
            self.rolling_returns = self.rolling_returns[-min_length:]
            self.conditions = self.conditions[-min_length:].reshape(-1, 1)
            
            print(f"{self.asset_name}: Performing full retraining using {min_length} rolling windows.")
            self.train()  # Full retraining on the updated rolling_returns and conditions.
            
            # Reset the online accumulator and update current_window.
            self.accumulated_online_returns = []
            self.current_window = self.rolling_returns[-1].copy().squeeze(-1)


    def generate_scenarios(self, save=True, num_scenarios=50000):
            self.generator.eval()
            all_generated_returns = []
            batch_size = 1000
            device = 'cuda' if self.cuda else 'cpu'
            
            # For example, use the mean condition from training data.
            # You could also allow this to be a parameter.
            cond_value = torch.tensor(self.conditions.mean(axis=0), dtype=torch.float32, device=device)
            cond_value = cond_value.unsqueeze(0).repeat(batch_size, 1)  # shape: (batch_size, cond_dim)
            
            with torch.no_grad():
                for _ in range(num_scenarios // batch_size):
                    z = torch.randn(batch_size, self.opt.latent_dim).to(device)
                    # Note: pass both noise and condition to the generator.
                    gen_returns = self.generator(z, cond_value).cpu().numpy()
                    gen_returns = self.scaler.inverse_transform(gen_returns)
                    all_generated_returns.append(gen_returns)
            all_generated_returns = np.vstack(all_generated_returns)
            
            save_dir = "generated_GAN_output"
            if save:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_final_scenarios.pt')
                torch.save(torch.tensor(all_generated_returns), save_path)
                print(f"Generated scenarios saved to: {save_path}")
            
            return all_generated_returns


class Generator(nn.Module):
    def __init__(self, opt, input_shape, cond_dim):
        super(Generator, self).__init__()
        self.opt = opt
        self.input_shape = input_shape
        self.cond_dim = cond_dim
        # Increase input dimension: latent_dim + cond_dim
        input_dim = opt.latent_dim + cond_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(input_shape)))
        )

    def forward(self, noise, condition):
        # Concatenate the noise and condition along the feature dimension.
        x = torch.cat((noise, condition), dim=1)
        returns = self.model(x)
        return returns.view(returns.size(0), *self.input_shape)

    

class Discriminator(nn.Module):
    def __init__(self, input_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.cond_dim = cond_dim
        # Increase input dimension: flattened returns + cond_dim
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
            nn.Linear(1000, 1)
        )

    def forward(self, returns, condition):
        x = returns.view(returns.size(0), -1)
        # Concatenate the flattened returns with the condition.
        x = torch.cat((x, condition), dim=1)
        validity = self.model(x)
        return validity

