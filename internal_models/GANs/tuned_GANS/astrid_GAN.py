import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

class AstridGAN:
    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        
        dir_path = os.path.join('generated_GAN_output', f"generated_returns_{self.asset_name}")
        os.makedirs(dir_path, exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate for generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate for discriminator")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval between sampling generated return sequences")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt
        self.rolling_returns, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(opt, self.input_shape)
        self.discriminator = Discriminator(self.input_shape)

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
        
        dataset = TensorDataset(rolling_returns_tensor)
        self.dataloader = DataLoader(dataset, batch_size=self.opt.batch_size, sampler=sampler, drop_last=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999))

    def train(self):
        self.setup()
        for epoch in range(self.opt.n_epochs):
            for i, (real_returns,) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns = real_returns.to(real_returns.device)
                
                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z)
                d_loss = -torch.mean(self.discriminator(real_returns)) + torch.mean(self.discriminator(gen_returns.detach()))
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator (update every 3 batches)
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                    gen_returns = self.generator(z)
                    g_loss = -torch.mean(self.discriminator(gen_returns))
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def online_training(self, new_return):
        """
        Perform an online update with the new daily return.
        Accumulate new returns until we have at least 252, then create multiple rolling windows
        from the accumulated data and trigger a full retraining.
        For fine-tuning, we further weight recent days in the current window.
        """
        # Scale the new return using the existing scaler.
        new_return_scaled = self.scaler.transform(np.array([[new_return]]))
        self.accumulated_online_returns.append(new_return_scaled)
        
        # Update current window: drop the oldest value and append the new return.
        self.current_window = np.concatenate((self.current_window[1:], new_return_scaled.reshape(1, 1)), axis=0)
        
        # If fewer than 252 new returns have been accumulated, perform online fine-tuning.
        if len(self.accumulated_online_returns) < self.opt.window_size:
            self.generator.train()
            self.discriminator.train()
            
            # Instead of a single sample, replicate the current window to create a dataset of length 252.
            # current_window has shape (252, 1); we replicate it along a new first dimension.
            replicated_window = np.repeat(self.current_window[np.newaxis, :, :], self.opt.window_size, axis=0)  # shape: (252, 252, 1)
            # Squeeze the last dimension to get shape: (252, 252)
            window_tensor = torch.tensor(replicated_window.squeeze(-1), dtype=torch.float32)  # Each sample is a 252-day sequence.
            window_tensor = window_tensor.to('cuda' if self.cuda else 'cpu')
            
            # Build weights for the 252-day window: a linear ramp from 1 (oldest) to 2 (newest).
            weights_array = np.linspace(1, 2, self.opt.window_size)
            weights_tensor = torch.tensor(weights_array, dtype=torch.float32)
            
            # Create a WeightedRandomSampler for fine-tuning.
            dataset = TensorDataset(window_tensor)
            sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=self.opt.window_size, replacement=True)
            fine_tune_loader = DataLoader(dataset, batch_size=self.opt.batch_size, sampler=sampler)
            
            online_epochs = 100  # Adjust as needed.
            for epoch in range(online_epochs):
                for (batch,) in fine_tune_loader:
                    batch_size = batch.size(0)
                    self.optimizer_D.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(batch.device)
                    gen_returns = self.generator(z)
                    d_loss = -torch.mean(self.discriminator(batch)) + torch.mean(self.discriminator(gen_returns.detach()))
                    d_loss.backward()
                    self.optimizer_D.step()
                    
                    self.optimizer_G.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(batch.device)
                    gen_returns = self.generator(z)
                    g_loss = -torch.mean(self.discriminator(gen_returns))
                    g_loss.backward()
                    self.optimizer_G.step()
            print(f"{self.asset_name}: Weighted online fine-tuning update completed for new return {new_return}.")
        else:
            # Full retraining: once at least 252 new returns have been accumulated, build new rolling windows.
            new_data = np.array(self.accumulated_online_returns)  # shape: (N, 1) with N >= 252.
            new_rolling_returns = []
            for i in range(len(new_data) - self.opt.window_size + 1):
                window = new_data[i:i+self.opt.window_size]
                new_rolling_returns.append(window)
            self.rolling_returns = np.array(new_rolling_returns)
            print(f"{self.asset_name}: Performing full retraining using {len(new_rolling_returns)} rolling windows.")
            self.train()  # Full retraining on the updated rolling_returns.
            # Reset the online accumulator and update current_window.
            self.accumulated_online_returns = []
            self.current_window = self.rolling_returns[-1].copy()

    def generate_scenarios(self, save=True, num_scenarios=50000):
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.opt.latent_dim).to('cuda' if self.cuda else 'cpu')
                gen_returns = self.generator(z).cpu().numpy()
                gen_returns = self.scaler.inverse_transform(gen_returns)
                all_generated_returns.append(gen_returns)

        all_generated_returns = np.vstack(all_generated_returns)
        save_dir = "generated_GAN_output"

        if save:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(torch.tensor(all_generated_returns), os.path.join(save_dir, f'generated_returns_{self.asset_name}_final_scenarios.pt'))

        return all_generated_returns


class Generator(nn.Module):
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.opt = opt
        self.input_shape = input_shape

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(input_shape)))
        )

    def forward(self, noise):
        returns = self.model(noise)
        return returns.view(returns.size(0), *self.input_shape)
    

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.9),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.9),
            nn.Linear(1000, 1)
        )

    def forward(self, returns):
        validity = self.model(returns.view(returns.size(0), -1))
        return validity
