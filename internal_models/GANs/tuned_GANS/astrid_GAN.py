import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler


class AstridGAN:
    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        
        dir_path = os.path.join('generated_GAN_output', f"generated_returns_{self.asset_name}")
        os.makedirs(dir_path, exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
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

        # Ensure that our dataset has at least 2 samples.
        rolling_returns = self.rolling_returns
        if rolling_returns.shape[0] < self.opt.batch_size:
            # Repeat the data so that we have at least opt.batch_size samples
            factor = self.opt.batch_size // rolling_returns.shape[0] + 1
            rolling_returns = np.repeat(rolling_returns, factor, axis=0)
        rolling_returns_tensor = torch.tensor(rolling_returns, dtype=torch.float32)
        self.dataloader = DataLoader(TensorDataset(rolling_returns_tensor), batch_size=self.opt.batch_size, shuffle=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999))

    def train(self):
        self.setup()
        for epoch in range(self.opt.n_epochs):
            for i, (real_returns,) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                valid = torch.ones(batch_size, 1).to(real_returns.device)
                fake = torch.zeros(batch_size, 1).to(real_returns.device)
                real_returns = real_returns.to(real_returns.device)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z)
                d_loss = -torch.mean(self.discriminator(real_returns)) + torch.mean(self.discriminator(gen_returns.detach()))
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator multiple times per discriminator step
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
        """
        # Scale the new return using the existing scaler
        new_return_scaled = self.scaler.transform(np.array([[new_return]]))
        # Append the scaled new return to the accumulated list
        self.accumulated_online_returns.append(new_return_scaled)
        
        # Update the current window by dropping the oldest value and appending the new return
        self.current_window = np.concatenate((self.current_window[1:], new_return_scaled.reshape(1, 1)), axis=0)
        
        # If we have not yet accumulated 252 new returns, perform online fine-tuning using current_window
        if len(self.accumulated_online_returns) <= self.opt.window_size:
            self.generator.train()
            self.discriminator.train()
            
            # Create a temporary tensor for the current window and ensure batch size > 1
            window_tensor = torch.tensor(self.current_window, dtype=torch.float32).view(1, self.opt.window_size)
            window_tensor = window_tensor.to('cuda' if self.cuda else 'cpu')
            if window_tensor.size(0) < 2:
                window_tensor = window_tensor.repeat(2, 1)  # replicate to avoid BatchNorm issues
            
            online_epochs = 5  # adjust the number of fine-tuning epochs as needed
            for epoch in range(online_epochs):
                batch_size = window_tensor.size(0)
                # Train Discriminator
                self.optimizer_D.zero_grad()
                valid = torch.ones(batch_size, 1).to(window_tensor.device)
                fake = torch.zeros(batch_size, 1).to(window_tensor.device)
                real_returns = window_tensor
                z = torch.randn(batch_size, self.opt.latent_dim).to(window_tensor.device)
                gen_returns = self.generator(z)
                d_loss = -torch.mean(self.discriminator(real_returns)) + torch.mean(self.discriminator(gen_returns.detach()))
                d_loss.backward()
                self.optimizer_D.step()
                
                # Train Generator
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.opt.latent_dim).to(window_tensor.device)
                gen_returns = self.generator(z)
                g_loss = -torch.mean(self.discriminator(gen_returns))
                g_loss.backward()
                self.optimizer_G.step()
            print(f"{self.asset_name}: Online fine-tuning update completed for new return {new_return}.")
        else:
            # Once we've accumulated at least 252 new returns, create multiple rolling windows
            new_data = np.array(self.accumulated_online_returns)  # shape: (N, 1) with N >= 252
            new_rolling_returns = []
            for i in range(len(new_data) - self.opt.window_size + 1):
                window = new_data[i:i+self.opt.window_size]
                new_rolling_returns.append(window)
            self.rolling_returns = np.array(new_rolling_returns)
            print(f"{self.asset_name}: Performing full retraining using {len(new_rolling_returns)} rolling windows.")
            self.train()  # Full retraining on the updated rolling_returns
            # Reset the accumulated online returns buffer and update current_window
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

            # Save the tensor in the specified directory
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
