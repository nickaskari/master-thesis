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

class BiGAN:
    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
        parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
        parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval between sampling generated return sequences")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt

        if isinstance(returns_df, pd.Series):
            returns_df = returns_df.to_frame()
        self.rolling_returns, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.encoder = Encoder(opt, self.input_shape)
        self.generator = Generator(opt, self.input_shape)
        self.discriminator = Discriminator(opt, self.input_shape)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.optimizer_E = None

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
            self.encoder.cuda()
            self.discriminator.cuda()

        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
        self.dataloader = DataLoader(TensorDataset(rolling_returns_tensor), batch_size=self.opt.batch_size, shuffle=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))
        self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))

    def train(self):
        self.setup()

        for epoch in range(self.opt.n_epochs):
            for i, (real_returns,) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z)
                encoded_real = self.encoder(real_returns)
                real_pair = self.discriminator(real_returns, encoded_real)
                fake_pair = self.discriminator(gen_returns.detach(), z)
                d_loss = -torch.mean(real_pair) + torch.mean(fake_pair)
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator & Encoder
                self.optimizer_G.zero_grad()
                self.optimizer_E.zero_grad()
                encoded_real = self.encoder(real_returns)
                fake_pair = self.discriminator(self.generator(z), z)
                real_pair = self.discriminator(real_returns, encoded_real)
                g_e_loss = -torch.mean(real_pair) - torch.mean(fake_pair)
                g_e_loss.backward()
                self.optimizer_G.step()
                self.optimizer_E.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G+E loss: {g_e_loss.item()}]")

    def generate_scenarios(self, num_scenarios=10000):
        self.generator.eval()  # Set generator to evaluation mode

        all_generated_returns = []
        batch_size = 1000  # Generate in batches to avoid memory issues

        with torch.no_grad():
            for _ in range(max(1, num_scenarios // batch_size)):  # Ensure loop runs at least once
                z = torch.normal(0, 0.02, size=(batch_size, self.opt.latent_dim)).to('cuda' if self.cuda else 'cpu')
                gen_returns = self.generator(z).cpu().numpy()  # Convert to NumPy

                print("Generated returns shape:", gen_returns.shape)  # Debugging print

                if gen_returns.size == 0:
                    print("Warning: Generated returns are empty!")
                    continue

                # Check if scaler is fitted before inverse transform
                if not hasattr(self.scaler, "mean_"):
                    raise ValueError("Scaler was not fitted! Check data preprocessing.")

                gen_returns_np = self.scaler.inverse_transform(gen_returns.reshape(gen_returns.shape[0], -1))
                all_generated_returns.append(gen_returns_np)  # Keep as NumPy array

        if not all_generated_returns:  # If still empty, throw an error
            raise ValueError("No generated returns were collected. Check generator training!")

        # Concatenate all batches into a single NumPy array before converting to tensor
        all_generated_returns = np.vstack(all_generated_returns)
        all_generated_returns = torch.tensor(all_generated_returns, dtype=torch.float32)  # Convert to PyTorch tensor

        # Save as a .pt file
        torch.save(all_generated_returns, f'generated_returns_{self.asset_name}/final_scenarios.pt')
        print(f"Saved {all_generated_returns.shape[0]} scenarios for {self.asset_name}.")


class Generator(nn.Module):
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, int(np.prod(input_shape)))
        )
        self.input_shape = input_shape
    def forward(self, noise):
        return self.model(noise).view(noise.size(0), *self.input_shape)

class Encoder(nn.Module):
    def __init__(self, opt, input_shape):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, opt.latent_dim)
        )
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class Discriminator(nn.Module):
    def __init__(self, opt, input_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)) + opt.latent_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
    def forward(self, x, z):
        combined = torch.cat((x.view(x.size(0), -1), z), dim=1)
        return self.model(combined)
