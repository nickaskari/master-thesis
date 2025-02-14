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

class WGAN_GP:

    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.00002, help="learning rate for RMSprop")
        parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
        parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval between sampling generated return sequences")
        parser.add_argument("--lambda_gp", type=float, default=20, help="gradient penalty coefficient")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt

        if isinstance(returns_df, pd.Series):
            returns_df = returns_df.to_frame()
        self.rolling_returns, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(opt, self.input_shape)
        self.critic = Critic(self.input_shape)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_C = None

    def create_rolling_returns(self, returns_df):
        window_size = self.opt.window_size
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_df.values.reshape(-1, 1))

        rolling_returns = []
        for i in range(len(scaled_returns) - window_size):
            window = scaled_returns[i:i + window_size]
            rolling_returns.append(window)
        return np.array(rolling_returns), scaler

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1).to(real_samples.device)
        
        # Reshape fake_samples to match real_samples
        fake_samples = fake_samples.view(fake_samples.size(0), real_samples.size(1), -1)

        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.critic(interpolates)
        
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(d_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generate_scenarios(self, num_scenarios=10000):
        self.generator.eval()  # Set generator to evaluation mode

        all_generated_returns = []
        batch_size = 1000  # Generate in batches to avoid memory issues

        with torch.no_grad():
            for _ in range(max(1, num_scenarios // batch_size)):  # Ensure loop runs at least once
                z = torch.normal(0, 0.02, size=(batch_size, self.opt.latent_dim)).to('cuda' if self.cuda else 'cpu')
                gen_returns = self.generator(z).cpu().numpy()

                # Debugging: Print generated returns shape
                print(f"Generated batch shape: {gen_returns.shape}")

                if gen_returns.size == 0:
                    print("Warning: Generated returns are empty!")
                    continue

                # Check if scaler is fitted before inverse transform
                if not hasattr(self.scaler, "mean_"):
                    raise ValueError("Scaler was not fitted! Check data preprocessing.")

                gen_returns = self.scaler.inverse_transform(gen_returns.reshape(gen_returns.shape[0], -1))
                all_generated_returns.append(gen_returns)

        # Debugging: Check list content
        if len(all_generated_returns) == 0:
            raise ValueError("No valid scenarios were generated! Check GAN training or batch size.")

        # Concatenate all batches into a single array
        all_generated_returns = np.vstack(all_generated_returns)

        # Convert to PyTorch tensor before saving
        all_generated_returns = torch.tensor(all_generated_returns, dtype=torch.float32)
        torch.save(all_generated_returns, f'generated_returns_{self.asset_name}/final_scenarios.pt')
        print(f"Saved {all_generated_returns.shape[0]} scenarios for {self.asset_name}.")



    def setup(self):
        if self.cuda:
            self.generator.cuda()
            self.critic.cuda()

        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
        self.dataloader = DataLoader(TensorDataset(rolling_returns_tensor), batch_size=self.opt.batch_size, shuffle=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))

    def train(self):
        self.setup()

        n_critic = 5  # Reduce to balance generator updates
        for epoch in range(self.opt.n_epochs):
            for i, (real_returns,) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)

                # Train Critic
                self.optimizer_C.zero_grad()
                z = torch.normal(0, 1.0, size=(batch_size, self.opt.latent_dim)).to(real_returns.device)
                gen_returns = self.generator(z)
                real_validity = self.critic(real_returns)
                fake_validity = self.critic(gen_returns.detach())
                gradient_penalty = self.compute_gradient_penalty(real_returns, gen_returns)
                c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.opt.lambda_gp * gradient_penalty

                c_loss.backward()
                self.optimizer_C.step()

                # Train Generator every n_critic steps
                if i % n_critic == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z)
                    fake_validity = self.critic(gen_returns)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [C loss: {c_loss.item()}] [G loss: {g_loss.item()}]")

                if (epoch * len(self.dataloader) + i) % self.opt.sample_interval == 0:
                    torch.save(gen_returns, f"generated_returns_{self.asset_name}/returns_{epoch}_{i}.pt")

class Generator(nn.Module):
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.opt = opt
        self.input_shape = input_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
          # if normalize:
           #     layers.append(nn.BatchNorm1d(out_feat, 0.8))  # Normalize for stability
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # Changed from ReLU to LeakyReLU
            return layers

        self.model = nn.Sequential(
            *block(self.opt.latent_dim, 512),
            *block(512, 512),
            *block(512, 512),
            *block(512, 512),
            nn.Linear(512, int(np.prod(self.input_shape))),
            nn.Tanh()  # Helps with stable value scaling
        )

    def forward(self, noise):
        returns = self.model(noise)
        return returns.view(returns.size(0), *self.input_shape)



class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.input_shape = input_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.input_shape)), 1024),  # Increased layer size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # No Sigmoid for Wasserstein loss
        )

    def forward(self, returns):
        validity = self.model(returns.view(returns.size(0), -1))
        return validity
