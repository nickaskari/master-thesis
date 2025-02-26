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
from torch.nn.utils import spectral_norm

class SolveigGAN:
    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=1500, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate for generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate for discriminator")
        parser.add_argument("--latent_dim", type=int, default=1200, help="dimensionality of the latent space")
        parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
        parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty coefficient")

        opt, _ = parser.parse_known_args()
        self.opt = opt
        self.rolling_returns, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(opt, self.input_shape)
        self.critic = Critic(self.input_shape)  # Renamed from Discriminator to Critic

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
        batch_size = real_samples.size(0)
        
        # Ensure fake_samples matches real_samples' shape
        fake_samples = fake_samples.view_as(real_samples)  

        # Fix: Ensure alpha matches the shape of real_samples
        alpha = torch.rand(batch_size, 1, 1).to(real_samples.device)  
        alpha = alpha.expand_as(real_samples)  

        # Compute interpolates correctly
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        d_interpolates = self.critic(interpolates)

        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(d_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def setup(self):
        if self.cuda:
            self.generator.cuda()
            self.critic.cuda()

        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
        self.dataloader = DataLoader(TensorDataset(rolling_returns_tensor), batch_size=self.opt.batch_size, shuffle=True)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999))

    def train(self):
        self.setup()
        n_critic = 5  # Train the critic 5 times for each generator step

        for epoch in range(self.opt.n_epochs):
            for i, (real_returns,) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns = real_returns.to(real_returns.device)

                # Train Critic
                for _ in range(n_critic):
                    self.optimizer_C.zero_grad()
                    z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                    gen_returns = self.generator(z)

                    real_validity = self.critic(real_returns)
                    fake_validity = self.critic(gen_returns.detach())
                    gradient_penalty = self.compute_gradient_penalty(real_returns, gen_returns)

                    c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.opt.lambda_gp * gradient_penalty
                    c_loss.backward()
                    self.optimizer_C.step()

                # Train Generator every n_critic steps
                self.optimizer_G.zero_grad()
                gen_returns = self.generator(z)
                g_loss = -torch.mean(self.critic(gen_returns))
                g_loss.backward()
                self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [C loss: {c_loss.item()}] [G loss: {g_loss.item()}]")

    def generate_scenarios(self, num_scenarios=50000):
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
        torch.save(torch.tensor(all_generated_returns), f'generated_returns_{self.asset_name}/final_scenarios.pt')


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


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.input_shape = input_shape

        self.model = nn.Sequential(
            spectral_norm(nn.Linear(int(np.prod(input_shape)), 800)),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(800, 800)),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(800, 800)),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(800, 1))  # Output is a real number, no sigmoid
        )

    def forward(self, returns):
        validity = self.model(returns.view(returns.size(0), -1))
        return validity