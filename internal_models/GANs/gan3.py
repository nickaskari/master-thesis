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


def minibatch_discrimination(x, num_kernels=50, kernel_dim=2):  # Reduce to 50*2 = 100 features
    batch_size, feature_dim = x.shape
    
    T = nn.Linear(feature_dim, num_kernels * kernel_dim, bias=False).to(x.device)
    M = T(x).view(batch_size, num_kernels, kernel_dim)
    
    M_exp = M.unsqueeze(0).expand(batch_size, batch_size, num_kernels, kernel_dim)
    M_diff = torch.abs(M_exp - M_exp.transpose(0, 1))
    
    return torch.sum(torch.exp(-M_diff), dim=3).sum(dim=2)  # Output: (batch_size, 100)




class GAN3:
    def __init__(self, returns_df, asset_name):
        self.returns_df = returns_df
        self.asset_name = asset_name
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate for generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate for discriminator")
        parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
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

        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
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

                #real_loss = F.binary_cross_entropy_with_logits(self.discriminator(real_returns), valid)
                #fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(gen_returns.detach()), fake)
                #d_loss = (real_loss + fake_loss) / 2
                d_loss = -torch.mean(self.discriminator(real_returns)) + torch.mean(self.discriminator(gen_returns.detach()))


                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator multiple times per discriminator step
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z)
                    #g_loss = F.binary_cross_entropy_with_logits(self.discriminator(gen_returns), valid)
                    g_loss = -torch.mean(self.discriminator(gen_returns))
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

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
        '''
        New Generator stuff
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, int(np.prod(input_shape)))
        )
        '''


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
            nn.Dropout(0.8),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.8),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.8),
            nn.Linear(1000, 1)  # Single output neuron for real/fake classification
        )

    def forward(self, returns):
        validity = self.model(returns.view(returns.size(0), -1))
        return validity
