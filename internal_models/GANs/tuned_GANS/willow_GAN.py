import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer

# SPECIALIZED FOR EONIA
class WillowGAN:
    def __init__(self, returns_df, asset_name, lambda_decay=0.99):
        self.returns_df = returns_df
        self.asset_name = asset_name
        self.lambda_decay = lambda_decay  # Stronger recency bias
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs")
        parser.add_argument("--batch_size", type=int, default=128, help="batch size")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate discriminator")
        parser.add_argument("--latent_dim", type=int, default=2000, help="latent space size")
        parser.add_argument("--window_size", type=int, default=252, help="rolling window")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt
        self.rolling_returns, self.labels, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(opt, self.input_shape)
        self.discriminator = Discriminator(self.input_shape)

    def sample_latent(self, batch_size, latent_dim):
        """ Biased latent space towards recent returns """
        recent_returns = self.returns_df.values[-latent_dim:]
        noise = torch.randn(batch_size, latent_dim, dtype=torch.float32)
        bias = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        return 0.7 * noise + 0.3 * bias  # Shift noise towards recent trend

    def create_rolling_returns(self, returns_df):
        """ Apply nonlinear scaling and detect jumps using EVT. """
        window_size = self.opt.window_size
        scaler = PowerTransformer()  # Nonlinear transformation
        scaled_returns = scaler.fit_transform(returns_df.values.reshape(-1, 1))

        rolling_returns, trends = [], []
        for i in range(len(scaled_returns) - window_size):
            window = scaled_returns[i:i + window_size]
            rolling_returns.append(window)
            trend = np.polyfit(np.arange(window_size), window.flatten(), 1)[0]
            trends.append(trend)

        # Extreme Value Theory (EVT) for jump detection
        threshold = np.percentile(trends, 95)  # Top 5% is a jump
        labels = [(0 if t > threshold else 1 if t < -threshold else 2) for t in trends]

        return np.array(rolling_returns), np.array(labels), scaler

    def train(self):
        self.setup()
        for epoch in range(self.opt.n_epochs):
            for i, (real_returns, labels) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns, labels = real_returns.to('cuda' if self.cuda else 'cpu'), labels.to('cuda' if self.cuda else 'cpu')

                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = self.sample_latent(batch_size, self.opt.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z, labels)

                d_loss = -torch.mean(self.discriminator(real_returns, labels)) + torch.mean(self.discriminator(gen_returns.detach(), labels))
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator with Recency Loss
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z, labels)
                    g_loss = -torch.mean(self.discriminator(gen_returns, labels))
                    recency_loss = F.mse_loss(gen_returns.mean(dim=1), real_returns.mean(dim=1))  # Match recent mean trend
                    total_loss = g_loss + 0.1 * recency_loss  # Weighted loss
                    total_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Recency Loss: {recency_loss.item()}]")

    def setup(self):
        """ Set up dataloader with strong recency bias. """
        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        
        time_indices = np.arange(len(self.rolling_returns))
        sample_weights = np.exp(-self.lambda_decay * time_indices[::-1])  # More emphasis on recent data
        sample_weights /= np.sum(sample_weights)

        sampler = WeightedRandomSampler(sample_weights, len(self.rolling_returns), replacement=True)

        self.dataloader = DataLoader(
            TensorDataset(rolling_returns_tensor, labels_tensor),
            batch_size=self.opt.batch_size,
            sampler=sampler
        )

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999))

    def generate_scenarios(self, num_scenarios=50000):
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.opt.latent_dim).to('cuda' if self.cuda else 'cpu')
                labels = torch.randint(0, 3, (batch_size,)).to('cuda' if self.cuda else 'cpu')

                gen_returns = self.generator(z, labels).cpu().numpy()

                # Print debugging info BEFORE transformation
                print(f"Generated Returns (first 10 values): {gen_returns.flatten()[:10]}")
                print(f"Contains NaN before transformation: {np.isnan(gen_returns).any()}")

                # Clip extreme values before inverse transformation
                gen_returns = np.clip(gen_returns, -1e3, 1e3)

                # Print debugging info AFTER clipping
                print(f"Generated Returns after Clipping (first 10 values): {gen_returns.flatten()[:10]}")
                print(f"Contains NaN after clipping: {np.isnan(gen_returns).any()}")

                try:
                    gen_returns = self.scaler.inverse_transform(gen_returns.reshape(-1, 1)).reshape(batch_size, -1)
                except ValueError:
                    print("Warning: Inverse transform failed due to extreme values")
                    gen_returns = np.nan_to_num(gen_returns, nan=np.median(self.returns_df.values))

                # Print debugging info AFTER inverse transform
                print(f"Generated Returns after Inverse Transform (first 10 values): {gen_returns.flatten()[:10]}")
                print(f"Contains NaN after inverse transform: {np.isnan(gen_returns).any()}")

                all_generated_returns.append(gen_returns)

        all_generated_returns = np.vstack(all_generated_returns)
        torch.save(torch.tensor(all_generated_returns), f'generated_returns_{self.asset_name}/final_scenarios.pt')




class Generator(nn.Module):
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size=opt.latent_dim, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, input_shape[0])
    
    def forward(self, noise, labels):
        noise = noise.unsqueeze(1)
        lstm_output, _ = self.lstm(noise)
        return self.fc(lstm_output.squeeze(1))

'''
class Generator(nn.Module):
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.opt = opt
        self.input_shape = input_shape

        
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, int(np.prod(input_shape)))
        )
        
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
        
    def forward(self, noise):
        returns = self.model(noise)
        return returns.view(returns.size(0), *self.input_shape)
        '''
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.label_embedding = nn.Embedding(3, 10)  # 3 regimes mapped to 10-dim embedding

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)) + 10, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.8),
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.8),
            nn.Linear(500, 500),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.8),
            nn.Linear(500, 1)
        )

    def forward(self, returns, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat((returns.view(returns.size(0), -1), label_embedding), dim=1)
        return self.model(x)
