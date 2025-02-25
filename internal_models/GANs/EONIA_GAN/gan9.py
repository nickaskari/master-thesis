import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler


class GAN9:
    def __init__(self, returns_df, asset_name, lambda_decay=0.1):
        self.returns_df = returns_df
        self.asset_name = asset_name
        self.lambda_decay = lambda_decay  # Decay rate for recency bias
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=1800, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate for generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate for discriminator")
        parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
        parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt
        self.rolling_returns, self.labels, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = Generator(opt, self.input_shape)
        self.discriminator = Discriminator(self.input_shape)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None

    def create_rolling_returns(self, returns_df):
        """ Prepares rolling return windows and assigns regime labels using percentile-based thresholds. """
        window_size = self.opt.window_size
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_df.values.reshape(-1, 1))

        rolling_returns, trends = [], []
        
        # Step 1: Compute trends first
        for i in range(len(scaled_returns) - window_size):
            window = scaled_returns[i:i + window_size]
            rolling_returns.append(window)
            trend = np.polyfit(np.arange(window_size), window.flatten(), 1)[0]
            trends.append(trend)

        # Step 2: Compute dynamic thresholds (percentiles)
        rising_threshold = np.percentile(trends, 99) 
        falling_threshold = np.percentile(trends, 1) 
        
        labels = []
        for trend in trends:
            if trend > rising_threshold:
                labels.append(0)  # Rising
            elif trend < falling_threshold:
                labels.append(1)  # Falling
            else:
                labels.append(2)  # Stable

        return np.array(rolling_returns), np.array(labels), scaler


    def setup(self):
        """ Sets up the data loader with weighted sampling for recency bias. """
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        rolling_returns_tensor = torch.tensor(self.rolling_returns, dtype=torch.float32)
        labels_tensor = torch.tensor(self.labels, dtype=torch.long)

        # Apply exponentially decayed sampling for recency bias
        num_samples = len(self.rolling_returns)
        time_indices = np.arange(num_samples)  # 0 = most recent, num_samples-1 = oldest
        sample_weights = np.exp(-self.lambda_decay * time_indices[::-1])  # Reverse order
        sample_weights /= np.sum(sample_weights)  # Normalize to sum to 1

        sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)

        self.dataloader = DataLoader(
            TensorDataset(rolling_returns_tensor, labels_tensor),
            batch_size=self.opt.batch_size,
            sampler=sampler  # Uses weighted sampling
        )

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr_d, betas=(0.5, 0.999))

    def train(self):
        """ Training loop for Regime-Switching GAN with Recency Bias. """
        self.setup()
        for epoch in range(self.opt.n_epochs):
            for i, (real_returns, labels) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                valid = torch.ones(batch_size, 1).to(real_returns.device)
                fake = torch.zeros(batch_size, 1).to(real_returns.device)
                real_returns = real_returns.to(real_returns.device)
                labels = labels.to(real_returns.device)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.opt.latent_dim).to(real_returns.device)
                gen_returns = self.generator(z, labels)

                d_loss = -torch.mean(self.discriminator(real_returns, labels)) + torch.mean(self.discriminator(gen_returns.detach(), labels))
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator multiple times per discriminator step
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z, labels)
                    g_loss = -torch.mean(self.discriminator(gen_returns, labels))
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def generate_scenarios(self, num_scenarios=50000):
        """ Generates new return sequences based on trained model. """
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.opt.latent_dim).to('cuda' if self.cuda else 'cpu')

                # Generate samples for each regime
                labels = torch.randint(0, 3, (batch_size,)).to('cuda' if self.cuda else 'cpu')
                gen_returns = self.generator(z, labels).cpu().numpy()
                gen_returns = self.scaler.inverse_transform(gen_returns)
                all_generated_returns.append(gen_returns)

        all_generated_returns = np.vstack(all_generated_returns)
        torch.save(torch.tensor(all_generated_returns), f'generated_returns_{self.asset_name}/final_scenarios.pt')



class Generator(nn.Module):
    # LSTM GENERATOR
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size=opt.latent_dim + 10, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, input_shape[0])
        self.label_embedding = nn.Embedding(3, 10)
    
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(1)
        x = torch.cat((noise, label_embedding), dim=1)
        returns = self.model(x)
        noise = noise.unsqueeze(1)
        returns = returns.unsqueeze(-1)  # Expands last dimension to match label embedding
        lstm_input = torch.cat((returns, label_embedding), dim=2)
        lstm_output, _ = self.lstm(lstm_input)
        return self.fc(lstm_output.squeeze(1))


'''
class Generator(nn.Module):
    """ Generator for Regime-Switching GAN. """
    def __init__(self, opt, input_shape):
        super(Generator, self).__init__()
        self.opt = opt
        self.input_shape = input_shape
        self.label_embedding = nn.Embedding(3, 10)  # 3 regimes, embedding into 10 dimensions

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim + 10, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, int(np.prod(input_shape)))
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat((noise, label_embedding), dim=1)
        returns = self.model(x)
        return returns.view(returns.size(0), *self.input_shape)

'''
class Discriminator(nn.Module):
    """ Discriminator for Regime-Switching GAN. """
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.label_embedding = nn.Embedding(3, 10)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)) + 10, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1000, 1)
        )

    def forward(self, returns, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat((returns.view(returns.size(0), -1), label_embedding), dim=1)
        validity = self.model(x)
        return validity
