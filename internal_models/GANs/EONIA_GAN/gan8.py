import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import PowerTransformer
from scipy.stats import norm, genpareto

class GAN8:
    def __init__(self, returns_df, asset_name, lambda_decay=0.3):
        self.returns_df = returns_df
        self.asset_name = asset_name
        self.lambda_decay = lambda_decay  # Stronger recency bias
        os.makedirs(f"generated_returns_{self.asset_name}", exist_ok=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs")
        parser.add_argument("--batch_size", type=int, default=128, help="batch size")
        parser.add_argument("--lr_g", type=float, default=0.0002, help="learning rate generator")
        parser.add_argument("--lr_d", type=float, default=0.00005, help="learning rate discriminator")
        parser.add_argument("--latent_dim", type=int, default=100, help="latent space size")
        parser.add_argument("--window_size", type=int, default=252, help="rolling window")
        
        opt, _ = parser.parse_known_args()
        self.opt = opt
        self.rolling_returns, self.labels, self.scaler = self.create_rolling_returns(returns_df)
        self.input_shape = (opt.window_size,)
        self.cuda = torch.cuda.is_available()

        self.generator = LSTMGenerator(opt, self.input_shape)
        self.discriminator = Discriminator(self.input_shape)

    def sample_latent(self, batch_size, latent_dim):
        """ Mixture of Gaussians latent space """
        if np.random.rand() < 0.3:
            return torch.tensor(genpareto.rvs(0.1, size=(batch_size, latent_dim)), dtype=torch.float32)
        else:
            return torch.randn(batch_size, latent_dim, dtype=torch.float32)

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

                # Train Generator
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z, labels)
                    g_loss = -torch.mean(self.discriminator(gen_returns, labels))
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

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
        """ Generate new return sequences based on the trained model. """
        self.generator.eval()
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = self.sample_latent(batch_size, self.opt.latent_dim).to('cuda' if self.cuda else 'cpu')
                labels = torch.randint(0, 3, (batch_size,)).to('cuda' if self.cuda else 'cpu')
                gen_returns = self.generator(z, labels).cpu().numpy()
                gen_returns = self.scaler.inverse_transform(gen_returns)
                all_generated_returns.append(gen_returns)

        all_generated_returns = np.vstack(all_generated_returns)
        torch.save(torch.tensor(all_generated_returns), f'generated_returns_{self.asset_name}/final_scenarios.pt')

class LSTMGenerator(nn.Module):
    def __init__(self, opt, input_shape):
        super(LSTMGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size=opt.latent_dim + 10, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, input_shape[0])
        self.label_embedding = nn.Embedding(3, 10)
    
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(1)
        noise = noise.unsqueeze(1)
        lstm_input = torch.cat((noise, label_embedding), dim=2)
        lstm_output, _ = self.lstm(lstm_input)
        return self.fc(lstm_output.squeeze(1))

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape[0], hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)
        self.label_embedding = nn.Embedding(3, 10)

    def forward(self, returns, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(1)
        lstm_input = torch.cat((returns.unsqueeze(1), label_embedding), dim=2)
        lstm_output, _ = self.lstm(lstm_input)
        return self.fc(lstm_output.squeeze(1))