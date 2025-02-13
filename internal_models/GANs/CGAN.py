import argparse
import os
import numpy as np
import math
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# THE C IS MISSING GAN - THE GAN IS FUCKED

os.makedirs("generated_returns", exist_ok=True)
# THIS IS CURRENTLY NOT A CGAN, BUT A NORMAL GAN

# LOADING DATA SET FROM RETURNS_DF
returns_df = pd.read_csv('../../data/final_daily_returns_asset_classes.csv', index_col=0, parse_dates=True)

# ROLLING RETURNS
window_size = 252  

rolling_returns = []
for i in range(len(returns_df) - window_size):
    window = returns_df.iloc[i:i + window_size].values.flatten()  # Flatten to create a 1D vector
    rolling_returns.append(window)

rolling_returns = np.array(rolling_returns)



# DEFINING PARAMTERS

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_assets", type=int, default=7, help="number of assets in the portfolio")
parser.add_argument("--window_size", type=int, default=252, help="size of the rolling window in days (1 year)")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between sampling generated return sequences")
opt = parser.parse_args()
print(opt)

# Each sample is a 1-year return sequence for 7 assets
input_shape = (opt.window_size * opt.n_assets,)  # 252 * 7 = 1764 features

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Consists of 4 layers
        self.model = nn.Sequential(
            *block(opt.latent_dim, 512, normalize=False),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, int(np.prod(input_shape))),
        )


    # Propagates noise input through the network
    def forward(self, noise):
        returns = self.model(noise)
        returns = returns.view(returns.size(0), *input_shape)
        return returns


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # This has three layers
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # Outputs a probability (real/fake)
        )

    def forward(self, returns):
        validity = self.model(returns.view(returns.size(0), -1))
        return validity



# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# For Vizualization
print(generator)
print(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


from torch.utils.data import TensorDataset, DataLoader

rolling_returns_tensor = torch.tensor(rolling_returns, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(rolling_returns_tensor), batch_size=opt.batch_size, shuffle=True)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (real_returns,) in enumerate(dataloader):

        batch_size = real_returns.size(0)

        # Ground truths
        valid = torch.ones((batch_size, 1), requires_grad=False).to(real_returns.device)
        fake = torch.zeros((batch_size, 1), requires_grad=False).to(real_returns.device)

        # Configure input
        real_returns = real_returns.to(real_returns.device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(batch_size, opt.latent_dim).to(real_returns.device)

        # Generate a batch of synthetic returns
        gen_returns = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_returns)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real loss
        validity_real = discriminator(real_returns)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Fake loss
        validity_fake = discriminator(gen_returns.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save generated returns periodically
        if (epoch * len(dataloader) + i) % opt.sample_interval == 0:
            torch.save(gen_returns, f"generated_returns/returns_{epoch}_{i}.pt")


# Load generated returns
gen_returns = torch.load('generated_returns/returns_0_0.pt')  # Example file
# Reshape gen_returns to (batch_size, 252 days, 7 assets)
gen_returns = gen_returns.view(gen_returns.size(0), 252, 7)
gen_returns = gen_returns.cpu().detach().numpy()
print(gen_returns.shape)

# Plot distribution for each asset
for asset in range(8):
    plt.figure(figsize=(8, 4))
    sns.histplot(gen_returns[:, :, asset].flatten(), bins=50, kde=True)
    plt.title(f'Distribution of Returns for Asset {asset+1}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.show()