import numpy as np
import pandas as pd
from scipy.stats import rankdata
from copulae import StudentCopula
from tqdm import tqdm
import os

from dotenv.main import load_dotenv
load_dotenv(override=True)

class CopulaGAN:
    def __init__(
        self, 
        gan_samples,       # 3D array with shape (n_simulations, n_days, n_assets)
        weights,           # Portfolio weights for the n_assets
        confidence=0.995
    ):
        self.gan_samples = gan_samples 
        self.weights = weights
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = self.assets_0 * float(os.getenv("FRAC_LIABILITIES"))
        # Now derive simulation parameters from the input shape:
        self.n_simulations = gan_samples.shape[0]
        self.n_days = gan_samples.shape[1]
        self.n_assets = gan_samples.shape[2]
        self.confidence = confidence

    def transform_to_uniform(self):
        """
        Rank-based transform each asset's samples to [0,1].
        Returns array of shape (n_simulations, n_days, n_assets).
        """
        # Flatten the first two dimensions
        N_total = self.n_simulations * self.n_days
        reshaped = self.gan_samples.reshape(N_total, self.n_assets)
        uniform_data = np.zeros_like(reshaped)
        for i in range(self.n_assets):
            # rankdata returns ranks 1..N_total; divide by (N_total+1) to map into (0,1)
            uniform_data[:, i] = rankdata(reshaped[:, i]) / (N_total + 1)
        # Reshape back to (n_simulations, n_days, n_assets)
        return uniform_data.reshape(self.n_simulations, self.n_days, self.n_assets)

    def fit_student_copula(self, uniform_data):
        """
        Fit a Student copula to the flattened uniform data.
        uniform_data: shape (n_simulations, n_days, n_assets)
        """
        # Flatten uniform_data to 2D: (n_simulations*n_days, n_assets)
        flattened = uniform_data.reshape(-1, self.n_assets)
        t_cop = StudentCopula(dim=self.n_assets)
        t_cop.fit(flattened)
        return t_cop

    def generate_uniform_samples(self, t_cop):
        """
        Generate (n_simulations*n_days) x n_assets uniform samples from the copula.
        Reshape to (n_simulations, n_days, n_assets).
        """
        samples = t_cop.random(self.n_simulations * self.n_days)
        return samples.reshape(self.n_simulations, self.n_days, self.n_assets)

    def invert_to_gan_marginals(self, uniform_samples):
        """
        For each asset, invert the uniform draws using the empirical CDF
        from the original GAN samples.
        uniform_samples: shape (n_simulations, n_days, n_assets)
        Returns simulated returns with the same shape.
        """
        simulated_returns = np.zeros_like(uniform_samples)
        # Flatten the original GAN samples over simulations and days
        flattened_gan = self.gan_samples.reshape(-1, self.n_assets)
        N_total = flattened_gan.shape[0]
        
        for i in range(self.n_assets):
            sorted_gan = np.sort(flattened_gan[:, i])
            # For each asset, convert uniform values to indices in [0, N_total-1]
            indices = (uniform_samples[..., i] * (N_total - 1)).astype(int)
            simulated_returns[..., i] = sorted_gan[indices]
        return simulated_returns

    def compute_portfolio(self, simulated_returns):
        """
        Compute the daily portfolio returns, compound them to get cumulative returns,
        and then calculate the change in Balance of Fund (BOF).
        simulated_returns: shape (n_simulations, n_days, n_assets)
        """
        # Calculate daily portfolio returns using weights
        portfolio_returns = np.sum(simulated_returns * self.weights, axis=2)
        # Compound daily returns to get cumulative return per simulation
        portfolio_cum = np.prod(1 + portfolio_returns, axis=1) - 1

        # Use the last asset as the liability driver (e.g. EONIA)
        eonia_returns = simulated_returns[:, :, -1]
        eonia_cum = np.prod(1 + eonia_returns, axis=1) - 1

        assets_t1 = self.assets_0 * (1 + portfolio_cum)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_cum)

        bof_t1 = assets_t1 - liabilities_t1
        bof_0 = self.assets_0 - self.liabilities_0
        bof_change = bof_t1 - bof_0
        return bof_change

    def calculate_distribution_and_scr(self):
        with tqdm(total=5, desc="Monte Carlo w/ GAN + Copula", unit="step") as pbar:
            # 1) Transform to uniform marginals
            uniform_data = self.transform_to_uniform()
            pbar.update(1)

            # 2) Fit Student copula on flattened uniform data
            t_cop = self.fit_student_copula(uniform_data)
            pbar.update(1)

            # 3) Generate correlated uniform samples
            uniform_samples = self.generate_uniform_samples(t_cop)
            pbar.update(1)

            # 4) Invert the uniform samples to GAN marginal distributions
            simulated_returns = self.invert_to_gan_marginals(uniform_samples)
            pbar.update(1)

            # 5) Compute portfolio BOF changes & SCR
            bof_change = self.compute_portfolio(simulated_returns)
            scr = np.percentile(bof_change, 100 * (1 - self.confidence))
            pbar.update(1)

        return bof_change, scr