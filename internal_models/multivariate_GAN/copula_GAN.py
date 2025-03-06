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
        gan_samples,       # List or array of shape (N, 7) with GAN samples for 7 assets
        weights,           # Portfolio weights for the 7 assets
        confidence=0.995
    ):

        self.gan_samples = gan_samples 
        self.weights = weights
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = self.assets_0 * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.n_days = int(os.getenv("N_DAYS"))
        self.confidence = confidence
        self.n_assets = gan_samples.shape[1]

    def transform_to_uniform(self):
        """
        Rank-based transform each asset's samples to [0,1].
        Returns array shape (N, 7).
        """
        N = self.gan_samples.shape[0]
        uniform_data = np.zeros_like(self.gan_samples)
        for i in range(self.n_assets):
            uniform_data[:, i] = rankdata(self.gan_samples[:, i]) / (N + 1)
        return uniform_data

    def fit_student_copula(self, uniform_data):
        """
        Fit a Student copula to the Nx7 uniform data.
        """
        t_cop = StudentCopula(dim=self.n_assets)
        t_cop.fit(uniform_data)
        return t_cop

    def generate_uniform_samples(self, t_cop):
        """
        Generate (n_simulations*n_days) x 7 uniform samples from the copula.
        Reshape to (n_simulations, n_days, 7).
        """
        samples = t_cop.random(self.n_simulations * self.n_days)
        return samples.reshape(self.n_simulations, self.n_days, self.n_assets)

    def invert_to_gan_marginals(self, uniform_samples):
        """
        For each asset (column), invert the uniform draws using the empirical CDF
        from the original GAN samples. A simple approach is to sort the GAN samples
        and index by the uniform quantile.
        """
        simulated_returns = np.zeros_like(uniform_samples)
        N = self.gan_samples.shape[0]

        for i in range(self.n_assets):
            sorted_gan = np.sort(self.gan_samples[:, i])  # ascending
            # uniform_samples[..., i] is shape (n_simulations, n_days)
            # Convert uniform in [0,1] -> index in [0, N-1]
            # We'll do nearest integer indexing here; you could do interpolation for more precision.
            indices = (uniform_samples[..., i] * (N - 1)).astype(int)
            simulated_returns[..., i] = sorted_gan[indices]
        return simulated_returns

    def compute_portfolio(self, simulated_returns):
        """
        Multiply each day’s returns by weights, get daily returns, compound.
        Returns the final portfolio level minus initial level (bof_change).
        """
        portfolio_returns = np.sum(simulated_returns * self.weights, axis=2)
        # Compound daily returns -> final growth factor - 1
        portfolio_cum = np.prod(1 + portfolio_returns, axis=1) - 1

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
            # 1) Uniform transform
            uniform_data = self.transform_to_uniform()
            pbar.update(1)

            # 2) Fit Student Copula
            t_cop = self.fit_student_copula(uniform_data)
            pbar.update(1)

            # 3) Generate correlated uniform samples
            uniform_samples = self.generate_uniform_samples(t_cop)
            pbar.update(1)

            # 4) Invert to each asset's GAN marginal
            simulated_returns = self.invert_to_gan_marginals(uniform_samples)
            pbar.update(1)

            # 5) Compute portfolio BOF changes & SCR
            bof_change = self.compute_portfolio(simulated_returns)
            scr = np.percentile(bof_change, 100 * (1 - self.confidence))
            pbar.update(1)

        return bof_change, scr
