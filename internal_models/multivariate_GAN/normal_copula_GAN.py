import numpy as np
import pandas as pd
from scipy.stats import rankdata
from copulae import GaussianCopula
from tqdm import tqdm
import os
from dotenv.main import load_dotenv
load_dotenv(override=True)

class CopulaGANNormal:
    def __init__(
        self, 
        gan_samples,       # 2D array with shape (n_simulations, n_assets)
        weights,           # Portfolio weights for the n_assets
        confidence=0.995
    ):
        """
        gan_samples: shape (n_simulations, n_assets) - aggregated (e.g., cumulative) returns
        weights: array of length n_assets
        confidence: e.g., 0.995 for 99.5% VaR
        """
        self.gan_samples = gan_samples 
        self.weights = weights
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = self.assets_0 * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = gan_samples.shape[0]
        self.n_assets = gan_samples.shape[1]
        self.confidence = confidence

    def transform_to_uniform(self):
        """
        Rank-based transform each asset's returns to [0,1].
        """
        N = self.n_simulations
        uniform_data = np.zeros_like(self.gan_samples)
        for i in range(self.n_assets):
            # rankdata returns ranks 1..N; divide by (N+1) to map into (0,1)
            uniform_data[:, i] = rankdata(self.gan_samples[:, i]) / (N + 1)
        return uniform_data

    def fit_normal_copula(self, uniform_data):
        """
        Fit a Gaussian copula to the uniform data.
        uniform_data: shape (n_simulations, n_assets)
        """
        normal_cop = GaussianCopula(dim=self.n_assets)
        normal_cop.fit(uniform_data)  # MLE for correlation matrix
        return normal_cop

    def generate_uniform_samples(self, normal_cop):
        """
        Generate n_simulations x n_assets uniform samples from the fitted Gaussian copula.
        """
        samples = normal_cop.random(self.n_simulations)
        return samples

    def invert_to_gan_marginals(self, uniform_samples):
        """
        For each asset, invert the uniform draws using the empirical CDF
        from the original aggregated GAN samples.
        uniform_samples: shape (n_simulations, n_assets)
        """
        simulated_returns = np.zeros_like(uniform_samples)
        N = self.n_simulations
        for i in range(self.n_assets):
            sorted_gan = np.sort(self.gan_samples[:, i])
            # Map uniform values to indices in [0, N-1]
            indices = (uniform_samples[:, i] * (N - 1)).astype(int)
            simulated_returns[:, i] = sorted_gan[indices]
        return simulated_returns

    def compute_portfolio(self, simulated_returns):
        """
        Compute portfolio cumulative return (weighted sum) and then BOF change.
        simulated_returns: shape (n_simulations, n_assets)
        """
        portfolio_cum = np.sum(simulated_returns * self.weights, axis=1)
        eonia_cum = simulated_returns[:, -1]  # if last asset is EONIA
        assets_t1 = self.assets_0 * (1 + portfolio_cum)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_cum)
        bof_t1 = assets_t1 - liabilities_t1
        bof_0 = self.assets_0 - self.liabilities_0
        bof_change = bof_t1 - bof_0
        return bof_change

    def calculate_distribution_and_scr(self):
        with tqdm(total=5, desc="Monte Carlo w/ GAN + Gaussian Copula", unit="step") as pbar:
            # 1) Transform to uniform marginals
            uniform_data = self.transform_to_uniform()
            pbar.update(1)

            # 2) Fit the Gaussian copula
            normal_cop = self.fit_normal_copula(uniform_data)
            pbar.update(1)

            # 3) Generate correlated uniform samples
            uniform_samples = self.generate_uniform_samples(normal_cop)
            pbar.update(1)

            # 4) Invert to GAN marginals
            simulated_returns = self.invert_to_gan_marginals(uniform_samples)
            pbar.update(1)

            # 5) Compute BOF changes & SCR
            bof_change = self.compute_portfolio(simulated_returns)
            scr = np.percentile(bof_change, 100 * (1 - self.confidence))
            pbar.update(1)

        return bof_change, scr
