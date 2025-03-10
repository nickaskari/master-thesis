import numpy as np
import os
from dotenv.main import load_dotenv
load_dotenv(override=True)

class SimpleGANPortfolio:
    def __init__(self, gan_samples, weights, confidence=0.995):
        """
        gan_samples: Aggregated GAN output with cumulative returns, shape (n_simulations, n_assets)
        weights: Array-like of length n_assets representing portfolio weights.
        confidence: Confidence level for VaR/SCR calculation (e.g., 0.995 for 99.5% VaR).
        """
        self.gan_samples = gan_samples
        self.weights = np.array(weights)
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = self.assets_0 * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = self.gan_samples.shape[0]
        self.n_assets = self.gan_samples.shape[1]
        self.confidence = confidence

    def compute_portfolio(self):
        """
        Compute the portfolio cumulative return using the weighted sum of GAN-generated cumulative returns.
        Assume that the last asset (column) is used as the liability driver.
        
        Returns:
          bof_change: Array of BOF changes for each simulation.
        """
        # Compute the portfolio cumulative return as the weighted sum across assets.
        portfolio_cum = np.sum(self.gan_samples * self.weights, axis=1)
        # Use the last asset's cumulative return as the liability driver (e.g., EONIA)
        eonia_cum = self.gan_samples[:, -1]
        
        assets_t1 = self.assets_0 * (1 + portfolio_cum)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_cum)
        
        bof_t1 = assets_t1 - liabilities_t1
        bof_0 = self.assets_0 - self.liabilities_0
        
        bof_change = bof_t1 - bof_0
        return bof_change

    def calculate_distribution_and_scr(self):
        """
        Compute the distribution of BOF changes and the SCR (e.g., the 100*(1-confidence) percentile).
        
        Returns:
          bof_change: Array of BOF changes per simulation.
          scr: The SCR value computed as the appropriate percentile of bof_change.
        """
        bof_change = self.compute_portfolio()
        scr = np.percentile(bof_change, 100 * (1 - self.confidence))
        return bof_change, scr