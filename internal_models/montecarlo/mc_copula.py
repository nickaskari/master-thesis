import pandas as pd  
import numpy as np
from scipy.stats import t, rankdata
from copulae import StudentCopula
import matplotlib.pyplot as plt
from dotenv.main import load_dotenv
load_dotenv(override=True)
import os

class MonteCarloCopula:

    def __init__(self, returns_df, weights):
        self.returns_df = returns_df
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = int(os.getenv("INIT_ASSETS")) * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.n_days = int(os.getenv("N_DAYS"))
        self.asset_classes = returns_df.columns
        self.weights = weights

    def transform_to_uniorm_marginal(self):
        return self.returns_df.apply(lambda x: rankdata(x) / (len(x) + 1), axis=0)
    
    def fit_student_copula(self, uniform_data):
        ndim = uniform_data.shape[1]
        t_cop = StudentCopula(dim=ndim)
        t_cop.fit(uniform_data)
        return t_cop

    def generate_uniform_samples(self, uniform_data):
        t_cop = self.fit_student_copula(uniform_data)
        samples = t_cop.random(self.n_simulations * self.n_days).to_numpy()
        
        reshaped_samples = samples.reshape(self.n_simulations, self.n_days, uniform_data.shape[1])
        
        return reshaped_samples


    def get_simulated_returns(self, uniform_data):
        simulated_uniforms = self.generate_uniform_samples(uniform_data)
        simulated_returns = np.zeros_like(simulated_uniforms)

        # Fit Student-t marginals to each asset's returns
        for i, col in enumerate(self.returns_df.columns):
            params = t.fit(self.returns_df[col])  # Fit Student-t distribution for each asset
            simulated_returns[:, :, i] = t.ppf(simulated_uniforms[:, :, i], *params)

        return simulated_returns
    
    def get_cumulative_returns(self, simulated_returns):
        portfolio_returns = np.sum(simulated_returns * self.weights, axis=2)
        cumulative_returns = np.prod(1 + portfolio_returns, axis=1) - 1
        return cumulative_returns
    
    def calculate_distribution_and_scr(self):
        BOF_0 = self.assets_0 - self.liabilities_0  
        uniform_data = self.transform_to_uniorm_marginal()
        simulated_returns = self.get_simulated_returns(uniform_data)
        simulated_cumulative_returns = self.get_cumulative_returns(simulated_returns)

        # Extract cumulative returns for EONIA (7th asset, index 6)
        simulated_returns_eonia = simulated_returns[:, :, 6]  # Last day cumulative returns for EONIA
        # Compute cumulative returns for EONIA over 252 trading days
        cumulative_returns_eonia = np.prod(1 + simulated_returns_eonia, axis=1) - 1  # Shape: (num_simulations,)


        assets_t1 = self.assets_0 * (1 + simulated_cumulative_returns)  # Assets after 1 year
        liabilities_t1 = self.liabilities_0 * (1 + cumulative_returns_eonia)  # Liabilities after 1 year

        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0

        scr = np.percentile(bof_change, 100 * (1 - 0.995))

        return bof_change, scr

    def plot_bof_var(self, bof_change, scr):
        plt.figure(figsize=(10, 6))
        plt.hist(bof_change / 1e6, bins=500, alpha=0.7, color='c', label='Change in BOF Distribution', density=True)
        plt.axvline(np.percentile(bof_change, 0.5) / 1e6, color='r', linestyle='--', label=f'VaR (99.5%): {scr / 1e6:.2f}M')
        plt.title('Distribution of Change in BOF after 1 Year: Monte Carlo')
        plt.xlabel('Change in BOF (Millions)')
        plt.ylabel('Density')
        
        plt.xlim(-1, 1)
        plt.xticks(np.arange(-1, 1, 0.5))  # Increments of 0.5 within the range [-2, 2]

        plt.legend()
        plt.grid(True)
        plt.show()

