import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from dotenv.main import load_dotenv
load_dotenv(override=True)
import os

class MonteCarloVanillaGBM:

    def __init__(self, returns_df, weights):
        self.returns_df = returns_df
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = int(os.getenv("INIT_ASSETS")) * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.asset_classes = returns_df.columns
        self.weights = weights

        # PARAMTERS FOR GBM
        self.T = int(os.getenv("N_DAYS"))
        num_assets = len(returns_df.columns)
        self.S0 = np.array([100] * num_assets)  
        self.dt = 1

        self.mu = returns_df.mean().values  # Mean daily return for each asset
        self.sigma = returns_df.std().values


    def simulate_assets(self):
        simulated_final_prices = {}

        time = np.linspace(0, self.T, self.T)

        np.random.seed(42)
        for i, asset in enumerate(self.asset_classes):
            Z = np.random.normal(0, 1, (self.T, self.n_simulations))  # Independent shocks
            r_sim = (self.mu[i] - 0.5 * self.sigma[i]**2) * self.dt + self.sigma[i] * np.sqrt(self.dt) * Z  # Log returns

            S = np.zeros((self.T, self.n_simulations))
            S[0, :] = self.S0[i]

            for t in range(1, self.T):
                S[t, :] = S[t-1, :] * np.exp(r_sim[t, :])

            simulated_final_prices[asset] = S[-1, :] # final prices

            expected_value_gbm = self.S0[i] * np.exp(self.mu[i] * time)

            '''
            plt.figure(figsize=(10, 6))
            for path in S.T:
                plt.plot(time, path, color='lightgray', linewidth=0.5)
            plt.plot(time, expected_value_gbm, color='red',
                    linewidth=2, label='Expected Value')
            plt.title(f'Geometric Brownian Motion Sample Paths for {asset}')
            plt.xlabel('Time')
            plt.ylabel('S(t)')
            plt.legend()
            plt.grid(True)
            '''

            plt.show()

        return simulated_final_prices
    
    def calculate_distribution_and_scr(self):
        BOF_0 = self.assets_0 - self.liabilities_0
        stored_simulated_data = self.simulate_assets()

        yearly_returns = {
            asset: stored_simulated_data[asset] / 100 - 1 for asset in self.asset_classes
        }

        returns_matrix = np.array(list(yearly_returns.values()))  
        portfolio_returns = np.dot(self.weights, returns_matrix) 

        eonia_returns = yearly_returns["EONIA"]

        

        assets_t1 = self.assets_0 * (1 + portfolio_returns)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_returns)

        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0

        scr = np.percentile(bof_change, 100 * (1 - 0.995))

        return bof_change, scr