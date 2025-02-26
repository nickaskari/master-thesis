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
        self.n_days = int(os.getenv("N_DAYS"))
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

        np.random.seed(42)
        for i, asset in enumerate(asset_names):
            Z = np.random.normal(0, 1, (T, N_sim))  # Independent shocks
            r_sim = (mu[i] - 0.5 * sigma[i]**2) * dt + sigma[i] * np.sqrt(dt) * Z  # Log returns

            S = np.zeros((T, N_sim))
            S[0, :] = S0[i]

            for t in range(1, T):
                S[t, :] = S[t-1, :] * np.exp(r_sim[t, :])

            simulated_final_prices[asset] = S[-1, :] # final prices

            expected_value_gbm = S0[i] * np.exp(mu[i] * time)

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

            plt.show()
        return
    
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
