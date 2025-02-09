import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from dotenv.main import load_dotenv
load_dotenv(override=True)

import os

class HistoricalSimulation:

    def __init__(self, returns_df, weights):
        self.returns_df = returns_df
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = int(os.getenv("INIT_ASSETS")) * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.n_days = int(os.getenv("N_DAYS"))
        self.asset_classes = returns_df.columns
        self.weights = weights

    def calculate_distribution_and_scr(self):
        BOF_0 = self.assets_0 - self.liabilities_0  

        PnL_portfolio_daily = (self.weights * self.returns_df.values).sum(axis=1)
        PnL_portfolio_1yr = pd.Series(PnL_portfolio_daily).rolling(window=252).sum().dropna()

        eonia_returns = self.returns_df.iloc[:, 6]  # EONIA daily returns
        PnL_eonia_1yr = eonia_returns.rolling(window=252).sum().dropna()

        # Align lengths of PnL_portfolio_1yr and PnL_eonia_1yr
        min_length = min(len(PnL_portfolio_1yr), len(PnL_eonia_1yr))
        PnL_portfolio_1yr = PnL_portfolio_1yr[-min_length:]
        PnL_eonia_1yr = PnL_eonia_1yr[-min_length:]

        assets_t1 = self.assets_0 * (1 + PnL_portfolio_1yr.values)
        liabilities_t1 = self.liabilities_0 * (1 + PnL_eonia_1yr.values)

        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0

        scr = np.percentile(bof_change, 100 * (1 - 0.995))

        return bof_change, scr

    def plot_bof_var(self, bof_change, scr):
        plt.figure(figsize=(10, 6))
        plt.hist(bof_change / 1e6, bins=100, alpha=0.7, color='c', label='Change in BOF Distribution', density=True)
        plt.axvline(scr / 1e6, color='r', linestyle='--', label=f'VaR (99.5%): {scr / 1e6:.2f}M')
        plt.title('Distribution of Change in BOF after 1 Year (Historical Simulation)')
        plt.xlabel('Change in BOF (Millions)')
        plt.ylabel('Density')

        plt.xlim(-1, 1)
        plt.xticks(np.arange(-1, 1.1, 0.5))  # Increments of 0.5 within the range [-1, 1]

        plt.legend()
        plt.grid(True)
        plt.show()