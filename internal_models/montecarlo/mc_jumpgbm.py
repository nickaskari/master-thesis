import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, rankdata, norm, poisson, laplace, uniform
from dotenv.main import load_dotenv
load_dotenv(override=True)
import os
from tqdm import tqdm

class MonteCarloJumpGBM:

    def __init__(self, returns_df, weights):
        self.returns_df = returns_df
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = int(os.getenv("INIT_ASSETS")) * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.asset_classes = returns_df.columns
        self.weights = weights

        # PARAMTERS FOR GBM
        self.correlation_matrix = returns_df.corr()
        self.mu = returns_df.mean()
        self.sigma = returns_df.std()
        self.n_assets = returns_df.shape[1]
        self.T = int(os.getenv("N_DAYS"))
        self.dt = 1

        # Set threshold for identifying jumps (3-sigma rule)
        self.k = 3


    def simulate_gbm_with_jumps(self, jump_params):
        S0 = np.full(self.n_assets, 100)

        # Cholesky decomposition for correlated Brownian motions
        L = np.linalg.cholesky(self.correlation_matrix)

        paths = np.zeros((self.T, self.n_assets))
        paths[0, :] = S0

        for t in range(1, self.T):
            # Generate correlated Brownian motions
            Z = np.random.normal(size=self.n_assets)
            dW = np.dot(L, Z) * np.sqrt(self.dt)

            # Initialize jump process
            J = np.zeros(self.n_assets)
            
            
            for i, name in enumerate(self.returns_df.columns):
                # Poisson-distributed number of jumps
                num_jumps = np.random.poisson(jump_params[name]['lambda'] * self.dt)
                if num_jumps > 0:
                    # Sample jump size from specified distribution
                    J[i] = np.sum(jump_params[name]['jump_distribution'].rvs(size=num_jumps))
        
            # GBM with jumps formula
            paths[t, :] = paths[t-1, :] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW + J)

        return pd.DataFrame(paths, columns=self.returns_df.columns)
    
    def jump_params_est(self):
        jumps_df = (self.returns_df - self.mu).abs() > (self.k * self.sigma)

        # Estimate jump intensity (λ) as the fraction of days with jumps
        lambda_est = jumps_df.mean()

        # Extract jump sizes
        jump_sizes = {
            asset: self.returns_df[asset][jumps_df[asset]]
                    for asset in self.returns_df.columns
            }

        # Fit normal distribution to jump sizes
        jump_params_est = {}
        for asset, jumps in jump_sizes.items():
            if len(jumps) > 1:  # Ensure there are enough jumps to estimate
                loc, scale = norm.fit(jumps)  # Fit normal distribution
            else:
                loc, scale = 0, self.sigma[asset]  # Default if no jumps detected
                print(f'No jumps detected for {asset}. Using default parameters.')

            jump_params_est[asset] = {'lambda': lambda_est[asset],
                                    'jump_distribution': norm(loc=loc, scale=scale)}
        
        return jump_params_est
    
    def get_all_simulated_paths(self):
        jump_params_est = self.jump_params_est()

        simulated_paths = {}
        for i in tqdm(range(self.n_simulations), desc="MonteCarlo GBM w Jumps", unit="sim"):
            simulated_paths[i] = self.simulate_gbm_with_jumps(jump_params=jump_params_est)
        
        return simulated_paths   
    
    def calculate_distribution_and_scr(self):
        BOF_0 = self.assets_0 - self.liabilities_0

        simulated_results = self.get_all_simulated_paths()
        end_prices = pd.DataFrame({i: simulated_results[i].iloc[-1] for i in range(self.n_simulations)}).T

        portfolio_end_values = end_prices.dot(self.weights)

        portfolio_returns = (portfolio_end_values - 100) / 100

        eonia_end_prices = pd.Series({i: simulated_results[i]["EONIA"].iloc[-1] for i in range(self.n_simulations)})

        # Compute EONIA returns (assuming initial price is 100)
        eonia_returns = (eonia_end_prices - 100) / 100  

        assets_t1 = self.assets_0 * (1 + portfolio_returns)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_returns)

        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0

        scr = np.percentile(bof_change, 100 * (1 - 0.995))

        return bof_change, scr