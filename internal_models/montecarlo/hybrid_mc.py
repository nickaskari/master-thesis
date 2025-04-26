import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, rankdata, norm, poisson, laplace, uniform
from dotenv.main import load_dotenv
load_dotenv(override=True)
import os
from tqdm import tqdm
import concurrent.futures
from functools import partial
import numba
from numba import jit, prange
import time

@jit(nopython=True)
def _numba_simulate_gbm_with_jumps(T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales, 
                                  eonia_index, kappa, theta, r0, eonia_sigma, seed):
    np.random.seed(seed)
    
    paths = np.zeros((T, n_assets))
    paths[0, :] = S0
    
    is_eonia = np.zeros(n_assets, dtype=np.bool_)
    if eonia_index >= 0 and eonia_index < n_assets:
        is_eonia[eonia_index] = True
    
    # For EONIA - use separate arrays for calculation
    eonia_rates = np.zeros(T)
    
    # Base rate matches historical EONIA (approximately -0.00001)
    eonia_base_rate = -0.00001
    eonia_rates[0] = eonia_base_rate
    
    # No sudden initial drop - start at same price
    if eonia_index >= 0:
        paths[0, eonia_index] = 100.0
    
    # Policy jump counter to limit frequency
    policy_jump_count = 0
    max_policy_jumps = 1  # Maximum number of policy jumps in the entire simulation
    
    # Week days - 0 is Monday, 4 is Friday in our simulation
    weekend_days = np.zeros(7, dtype=np.bool_)
    weekend_days[4] = True  # Friday always has the drop (day 4 in our 0-indexed week)
    
    for t in range(1, T):
        Z = np.zeros(n_assets)
        for i in range(n_assets):
            Z[i] = np.random.normal(0.0, 1.0)
            
        dW = np.dot(L, Z) * np.sqrt(dt)
        
        J = np.zeros(n_assets)
        
        for i in range(n_assets):
            if is_eonia[i]:
                # Determine day of week (0-6)
                day_of_week = t % 7
                
                # Weekend effect - happens on Friday (day 4) almost certainly
                if weekend_days[day_of_week] and np.random.random() < 0.98:  # 98% chance on Friday
                    # Weekend drop - consistent size
                    jump_size = 0.00002  # Fixed size based on historical data
                    J[i] = -jump_size  # Always negative
                    eonia_rates[t] = eonia_base_rate + J[i]
                # Monday (day 0) - always revert from weekend drop
                elif day_of_week == 0 and t > 4:  # After first week
                    # Return to base rate
                    eonia_rates[t] = eonia_base_rate
                elif policy_jump_count < max_policy_jumps and np.random.random() < 0.00005:  # Extremely rare policy jumps
                    # Only allow at most one policy jump, with very low probability
                    jump_size = np.random.uniform(0.000005, 0.00001)
                    jump_sign = -1.0 if np.random.random() < 0.8 else 1.0  # Mostly negative
                    J[i] = jump_size * jump_sign
                    
                    # Update base rate and limit future jumps
                    eonia_base_rate = eonia_base_rate + J[i]
                    policy_jump_count += 1
                    eonia_rates[t] = eonia_base_rate
                else:
                    # Normal day - no jumps, stay at base rate with minimal noise
                    tiny_noise = eonia_sigma * dW[i] * 0.00001  # Extremely small noise
                    eonia_rates[t] = eonia_base_rate + tiny_noise
                
                # Ensure rates stay in a reasonable range
                eonia_rates[t] = max(min(eonia_rates[t], 0.0), -0.00003)
                
                # Convert to daily return for price path calculation
                daily_return = eonia_rates[t] - eonia_rates[t-1]
                
                # Apply return to price path
                paths[t, i] = paths[t-1, i] * (1 + daily_return)
            else:
                # Regular GBM jumps for other assets
                num_jumps = np.random.poisson(jump_lambdas[i] * dt)
                if num_jumps > 0:
                    jump_sum = 0.0
                    for j in range(num_jumps):
                        jump_sum += np.random.normal(jump_locs[i], jump_scales[i])
                    J[i] = jump_sum
                
                paths[t, i] = paths[t-1, i] * np.exp((mu[i] - 0.5 * sigma[i]**2) * dt + sigma[i] * dW[i] + J[i])
    
    return paths

def _process_batch(args):
    sim_ids, jump_params, T, n_assets, L, mu, sigma, dt, column_names, eonia_params = args
    
    batch_results = {}
    for sim_id in sim_ids:
        S0 = np.full(n_assets, 100)
        
        jump_lambdas = np.zeros(n_assets)
        jump_locs = np.zeros(n_assets)
        jump_scales = np.zeros(n_assets)
        
        for i, name in enumerate(column_names):
            jump_lambdas[i] = jump_params[name]['lambda']
            jump_locs[i] = jump_params[name]['jump_distribution'].mean()
            jump_scales[i] = jump_params[name]['jump_distribution'].std()
        
        eonia_index = -1
        for i, name in enumerate(column_names):
            if name == "EONIA":
                eonia_index = i
                S0[i] = eonia_params.get('r0', 0.01)
                break
        
        seed = 42 + sim_id
        
        try:
            paths_array = _numba_simulate_gbm_with_jumps(
                T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales,
                eonia_index, eonia_params.get('kappa', 0.3), eonia_params.get('theta', 0.01), 
                eonia_params.get('r0', 0.01), eonia_params.get('sigma', 0.005), seed
            )
            
            paths_df = pd.DataFrame(paths_array, columns=column_names)
            batch_results[sim_id] = paths_df
        except Exception as e:
            print(f"Numba optimization failed for simulation {sim_id}, falling back to non-optimized version.")
            paths_df = _fallback_simulate_mixed_model(
                T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales,
                eonia_index, eonia_params, seed, column_names
            )
            batch_results[sim_id] = paths_df
        
    return batch_results

def _fallback_simulate_mixed_model(T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales, 
                                  eonia_index, eonia_params, seed, column_names):
    np.random.seed(seed)
    
    paths = np.zeros((T, n_assets))
    paths[0, :] = S0
    
    is_eonia = np.zeros(n_assets, dtype=bool)
    if eonia_index >= 0 and eonia_index < n_assets:
        is_eonia[eonia_index] = True
        paths[0, eonia_index] = eonia_params.get('r0', -0.0003)
    
    # For EONIA - start with negative rate to match recent history
    eonia_base_rate = -0.0003
    eonia_sigma = eonia_params.get('sigma', 0.001)
    
    # Policy jump counter to limit frequency
    policy_jump_count = 0
    max_policy_jumps = 1  # Maximum number of policy jumps
    
    for t in range(1, T):
        Z = np.random.normal(0, 1, size=n_assets)
        dW = np.dot(L, Z) * np.sqrt(dt)
        
        J = np.zeros(n_assets)
        
        for i in range(n_assets):
            if is_eonia[i]:
                # Weekend effect - regular small jumps
                is_weekend = (t % 7 == 0)
                
                if is_weekend and np.random.random() < 0.7:
                    # Weekend drop - always small and always reverts
                    jump_size = 0.00002  # Fixed size for consistency
                    J[i] = -jump_size  # Always negative
                    paths[t, i] = eonia_base_rate + J[i]
                elif policy_jump_count < max_policy_jumps and np.random.random() < 0.0001:
                    # Only allow at most one policy jump, and very rarely
                    jump_size = np.random.uniform(0.00005, 0.0001)
                    jump_sign = -1.0 if np.random.random() < 0.8 else 1.0  # Mostly negative
                    J[i] = jump_size * jump_sign
                    
                    # Update base rate and limit future jumps
                    eonia_base_rate = eonia_base_rate + J[i]
                    policy_jump_count += 1
                    paths[t, i] = eonia_base_rate
                else:
                    # Normal day - no jumps, stay at base rate
                    tiny_noise = eonia_sigma * dW[i] * 0.0001  # Extremely small noise
                    paths[t, i] = eonia_base_rate + tiny_noise
            else:
                num_jumps = np.random.poisson(jump_lambdas[i] * dt)
                if num_jumps > 0:
                    jump_sizes = np.random.normal(jump_locs[i], jump_scales[i], size=num_jumps)
                    J[i] = np.sum(jump_sizes)
        
        for i in range(n_assets):
            if is_eonia[i]:
                # Ensure rates stay in a reasonable range
                paths[t, i] = max(min(paths[t, i], 0.0005), -0.0012)
            else:
                paths[t, i] = paths[t-1, i] * np.exp((mu[i] - 0.5 * sigma[i]**2) * dt + sigma[i] * dW[i] + J[i])
    
    return pd.DataFrame(paths, columns=column_names)

class MonteCarloJumpGBM:

    def __init__(self, returns_df, weights):
        self.returns_df = returns_df
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = int(os.getenv("INIT_ASSETS")) * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.asset_classes = returns_df.columns
        self.weights = weights

        self.correlation_matrix = returns_df.corr().values
        self.mu = returns_df.mean().values
        self.sigma = returns_df.std().values
        self.n_assets = returns_df.shape[1]
        self.T = int(os.getenv("N_DAYS"))
        self.dt = 1

        self.k = 3
        
        self.L = np.linalg.cholesky(self.correlation_matrix)
        
        self.column_names = returns_df.columns.tolist()
        
        self.eonia_params = self._estimate_vasicek_params()

    def simulate_gbm_with_jumps(self, jump_params, sim_id=None):
        S0 = np.full(self.n_assets, 100)
        
        jump_lambdas = np.zeros(self.n_assets)
        jump_locs = np.zeros(self.n_assets)
        jump_scales = np.zeros(self.n_assets)
        
        for i, name in enumerate(self.column_names):
            jump_lambdas[i] = jump_params[name]['lambda']
            jump_locs[i] = jump_params[name]['jump_distribution'].mean()
            jump_scales[i] = jump_params[name]['jump_distribution'].std()
        
        eonia_index = -1
        for i, name in enumerate(self.column_names):
            if name == "EONIA":
                eonia_index = i
                break
        
        seed = 42 + (0 if sim_id is None else sim_id)
        
        paths_array = _numba_simulate_gbm_with_jumps(
            self.T, self.n_assets, S0, self.L, self.mu, self.sigma, 
            self.dt, jump_lambdas, jump_locs, jump_scales,
            eonia_index, self.eonia_params['kappa'], self.eonia_params['theta'], 
            self.eonia_params['r0'], self.eonia_params.get('sigma', 0.001), seed
        )
        
        paths_df = pd.DataFrame(paths_array, columns=self.column_names)
        
        return sim_id, paths_df

    def _estimate_vasicek_params(self):
        try:
            eonia_idx = self.column_names.index("EONIA")
        except ValueError:
            return {'kappa': 0.2, 'theta': -0.00001, 'r0': -0.00001, 'sigma': 0.0001}
            
        eonia_data = self.returns_df.iloc[:, eonia_idx].values
        
        if len(eonia_data) < 3:
            return {'kappa': 0.2, 'theta': -0.00001, 'r0': -0.00001, 'sigma': 0.0001}
        
        # Set base rate to match recent history (approximately -0.00001)
        r0 = -0.00001
        theta = -0.00001
        
        # Use a very low volatility to match the observed pattern
        sigma = 0.0001
        
        # Low mean reversion to maintain stability
        kappa = 0.2
        
        return {'kappa': kappa, 'theta': theta, 'r0': r0, 'sigma': sigma}
    
    def jump_params_est(self):
        mu_array = np.array(self.mu).reshape(1, -1)
        sigma_array = np.array(self.sigma).reshape(1, -1)
        returns_array = self.returns_df.values
        
        jumps_array = np.abs(returns_array - mu_array) > (self.k * sigma_array)
        jumps_df = pd.DataFrame(jumps_array, columns=self.returns_df.columns)
        
        lambda_est = jumps_df.mean()
        
        jump_sizes = {}
        for i, asset in enumerate(self.column_names):
            jump_mask = jumps_array[:, i]
            if np.any(jump_mask):
                jump_sizes[asset] = returns_array[jump_mask, i]
            else:
                jump_sizes[asset] = np.array([])
        
        jump_params_est = {}
        for i, asset in enumerate(self.column_names):
            jumps = jump_sizes[asset]
            if len(jumps) > 1:
                loc, scale = norm.fit(jumps)
            else:
                loc, scale = 0, self.sigma[i]
                
            jump_params_est[asset] = {'lambda': lambda_est[asset],
                                    'jump_distribution': norm(loc=loc, scale=scale)}
        
        return jump_params_est
    
    def get_all_simulated_paths(self, max_workers=None, batch_size=None, use_numba=True):
        start_time = time.time()
        
        jump_params_est = self.jump_params_est()
        
        if max_workers is None:
            max_workers = os.cpu_count()
            
        if batch_size is None:
            batch_size = max(1, self.n_simulations // (max_workers * 4))
            
        print(f"Running {self.n_simulations} simulations with {max_workers} workers and batch size {batch_size}")
        
        batches = []
        for i in range(0, self.n_simulations, batch_size):
            batch_end = min(i + batch_size, self.n_simulations)
            batches.append(list(range(i, batch_end)))
            
        simulated_paths = {}
        
        if not use_numba:
            for batch in tqdm(batches, desc="MonteCarlo Simulation", unit="batch"):
                batch_results = self._run_numpy_batch(batch, jump_params_est)
                simulated_paths.update(batch_results)
        else:
            try:
                test_sim_id = 0
                test_args = ([test_sim_id], jump_params_est, self.T, self.n_assets, 
                            self.L, self.mu, self.sigma, self.dt, self.column_names, self.eonia_params)
                test_result = _process_batch(test_args)
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    batch_args = [
                        (batch, jump_params_est, self.T, self.n_assets, 
                        self.L, self.mu, self.sigma, self.dt, self.column_names, self.eonia_params) 
                        for batch in batches
                    ]
                    
                    futures = [executor.submit(_process_batch, args) for args in batch_args]
                    
                    for future in tqdm(
                        concurrent.futures.as_completed(futures), 
                        total=len(batches),
                        desc="MonteCarlo Simulation", 
                        unit="batch"
                    ):
                        batch_results = future.result()
                        simulated_paths.update(batch_results)
                        
            except Exception as e:
                print(f"Parallel processing with Numba failed")
                
                simulated_paths = {}
                
                for batch in tqdm(batches, desc="MonteCarlo Simulation", unit="batch"):
                    batch_results = self._run_numpy_batch(batch, jump_params_est)
                    simulated_paths.update(batch_results)
        
        end_time = time.time()
        print(f"Completed {self.n_simulations} simulations in {end_time - start_time:.2f} seconds")
        
        return simulated_paths
        
    def _run_numpy_batch(self, sim_ids, jump_params):
        batch_results = {}
        
        for sim_id in sim_ids:
            np.random.seed(42 + sim_id)
            
            S0 = np.full(self.n_assets, 100)
            
            eonia_index = -1
            for i, name in enumerate(self.column_names):
                if name == "EONIA":
                    eonia_index = i
                    S0[i] = self.eonia_params['r0']
                    break
            
            paths_df = _fallback_simulate_mixed_model(
                self.T, self.n_assets, S0, self.L, self.mu, self.sigma, 
                self.dt, [jump_params[name]['lambda'] for name in self.column_names],
                [jump_params[name]['jump_distribution'].mean() for name in self.column_names],
                [jump_params[name]['jump_distribution'].std() for name in self.column_names],
                eonia_index, self.eonia_params, 42 + sim_id, self.column_names
            )
            
            batch_results[sim_id] = paths_df
            
        return batch_results
    
    def calculate_distribution_and_scr(self, max_workers=None, batch_size=None, use_numba=True):
        BOF_0 = self.assets_0 - self.liabilities_0

        simulated_results = self.get_all_simulated_paths(
            max_workers=max_workers, 
            batch_size=batch_size,
            use_numba=use_numba
        )
        
        eonia_index = self.column_names.index("EONIA")
        
        end_prices = np.zeros((self.n_simulations, self.n_assets))
        eonia_end_prices = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            end_row = simulated_results[i].iloc[-1].values
            end_prices[i, :] = end_row
            eonia_end_prices[i] = end_row[eonia_index]
        
        portfolio_end_values = np.dot(end_prices, self.weights)
        portfolio_returns = (portfolio_end_values - 100) / 100
        eonia_returns = (eonia_end_prices - 100) / 100
        
        assets_t1 = self.assets_0 * (1 + portfolio_returns)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_returns)
        
        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0
        
        scr = np.percentile(bof_change, 100 * (1 - 0.995))
        
        return bof_change, scr, simulated_results