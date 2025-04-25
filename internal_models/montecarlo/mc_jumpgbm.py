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

# Define a numba-optimized function for the simulation
@jit(nopython=True)
def _numba_simulate_gbm_with_jumps(T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales, seed):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    paths = np.zeros((T, n_assets))
    paths[0, :] = S0
    
    for t in range(1, T):
        # Generate correlated Brownian motions
        # Use the correct syntax for Numba-compatible random number generation
        Z = np.zeros(n_assets)
        for i in range(n_assets):
            Z[i] = np.random.normal(0.0, 1.0)  # Explicitly provide loc and scale parameters
            
        dW = np.dot(L, Z) * np.sqrt(dt)
        
        # Initialize jump process
        J = np.zeros(n_assets)
        
        for i in range(n_assets):
            # Poisson-distributed number of jumps
            num_jumps = np.random.poisson(jump_lambdas[i] * dt)
            if num_jumps > 0:
                # Sample jump size from normal distribution
                # Use a loop to generate multiple normal samples and sum them
                jump_sum = 0.0
                for j in range(num_jumps):
                    jump_sum += np.random.normal(jump_locs[i], jump_scales[i])
                J[i] = jump_sum
        
        # GBM with jumps formula
        paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + J)
    
    return paths

# Define a helper function at module level for multiprocessing
def _process_batch(args):
    """
    Process a batch of simulations.
    
    Args:
        args: A tuple containing (sim_ids, jump_params, T, n_assets, L, mu, sigma, dt, column_names)
        
    Returns:
        dict: Mapping of simulation IDs to their results
    """
    sim_ids, jump_params, T, n_assets, L, mu, sigma, dt, column_names = args
    
    batch_results = {}
    for sim_id in sim_ids:
        # Prepare parameters for numba function
        S0 = np.full(n_assets, 100)
        
        # Extract jump parameters into numpy arrays for numba compatibility
        jump_lambdas = np.zeros(n_assets)
        jump_locs = np.zeros(n_assets)
        jump_scales = np.zeros(n_assets)
        
        for i, name in enumerate(column_names):
            jump_lambdas[i] = jump_params[name]['lambda']
            jump_locs[i] = jump_params[name]['jump_distribution'].mean()
            jump_scales[i] = jump_params[name]['jump_distribution'].std()
        
        # Generate a seed based on sim_id for reproducibility
        seed = 42 + sim_id
        
        # Call the numba-optimized function
        try:
            paths_array = _numba_simulate_gbm_with_jumps(
                T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales, seed
            )
            
            # Convert back to DataFrame
            paths_df = pd.DataFrame(paths_array, columns=column_names)
            batch_results[sim_id] = paths_df
        except Exception as e:
            # If Numba fails, fall back to a non-optimized version
            print(f"Numba optimization failed for simulation {sim_id}, falling back to non-optimized version. Error: {str(e)}")
            paths_df = _fallback_simulate_gbm_with_jumps(
                T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales, seed, column_names
            )
            batch_results[sim_id] = paths_df
        
    return batch_results

# Fallback simulation function without Numba (pure NumPy)
def _fallback_simulate_gbm_with_jumps(T, n_assets, S0, L, mu, sigma, dt, jump_lambdas, jump_locs, jump_scales, seed, column_names):
    """
    Non-Numba fallback implementation of the GBM with jumps simulation
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    paths = np.zeros((T, n_assets))
    paths[0, :] = S0
    
    for t in range(1, T):
        # Generate correlated Brownian motions
        Z = np.random.normal(0, 1, size=n_assets)
        dW = np.dot(L, Z) * np.sqrt(dt)
        
        # Initialize jump process
        J = np.zeros(n_assets)
        
        for i in range(n_assets):
            # Poisson-distributed number of jumps
            num_jumps = np.random.poisson(jump_lambdas[i] * dt)
            if num_jumps > 0:
                # Sample jump size from normal distribution
                jump_sizes = np.random.normal(jump_locs[i], jump_scales[i], size=num_jumps)
                J[i] = np.sum(jump_sizes)
        
        # GBM with jumps formula
        paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + J)
    
    return pd.DataFrame(paths, columns=column_names)

class MonteCarloJumpGBM:

    def __init__(self, returns_df, weights):
        self.returns_df = returns_df
        self.assets_0 = int(os.getenv("INIT_ASSETS"))
        self.liabilities_0 = int(os.getenv("INIT_ASSETS")) * float(os.getenv("FRAC_LIABILITIES"))
        self.n_simulations = int(os.getenv("N_SIMULATIONS"))
        self.asset_classes = returns_df.columns
        self.weights = weights

        # PARAMTERS FOR GBM
        self.correlation_matrix = returns_df.corr().values  # Convert to numpy array for numba
        self.mu = returns_df.mean().values  # Convert to numpy array for numba
        self.sigma = returns_df.std().values  # Convert to numpy array for numba
        self.n_assets = returns_df.shape[1]
        self.T = int(os.getenv("N_DAYS"))
        self.dt = 1

        # Set threshold for identifying jumps (3-sigma rule)
        self.k = 3
        
        # Pre-compute Cholesky decomposition (only need to do this once)
        self.L = np.linalg.cholesky(self.correlation_matrix)
        
        # Store column names for later reconstruction
        self.column_names = returns_df.columns.tolist()

    def simulate_gbm_with_jumps(self, jump_params, sim_id=None):
        """
        Simulate a single path of geometric Brownian motion with jumps.
        
        Parameters:
        jump_params (dict): Dictionary of jump parameters for each asset
        sim_id (int, optional): Simulation ID for tracking
        
        Returns:
        tuple: (sim_id, paths_df) - Simulation ID and DataFrame with paths
        """
        # Prepare parameters for numba function
        S0 = np.full(self.n_assets, 100)
        
        # Extract jump parameters into numpy arrays for numba compatibility
        jump_lambdas = np.zeros(self.n_assets)
        jump_locs = np.zeros(self.n_assets)
        jump_scales = np.zeros(self.n_assets)
        
        for i, name in enumerate(self.column_names):
            jump_lambdas[i] = jump_params[name]['lambda']
            jump_locs[i] = jump_params[name]['jump_distribution'].mean()
            jump_scales[i] = jump_params[name]['jump_distribution'].std()
        
        # Generate a seed based on sim_id for reproducibility
        seed = 42 + (0 if sim_id is None else sim_id)
        
        # Call the numba-optimized function
        paths_array = _numba_simulate_gbm_with_jumps(
            self.T, self.n_assets, S0, self.L, self.mu, self.sigma, 
            self.dt, jump_lambdas, jump_locs, jump_scales, seed
        )
        
        # Convert back to DataFrame for compatibility with rest of the code
        paths_df = pd.DataFrame(paths_array, columns=self.column_names)
        
        return sim_id, paths_df

    def jump_params_est(self):
        # Vectorized operations for identifying jumps
        mu_array = np.array(self.mu).reshape(1, -1)  # Convert to 2D for broadcasting
        sigma_array = np.array(self.sigma).reshape(1, -1)
        returns_array = self.returns_df.values
        
        # Detect jumps (vectorized comparison)
        jumps_array = np.abs(returns_array - mu_array) > (self.k * sigma_array)
        jumps_df = pd.DataFrame(jumps_array, columns=self.returns_df.columns)
        
        # Estimate jump intensity (λ) as the fraction of days with jumps (vectorized)
        lambda_est = jumps_df.mean()
        
        # Extract jump sizes with vectorized operations
        jump_sizes = {}
        for i, asset in enumerate(self.column_names):
            jump_mask = jumps_array[:, i]
            if np.any(jump_mask):
                jump_sizes[asset] = returns_array[jump_mask, i]
            else:
                jump_sizes[asset] = np.array([])
        
        # Fit normal distribution to jump sizes
        jump_params_est = {}
        for i, asset in enumerate(self.column_names):
            jumps = jump_sizes[asset]
            if len(jumps) > 1:  # Ensure there are enough jumps to estimate
                loc, scale = norm.fit(jumps)  # Fit normal distribution
            else:
                # Use column index to access sigma values correctly
                loc, scale = 0, self.sigma[i]  # Default if no jumps detected
                print(f'No jumps detected for {asset}. Using default parameters.')
                
            jump_params_est[asset] = {'lambda': lambda_est[asset],
                                    'jump_distribution': norm(loc=loc, scale=scale)}
        
        return jump_params_est
    
    def get_all_simulated_paths(self, max_workers=None, batch_size=None, use_numba=True):
        """
        Run Monte Carlo simulations in parallel using concurrent.futures with batch processing.
        
        Parameters:
        max_workers (int, optional): The maximum number of worker processes to use.
                                    If None, it will default to the number of CPUs.
        batch_size (int, optional): Number of simulations to batch together for each worker.
                                    If None, it will be calculated automatically.
        use_numba (bool): Whether to use Numba optimization. If False, will use NumPy only.
        
        Returns:
        dict: Dictionary of simulated paths
        """
        start_time = time.time()
        
        # Estimate jump parameters once (shared across all simulations)
        jump_params_est = self.jump_params_est()
        
        # Default to CPU count if max_workers not specified
        if max_workers is None:
            max_workers = os.cpu_count()
            
        # Calculate optimal batch size if not specified
        if batch_size is None:
            # A simple heuristic: aim for ~4 tasks per worker
            batch_size = max(1, self.n_simulations // (max_workers * 4))
            
        print(f"Running {self.n_simulations} simulations with {max_workers} workers and batch size {batch_size}")
        print(f"Using {'Numba optimization' if use_numba else 'NumPy only (no Numba)'}")
        
        # Create batches of simulation IDs
        batches = []
        for i in range(0, self.n_simulations, batch_size):
            batch_end = min(i + batch_size, self.n_simulations)
            batches.append(list(range(i, batch_end)))
            
        simulated_paths = {}
        
        if not use_numba:
            # If not using Numba, perform simulations sequentially with NumPy only
            print("Using NumPy implementation (no Numba)...")
            for batch in tqdm(batches, desc="MonteCarlo GBM w Jumps", unit="batch"):
                batch_results = self._run_numpy_batch(batch, jump_params_est)
                simulated_paths.update(batch_results)
        else:
            # Try to use parallel processing with Numba
            try:
                # Test Numba function with a small sample to ensure it works
                test_sim_id = 0
                test_args = ([test_sim_id], jump_params_est, self.T, self.n_assets, 
                            self.L, self.mu, self.sigma, self.dt, self.column_names)
                test_result = _process_batch(test_args)
                print("Numba optimization test successful. Proceeding with parallel processing.")
                
                # Use ProcessPoolExecutor for CPU-bound tasks
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Prepare arguments for the _process_batch function
                    batch_args = [
                        (batch, jump_params_est, self.T, self.n_assets, 
                        self.L, self.mu, self.sigma, self.dt, self.column_names) 
                        for batch in batches
                    ]
                    
                    # Submit all batches to the executor
                    futures = [executor.submit(_process_batch, args) for args in batch_args]
                    
                    # Process results as they complete
                    for future in tqdm(
                        concurrent.futures.as_completed(futures), 
                        total=len(batches),
                        desc="MonteCarlo GBM w Jumps", 
                        unit="batch"
                    ):
                        batch_results = future.result()
                        simulated_paths.update(batch_results)
                        
            except Exception as e:
                print(f"Parallel processing with Numba failed: {str(e)}")
                print("Falling back to sequential NumPy implementation...")
                
                # Clear any partial results
                simulated_paths = {}
                
                # Fall back to NumPy implementation
                for batch in tqdm(batches, desc="MonteCarlo GBM w Jumps", unit="batch"):
                    batch_results = self._run_numpy_batch(batch, jump_params_est)
                    simulated_paths.update(batch_results)
        
        end_time = time.time()
        print(f"Completed {self.n_simulations} simulations in {end_time - start_time:.2f} seconds")
        
        return simulated_paths
        
    def _run_numpy_batch(self, sim_ids, jump_params):
        """
        Run a batch of simulations using NumPy (no Numba optimization).
        
        Parameters:
        sim_ids (list): List of simulation IDs to process
        jump_params (dict): Jump parameters for the simulations
        
        Returns:
        dict: Dictionary mapping simulation IDs to results
        """
        batch_results = {}
        
        for sim_id in sim_ids:
            # Set seed for reproducibility
            np.random.seed(42 + sim_id)
            
            # Initialize paths
            S0 = np.full(self.n_assets, 100)
            paths = np.zeros((self.T, self.n_assets))
            paths[0, :] = S0
            
            # Prepare jump parameters
            jump_lambdas = np.zeros(self.n_assets)
            jump_locs = np.zeros(self.n_assets)
            jump_scales = np.zeros(self.n_assets)
            
            for i, name in enumerate(self.column_names):
                jump_lambdas[i] = jump_params[name]['lambda']
                jump_locs[i] = jump_params[name]['jump_distribution'].mean()
                jump_scales[i] = jump_params[name]['jump_distribution'].std()
            
            # Simulate path
            for t in range(1, self.T):
                # Generate correlated Brownian motions
                Z = np.random.normal(0, 1, size=self.n_assets)
                dW = np.dot(self.L, Z) * np.sqrt(self.dt)
                
                # Initialize jump process
                J = np.zeros(self.n_assets)
                
                for i in range(self.n_assets):
                    # Poisson-distributed number of jumps
                    num_jumps = np.random.poisson(jump_lambdas[i] * self.dt)
                    if num_jumps > 0:
                        # Sample jump size from normal distribution
                        jump_sizes = np.random.normal(jump_locs[i], jump_scales[i], size=num_jumps)
                        J[i] = np.sum(jump_sizes)
                
                # GBM with jumps formula
                paths[t, :] = paths[t-1, :] * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW + J)
            
            # Convert to DataFrame and store
            paths_df = pd.DataFrame(paths, columns=self.column_names)
            batch_results[sim_id] = paths_df
            
        return batch_results
    
    def calculate_distribution_and_scr(self, max_workers=None, batch_size=None, use_numba=True):
        """
        Calculate the distribution of Basic Own Funds (BOF) changes and Solvency Capital Requirement (SCR).
        
        Parameters:
        max_workers (int, optional): The maximum number of worker processes to use for simulations.
        batch_size (int, optional): Number of simulations to batch together for each worker.
        use_numba (bool): Whether to use Numba optimization. If False, will use NumPy only.
        
        Returns:
        tuple: (bof_change, scr) - The distribution of BOF changes and the SCR value
        """
        BOF_0 = self.assets_0 - self.liabilities_0

        # Get simulation results with the optimized parallel implementation
        simulated_results = self.get_all_simulated_paths(
            max_workers=max_workers, 
            batch_size=batch_size,
            use_numba=use_numba
        )
        
        # Vectorized extraction of end prices
        eonia_index = self.column_names.index("EONIA")  # Find EONIA index
        
        # Preallocate arrays for better performance
        end_prices = np.zeros((self.n_simulations, self.n_assets))
        eonia_end_prices = np.zeros(self.n_simulations)
        
        # Extract end prices in a more efficient way
        for i in range(self.n_simulations):
            end_row = simulated_results[i].iloc[-1].values
            end_prices[i, :] = end_row
            eonia_end_prices[i] = end_row[eonia_index]
        
        # Vectorized calculations
        portfolio_end_values = np.dot(end_prices, self.weights)
        portfolio_returns = (portfolio_end_values - 100) / 100
        eonia_returns = (eonia_end_prices - 100) / 100
        
        assets_t1 = self.assets_0 * (1 + portfolio_returns)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_returns)
        
        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0
        
        # Calculate SCR (99.5th percentile of losses)
        scr = np.percentile(bof_change, 100 * (1 - 0.995))
        
        return bof_change, scr

