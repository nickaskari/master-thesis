import pandas as pd  
import numpy as np
from scipy.stats import norm, rankdata
from numba import jit, prange
import time
from tqdm import tqdm
from copulae import ClaytonCopula

@jit(nopython=True, cache=True)
def hull_white_jump_simulation(T, r0, kappa, theta, sigma, jump_intensity, jump_mean, jump_std, seed=42):
    np.random.seed(seed)
    dt = 1.0 / T
    rates = np.zeros(T)
    rates[0] = r0
    dW = np.random.normal(0, np.sqrt(dt), T-1)
    jump_randoms = np.random.random(T-1)
    jump_sizes = np.random.normal(jump_mean, jump_std, T-1)
    
    for t in range(1, T):
        drift = kappa * (theta - rates[t-1]) * dt
        diffusion = sigma * dW[t-1]
        
        jump = 0.0
        if jump_randoms[t-1] < jump_intensity * dt:
            jump = jump_sizes[t-1]
        
        rates[t] = rates[t-1] + drift + diffusion + jump
    
    return rates

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def simulate_independent_paths(n_sims, T, n_assets, S0, mu, sigma, dt, 
                              jump_lambdas, jump_locs, jump_scales, eonia_index, 
                              hw_r0, hw_kappa, hw_theta, hw_sigma, hw_jump_intensity, 
                              hw_jump_mean, hw_jump_std, seed):
    np.random.seed(seed)
    all_paths = np.zeros((n_sims, T, n_assets))
    sqrt_dt = np.sqrt(dt)
    
    for sim in prange(n_sims):
        all_paths[sim, 0, :] = S0
        
        if eonia_index >= 0:
            eonia_rates = hull_white_jump_simulation(
                T, hw_r0, hw_kappa, hw_theta, hw_sigma, 
                hw_jump_intensity, hw_jump_mean, hw_jump_std, seed + sim
            )
        
        for t in range(1, T):
            for i in range(n_assets):
                if i == eonia_index and eonia_index >= 0:
                    daily_return = eonia_rates[t] - eonia_rates[t-1]
                    all_paths[sim, t, i] = all_paths[sim, t-1, i] * (1 + daily_return)
                else:
                    dW = np.random.normal(0, sqrt_dt)
                    
                    J_multiplier = 1.0
                    if np.random.random() < jump_lambdas[i] * dt:
                        log_jump = jump_locs[i] + jump_scales[i] * np.random.randn()
                        J_multiplier = np.exp(log_jump)
                    
                    kappa_jump = np.exp(jump_locs[i] + 0.5 * jump_scales[i]**2) - 1
                    drift = (mu[i] - jump_lambdas[i] * kappa_jump - 0.5 * sigma[i]**2) * dt
                    diffusion = sigma[i] * dW
                    
                    all_paths[sim, t, i] = all_paths[sim, t-1, i] * np.exp(drift + diffusion) * J_multiplier
    
    return all_paths
"""
def apply_clayton_copula(returns, theta=4.0):
    n_sims, n_assets = returns.shape
    
    uniform_samples = np.zeros_like(returns)
    for i in range(n_assets):
        ranks = rankdata(returns[:, i], method='average')
        uniform_samples[:, i] = ranks / (n_sims + 1)
    
    U = np.random.uniform(0, 1, (n_sims, n_assets))
    clayton_uniform = np.zeros_like(U)
    clayton_uniform[:, 0] = U[:, 0]
    
    for i in range(1, n_assets):
        clayton_uniform[:, i] = ((U[:, i] ** (-theta/(1+theta*(i))) - 1) *
                                clayton_uniform[:, 0] ** (-theta) + 1) ** (-1/theta)
    
    copula_returns = np.zeros_like(returns)
    for i in range(n_assets):
        sorted_original = np.sort(returns[:, i])
        indices = np.clip((clayton_uniform[:, i] * (n_sims - 1)).astype(int), 0, n_sims - 1)
        copula_returns[:, i] = sorted_original[indices]
    
    return copula_returns
"""
def apply_clayton_copula(returns, theta=1):
    n_sims, n_assets = returns.shape
    
    uniform_samples = np.zeros_like(returns)

    uniform_samples = np.zeros_like(returns)
    for i in range(n_assets):
        ranks = rankdata(returns[:, i], method='average')
        uniform_samples[:, i] = ranks / (n_sims + 1)

    copula = ClaytonCopula(dim=n_assets)
    copula.params = theta
    clayton_uniform = copula.random(n_sims)

    copula_samples = np.zeros_like(returns)
    for i in range(n_assets):
        sorted_original = np.sort(returns[:, i])
        indices = np.clip((clayton_uniform[:, i] * (n_sims - 1)).astype(int), 0, n_sims - 1)
        copula_samples[:, i] = sorted_original[indices]

    return copula_samples

def estimate_kappa_mle(rate_series):
    rates = rate_series.dropna().values
    if len(rates) < 10:
        return 0.2
    
    dr = np.diff(rates)
    r_lag = rates[:-1]
    
    X = np.column_stack([np.ones(len(r_lag)), r_lag])
    beta = np.linalg.lstsq(X, dr, rcond=None)[0]
    kappa = -beta[1]
    
    return max(0.01, min(kappa, 1.0))

class MonteCarloJumpGBM:
    
    def __init__(self, returns_df, weights, kappa=None, clayton_theta=3.0):
        self.returns_df = returns_df
        self.weights = weights
        self.assets_0 = 1000000
        self.liabilities_0 = 880000
        self.n_simulations = 10000
        self.T = 252
        self.dt = 1
        self.k = 3
        self.clayton_theta = clayton_theta
        
        self.mu = returns_df.mean().values
        self.sigma = returns_df.std().values
        self.n_assets = len(returns_df.columns)
        self.column_names = returns_df.columns.tolist()
        
        self.eonia_index = self.column_names.index("EONIA") if "EONIA" in self.column_names else -1
        self.kappa = kappa
        self.hull_white_params = self._estimate_hull_white_params()
    
    def _estimate_hull_white_params(self):
        if self.eonia_index >= 0:
            eonia_series = self.returns_df.iloc[:, self.eonia_index]
            
            kappa_val = self.kappa if self.kappa is not None else estimate_kappa_mle(eonia_series)
            
            eonia_mean = eonia_series.mean()
            eonia_std = eonia_series.std()
            jumps_mask = np.abs(eonia_series - eonia_mean) > (3 * eonia_std)
            jump_freq = np.mean(jumps_mask) if len(jumps_mask) > 0 else 0.01
            
            if np.any(jumps_mask):
                jumps = eonia_series[jumps_mask]
                jump_mean, jump_std = jumps.mean(), jumps.std()
            else:
                jump_mean, jump_std = 0.0, 0.001
            
            return {
                'kappa': kappa_val,
                'theta': float(eonia_mean) if not pd.isna(eonia_mean) else -0.00001,
                'r0': float(eonia_series.iloc[0]) if not pd.isna(eonia_series.iloc[0]) else -0.00001,
                'sigma': float(eonia_std * np.sqrt(252)) if not pd.isna(eonia_std) else 0.0001,
                'jump_intensity': jump_freq * 252,
                'jump_mean': float(jump_mean) if not pd.isna(jump_mean) else 0.0,
                'jump_std': float(jump_std) if not pd.isna(jump_std) else 0.001
            }
        
        kappa_val = self.kappa if self.kappa is not None else 0.2
        return {
            'kappa': kappa_val, 'theta': -0.00001, 'r0': -0.00001, 'sigma': 0.0001,
            'jump_intensity': 2.5, 'jump_mean': 0.0, 'jump_std': 0.001
        }
    
    def _estimate_jump_params(self):
        jump_params = {}
        returns_array = self.returns_df.values
        
        for i, asset in enumerate(self.column_names):
            jumps_mask = np.abs(returns_array[:, i] - self.mu[i]) > (self.k * self.sigma[i])
            jump_freq = np.mean(jumps_mask)
            
            if np.any(jumps_mask):
                jumps = returns_array[jumps_mask, i]
                jump_mean, jump_std = norm.fit(jumps) if len(jumps) > 1 else (0, self.sigma[i])
            else:
                jump_mean, jump_std = 0, self.sigma[i]
            
            jump_params[asset] = {
                'lambda': jump_freq,
                'mean': jump_mean,
                'std': jump_std
            }
        
        return jump_params
    
    def run_simulation(self):
        jump_params = self._estimate_jump_params()
        jump_lambdas = np.array([jump_params[name]['lambda'] for name in self.column_names])
        jump_locs = np.array([jump_params[name]['mean'] for name in self.column_names])
        jump_scales = np.array([jump_params[name]['std'] for name in self.column_names])
        
        S0 = np.full(self.n_assets, 100.0)
        hw_params = self.hull_white_params
        
        paths = simulate_independent_paths(
            self.n_simulations, self.T, self.n_assets, S0,
            self.mu, self.sigma, self.dt,
            jump_lambdas, jump_locs, jump_scales, self.eonia_index,
            hw_params['r0'], hw_params['kappa'], hw_params['theta'], hw_params['sigma'],
            hw_params['jump_intensity'], hw_params['jump_mean'], hw_params['jump_std'], 42
        )
        
        cumulative_returns = (paths[:, -1, :] / paths[:, 0, :]) - 1
        clayton_returns = apply_clayton_copula(cumulative_returns, self.clayton_theta)
        
        paths[:, -1, :] = paths[:, 0, :] * (1 + clayton_returns)
        
        return paths
    
    def calculate_scr(self):
        paths = self.run_simulation()
        
        end_prices = paths[:, -1, :]
        
        portfolio_returns = (np.dot(end_prices, self.weights) - 100) / 100
        
        if self.eonia_index >= 0:
            eonia_returns = (end_prices[:, self.eonia_index] - 100) / 100
        else:
            eonia_returns = np.zeros(self.n_simulations)
        
        BOF_0 = self.assets_0 - self.liabilities_0
        assets_t1 = self.assets_0 * (1 + portfolio_returns)
        liabilities_t1 = self.liabilities_0 * (1 + eonia_returns)
        bof_t1 = assets_t1 - liabilities_t1
        bof_change = bof_t1 - BOF_0
        
        scr = np.percentile(bof_change, 0.5)
        
        return {
            'scr': scr,
            'bof_change': bof_change,
            'mean_bof': np.mean(bof_change),
            'std_bof': np.std(bof_change),
            'var_95': np.percentile(bof_change, 5),
            'var_99': np.percentile(bof_change, 1)
        }