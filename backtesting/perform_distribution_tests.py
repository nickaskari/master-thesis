import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skewtest
from backtesting.distribution_tests.pca_plot import pca_plot
from backtesting.distribution_tests.excess_kurtosis import assess_fat_tails
from backtesting.distribution_tests.quantile_quantile_plot import qq_plot
import torch

def perform_distribution_tests(generated_returns, empirical_returns_rolling, asset_name, verbose=True, bof=False):
    """
    Perform a suite of distribution tests and visualizations on a given asset.
    
    Inputs:
      - generated_returns: a torch.Tensor with shape [10000, 252] representing GAN-generated one-year 
                           return scenarios.
      - empirical_returns_rolling: a 2D NumPy array (e.g., from create_rolling_empirical) containing rolling 
                                   one-year empirical return sequences.
      - asset_name: a label for the asset.
      - bof: Boolean flag to indicate if the tests are for BOF distributions (affects the Q-Q plot and PCA).
      - verbose: If True, prints diagnostic information.
    
    Returns:
      A flat dictionary containing computed statistics.
    """
    significance_level = float(os.getenv("SIGNIFICANCE_LEVEL", "0.05"))
    
    # Convert generated_returns to numpy if needed.
    if isinstance(generated_returns, torch.Tensor):
        generated_array = generated_returns.cpu().numpy()  # Expected shape: (10000, 252)
    else:
        generated_array = np.array(generated_returns)
    
    # Flatten both distributions to obtain overall 1D arrays.
    generated_flat = generated_array.flatten()
    empirical_flat = np.array(empirical_returns_rolling).flatten()
    
    # Compute moments for generated data.
    gen_mean = np.mean(generated_flat)
    gen_std = np.std(generated_flat, ddof=1)
    gen_skew = np.mean((generated_flat - gen_mean)**3) / (gen_std**3)
    
    # Compute moments for empirical data.
    emp_mean = np.mean(empirical_flat)
    emp_std = np.std(empirical_flat, ddof=1)
    emp_skew = np.mean((empirical_flat - emp_mean)**3) / (emp_std**3)
    
    if verbose:
        print("="*150)
        print(f"Distribution Tests for {asset_name}")
        print("="*150)
        print("Overall Moments Comparison:")
        print(f"Generated -> Mean: {gen_mean:.4f}, Std: {gen_std:.4f}, Skewness: {gen_skew:.4f}")
        print(f"Empirical -> Mean: {emp_mean:.4f}, Std: {emp_std:.4f}, Skewness: {emp_skew:.4f}")
    
    # Perform skewtest on generated distribution.
    from scipy.stats import skewtest
    skew_stat, skew_pvalue = skewtest(generated_flat)
    if skew_pvalue < significance_level:
        skew_interpretation = "Significant skewness detected (distribution is asymmetric)."
    else:
        skew_interpretation = "No significant skewness detected (cannot reject symmetry)."
    
    if verbose:
        print("\nSkewness Test on Generated Data:")
        print(f"Test Statistic: {skew_stat:.4f}")
        print(f"p-value: {skew_pvalue:.4f}")
        print("Skewness Interpretation:", skew_interpretation)
        print("\nGenerating Q-Q Plot comparing Generated vs. Empirical Distributions...")
    qq_plot(generated_flat, empirical_flat, bof=bof)
    
    # For BOF, perform PCA on flat data by reshaping to (-1, 1)
    if bof:
        # Call the PCA plotting function (which should be defined elsewhere)
        print("empirical returns rolling shape", empirical_returns_rolling.shape)
        explained_variance = pca_plot(empirical_returns_rolling, generated_returns, title=f"PCA of {asset_name} BOF: Empirical vs Generated")
        pca_generated_pc1 = explained_variance[0]
        pca_generated_pc2 = explained_variance[1] if explained_variance[1] is not None else None
    else:
        pca_generated_pc1 = None
        pca_generated_pc2 = None

    if verbose:
        print("\nAssessing Fat Tails via Rolling Windows (Excess Kurtosis Comparison)...")
    fat_tail_results = assess_fat_tails(generated_array, empirical_returns_rolling)
    
    # Create a flat dictionary with all results.
    results = {
        "asset_name": asset_name,
        "significance_level": significance_level,
        "gen_mean": gen_mean,
        "gen_std": gen_std,
        "gen_skew": gen_skew,
        "emp_mean": emp_mean,
        "emp_std": emp_std,
        "emp_skew": emp_skew,
        "skew_stat": skew_stat,
        "skew_pvalue": skew_pvalue,
        "skew_interpretation": skew_interpretation,
        "pca_generated_pc1": pca_generated_pc1,
        "pca_generated_pc2": pca_generated_pc2,
    }
    
    # Flatten fat tail results into the dictionary.
    if isinstance(fat_tail_results, dict):
        for key, value in fat_tail_results.items():
            results[f"fat_tail_{key}"] = value
    else:
        results["fat_tail_assessment"] = fat_tail_results

    if verbose:
        print("="*150)
    return results
