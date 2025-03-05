import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skewtest
from backtesting.distribution_test.excess_kurtosis import assess_fat_tails
from backtesting.distribution_test.quantile_quantile_plot import qq_plot
import torch


def perform_distribution_tests(generated_tensor, empirical_returns_rolling, asset_name="Asset"):
    """
    Perform a suite of distribution tests and visualizations on a given asset.
    
    Inputs:
      - generated_tensor: a torch.Tensor with shape [10000, 252] representing GAN-generated one-year 
                          return scenarios.
      - empirical_returns_rolling: a 2D NumPy array (e.g., from create_rolling_empirical) containing rolling 
                                   one-year empirical return sequences.
      - asset_name: a label for the asset.
    
    The function:
      1. Flattens the generated and empirical rolling data to obtain overall 1D distributions.
      2. Computes and prints key moments (mean, std, skewness, excess kurtosis).
      3. Performs a skewness test (skewtest) on the generated distribution.
      4. Generates a Q-Q plot comparing generated vs. empirical distributions.
      5. Assesses fat tails by comparing the average excess kurtosis from the rolling windows.
    
    Returns:
      A dictionary containing all computed statistics and interpretations.
    """
    significance_level = float(os.getenv("SIGNIFICANCE_LEVEL", "0.05"))
    
    # Convert generated_tensor to numpy if it is a torch tensor.
    if isinstance(generated_tensor, torch.Tensor):
        generated_array = generated_tensor.cpu().numpy()  # shape: (10000, 252)
    else:
        generated_array = np.array(generated_tensor)
    
    # Flatten both distributions to obtain overall 1D arrays.
    generated_flat = generated_array.flatten()
    empirical_flat = np.array(empirical_returns_rolling).flatten()
    
    # Compute moments for generated distribution:
    gen_mean = np.mean(generated_flat)
    gen_std = np.std(generated_flat, ddof=1)
    gen_skew = np.mean((generated_flat - gen_mean)**3) / (gen_std**3)
    gen_excess_kurt = kurtosis(generated_flat, fisher=True, bias=False)
    
    # Compute moments for empirical distribution:
    emp_mean = np.mean(empirical_flat)
    emp_std = np.std(empirical_flat, ddof=1)
    emp_skew = np.mean((empirical_flat - emp_mean)**3) / (emp_std**3)
    emp_excess_kurt = kurtosis(empirical_flat, fisher=True, bias=False)
    
    print("="*50)
    print(f"Distribution Tests for {asset_name}")
    print("="*50)
    print("Overall Moments Comparison:")
    print(f"Generated -> Mean: {gen_mean:.4f}, Std: {gen_std:.4f}, Skewness: {gen_skew:.4f}, Excess Kurtosis: {gen_excess_kurt:.4f}")
    print(f"Empirical -> Mean: {emp_mean:.4f}, Std: {emp_std:.4f}, Skewness: {emp_skew:.4f}, Excess Kurtosis: {emp_excess_kurt:.4f}")
    
    # Perform skewtest on generated distribution:
    skew_stat, skew_pvalue = skewtest(generated_flat)
    print("\nSkewness Test on Generated Data:")
    print(f"Test Statistic: {skew_stat:.4f}")
    print(f"p-value: {skew_pvalue:.4f}")
    if skew_pvalue < significance_level:
        skew_interpretation = "Significant skewness detected (distribution is asymmetric)."
    else:
        skew_interpretation = "No significant skewness detected (cannot reject symmetry)."
    print("Skewness Interpretation:", skew_interpretation)
    
    # Create a Q-Q plot comparing the overall generated and empirical distributions.
    print("\nGenerating Q-Q Plot comparing Generated vs. Empirical Distributions...")
    qq_plot(generated_flat, empirical_flat)
    
    # Assess fat tails using the rolling windows.
    print("\nAssessing Fat Tails via Rolling Windows (Excess Kurtosis Comparison)...")
    fat_tail_results = assess_fat_tails(generated_array, empirical_returns_rolling)
    
    results = {
        "asset_name": asset_name,
        "significance_level": significance_level,
        "generated_moments": {"mean": gen_mean, "std": gen_std, "skewness": gen_skew, "excess_kurtosis": gen_excess_kurt},
        "empirical_moments": {"mean": emp_mean, "std": emp_std, "skewness": emp_skew, "excess_kurtosis": emp_excess_kurt},
        "skewtest": {"statistic": skew_stat, "p_value": skew_pvalue, "interpretation": skew_interpretation},
        "fat_tail_assessment": fat_tail_results
    }
    
    print("="*50)
    return results