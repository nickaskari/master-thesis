import numpy as np
from scipy.stats import kurtosis

def assess_fat_tails(generated_returns, empirical_returns_rolling):
    """
    Assess the stylized fact of fat tails by comparing the overall excess kurtosis of 
    GAN-generated one-year ahead return distributions to that of empirical rolling one-year returns.
    
    Parameters
    ----------
    generated_returns : array-like, shape (num_scenarios, num_days)
        GAN-generated return distributions for one-year ahead (e.g., 252 days).
    empirical_returns_rolling : array-like, shape (num_windows, window_size)
        Rolling one-year (e.g., 252-day) sequences from your empirical data.
    
    Returns
    -------
    result : dict
        A dictionary containing:
          - generated_excess_kurtosis: Overall excess kurtosis for the generated data.
          - empirical_excess_kurtosis: Overall excess kurtosis for the empirical data.
          - difference: Difference (generated minus empirical).
          - interpretation: Text interpretation of the difference.
    """
    # Flatten the arrays to compute overall distribution moments.
    gen_flat = np.array(generated_returns).flatten()
    emp_flat = np.array(empirical_returns_rolling).flatten()
    
    # Compute overall excess kurtosis. With fisher=True, a normal distribution has 0 excess kurtosis.
    generated_excess_kurtosis = kurtosis(gen_flat, fisher=True, bias=False)
    empirical_excess_kurtosis = kurtosis(emp_flat, fisher=True, bias=False)
    
    diff = generated_excess_kurtosis - empirical_excess_kurtosis
    
    print("Overall Mean Generated Excess Kurtosis:", generated_excess_kurtosis)
    print("Overall Mean Empirical Excess Kurtosis:", empirical_excess_kurtosis)
    print("Difference (Generated - Empirical):", diff)
    
    # Interpretation (using rough benchmarks - adjust based on your domain knowledge):
    if np.abs(diff) < 1:
        interpretation = ("The generated distribution's tail heaviness is close to the empirical benchmark.")
    elif diff > 1:
        interpretation = ("The generated distribution exhibits significantly heavier tails than the empirical data.")
    else:
        interpretation = ("The generated distribution exhibits significantly lighter tails than the empirical data.")
    print("Interpretation:", interpretation)
    
    result = {
        "generated_excess_kurtosis": generated_excess_kurtosis,
        "empirical_excess_kurtosis": empirical_excess_kurtosis,
        "difference": diff,
        "interpretation": interpretation
    }
    
    return result