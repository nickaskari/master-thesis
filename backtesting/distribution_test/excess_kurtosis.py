import numpy as np
from scipy.stats import kurtosis

def assess_fat_tails(generated_returns, empirical_returns_rolling):
    """
    Assess the stylized fact of fat tails by comparing the excess kurtosis of GAN-generated 
    one-year ahead return distributions to that of empirical rolling one-year returns.
    
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
          - mean_generated_excess_kurtosis: Average excess kurtosis across generated scenarios.
          - mean_empirical_excess_kurtosis: Average excess kurtosis across empirical windows.
          - difference: Difference (generated minus empirical).
          - generated_excess_kurtosis_list: Excess kurtosis for each GAN scenario.
          - empirical_excess_kurtosis_list: Excess kurtosis for each empirical window.
    """

    gen_returns = np.array(generated_returns)
    emp_returns = np.array(empirical_returns_rolling)
    
    # fisher=True gives excess kurtosis (i.e. normal => 0), and bias=False uses an unbiased estimator.
    gen_excess_kurtosis = kurtosis(gen_returns, fisher=True, bias=False, axis=1)
    emp_excess_kurtosis = kurtosis(emp_returns, fisher=True, bias=False, axis=1)
    
    mean_gen_kurt = np.mean(gen_excess_kurtosis)
    mean_emp_kurt = np.mean(emp_excess_kurtosis)
    diff = mean_gen_kurt - mean_emp_kurt
    
    print("Mean Generated Excess Kurtosis:", mean_gen_kurt)
    print("Mean Empirical Excess Kurtosis:", mean_emp_kurt)
    print("Difference (Generated - Empirical):", diff)
    
    # Interpretation (using a rough benchmark):
    # Empirical daily returns typically have excess kurtosis in the range of 4-8 (when aggregated to a 
    # yearly scale the value might change, so use your domain knowledge).
    if np.abs(diff) < 1:
        interpretation = ("The proposed distributions are close to the empirical benchmark "
                          "in terms of tail heaviness.")
    elif diff > 1:
        interpretation = ("The proposed distributions exhibit significantly heavier tails than "
                          "the empirical data.")
    else:
        interpretation = ("The proposed distributions exhibit significantly lighter tails than "
                          "the empirical data.")
    print("Interpretation:", interpretation)
    
    result = {
        "mean_generated_excess_kurtosis": mean_gen_kurt,
        "mean_empirical_excess_kurtosis": mean_emp_kurt,
        "difference": diff,
        "generated_excess_kurtosis_list": gen_excess_kurtosis,
        "empirical_excess_kurtosis_list": emp_excess_kurtosis,
        "interpretation": interpretation
    }
    
    return result
