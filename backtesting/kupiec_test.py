import numpy as np
import scipy.stats as stats

def kupiec_pof_test(failures, VaR_confidence):
    """
    Parameters:
    failures (list or np.array): Binary sequence where 1 represents a VaR exception (failure) 
                                 and 0 represents no failure.
    VaR_confidence (float): The confidence level of the VaR model (e.g., 0.99 for 99% VaR).
    
    """
    T = len(failures)  
    x = np.sum(failures)  

    # Expected failure probability 
    p = 1 - VaR_confidence  
    
    # Observed failure probability
    p_hat = x / T  

    L0 = (1 - p) ** (T - x) * p ** x  
    L1 = (1 - p_hat) ** (T - x) * p_hat ** x  

    # Likelihood Ratio Statistic
    LR_pof = -2 * np.log(L0 / L1) if L1 > 0 else np.inf  

    p_value = 1 - stats.chi2.cdf(LR_pof, df=1)

    return LR_pof, p_value


################################################ SMALL TEST ################################################

# Example: Simulated sequence of VaR failures (1 = failure, 0 = no failure)
np.random.seed(42)  # Set seed for reproducibility
failures = np.random.choice([0, 1], size=250, p=[0.99, 0.01])  # Simulating 1% failure rate

# Run Kupiec's Proportion of Failures Test (for 99% VaR model)
LR_stat, p_val = kupiec_pof_test(failures, VaR_confidence=0.99)

# Display results
print(LR_stat, p_val)
