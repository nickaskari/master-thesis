import numpy as np
import scipy.stats as stats

def christoffersen_independence_test(failures):
    """
    Parameters:
    failures (list or np.array): Binary sequence where 1 represents a VaR exception (failure) 
                                 and 0 represents no failure.

    Returns:
    tuple: (Likelihood Ratio Statistic, p-value)
    """

    n00 = np.sum((failures[:-1] == 0) & (failures[1:] == 0)) 
    n01 = np.sum((failures[:-1] == 0) & (failures[1:] == 1)) 
    n10 = np.sum((failures[:-1] == 1) & (failures[1:] == 0)) 
    n11 = np.sum((failures[:-1] == 1) & (failures[1:] == 1)) 


    Pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0  # Probability of failure after a non-failure
    Pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0  # Probability of failure after a failure
    Pi = (n01 + n11) / (n00 + n01 + n10 + n11)  # Overall failure probability

    # Likelihood functions
    L0 = (1 - Pi) ** (n00 + n10) * Pi ** (n01 + n11)  
    L1 = (1 - Pi_0) ** n00 * Pi_0 ** n01 * (1 - Pi_1) ** n10 * Pi_1 ** n11  

    # Likelihood Ratio Statistic
    LR_ind = -2 * np.log(L0 / L1) if L1 > 0 else np.inf

    # Compute p-value from Chi-square distribution (1 degree of freedom)
    p_value = 1 - stats.chi2.cdf(LR_ind, df=1)

    return LR_ind, p_value

################################################ SMALL TEST ################################################

# Example: Simulated sequence of VaR failures (1 = failure, 0 = no failure)
np.random.seed(42)  # Set seed for reproducibility
failures = np.random.choice([0, 1], size=250, p=[0.95, 0.05])  # Simulating 5% failure rate

# Run Christoffersen's Independence Test
LR_stat, p_val = christoffersen_independence_test(failures)

print(LR_stat, p_val)
