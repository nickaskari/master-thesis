import numpy as np
import pandas as pd

def update_condition_matrix(historical_returns, test_returns, quarter_length=63):
    """
    Update the condition matrix for a given asset based on historical returns and out-of-sample test returns.
    
    The condition for the first quarter (Q0) is computed from the last quarter_length days 
    of historical_returns. For each quarter in test_returns, the condition is the cumulative return
    over that quarter.
    
    Parameters
    ----------
    historical_returns : pd.Series or array-like
        Historical returns (used during training).
    test_returns : array-like of shape (n_quarters, quarter_length)
        Out-of-sample test returns organized by quarter.
    quarter_length : int, default=63
        Number of days that define a quarter.
    
    Returns
    -------
    conditions : np.ndarray of shape (n_quarters+1, 1)
        The first row is the condition from the historical data (Q0), and each subsequent row is
        the cumulative return for each quarter from test_returns.
    """
    # Ensure historical_returns is a pandas Series.
    if not isinstance(historical_returns, pd.Series):
        historical_returns = pd.Series(historical_returns)
    
    # Compute the base condition from the last quarter_length days of historical_returns.
    base_window = historical_returns.iloc[-quarter_length:]
    cond_base = np.prod(1 + base_window) - 1  # Cumulative return
    
    conditions = [cond_base]
    
    # Ensure test_returns is a NumPy array.
    test_returns = np.array(test_returns)
    
    # For each quarter in test_returns, compute the cumulative return.
    for i in range(test_returns.shape[0]):
        quarter = test_returns[i]
        cond = np.prod(1 + quarter) - 1
        conditions.append(cond)
    
    return np.array(conditions).reshape(-1, 1)

# Example usage:
# Suppose historical_returns is a pandas Series of daily returns.
# And test_returns is a 2D array of shape (n_quarters, quarter_length)
# conditions = update_condition_matrix(historical_returns, test_returns, quarter_length=63)
# print(conditions)
