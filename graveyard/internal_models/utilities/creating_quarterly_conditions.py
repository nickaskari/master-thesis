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
    test_returns : array-like
        Out-of-sample test returns. This can be a pandas DataFrame/Series or a 1D array 
        containing daily returns for the full out-of-sample period (e.g. 252 days).
    quarter_length : int, default=63
        Number of days that define a quarter.
    
    Returns
    -------
    conditions : np.ndarray of shape (n_quarters+1, 1)
        The first row is the condition from the historical data (Q0), and each subsequent row is
        the cumulative return for each quarter computed from test_returns.
    """
    # Ensure historical_returns is a pandas Series.
    if not isinstance(historical_returns, pd.Series):
        historical_returns = pd.Series(historical_returns)
    
    # Compute the base condition from the last quarter_length days of historical_returns.
    base_window = historical_returns.iloc[-quarter_length:]
    cond_base = np.prod(1 + base_window) - 1  # Cumulative return
    
    conditions = [cond_base]
    
    # Convert test_returns to a NumPy array.
    test_returns = np.array(test_returns)
    
    # If test_returns is 1D, assume it contains daily returns and divide it into quarters.
    if test_returns.ndim == 1:
        total_days = len(test_returns)
        n_quarters = total_days // quarter_length
        # Truncate any extra days that don't form a complete quarter.
        test_returns = test_returns[:n_quarters * quarter_length].reshape(n_quarters, quarter_length)
    
    # If test_returns is 2D but not of shape (n_quarters, quarter_length), try to handle it.
    elif test_returns.ndim == 2:
        if test_returns.shape[1] != quarter_length:
            # If the number of rows is equal to quarter_length, assume the days are in rows and transpose.
            if test_returns.shape[0] == quarter_length:
                test_returns = test_returns.T
            else:
                # Otherwise, truncate columns if there are extra days.
                test_returns = test_returns[:, :quarter_length]
    
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


def update_condition_matrix_volatility(historical_returns, test_returns, quarter_length=63):
    """
    Update the condition matrix for a given asset based on historical returns and out-of-sample test returns,
    using volatility (standard deviation) as the condition instead of cumulative return.
    
    The condition for the first quarter (Q0) is computed as the volatility (std) of the last quarter_length days 
    of historical_returns. For each quarter in test_returns, the condition is the volatility (std) over that quarter.
    
    Parameters
    ----------
    historical_returns : pd.Series or array-like
        Historical returns (used during training).
    test_returns : array-like
        Out-of-sample test returns. This can be a pandas DataFrame/Series or a 1D array 
        containing daily returns for the full out-of-sample period (e.g. 252 days).
    quarter_length : int, default=63
        Number of days that define a quarter.
    
    Returns
    -------
    conditions : np.ndarray of shape (n_quarters+1, 1)
        The first row is the condition from historical data (Q0), computed as the volatility (std) over the last quarter_length days.
        Each subsequent row is the volatility (std) for each quarter computed from test_returns.
    """
    # Ensure historical_returns is a pandas Series.
    if not isinstance(historical_returns, pd.Series):
        historical_returns = pd.Series(historical_returns)
    
    # Compute the base condition from the last quarter_length days of historical_returns.
    base_window = historical_returns.iloc[-quarter_length:]
    cond_base = base_window.std()  # Volatility as the standard deviation
    
    conditions = [cond_base]
    
    # Convert test_returns to a NumPy array.
    test_returns = np.array(test_returns)
    
    # If test_returns is 1D, assume it contains daily returns and divide it into quarters.
    if test_returns.ndim == 1:
        total_days = len(test_returns)
        n_quarters = total_days // quarter_length
        # Truncate any extra days that don't form a complete quarter.
        test_returns = test_returns[:n_quarters * quarter_length].reshape(n_quarters, quarter_length)
    
    # If test_returns is 2D but not of shape (n_quarters, quarter_length), try to handle it.
    elif test_returns.ndim == 2:
        if test_returns.shape[1] != quarter_length:
            # If the number of rows is equal to quarter_length, assume the days are in rows and transpose.
            if test_returns.shape[0] == quarter_length:
                test_returns = test_returns.T
            else:
                # Otherwise, truncate columns if there are extra days.
                test_returns = test_returns[:, :quarter_length]
    
    # For each quarter in test_returns, compute the volatility.
    for i in range(test_returns.shape[0]):
        quarter = test_returns[i]
        cond = np.std(quarter) # Standard deviation over the quarter
        if cond > 0.05:
            cond *= 100
        conditions.append(cond)
    
    return np.array(conditions).reshape(-1, 1)
