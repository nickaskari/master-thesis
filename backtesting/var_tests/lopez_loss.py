import numpy as np

def lopez_average_loss(returns, var_forecast):
    """
    Computes the average Lopez loss for VaR backtesting.
    
    The Lopez loss for day t is defined as:
        L_t = 1 + ((r_t - VaR_t)^2 / VaR_t^2)   if r_t < VaR_t,
              0                              otherwise.
              
    If the provided var_forecast is shorter than the returns array,
    it is assumed that each VaR value in var_forecast applies to a block
    of returns (stretched evenly across the sample).
    
    Parameters:
    -----------
    returns : array-like of shape (n,)
        Out-of-sample realized returns (or negative PnL).
    var_forecast : float or array-like
        - If a single float, that VaR is used for all days.
        - If an array of length m (m <= n), the VaR forecast is assumed
          to be constant for each block of returns. If m == n, each day's 
          VaR is used individually.
    
    Returns:
    --------
    avg_loss : float
        The sample average Lopez loss.
    """
    returns = np.asarray(returns)
    n = len(returns)
    
    if np.isscalar(var_forecast):
        var_forecast_full = np.full(n, var_forecast, dtype=float)
    else:
        var_forecast = np.asarray(var_forecast, dtype=float)
        m = len(var_forecast)
        if m == n:
            var_forecast_full = var_forecast
        elif m < n:
            indices = (np.floor(np.arange(n) * m / n)).astype(int)
            var_forecast_full = var_forecast[indices]
        else:
            raise ValueError("Length of var_forecast cannot exceed the length of returns.")
    
    L = np.zeros(n)
    mask = returns < var_forecast_full

    L[mask] = 1 + ((returns[mask] - var_forecast_full[mask])**2) / (var_forecast_full[mask]**2)
    
    avg_loss = L.mean()
    return avg_loss