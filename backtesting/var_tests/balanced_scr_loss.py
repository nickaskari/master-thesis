import numpy as np


'''
This is a subjective test, to better capture what insurance companies really want. A low SCR, whilst still being a valid value-at-risk estimate. 

Should punish being to conservative.
'''

def balanced_scr_loss(BOF, VaR, alpha=0.5):
    """
    Computes a balanced loss metric for level-based VaR that penalizes both
    over-conservatism (VaR significantly above BOF) and under-conservatism
    (breaches where BOF falls below VaR).

    Parameters
    ----------
    BOF : array-like of shape (n,)
        Realized Balance of Fund levels over time.
    VaR : float or array-like
        VaR level(s). If a scalar, that value is used for all days.
        If an array with m <= n, then each VaR value is applied evenly over the n days.
    alpha : float, default 0.5
        Weighting factor (0 <= alpha <= 1):
          - alpha=1: only penalizes over-conservatism (capital inefficiency),
          - alpha=0: only penalizes breaches (under-conservatism),
          - intermediate values trade off the two.

    Returns
    -------
    float
        The average balanced loss metric.
        
    For each day t, let:
      - OCP_t = max(VaR_t - BOF_t, 0)
      - UCP_t = { ((VaR_t - BOF_t)^2) / (VaR_t^2)  if BOF_t < VaR_t, else 0 }
      and the daily loss is:
      
          L_t = alpha * OCP_t + (1 - alpha) * UCP_t.
          
    The final metric is the average of L_t over all days.
    """
    BOF = np.asarray(BOF, dtype=float)
    n = len(BOF)
    
    # Expand VaR to full length if necessary:
    if np.isscalar(VaR):
        VaR_full = np.full(n, VaR, dtype=float)
    else:
        VaR = np.asarray(VaR, dtype=float)
        m = len(VaR)
        if m == n:
            VaR_full = VaR
        elif m < n:
            # Stretch VaR by assigning each block of returns the same VaR value.
            indices = (np.floor(np.arange(n) * m / n)).astype(int)
            VaR_full = VaR[indices]
        else:
            raise ValueError("Length of VaR cannot exceed the length of BOF.")
    
    # Over-Conservatism Penalty: when VaR is above BOF.
    ocp = np.clip(abs(VaR_full - BOF), 0, None)
    
    # Under-Conservatism Penalty: when BOF is below VaR.
    mask = BOF < VaR_full
    ucp = np.zeros(n, dtype=float)
    ucp[mask] = ((VaR_full[mask] - BOF[mask])**2) / (VaR_full[mask]**2)
    
    # Combined daily loss
    daily_loss = alpha * ocp + (1 - alpha) * ucp
    
    M = daily_loss.mean()
    
    return M