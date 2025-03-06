import numpy as np

def clean_eonia(gan_samples):
    """
    Clean the EONIA asset in the 3D gan_samples array (shape: n_simulations, n_days, n_assets).
    Replaces NaNs with the median and infinities with small numbers.
    """
    eonia = gan_samples[:, :, 6]
    median_val = np.nanmedian(eonia)
    eonia_clean = np.nan_to_num(eonia, nan=median_val, posinf=1e-6, neginf=-1e-6)
    gan_samples[:, :, 6] = eonia_clean
    return gan_samples
