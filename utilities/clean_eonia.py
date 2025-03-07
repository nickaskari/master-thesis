import numpy as np

def clean_eonia(eonia_sanples):

    median_val = np.nanmedian(eonia_sanples)
    eonia_clean = np.nan_to_num(eonia_sanples, nan=median_val, posinf=1e-6, neginf=-1e-6)
    eonia_sanples = eonia_clean
    return eonia_sanples
