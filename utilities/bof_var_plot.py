import matplotlib.pyplot as plt
import numpy as np

def plot_bof_var(self, bof_change, scr, title):
    plt.figure(figsize=(10, 6))
    plt.hist(bof_change / 1e6, bins=500, alpha=0.7, color='c', label='Change in BOF Distribution', density=True)
    plt.axvline(np.percentile(bof_change, 0.5) / 1e6, color='r', linestyle='--', label=f'VaR (99.5%): {scr / 1e6:.2f}M')
    plt.title(title)
    plt.xlabel('Change in BOF (Millions)')
    plt.ylabel('Density')
    
    plt.xlim(-1, 1)
    plt.xticks(np.arange(-1, 1, 0.5))  # Increments of 0.5 within the range [-2, 2]

    plt.legend()
    plt.grid(False)
    plt.show()
