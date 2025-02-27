import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_bof_var(bof_change, scr, title, bof_0):
    scr_percentage = (scr / bof_0) * 100  # Calculate SCR as % of BOF_0
    bof_change_millions = bof_change / 1e6  # Convert to millions

    plt.figure(figsize=(10, 6))

    # Original histogram (unchanged)
    plt.hist(bof_change_millions, bins=500, alpha=0.7, color='c', label='Change in BOF Distribution', density=True)
    
    # KDE curve added for extra visualization
    sns.kdeplot(bof_change_millions, color='b', linewidth=2, label="KDE Estimate")

    # SCR (99.5% VaR) vertical line
    plt.axvline(scr / 1e6, color='r', linestyle='--', linewidth=2)

    # Title and labels
    plt.title(title)
    plt.xlabel('Change in BOF (Millions)')
    plt.ylabel('Density')

    # X-axis limits and ticks
    plt.xlim(-1, 1)
    plt.xticks(np.arange(-1, 1.1, 0.5))  # Increments of 0.5 within range [-1,1]

    # Legend with SCR information
    plt.legend([
        'Change in BOF Distribution',
        'KDE Estimate',
        f'VaR (99.5%): {scr / 1e6:.2f}M ({scr_percentage:.2f}%)'
    ])

    plt.grid(False)
    plt.show()