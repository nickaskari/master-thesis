import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

def plot_bof_var(bof_change, scr, title):
    assets_0 = int(os.getenv("INIT_ASSETS"))

    bof_change = np.array(bof_change)

    scr_percentage = (scr / assets_0) * 100  # Calculate SCR as % of assets_0
    bof_change_millions = bof_change / 1e6  # Convert to millions
    scr_millions = scr / 1e6

    plt.figure(figsize=(10, 6))

    plt.hist(bof_change_millions, bins=500, alpha=0.7, color='c', label='Change in BOF Distribution', density=True)
    sns.kdeplot(bof_change_millions, color='b', linewidth=2, label="KDE Estimate")
    plt.axvline(scr / 1e6, color='r', linestyle='--', linewidth=2)

    plt.title(title)
    plt.xlabel('Change in BOF (Millions)')
    plt.ylabel('Density')

    plt.xlim(-1, 1)
    plt.xticks(np.arange(-1, 1.1, 0.5))  # Increments of 0.5 within range [-1,1]

    plt.legend([
        'Change in BOF Distribution',
        'KDE Estimate',
        f'VaR (99.5%): {scr / 1e6:.2f}M ({scr_percentage:.2f}%)'
    ])

    plt.grid(False)
    plt.show()


def plot_3d_density(data_collection, x_range=np.linspace(-1, 1, 200)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, data in enumerate(data_collection):
        # Evaluate the stored KDE function on our x_range
        density = data['kde'](x_range)
        
        # Plot the density curve in 3D
        ax.plot(x_range, np.ones_like(x_range) * i, density, linewidth=2)
        
        # Mark SCR point
        scr_val = data['scr_millions']
        if scr_val >= -1 and scr_val <= 1:
            ax.scatter([scr_val], [i], [0], color='red', s=40)
    
    # Set labels and remove unnecessary elements
    ax.set_xlabel('Change in Millions')
    ax.set_yticks([])  # No labels for y-axis
    ax.set_zlabel('Density')
    
    # Set view angle for better visualization
    ax.view_init(30, 45)
    
    return fig