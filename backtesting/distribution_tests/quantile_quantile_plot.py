import numpy as np
import matplotlib.pyplot as plt

def qq_plot(generated, empirical, bof=False):
    """
    Create a Q-Q plot comparing generated vs. empirical distributions.
    If bof=True, also display the distribution histograms side by side.
    
    Parameters:
      generated: array-like, generated data.
      empirical: array-like, empirical data.
      bof: boolean, if True, create an additional distribution plot.
    """
    quantiles = np.linspace(0, 1, 100)
    gen_quantiles = np.quantile(generated, quantiles)
    emp_quantiles = np.quantile(empirical, quantiles)
    
    if bof:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Q-Q plot on the left subplot
        axes[0].plot(emp_quantiles, gen_quantiles, 'o', label='Quantile Points')
        axes[0].plot([min(emp_quantiles), max(emp_quantiles)], 
                     [min(emp_quantiles), max(emp_quantiles)], 
                     'r--', label='45-degree line')
        axes[0].set_xlabel("Empirical Quantiles")
        axes[0].set_ylabel("Generated Quantiles")
        axes[0].set_title(r"Q-Q Plot: Generated vs Empirical $\Delta$ BOF Distribution")
        axes[0].legend()
        
        # Distribution plot on the right subplot
        axes[1].hist(empirical, bins=500, alpha=0.5, density=True, label='Empirical')
        axes[1].hist(generated, bins=500, alpha=0.5, density=True, label='Generated')
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Density")
        axes[1].set_title(r"Distribution: Empirical vs Generated $\Delta$ BOF")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        plt.plot(emp_quantiles, gen_quantiles, 'o', label='Quantile Points')
        plt.plot([min(emp_quantiles), max(emp_quantiles)], 
                 [min(emp_quantiles), max(emp_quantiles)], 'r--', label='45-degree line')
        plt.xlabel("Empirical Quantiles")
        plt.ylabel("Generated Quantiles")
        plt.title("Q-Q Plot: Generated vs Empirical Return Distribution")
        plt.legend()
        plt.show()
