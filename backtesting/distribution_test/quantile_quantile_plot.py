import numpy as np
import matplotlib.pyplot as plt

def qq_plot(generated, empirical):
    """
    Create a Q-Q plot comparing generated vs. empirical return distributions.
    """
    quantiles = np.linspace(0, 1, 100)
    gen_quantiles = np.quantile(generated, quantiles)
    emp_quantiles = np.quantile(empirical, quantiles)
    
    plt.figure(figsize=(8, 6))
    plt.plot(emp_quantiles, gen_quantiles, 'o', label='Quantile Points')
    plt.plot([min(emp_quantiles), max(emp_quantiles)], [min(emp_quantiles), max(emp_quantiles)], 
             'r--', label='45-degree line')
    plt.xlabel("Empirical Quantiles")
    plt.ylabel("Generated Quantiles")
    plt.title("Q-Q Plot: Generated vs Empirical Return Distribution")
    plt.legend()
    plt.show()