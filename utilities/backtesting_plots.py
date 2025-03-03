import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Plotting Forcasted Distribution Against Test

def backtest_var_single_asset(test_returns, generated_returns, asset_name, confidence_level=0.995):
    
    var_threshold = np.percentile(generated_returns.flatten(), 100 * (1 - confidence_level))

    failures = (test_returns < var_threshold).astype(int)  # Convert boolean to int (1 for failure, 0 otherwise)
    failure_count = failures.sum()


    plt.figure(figsize=(12, 6))
    plt.plot(test_returns.index, test_returns, label="Test Returns", color="blue", alpha=0.7)
    plt.axhline(var_threshold, color="red", linestyle="dashed", linewidth=2, label=f"VaR {confidence_level * 100:.1f}%")

    plt.scatter(test_returns.index[failures == 1], test_returns[failures == 1], 
                color="red", label="VaR Breach", marker="o", zorder=3)

    plt.title(f"Backtesting VaR for {asset_name} (99.5%)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    print(f"\n📊 VaR Backtesting Summary for {asset_name}:")
    print(f"VaR {confidence_level * 100:.1f}% threshold: {var_threshold:.6f}")
    print(f"Failures (breaches below VaR): {failure_count} times")

    return failures