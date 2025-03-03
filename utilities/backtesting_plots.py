import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Plotting Forcasted Distribution Against Test


def backtest_var_single_asset(test_returns, generated_returns, asset_name, confidence_level=0.995):
    """
    Backtests VaR (99.5%) for a single asset by comparing test period returns against 
    the estimated VaR from the generated return distribution.

    Parameters:
    - test_returns (Series): Pandas Series of actual test period returns (indexed by date).
    - generated_returns (array-like): Simulated/generated returns for the asset.
    - asset_name (str): The asset name.
    - confidence_level (float): The confidence level for VaR (default 99.5%).

    Returns:
    - failures (np.array): Binary sequence where 1 represents a VaR exception (failure) 
                           and 0 represents no failure.

    Outputs:
    - Time-series plot of test returns with VaR overlay and failures highlighted.
    - Summary of failures (how often actual losses exceed the VaR).
    """
    
    # Compute 99.5% VaR from generated returns
    var_threshold = np.percentile(generated_returns.flatten(), 100 * (1 - confidence_level))

    # Identify failures (when actual test return < VaR)
    failures = (test_returns < var_threshold).astype(int)  # Convert boolean to int (1 for failure, 0 otherwise)
    failure_count = failures.sum()

    # Plot test returns
    plt.figure(figsize=(12, 6))
    plt.plot(test_returns.index, test_returns, label="Test Returns", color="blue", alpha=0.7)

    # Plot VaR threshold line
    plt.axhline(var_threshold, color="red", linestyle="dashed", linewidth=2, label=f"VaR {confidence_level * 100:.1f}%")

    # Highlight failures
    plt.scatter(test_returns.index[failures == 1], test_returns[failures == 1], 
                color="red", label="VaR Breach", marker="o", zorder=3)

    # Formatting
    plt.title(f"Backtesting VaR for {asset_name} (99.5%)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print failure count summary
    print(f"\n📊 VaR Backtesting Summary for {asset_name}:")
    print(f"VaR {confidence_level * 100:.1f}% threshold: {var_threshold:.6f}")
    print(f"Failures (breaches below VaR): {failure_count} times")

    return failures