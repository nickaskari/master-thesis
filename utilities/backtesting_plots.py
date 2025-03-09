import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import numpy as np

def calculate_var_threshold(generated_returns, confidence_level=0.995):

    generated_returns = np.nan_to_num(generated_returns, nan=0.0, posinf=0.0, neginf=0.0)

    if np.all(generated_returns == 0):  # If all values are 0, VaR should be 0
        return 0.0

    var_threshold = np.percentile(generated_returns.flatten(), 100 * (1 - confidence_level))

    return var_threshold


# Plotting Forcasted Distribution Against Test

# Create support for multi VaR

def backtest_var_single_asset(test_returns, generated_returns, asset_name, confidence_level=0.995, verbose=True):
    
    var_threshold = calculate_var_threshold(generated_returns, confidence_level)

    failures = (test_returns < var_threshold).astype(int)  # Convert boolean to int (1 for failure, 0 otherwise)
    failure_count = failures.sum()

    if verbose:
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

def backtest_var_bof_value(
    model_name,
    test_returns,
    generated_bof_levels,
    weights,
    assets_0,
    liabilities_0,
    confidence_level=0.995,
    verbose=True
):
    """
    Backtest VaR for a Balance of Fund (BOF = assets - liabilities) portfolio using
    the actual BOF levels rather than BOF returns.
    """

    eonia = test_returns.iloc[:, -1]

    portfolio_returns = (test_returns * weights).sum(axis=1)
    
    portfolio_value = assets_0 * (1 + portfolio_returns).cumprod()
    
    liabilities = liabilities_0 * (1 + eonia).cumprod()
    
    bof = portfolio_value - liabilities

    bof_0 = assets_0 - liabilities_0

    bof = bof - bof_0 # Delta BOF

    if model_name == 'Standard Formula':
        var_threshold = generated_bof_levels # Since the standard formula does not produce distribution, a VaR is passed instead.
    else:
        var_threshold = calculate_var_threshold(generated_bof_levels, confidence_level)

    failures = (bof < var_threshold).astype(int)
    failure_count = failures.sum()

    if verbose:
        plt.figure(figsize=(12, 6))
        plt.plot(bof.index, bof, label="BOF Level", color="blue", alpha=0.7)
        plt.axhline(var_threshold, color="red", linestyle="dashed", linewidth=2,
                    label=f"VaR Level ({confidence_level * 100:.1f}%)")
        
        # Mark breaches
        breach_dates = bof.index[failures == 1]
        plt.scatter(breach_dates, bof[failures == 1],
                    color="red", label="VaR Breach", marker="o", zorder=3)
        
        plt.title(f"Backtesting VaR for {model_name} (Level-based)", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel(r"$\Delta$ BOF")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 VaR Backtesting Summary for {model_name}:")
        print(f"• Confidence Level: {confidence_level * 100:.1f}%")
        print(f"• VaR Threshold (Level): {var_threshold:.6f}")
        print(f"• Failures (BOF below VaR): {failure_count} times")
    
    return failures, bof