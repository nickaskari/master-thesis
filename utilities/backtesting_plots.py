import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.cm as cm

def calculate_var_threshold(generated_returns, confidence_level=0.995):

    generated_returns = np.nan_to_num(generated_returns, nan=0.0, posinf=0.0, neginf=0.0)

    if np.all(generated_returns == 0):  # If all values are 0, VaR should be 0
        return 0.0

    var_threshold = np.percentile(generated_returns.flatten(), 100 * (1 - confidence_level))

    return var_threshold


# Plotting Forcasted Distribution Against Test

# Create support for multi VaR

def backtest_var_single_asset(test_returns, generated_returns, asset_name, confidence_level=0.995, verbose=True, quarterly=False):
    
    if quarterly:
        # Calculate VaR threshold for each quarter (assume generated_returns is a list)
        var_thresholds = [calculate_var_threshold(gen_ret, confidence_level) for gen_ret in generated_returns]
        print("all var thresholds",var_thresholds)
        
        # Initialize a Series for failures matching the test_returns index.
        failures = pd.Series(0, index=test_returns.index)
        
        # Group test returns by calendar quarter.
        # Note: test_returns.index.quarter returns values 1,2,3,4.
        grouped = test_returns.groupby(test_returns.index.quarter)
        for q, group in grouped:
            # Map calendar quarter (1-4) to index (0-3)
            idx = q - 1
            if idx < len(var_thresholds):
                thr = var_thresholds[idx]
                fails = (group < thr).astype(int)
                failures.loc[group.index] = fails
            else:
                failures.loc[group.index] = 0
        failure_count = failures.sum()
        
        if verbose:
            plt.figure(figsize=(12, 6))
            plt.plot(test_returns.index, test_returns, label="Test Returns", color="blue", alpha=0.7)

            # Define a colormap and generate a list of colors for each threshold.
            n_thresholds = len(var_thresholds)
            colors = [cm.tab10(i) for i in range(n_thresholds)]

            # Plot a horizontal line for each quarter using the corresponding VaR threshold.
            for q, group in grouped:
                idx = q - 1  # Adjust index (assumes quarters start at 1)
                if idx < n_thresholds:
                    thr = var_thresholds[idx]
                    plt.hlines(
                        y=thr, 
                        xmin=group.index.min(), 
                        xmax=group.index.max(),
                        colors=colors[idx], 
                        linestyles="dashed", 
                        linewidth=2,
                        label=f"VaR Q{q} ({confidence_level * 100:.1f}%)"
                    )

            plt.scatter(
                test_returns.index[failures == 1], 
                test_returns[failures == 1],
                color="red", 
                label="VaR Breach", 
                marker="o", 
                zorder=3
            )
            plt.title(f"Backtesting VaR for {asset_name} (Quarterly)", fontsize=14)
            plt.xlabel("Date")
            plt.ylabel("Return")
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt.show()
                
    else:
        # Non-quarterly: calculate a single VaR threshold and return it in a list.
        threshold = calculate_var_threshold(generated_returns, confidence_level)
        var_thresholds = [threshold]
        failures = (test_returns < threshold).astype(int)
        failure_count = failures.sum()
        
        if verbose:
            plt.figure(figsize=(12, 6))
            plt.plot(test_returns.index, test_returns, label="Test Returns", color="blue", alpha=0.7)
            plt.axhline(threshold, color="red", linestyle="dashed", linewidth=2,
                        label=f"VaR ({confidence_level * 100:.1f}%)")
            plt.scatter(test_returns.index[failures == 1], test_returns[failures == 1],
                        color="red", label="VaR Breach", marker="o", zorder=3)
            plt.title(f"Backtesting VaR for {asset_name}", fontsize=14)
            plt.xlabel("Date")
            plt.ylabel("Return")
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt.show()

            print(f"\n📊 VaR Backtesting Summary for {asset_name}:")
            print(f"VaR ({confidence_level * 100:.1f}%): {threshold:.6f}")
            print(f"Failures (breaches below VaR): {failure_count} times")
    
    # Return both the failure indicators and the list of all calculated VaR thresholds.
    return failures, var_thresholds
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