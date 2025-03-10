import pandas as pd
from backtesting.perform_distribution_tests import perform_distribution_tests
from backtesting.perform_var_tests import perform_var_backtesting_tests
from utilities.backtesting_plots import backtest_var_bof_value, calculate_var_threshold
from utilities.gan_plotting import create_rolling_empirical
import numpy as np
# Make support for multi VaR

def get_empirical_delta_bof(returns_df, weights, assets_0, liabilities_0):
    eonia = returns_df.iloc[:, -1]
    portfolio_returns = (returns_df * weights).sum(axis=1)
    portfolio_value = assets_0 * (1 + portfolio_returns).cumprod()
    liabilities = liabilities_0 * (1 + eonia).cumprod()
    bof = portfolio_value - liabilities
    bof_0 = assets_0 - liabilities_0

    return  bof - bof_0 # Delta BOF

def run_all_tests_on_models(
        models,
        train_returns,
        test_returns,
        weights,
        assets_0,
        liabilities_0,
        verbose=False):
    """
    models: a dictionary (model_name, distribution)
    empirical_returns_rolling: your rolling empirical data
    """
    results_list_var = []
    results_list_dist = []

    empirical_delta_bof = get_empirical_delta_bof(train_returns, weights, assets_0, liabilities_0)
    rolling_windows = create_rolling_empirical(empirical_delta_bof, window_size=252)
    plot_rolling_windows_distribution(rolling_windows, num_windows=10)

    for model_name, dist in models.items():
        print_model_box(model_name)

        failures, bof = backtest_var_bof_value(model_name, test_returns, dist, weights, assets_0, liabilities_0, confidence_level=0.995,verbose=verbose)

        dist_results = perform_distribution_tests(
            dist, 
            create_rolling_empirical(empirical_delta_bof), 
            asset_name=model_name, 
            verbose=False,
            bof=True
        )
        
        var_results = perform_var_backtesting_tests(
            failures=failures, 
            returns=test_returns, 
            var_forecast=[calculate_var_threshold(dist)], 
            asset_name=model_name, 
            generated_returns=dist, 
            verbose=False,
            portfolio=True,
            weights=weights,
            bof=bof
        )

        results_list_var.append({
            "model_name": model_name,
            **var_results
        })
        results_list_dist.append({
            "model_name": model_name,
            **dist_results
        })

    var_df = pd.DataFrame(results_list_var)
    dist_df = pd.DataFrame(results_list_dist)

    # Return styled DataFrames without index (or raw DataFrames if preferred)
    return var_df.style.hide(), dist_df.style.hide()

def print_model_box(model_name):
    padding = 10
    box_width = len(model_name) + 2 * padding
    border = "+" + "-" * box_width + "+"
    empty_line = "|" + " " * box_width + "|"
    model_line = "|" + model_name.center(box_width) + "|"
    
    print(border)
    print(empty_line)
    print(model_line)
    print(empty_line)
    print(border)


import seaborn as sns
import matplotlib.pyplot as plt
def plot_rolling_windows_distribution(rolling_data, num_windows=10):
    """
    Plot density curves for a subset of rolling windows.
    
    Parameters:
      rolling_data: 2D array of shape (n_windows, window_size).
      num_windows: Number of windows to sample and plot.
    """
    n_windows = rolling_data.shape[0]
    # Choose indices evenly spaced across the windows.
    indices = np.linspace(0, n_windows - 1, num_windows, dtype=int)
    
    plt.figure(figsize=(10, 6))
    for idx in indices:
        window = rolling_data[idx]
        # Plot the kernel density estimate for the window.
        sns.kdeplot(window, label=f"Window {idx}", fill=False)
    
    plt.xlabel("Delta BOF")
    plt.ylabel("Density")
    plt.title("Density Estimates for Sampled Rolling Windows of Empirical Delta BOF")
    plt.legend()
    plt.tight_layout()
    plt.show()