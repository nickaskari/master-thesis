import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from scipy import stats
import os
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree


def create_rolling_empirical(returns_df, window_size=252):
    """Creates rolling 1-year (252-day) sequences from empirical data"""
    rolling_data = []
    for i in range(len(returns_df) - window_size):
        window = returns_df[i : i + window_size]
        rolling_data.append(window)
    return np.array(rolling_data)


def check_mode_collapse(real_returns, generated_returns):
    """
    Checks for mode collapse using:
    1. Variance comparison.
    2. Sample diversity (pairwise distance).
    3. PCA visualization (handling NaN issues).
    """

    real_var = np.nanvar(real_returns, axis=0).mean()  # Handle NaNs
    gen_var = np.nanvar(generated_returns, axis=0).mean()

    print(f"Variance of Real Data: {real_var:.6f}")
    print(f"Variance of Generated Data: {gen_var:.6f}")

    if gen_var < 0.5 * real_var:
        print("⚠️ Warning: Possible Mode Collapse - Low Variance in Generated Data")
    else:
        print("✅ Generated Data Shows Reasonable Variance")

    # Check for diversity using pairwise distances
    real_distances = pdist(real_returns, metric="euclidean")
    gen_distances = pdist(generated_returns, metric="euclidean")

    real_mean_dist = np.nanmean(real_distances)
    gen_mean_dist = np.nanmean(gen_distances)

    print(f"Mean Pairwise Distance (Real): {real_mean_dist:.6f}")
    print(f"Mean Pairwise Distance (Generated): {gen_mean_dist:.6f}")

    if gen_mean_dist < 0.5 * real_mean_dist:
        print("⚠️ Warning: Potential Mode Collapse - Samples are too similar")
    else:
        print("✅ Generated samples are reasonably diverse")

    # -------- Fix NaN issue before PCA -------- #
    real_returns = np.nan_to_num(real_returns, nan=np.nanmedian(real_returns))
    generated_returns = np.nan_to_num(generated_returns, nan=np.nanmedian(generated_returns))

    # PCA Projection for real vs generated data
    pca = PCA(n_components=2)
    
    real_pca = pca.fit_transform(real_returns)
    gen_pca = pca.transform(generated_returns)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], color="blue", label="Real Returns", alpha=0.5)
    plt.scatter(gen_pca[:, 0], gen_pca[:, 1], color="red", label="Generated Returns", alpha=0.5)
    plt.legend()
    plt.title("PCA Projection of Real vs Generated Returns")
    plt.show()

def analyse_assets(returns_df, precomputed_rolling_returns, test):
    asset_names = returns_df.columns

    results = []
    for asset_name in asset_names:
        # Print a boxed title before analyzing each asset
        title = f" ANALYZING ASSET: {asset_name} "
        print("\n" + "═" * (len(title) + 4))
        print(f"║{title.center(len(title) + 2)}║")
        print("═" * (len(title) + 4) + "\n")

        # Load the generated returns
        gen_returns = load_generated_returns(asset_name, test)
        gen_returns = gen_returns.view(gen_returns.size(0), 252).cpu().detach().numpy()

        empirical_returns = precomputed_rolling_returns[asset_name]

        # Perform mode collapse check
        check_mode_collapse(empirical_returns, gen_returns)

        if asset_name != 'EONIA':
            extreme_value_analysis(asset_name, precomputed_rolling_returns, test)
            nearest_distance_histogram(asset_name, precomputed_rolling_returns, test)

        result = wasserstein_distance_analysis(asset_name, precomputed_rolling_returns, test)
        results.append(result)


    wasserstein_distance_plot(results)


# ---------- HISTOGRAM PLOT FUNCTION ---------- #

def compute_return_statistics(data, asset_name):
    """
    Computes key return statistics for the given asset data.
    """
    def safe_stat(func, data, default=0):
        try:
            result = func(data)
            return result if np.isfinite(result) else default
        except Exception:
            return default

    stats_dict = {
        "Asset": asset_name,
        "Mean": safe_stat(np.mean, data),
        "Std Dev": safe_stat(np.std, data),
        "Skewness": safe_stat(stats.skew, data),
        "Kurtosis": safe_stat(stats.kurtosis, data),
        "99.5% VaR": safe_stat(lambda x: np.percentile(x, 0.5), data)
    }
    
    return stats_dict

def display_statistics(stats_list):
    """
    Displays the computed statistics for all assets in a structured table.
    """
    stats_df = pd.DataFrame(stats_list)
    print("\n📊 Return Statistics Summary:")
    print(stats_df.to_string(index=False))  # Pretty prints the table
    

def plot_histogram_distributions(returns_df, precomputed_rolling_returns, test, quarterly,scaled=True, bins=500, cols=3):
    asset_names = returns_df.columns
    num_assets = len(asset_names)
    
    # Grid layout configuration
    rows = (num_assets + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    stats_list = []  # Store statistics for later display

    for idx, asset_name in enumerate(asset_names):
        ax = axes[idx]

        # Load the generated returns
        gen_returns = load_generated_returns(asset_name, test, quarterly).cpu().detach().numpy()

        # Handle NaNs in generated returns
        if np.isnan(gen_returns).any() or np.isinf(gen_returns).any():
            print(f"⚠️ Warning: NaNs or Infs detected in generated returns for {asset_name}. Fixing...")
            gen_returns = np.nan_to_num(gen_returns, nan=np.nanmedian(returns_df[asset_name].values), posinf=1e-6, neginf=-1e-6)

        # Retrieve precomputed empirical returns
        empirical_returns = precomputed_rolling_returns[asset_name]

        # Ensure both have the same length by truncating
        min_length = min(len(empirical_returns.flatten()), len(gen_returns.flatten()))
        real_data = empirical_returns.flatten()[:min_length]
        generated_data = gen_returns.flatten()[:min_length]

        # Scaling
        if scaled:
            scaling_factor = 1e6  
            real_data_scaled = real_data * scaling_factor
            generated_data_scaled = generated_data * scaling_factor
            real_data_scaled = np.nan_to_num(real_data_scaled, nan=np.nanmedian(real_data_scaled))
            generated_data_scaled = np.nan_to_num(generated_data_scaled, nan=np.nanmedian(generated_data_scaled))
        else:
            real_data_scaled = real_data
            generated_data_scaled = generated_data

        # Compute common bin edges
        bin_min = min(real_data_scaled.min(), generated_data_scaled.min())
        bin_max = max(real_data_scaled.max(), generated_data_scaled.max())
        common_bins = np.linspace(bin_min, bin_max, bins)

        # Plot histogram
        ax.hist(real_data_scaled, bins=common_bins, alpha=0.6, color='blue', label=f"{asset_name} - Real", density=True)
        ax.hist(generated_data_scaled, bins=common_bins, alpha=0.6, color='red', label=f"{asset_name} - Generated", density=True)

        ax.set_title(f'{asset_name} Returns', fontsize=12)
        ax.set_xlabel('Return (Scaled)' if scaled else 'Return', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(False)

        # Compute and store statistics
        stats_real = compute_return_statistics(real_data_scaled, asset_name + " - Real")
        stats_generated = compute_return_statistics(generated_data_scaled, asset_name + " - Generated")

        stats_list.append(stats_real)
        stats_list.append(stats_generated)

    # Hide unused subplots
    for i in range(num_assets, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Display statistics separately
    display_statistics(stats_list)

# ---------- EXTREME VALUE PLOT FUNCTION ---------- #
def extreme_value_analysis(asset_name, precomputed_rolling_returns, test, quarterly):

    # Load generated returns
    gen_returns = load_generated_returns(asset_name, test, quarterly)
    gen_returns = gen_returns.view(gen_returns.size(0), 252).cpu().detach().numpy().flatten()
    
    # Get real returns from precomputed data
    real_returns = precomputed_rolling_returns[asset_name].flatten()
    
    # ---- Tail Quantile Plot ----
    quantiles = np.linspace(0.01, 0.99, 50)
    real_quantiles = np.quantile(real_returns, quantiles)
    gen_quantiles = np.quantile(gen_returns, quantiles)
    
    plt.figure(figsize=(8, 5))
    plt.plot(quantiles, real_quantiles, label='Real Returns', marker='o', linestyle='dashed')
    plt.plot(quantiles, gen_quantiles, label='Generated Returns', marker='s', linestyle='dashed')
    plt.xlabel('Quantile')
    plt.ylabel('Return')
    plt.title(f'Tail Quantile Plot for {asset_name}')
    plt.legend()
    plt.grid(False)
    plt.show()

def load_generated_returns(asset_name, test=False, quarterly=False):
    if test:
        load_dir = 'generated_CGAN_output_test'
    else:
        load_dir = 'generated_GAN_output'

    file_path = os.path.join(load_dir, f'generated_returns_{asset_name}_final_scenarios.pt')
    gen_returns = torch.load(file_path)

    return gen_returns

def wasserstein_distance_analysis(asset_name, precomputed_rolling_returns, test, quarterly):

    title = f" COMPUTING WASSERSTEIN DISTANCE: {asset_name} "
    print("\n" + "═" * (len(title) + 4))
    print(f"║{title.center(len(title) + 2)}║")
    print("═" * (len(title) + 4) + "\n")

    # Load the generated returns
    gen_returns = load_generated_returns(asset_name, test, quarterly)
    gen_returns = gen_returns.view(gen_returns.size(0), 252).cpu().detach().numpy().flatten()

    # Retrieve precomputed empirical returns
    empirical_returns = precomputed_rolling_returns[asset_name].flatten()

    # Ensure both have the same length by truncating
    min_length = min(len(empirical_returns), len(gen_returns))
    real_data = empirical_returns[:min_length]
    generated_data = gen_returns[:min_length]

    # Compute Wasserstein Distance
    w_distance = wasserstein_distance(real_data, generated_data)

    print(f"📊 Wasserstein Distance for {asset_name}: {w_distance:.6f}\n")

    return asset_name, w_distance  # Return values for external use

def wasserstein_distance_plot(results):
    asset_names, wasserstein_distances = zip(*results)

    plt.figure(figsize=(12, 6))
    plt.bar(asset_names, wasserstein_distances, color='purple', alpha=0.7)

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.ylabel("Wasserstein Distance", fontsize=14)
    plt.xlabel("Asset", fontsize=14)
    plt.title("Wasserstein Distance Between Empirical & Generated Returns", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def nearest_distance_histogram(asset_name, precomputed_rolling_returns, test, quarterly, bins=50):
    title = f" COMPUTING NEAREST DISTANCE HISTOGRAM: {asset_name} "
    print("\n" + "═" * (len(title) + 4))
    print(f"║{title.center(len(title) + 2)}║")
    print("═" * (len(title) + 4) + "\n")

    # Load generated returns
    gen_returns = load_generated_returns(asset_name, test, quarterly)
    gen_returns = gen_returns.view(gen_returns.size(0), 252).cpu().detach().numpy()

    # Retrieve empirical returns
    empirical_returns = precomputed_rolling_returns[asset_name]

    # Normalize data before distance computation
    real_data = (empirical_returns.flatten() - np.mean(empirical_returns)) / np.std(empirical_returns)
    generated_data = (gen_returns.flatten() - np.mean(gen_returns)) / np.std(gen_returns)

    # Build KDTree for nearest-neighbor search
    tree = cKDTree(real_data.reshape(-1, 1))
    distances, _ = tree.query(generated_data.reshape(-1, 1), k=1)

    # Print distance summary for debugging
    print(f"\n📊 Distance Summary for {asset_name}:")
    print(f"Min Distance: {distances.min():.6f}")
    print(f"Max Distance: {distances.max():.6f}")
    print(f"Mean Distance: {distances.mean():.6f}")
    print(f"Median Distance: {np.median(distances):.6f}")
    print(f"Standard Deviation: {distances.std():.6f}")

    # **Fix 1: Trim Outliers Beyond the 99.5th Percentile**
    threshold = np.percentile(distances, 99.5)  # Exclude extreme outliers
    distances = distances[distances <= threshold]

    # **Fix 2: Improved binning strategy**
    min_dist, max_dist = distances.min(), distances.max()
    bins = np.linspace(min_dist, max_dist, bins)  # Evenly spaced bins within a controlled range

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=bins, color="lightblue", edgecolor="black", alpha=0.7)

    # **Fix 3: Apply Symlog Scaling for Better Visibility**
    plt.xscale("symlog", linthresh=1e-4)  # Keeps small values visible without over-compressing larger ones

    plt.xlabel("Distance to nearest empirical data point", fontsize=12)
    plt.ylabel("Number of generated data points", fontsize=12)
    plt.title(f"Nearest Distance Histogram for {asset_name}", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
    
def extensive_plotting(scaled, returns_df, test=False, quarterly=False):
    precomputed_rolling_returns = {asset: create_rolling_empirical(returns_df[asset].values) for asset in returns_df.columns}
    
  
    # Call functions using precomputed returns
    plot_histogram_distributions(returns_df, precomputed_rolling_returns, test, scaled, quarterly, bins=500, cols=3)

    print("\n" + "=" * 50 + "\n")  

    analyse_assets(returns_df, precomputed_rolling_returns, test, quarterly)
    #extreme_value_analysis(returns_df, precomputed_rolling_returns)




