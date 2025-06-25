import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from sklearn.decomposition import PCA
import seaborn as sns

def plot_tail_comparison(real_returns, gen_returns, asset_name):
    """
    Create QQ plots to compare the tails of real and generated returns.
    Especially useful for Solvency II applications where tail risk is critical.
    """
    # Convert to numpy array if it's a pandas Series
    if isinstance(real_returns, pd.Series):
        real_flat = real_returns.values
    else:
        real_flat = real_returns.flatten() if hasattr(real_returns, 'flatten') else real_returns
        
    if isinstance(gen_returns, pd.Series):
        gen_flat = gen_returns.values
    else:
        gen_flat = gen_returns.flatten() if hasattr(gen_returns, 'flatten') else gen_returns
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left tail QQ plot (focused on losses)
    axes[0].set_title(f"Left Tail QQ Plot - {asset_name}")
    stats.probplot(real_flat, dist="norm", plot=axes[0])
    stats.probplot(gen_flat, dist="norm", plot=axes[0], plotkw={"marker": "x", "color": "red"})
    axes[0].set_xlabel("Theoretical Quantiles")
    axes[0].set_ylabel("Ordered Values")
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Direct comparison of empirical CDFs
    axes[1].set_title(f"Empirical CDF Comparison - {asset_name}")
    
    # Sort the data
    real_sorted = np.sort(real_flat)
    gen_sorted = np.sort(gen_flat)
    
    # Calculate the proportional ranks
    p_real = np.linspace(0, 1, len(real_sorted))
    p_gen = np.linspace(0, 1, len(gen_sorted))
    
    # Plot the CDFs
    axes[1].plot(real_sorted, p_real, 'b-', label='Real Returns')
    axes[1].plot(gen_sorted, p_gen, 'r-', label='Generated Returns')
    
    # Highlight the tails
    tail_pct = 0.05  # Focus on 5% tail
    axes[1].axhline(y=tail_pct, color='k', linestyle='--', alpha=0.5)
    axes[1].axhline(y=1-tail_pct, color='k', linestyle='--', alpha=0.5)
    
    # Zoom on left tail in a small subplot
    left_tail_ax = fig.add_axes([0.67, 0.2, 0.15, 0.15])  # [left, bottom, width, height]
    left_tail_ax.plot(real_sorted[:int(len(real_sorted)*tail_pct)], 
                     p_real[:int(len(p_real)*tail_pct)], 'b-')
    left_tail_ax.plot(gen_sorted[:int(len(gen_sorted)*tail_pct)], 
                     p_gen[:int(len(p_gen)*tail_pct)], 'r-')
    left_tail_ax.set_title('Left 5% Tail', fontsize=8)
    
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def explore_latent_space(model, asset_name, num_dimensions=5, num_samples=500):
    """
    Explore how perturbing individual dimensions in the latent space affects generated returns.
    
    This helps understand what "features" the GAN has learned to represent in its latent space.
    """
    # Get fixed condition (most recent)
    fixed_condition = torch.tensor(model.conditions[-1:], dtype=torch.float32)
    if model.cuda:
        fixed_condition = fixed_condition.cuda()
    fixed_condition = fixed_condition.repeat(num_samples, 1)
    
    # Create a baseline random noise
    np.random.seed(42)  # For reproducibility
    base_noise = torch.randn(1, model.latent_dim, device=fixed_condition.device)
    
    # Select a few dimensions to explore
    dims_to_explore = np.random.choice(model.latent_dim, num_dimensions, replace=False)
    
    # For each dimension, create variations and generate returns
    all_variations = []
    variation_values = np.linspace(-3, 3, num_samples)  # -3 to 3 standard deviations
    
    # Setup figure
    fig, axes = plt.subplots(num_dimensions, 2, figsize=(18, 4*num_dimensions))
    
    for i, dim in enumerate(dims_to_explore):
        # Create variations along this dimension
        noise_variations = []
        for val in variation_values:
            noise_var = base_noise.clone()
            noise_var[0, dim] = val
            noise_variations.append(noise_var)
        
        # Stack all variations
        noise_tensor = torch.cat(noise_variations, dim=0)
        
        # Generate returns with fixed condition but varying noise
        model.generator.eval()
        with torch.no_grad():
            gen_returns = model.generator(noise_tensor, fixed_condition)
            gen_returns = model.scaler.inverse_transform(gen_returns.cpu().numpy())
        
        # Calculate key metrics for each generated scenario
        metrics = []
        for j in range(len(gen_returns)):
            returns = gen_returns[j]
            metrics.append({
                'mean': np.mean(returns),
                'volatility': np.std(returns),
                'skew': np.mean(returns**3) / (np.std(returns)**3),
                'min': np.min(returns),
                'max': np.max(returns),
                'var_95': np.percentile(returns, 5),
                'cumulative': np.prod(1 + returns) - 1
            })
        
        # Plot relationship between latent dimension and metrics
        ax1, ax2 = axes[i]
        
        # First plot: Relationship with return properties
        ax1.plot(variation_values, [m['mean'] for m in metrics], 'r-', label='Mean Return')
        ax1.plot(variation_values, [m['volatility'] for m in metrics], 'b-', label='Volatility')
        ax1.plot(variation_values, [m['var_95'] for m in metrics], 'g-', label='VaR 95%')
        ax1.set_xlabel(f'Latent Dimension {dim} Value')
        ax1.set_ylabel('Return Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Dimension {dim} vs Return Properties')
        
        # Second plot: Sample paths at different points
        # Choose 5 samples along the dimension
        sample_indices = [0, num_samples//4, num_samples//2, 3*num_samples//4, num_samples-1]
        
        for idx in sample_indices:
            cumulative = (1 + gen_returns[idx]).cumprod()
            label = f'z_{dim}={variation_values[idx]:.1f}'
            ax2.plot(cumulative, label=label, alpha=0.7)
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Sample Paths for Different Values of Dimension {dim}')
    
    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def analyze_scr_attribution(asset_scenarios_dict, portfolio_weights=None):
    """
    Analyze how each asset contributes to the overall Solvency Capital Requirement.
    
    Parameters:
    - asset_scenarios_dict: Dictionary with asset names as keys and arrays of scenarios as values
    - portfolio_weights: Dictionary of asset weights (defaults to equal weighting)
    
    Returns:
    - Figure with SCR attribution analysis
    """
    assets = list(asset_scenarios_dict.keys())
    num_assets = len(assets)
    
    # Set up weights
    if portfolio_weights is None:
        portfolio_weights = {asset: 1.0/num_assets for asset in assets}
    
    # Get dimensions
    first_asset = assets[0]
    num_scenarios, horizon = asset_scenarios_dict[first_asset].shape
    
    # Calculate portfolio scenarios
    portfolio_scenarios = np.zeros((num_scenarios, horizon))
    for asset in assets:
        portfolio_scenarios += portfolio_weights[asset] * asset_scenarios_dict[asset]
    
    # Convert to cumulative returns
    portfolio_cum_returns = (1 + portfolio_scenarios).cumprod(axis=1)
    portfolio_final_values = portfolio_cum_returns[:, -1]
    
    # Calculate the SCR (99.5% VaR for Solvency II)
    scr_percentile = 0.5  # 99.5th percentile
    portfolio_scr = 1 - np.percentile(portfolio_final_values, scr_percentile)
    
    # Get the scenario that corresponds to the SCR (worst 99.5% scenario)
    scr_scenario_index = np.argsort(portfolio_final_values)[int(num_scenarios * scr_percentile / 100)]
    
    # Calculate the contribution of each asset to this worst scenario
    asset_contributions = {}
    asset_cum_returns = {}
    
    for asset in assets:
        # Get the cumulative returns for this asset in the SCR scenario
        asset_scenario = asset_scenarios_dict[asset][scr_scenario_index]
        asset_cum_return = (1 + asset_scenario).prod() - 1
        
        # Weight by portfolio allocation
        weighted_contribution = portfolio_weights[asset] * asset_cum_return
        
        asset_contributions[asset] = weighted_contribution
        asset_cum_returns[asset] = asset_cum_return
    
    # Convert to DataFrame for easier plotting
    contribution_df = pd.DataFrame({
        'Asset': list(asset_contributions.keys()),
        'Contribution': list(asset_contributions.values()),
        'Cumulative Return': list(asset_cum_returns.values()),
        'Portfolio Weight': [portfolio_weights[asset] for asset in assets]
    })
    
    # Calculate percentage contribution to SCR
    total_negative_contrib = contribution_df[contribution_df['Contribution'] < 0]['Contribution'].sum()
    contribution_df['SCR %'] = contribution_df['Contribution'] / abs(total_negative_contrib) * 100
    
    # Sort by contribution
    contribution_df = contribution_df.sort_values('Contribution')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # 1. Waterfall chart of contributions
    ax1 = axes[0, 0]
    
    # Create waterfall chart manually
    assets_ordered = contribution_df['Asset'].tolist()
    contributions = contribution_df['Contribution'].tolist()
    
    # Start at 1 (100% of portfolio value)
    cumulative = [1.0]
    for contrib in contributions:
        cumulative.append(cumulative[-1] + contrib)
    
    # Add final SCR value
    assets_ordered = ['Start'] + assets_ordered + ['SCR']
    
    # Plot bars
    for i in range(len(cumulative)-1):
        start = cumulative[i]
        end = cumulative[i+1]
        height = end - start
        color = 'r' if height < 0 else 'g'
        ax1.bar(i, height, bottom=start, color=color, alpha=0.7)
    
    # Connect points with a line
    ax1.plot(range(len(cumulative)), cumulative, 'k--', alpha=0.5)
    
    # Labels
    ax1.set_xticks(range(len(assets_ordered)))
    ax1.set_xticklabels(assets_ordered, rotation=45, ha='right')
    ax1.set_ylabel('Portfolio Value')
    ax1.set_title('Waterfall Chart of SCR Contributions')
    
    for i, asset in enumerate(assets_ordered):
        if i > 0 and i < len(assets_ordered)-1:
            contrib = contributions[i-1]
            ax1.text(i, cumulative[i] + (0.01 if contrib > 0 else -0.03), 
                   f'{contrib:.2%}', ha='center', va='center')
    
    ax1.text(len(assets_ordered)-1, cumulative[-1] + 0.01, 
           f'SCR: {portfolio_scr:.2%}', ha='center', va='center', fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    
    # 2. Bar chart of percentage contributions
    ax2 = axes[0, 1]
    neg_contrib_df = contribution_df[contribution_df['Contribution'] < 0].copy()
    neg_contrib_df = neg_contrib_df.sort_values('SCR %')
    
    sns.barplot(x='SCR %', y='Asset', data=neg_contrib_df, ax=ax2, palette='Reds')
    ax2.set_title('Percentage Contribution to SCR')
    ax2.set_xlabel('Contribution to SCR (%)')
    
    for i, row in enumerate(neg_contrib_df.itertuples()):
        ax2.text(row._2 + 1, i, f'{row._2:.1f}%', va='center')
    
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatterplot of risk vs return in SCR scenario
    ax3 = axes[1, 0]
    
    # Calculate standalone VaR for each asset
    standalone_vars = {}
    
    for asset in assets:
        asset_cum_returns = (1 + asset_scenarios_dict[asset]).cumprod(axis=1)[:, -1]
        standalone_vars[asset] = 1 - np.percentile(asset_cum_returns, scr_percentile)
    
    scatter_df = pd.DataFrame({
        'Asset': assets,
        'Standalone SCR': [standalone_vars[asset] for asset in assets],
        'SCR Contribution': [asset_contributions[asset] for asset in assets],
        'Weight': [portfolio_weights[asset] for asset in assets]
    })
    
    # Create scatter plot with size proportional to weight
    scatter = ax3.scatter(
        scatter_df['Standalone SCR'], 
        scatter_df['SCR Contribution'],
        s=scatter_df['Weight'] * 1000,  # Scale for visibility
        alpha=0.7
    )
    
    # Add asset labels
    for i, row in scatter_df.iterrows():
        ax3.annotate(row['Asset'], 
                   (row['Standalone SCR'], row['SCR Contribution']),
                   xytext=(5, 5), textcoords='offset points')
    
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Standalone SCR')
    ax3.set_ylabel('Contribution to Portfolio SCR')
    ax3.set_title('Asset Risk vs. SCR Contribution')
    ax3.grid(True, alpha=0.3)
    
    # Add a legend for bubble size
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, 
                                           num=4, func=lambda s: s/1000)
    ax3.legend(handles, labels, loc="upper right", title="Portfolio Weight")
    
    # 4. Cumulative return paths for each asset in the SCR scenario
    ax4 = axes[1, 1]
    
    for asset in assets:
        # Get the specific scenario that corresponds to the portfolio's SCR
        asset_scenario = asset_scenarios_dict[asset][scr_scenario_index]
        cum_return = (1 + asset_scenario).cumprod()
        ax4.plot(cum_return, label=asset)
    
    # Also plot the portfolio path
    portfolio_scenario = portfolio_scenarios[scr_scenario_index]
    portfolio_cum_return = (1 + portfolio_scenario).cumprod()
    ax4.plot(portfolio_cum_return, 'k-', linewidth=2, label='Portfolio')
    
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Cumulative Return')
    ax4.set_title(f'Asset Paths in the SCR Scenario (99.5% VaR)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

def visualize_stress_tests(gan_dict, portfolio_weights=None):
    """
    Visualize how the portfolio responds to specific stress test scenarios.
    
    Parameters:
    - gan_dict: Dictionary with asset names as keys and trained GAN models as values
    - portfolio_weights: Dictionary of asset weights (defaults to equal weighting)
    
    Returns:
    - Figure with stress test visualizations
    """
    assets = list(gan_dict.keys())
    num_assets = len(assets)
    
    # Set up weights
    if portfolio_weights is None:
        portfolio_weights = {asset: 1.0/num_assets for asset in assets}
    
    # Define stress scenarios (adjustments to the condition vector)
    # Each stress test modifies specific condition variables
    # Assuming condition vector elements are [cum_return, volatility, kurtosis, max_drawdown]
    stress_scenarios = {
        'Base': {}, # No adjustments
        'Market Crash': {
            'cum_return': -0.4,      # -40% cumulative return
            'volatility': 0.4,       # 40% volatility
            'max_drawdown': -0.5     # -50% max drawdown
        },
        'Prolonged Decline': {
            'cum_return': -0.25,     # -25% cumulative return
            'volatility': 0.2,       # 20% volatility
            'max_drawdown': -0.3     # -30% max drawdown
        },
        'Volatility Spike': {
            'volatility': 0.35,      # 35% volatility
            'kurtosis': 8            # High kurtosis
        },
        'Recovery': {
            'cum_return': 0.15,      # 15% cumulative return
            'volatility': 0.15       # 15% volatility
        }
    }
    
    # Map condition names to indices in the condition vector
    condition_indices = {
        'cum_return': 0,
        'volatility': 1,
        'kurtosis': 2,
        'max_drawdown': 3
    }
    
    # Generate returns for each scenario and each asset
    scenario_results = {}
    num_scenarios = 1000  # Number of scenarios per stress test
    
    for scenario_name, adjustments in stress_scenarios.items():
        asset_scenarios = {}
        
        for asset in assets:
            model = gan_dict[asset]
            
            # Get the latest condition and modify it according to the stress scenario
            base_condition = model.conditions[-1:].copy()
            
            for cond_name, value in adjustments.items():
                idx = condition_indices[cond_name]
                base_condition[0, idx] = value
            
            # Generate returns with this condition
            model.generator.eval()
            with torch.no_grad():
                z = torch.randn(num_scenarios, model.latent_dim, 
                              device='cuda' if model.cuda else 'cpu')
                cond = torch.tensor(base_condition, dtype=torch.float32, 
                                  device='cuda' if model.cuda else 'cpu').repeat(num_scenarios, 1)
                gen_returns = model.generator(z, cond)
                gen_returns = model.scaler.inverse_transform(gen_returns.cpu().numpy())
                
                asset_scenarios[asset] = gen_returns
        
        # Store the results for this scenario
        scenario_results[scenario_name] = asset_scenarios
    
    # Now create portfolio scenarios by aggregating asset scenarios
    portfolio_scenario_results = {}
    
    for scenario_name, asset_scenarios in scenario_results.items():
        portfolio_scenarios = np.zeros_like(asset_scenarios[assets[0]])
        
        for asset in assets:
            portfolio_scenarios += portfolio_weights[asset] * asset_scenarios[asset]
        
        portfolio_scenario_results[scenario_name] = portfolio_scenarios
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Violin plot of portfolio returns by scenario
    ax1 = fig.add_subplot(gs[0, :])
    
    # Prepare data for violin plot
    violin_data = []
    scenario_names = []
    
    for scenario_name, scenarios in portfolio_scenario_results.items():
        # Calculate key metrics for this scenario
        final_cum_returns = (1 + scenarios).cumprod(axis=1)[:, -1] - 1
        violin_data.append(final_cum_returns)
        scenario_names.append(scenario_name)
    
    # Create violin plot
    violin_parts = ax1.violinplot(violin_data, showmeans=True, showmedians=True)
    
    # Color the violins according to scenario severity
    colors = ['#4575b4', '#d73027', '#fc8d59', '#fee090', '#91bfdb']  # Blue to red gradient
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add horizontal line at 0
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add VaR markers for each scenario
    var_levels = [0.95, 0.99, 0.995]  # 95%, 99%, 99.5% VaR
    var_markers = ['o', 's', 'd']     # Different marker for each VaR level
    
    for i, scenario_data in enumerate(violin_data):
        for j, level in enumerate(var_levels):
            var = np.percentile(scenario_data, (1-level)*100)
            ax1.plot(i+1, var, marker=var_markers[j], color='k', 
                   markersize=8, label=f'{level*100}% VaR' if i == 0 else "")
            
            # Add text label for 99.5% VaR (Solvency II SCR)
            if level == 0.995:
                ax1.text(i+1, var - 0.03, f'{var:.2%}', ha='center')
    
    # Customize plot
    ax1.set_xticks(np.arange(1, len(scenario_names) + 1))
    ax1.set_xticklabels(scenario_names)
    ax1.set_ylabel('Cumulative Portfolio Return')
    ax1.set_title('Distribution of Portfolio Returns Under Different Stress Scenarios')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for VaR markers
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), title='VaR Levels', loc='upper right')
    
    # 2. Heatmap of average asset returns by scenario
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(assets), len(scenario_names)))
    
    for i, asset in enumerate(assets):
        for j, scenario_name in enumerate(scenario_names):
            asset_scenarios = scenario_results[scenario_name][asset]
            final_cum_returns = (1 + asset_scenarios).cumprod(axis=1)[:, -1] - 1
            heatmap_data[i, j] = np.mean(final_cum_returns)
    
    # Create custom diverging colormap (red for negative, blue for positive)
    cmap = LinearSegmentedColormap.from_list('RdBu_custom', ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6'])
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap=cmap, center=0,
              xticklabels=scenario_names, yticklabels=assets, ax=ax2)
    
    ax2.set_title('Average Asset Return by Stress Scenario')
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Asset')
    
    # 3. Line plot of SCR by scenario
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate SCR for each scenario
    scr_values = []
    
    for scenario_name in scenario_names:
        scenarios = portfolio_scenario_results[scenario_name]
        final_cum_returns = (1 + scenarios).cumprod(axis=1)[:, -1]
        scr = 1 - np.percentile(final_cum_returns, 0.5)  # 99.5% VaR
        scr_values.append(scr)
    
    # Create bar chart
    bars = ax3.bar(scenario_names, scr_values, color=colors)
    
    # Add values on top of bars
    for i, v in enumerate(scr_values):
        ax3.text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    ax3.set_title('Solvency Capital Requirement by Scenario')
    ax3.set_ylabel('SCR (99.5% VaR)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Sample paths for base and market crash scenarios
    ax4 = fig.add_subplot(gs[2, :])
    
    # Get base and crash scenarios
    base_scenarios = portfolio_scenario_results['Base']
    crash_scenarios = portfolio_scenario_results['Market Crash']
    
    # Plot sample paths
    num_paths = 50  # Number of sample paths to plot
    
    # Plot base scenarios
    for i in range(min(num_paths, len(base_scenarios))):
        cum_returns = (1 + base_scenarios[i]).cumprod()
        ax4.plot(cum_returns, color='blue', alpha=0.1)
    
    # Plot crash scenarios
    for i in range(min(num_paths, len(crash_scenarios))):
        cum_returns = (1 + crash_scenarios[i]).cumprod()
        ax4.plot(cum_returns, color='red', alpha=0.1)
    
    # Add reference line at 1.0
    ax4.axhline(y=1.0, color='k', linestyle='-', alpha=0.5)
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', alpha=0.5, label='Base Scenario'),
        Line2D([0], [0], color='red', alpha=0.5, label='Market Crash')
    ]
    ax4.legend(handles=legend_elements)
    
    ax4.set_title('Sample Portfolio Paths: Base vs Market Crash Scenarios')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Cumulative Return')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_training_convergence(training_metrics_dict):
    """
    Visualize how GAN training metrics evolved to assess convergence.
    
    Parameters:
    - training_metrics_dict: Dictionary with epoch numbers as keys and metrics as values.
                           Expected metrics: d_loss, g_loss, tail_penalty, structure_penalty
    
    Returns:
    - Figure with training convergence visualizations
    """
    # Extract metrics
    epochs = list(training_metrics_dict.keys())
    d_losses = [m['d_loss'] for m in training_metrics_dict.values()]
    g_losses = [m['g_loss'] for m in training_metrics_dict.values()]
    tail_penalties = [m['tail_penalty'] for m in training_metrics_dict.values()]
    structure_penalties = [m['structure_penalty'] for m in training_metrics_dict.values()]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Generator and Discriminator Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, d_losses, 'r-', label='Discriminator Loss')
    ax1.plot(epochs, g_losses, 'b-', label='Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator and Discriminator Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add a second y-axis for the ratio of g_loss to d_loss
    ratio = [g/d if d != 0 else np.nan for g, d in zip(g_losses, d_losses)]
    ax1_right = ax1.twinx()
    ax1_right.plot(epochs, ratio, 'g--', label='G/D Ratio')
    ax1_right.set_ylabel('G/D Ratio')
    ax1_right.legend(loc='lower right')
    
    # 2. Penalty Terms
    ax2 = axes[0, 1]
    ax2.plot(epochs, tail_penalties, 'g-', label='Tail Penalty')
    ax2.plot(epochs, structure_penalties, 'm-', label='Structure Penalty')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Penalty Value')
    ax2.set_title('Penalty Terms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Average of Losses to Show Trend
    window_size = max(5, len(epochs) // 20)  # 5% of total epochs or at least 5
    
    # Calculate rolling averages
    g_loss_roll = pd.Series(g_losses).rolling(window_size).mean().values
    d_loss_roll = pd.Series(d_losses).rolling(window_size).mean().values
    
    ax3 = axes[1, 0]
    ax3.plot(epochs[window_size-1:], g_loss_roll[~np.isnan(g_loss_roll)], 'b-', label='Generator Loss (MA)')
    ax3.plot(epochs[window_size-1:], d_loss_roll[~np.isnan(d_loss_roll)], 'r-', label='Discriminator Loss (MA)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (Moving Average)')
    ax3.set_title(f'Moving Average of Losses (Window={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap of correlation between metrics over time
    metrics = np.array([d_losses, g_losses, tail_penalties, structure_penalties]).T
    
    # Create non-overlapping epochs segments 
    num_segments = 5
    segment_size = len(epochs) // num_segments
    
    correlation_matrices = []
    segment_labels = []
    
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(epochs)
        
        segment_metrics = metrics[start_idx:end_idx]
        corr_matrix = np.corrcoef(segment_metrics.T)
        correlation_matrices.append(corr_matrix)
        
        segment_labels.append(f"Epochs {epochs[start_idx]}-{epochs[end_idx-1]}")
    
    # Create a plot grid for correlation matrices
    ax4 = axes[1, 1]
    
    # Use the last correlation matrix (representing the final training state)
    final_corr = correlation_matrices[-1]
    
    # Create heatmap
    sns.heatmap(final_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
              xticklabels=['D Loss', 'G Loss', 'Tail Pen', 'Struct Pen'],
              yticklabels=['D Loss', 'G Loss', 'Tail Pen', 'Struct Pen'],
              ax=ax4)
    
    ax4.set_title(f'Metric Correlations in Final Training Phase')
    
    plt.tight_layout()
    return fig