import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd



def analyze_feature_importance(gan_model, num_samples=100):
    """
    Analyze which features in the condition vector have the most impact
    on the generated scenarios.
    
    Parameters:
    - gan_model: Trained FashionGAN model
    - num_samples: Number of test samples to generate
    
    Returns:
    - Figure with feature importance visualization
    """
    # Get the latest condition as baseline
    baseline_condition = torch.tensor(gan_model.conditions[-1:], dtype=torch.float32)
    if gan_model.cuda:
        baseline_condition = baseline_condition.cuda()
    
    # Use the same latent noise vector for all tests to isolate condition effects
    z = torch.randn(num_samples, gan_model.latent_dim, device=baseline_condition.device)
    
    # Generate baseline samples
    gan_model.generator.eval()
    with torch.no_grad():
        # Repeat the condition for each sample
        repeated_condition = baseline_condition.repeat(num_samples, 1)
        baseline_returns = gan_model.generator(z, repeated_condition).cpu().numpy()
    
    # Identify feature names (modify according to your model's features)
    feature_names = []
    for lag in [1]:  # Modify based on your lag periods
        feature_names.extend([
            f'Lag{lag} Cum Return', 
            f'Lag{lag} Volatility', 
            f'Lag{lag} Kurtosis',
            f'Lag{lag} Max Drawdown', 
        ])
    
    # Test each feature's impact by perturbing it
    feature_impact = []
    
    for i in range(gan_model.cond_dim):
        perturbed_conditions = []
        
        # Create 3 perturbation levels: -1SD, +1SD, +2SD
        for perturbation in [-1, 1, 2]:
            # Copy the baseline condition
            perturbed_condition = baseline_condition.clone()
            
            # Get standard deviation for this feature across all conditions
            feature_std = torch.std(torch.tensor(gan_model.conditions[:, i], dtype=torch.float32))
            
            # Perturb the feature
            perturbed_condition[0, i] += perturbation * feature_std
            
            # Generate samples with perturbed condition
            with torch.no_grad():
                repeated_perturbed = perturbed_condition.repeat(num_samples, 1)
                perturbed_returns = gan_model.generator(z, repeated_perturbed).cpu().numpy()
            
            # Measure the difference from baseline (using mean absolute difference)
            difference = np.mean(np.abs(perturbed_returns - baseline_returns))
            perturbed_conditions.append((perturbation, difference))
        
        # Average the impact across all perturbations
        avg_impact = np.mean([diff for _, diff in perturbed_conditions])
        feature_impact.append(avg_impact)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Generate a bar chart of feature impacts
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, feature_impact, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Mean Absolute Impact on Generated Returns')
    plt.title('Condition Feature Importance in the GAN')
    plt.grid(True, alpha=0.3)
    
    # Add a second plot with a heatmap for a subset of outputs
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))
    
    # First plot: bar chart
    axs[0].barh(y_pos, feature_impact, align='center')
    axs[0].set_yticks(y_pos)
    axs[0].set_yticklabels(feature_names)
    axs[0].set_xlabel('Mean Absolute Impact on Generated Returns')
    axs[0].set_title('Condition Feature Importance in the GAN')
    axs[0].grid(True, alpha=0.3)
    
    # Second plot: detailed response heatmap
    # Let's analyze how the most important features affect key risk metrics
    
    # Find the top 3 most important features
    top_features_idx = np.argsort(feature_impact)[-3:]
    top_features = [feature_names[i] for i in top_features_idx]
    
    # Risk metrics to analyze
    risk_metrics = ['Mean Return', 'Volatility', 'VaR(95%)', 'VaR(99.5%)', 'Max Drawdown']
    
    # Create a matrix to hold the results
    response_matrix = np.zeros((len(top_features_idx), len(risk_metrics)))
    
    # For each top feature, measure the response of risk metrics to +2SD change
    for i, feature_idx in enumerate(top_features_idx):
        # Create perturbed condition with +2SD
        perturbed_condition = baseline_condition.clone()
        feature_std = torch.std(torch.tensor(gan_model.conditions[:, feature_idx], dtype=torch.float32))
        perturbed_condition[0, feature_idx] += 2 * feature_std
        
        # Generate samples
        with torch.no_grad():
            repeated_perturbed = perturbed_condition.repeat(num_samples, 1)
            perturbed_returns = gan_model.generator(z, repeated_perturbed).cpu().numpy()
        
        # Calculate risk metrics for perturbed returns
        for j, metric in enumerate(risk_metrics):
            if metric == 'Mean Return':
                baseline_value = np.mean(baseline_returns)
                perturbed_value = np.mean(perturbed_returns)
            elif metric == 'Volatility':
                baseline_value = np.std(baseline_returns.flatten())
                perturbed_value = np.std(perturbed_returns.flatten())
            elif metric == 'VaR(95%)':
                baseline_value = np.percentile(baseline_returns.flatten(), 5)
                perturbed_value = np.percentile(perturbed_returns.flatten(), 5)
            elif metric == 'VaR(99.5%)':
                baseline_value = np.percentile(baseline_returns.flatten(), 0.5)
                perturbed_value = np.percentile(perturbed_returns.flatten(), 0.5)
            elif metric == 'Max Drawdown':
                # Calculate max drawdown for each path
                baseline_md = []
                perturbed_md = []
                
                for path in baseline_returns:
                    cumulative = (1 + path).cumprod()
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    baseline_md.append(np.min(drawdown))
                
                for path in perturbed_returns:
                    cumulative = (1 + path).cumprod()
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    perturbed_md.append(np.min(drawdown))
                
                baseline_value = np.mean(baseline_md)
                perturbed_value = np.mean(perturbed_md)
            
            # Calculate percentage change
            if abs(baseline_value) > 1e-10:  # Avoid division by zero
                pct_change = (perturbed_value - baseline_value) / abs(baseline_value) * 100
            else:
                pct_change = 0
                
            response_matrix[i, j] = pct_change
    
    # Create heatmap
    sns.heatmap(response_matrix, annot=True, fmt=".1f", 
              xticklabels=risk_metrics, 
              yticklabels=top_features,
              cmap="RdBu_r", center=0, ax=axs[1])
    
    axs[1].set_title('% Change in Risk Metrics When Feature Increases by 2 Std Dev')
    
    plt.tight_layout()
    return fig

def visualize_stress_tests_2(gan_model, asset_name):
    """
    Visualize how the GAN responds to different market stress scenarios.
    This is particularly useful for Solvency II stress testing.
    
    Parameters:
    - gan_model: Trained FashionGAN model
    - asset_name: Name of the asset
    
    Returns:
    - Figure with stress test visualization
    """
    # Define stress scenarios (using realistic Solvency II-like shocks)
    stress_scenarios = {
        'Baseline': {
            'return': 0,
            'volatility': 0,
            'kurtosis': 0,
            'max_drawdown': 0
        },
        'Market Crash': {
            'return': -0.4,  # -40% annual return
            'volatility': 0.5,  # 50% higher volatility
            'kurtosis': 2,  # Increased tail risk
            'max_drawdown': -0.3  # 30% worse drawdown
        },
        'Volatility Spike': {
            'return': -0.15,  # -15% annual return
            'volatility': 1.0,  # 100% higher volatility
            'kurtosis': 1,  # Some increase in tail risk
            'max_drawdown': -0.2  # 20% worse drawdown
        },
        'Prolonged Downturn': {
            'return': -0.25,  # -25% annual return
            'volatility': 0.2,  # 20% higher volatility
            'kurtosis': 0.5,  # Some increase in tail risk
            'max_drawdown': -0.4  # 40% worse drawdown
        }
    }
    
    # Get feature mapping (adjust to match your model)
    feature_mapping = {
        'return': 0,        # Index of cum return in condition vector
        'volatility': 1,    # Index of volatility in condition vector
        'kurtosis': 2,      # Index of kurtosis in condition vector
        'max_drawdown': 3   # Index of max drawdown in condition vector
    }
    
    # Get the baseline condition (most recent condition)
    baseline_condition = gan_model.conditions[-1:].copy()
    
    # Sample size for each scenario
    num_samples = 1000
    
    # Generate samples for each stress scenario
    results = {}
    z = torch.randn(num_samples, gan_model.latent_dim, 
                  device='cuda' if gan_model.cuda else 'cpu')
    
    for scenario_name, scenario in stress_scenarios.items():
        # Create stressed condition
        stressed_condition = baseline_condition.copy()
        
        # Apply stress factors
        for factor_name, factor_value in scenario.items():
            if factor_name in feature_mapping:
                idx = feature_mapping[factor_name]
                
                # Get standard deviation for this feature
                feature_std = np.std(gan_model.conditions[:, idx])
                
                # Apply the stress as a multiple of standard deviation
                stressed_condition[0, idx] += factor_value * feature_std
        
        # Generate samples with stressed condition
        gan_model.generator.eval()
        with torch.no_grad():
            cond = torch.tensor(stressed_condition, dtype=torch.float32, 
                              device='cuda' if gan_model.cuda else 'cpu')
            cond = cond.repeat(num_samples, 1)
            gen_returns = gan_model.generator(z, cond)
            
            # Convert back to original scale
            gen_returns = gan_model.scaler.inverse_transform(gen_returns.cpu().numpy())
            
            results[scenario_name] = gen_returns
    
    # Calculate key risk metrics for each scenario
    metrics = {
        'Mean Return': lambda x: np.mean(x),
        'Volatility': lambda x: np.std(x.flatten()),
        'VaR (95%)': lambda x: np.percentile(x.flatten(), 5),
        'VaR (99.5%)': lambda x: np.percentile(x.flatten(), 0.5),
        'Expected Shortfall (99%)': lambda x: np.mean(x.flatten()[x.flatten() <= np.percentile(x.flatten(), 1)]),
        'Max Drawdown': lambda x: np.mean([np.min((1+path).cumprod() / np.maximum.accumulate((1+path).cumprod()) - 1) 
                                         for path in x])
    }
    
    # Compute metrics for each scenario
    metrics_results = {}
    for scenario, samples in results.items():
        metrics_results[scenario] = {metric: func(samples) for metric, func in metrics.items()}
    
    # Convert to DataFrame for visualization
    metrics_df = pd.DataFrame(metrics_results).T
    
    # Calculate percentage change from baseline
    for col in metrics_df.columns:
        baseline_value = metrics_df.loc['Baseline', col]
        if abs(baseline_value) > 1e-10:  # Avoid division by zero
            metrics_df[f'{col} %Change'] = (metrics_df[col] - baseline_value) / abs(baseline_value) * 100
        else:
            metrics_df[f'{col} %Change'] = 0
            
    # Create the visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Bar chart comparing VaR across scenarios
    axs[0, 0].bar(metrics_df.index, metrics_df['VaR (99.5%)'])
    axs[0, 0].set_title('99.5% VaR by Stress Scenario')
    axs[0, 0].set_ylabel('VaR (lower is worse)')
    axs[0, 0].grid(True, alpha=0.3)
    plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Add value labels to bars
    for i, v in enumerate(metrics_df['VaR (99.5%)']):
        axs[0, 0].text(i, v, f'{v:.2%}', ha='center', va='bottom' if v > 0 else 'top')
    
    # 2. Heatmap of percentage changes
    pct_change_cols = [col for col in metrics_df.columns if '%Change' in col]
    pct_change_df = metrics_df[pct_change_cols].copy()
    pct_change_df.columns = [col.replace(' %Change', '') for col in pct_change_df.columns]
    
    # Remove baseline row (will be all zeros)
    pct_change_df = pct_change_df.drop('Baseline')
    
    sns.heatmap(pct_change_df, annot=True, fmt=".1f", 
              cmap="RdBu_r", center=0, ax=axs[0, 1])
    axs[0, 1].set_title('% Change in Risk Metrics by Stress Scenario')
    
    # 3. Cumulative return paths for each scenario
    for i, (scenario, samples) in enumerate(results.items()):
        if i > 0:  # Skip baseline for clarity
            # Calculate cumulative returns for 20 random paths
            rand_indices = np.random.choice(len(samples), 20)
            cum_returns = [(1 + samples[idx]).cumprod() for idx in rand_indices]
            
            # Plot paths
            for path in cum_returns:
                axs[1, 0].plot(path, alpha=0.3)
            
            # Plot the median path with a thicker line
            median_path = np.median([(1 + samples[idx]).cumprod() for idx in range(len(samples))], axis=0)
            axs[1, 0].plot(median_path, linewidth=2, label=scenario)
    
    axs[1, 0].axhline(y=1.0, color='k', linestyle='-', alpha=0.5)
    axs[1, 0].set_title('Scenario Paths Comparison')
    axs[1, 0].set_ylabel('Cumulative Return')
    axs[1, 0].set_xlabel('Time Steps')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribution of final values by scenario
    for scenario, samples in results.items():
        final_values = (1 + samples).cumprod(axis=1)[:, -1]
        sns.kdeplot(final_values, ax=axs[1, 1], label=scenario)
    
    axs[1, 1].axvline(x=1.0, color='k', linestyle='-', alpha=0.5)
    axs[1, 1].set_title('Distribution of Final Values by Scenario')
    axs[1, 1].set_xlabel('Final Cumulative Return')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig