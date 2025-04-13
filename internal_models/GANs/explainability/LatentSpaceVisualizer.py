import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from tqdm import tqdm

class LatentSpaceVisualizer:
    def __init__(self, gan_model, n_samples=2000, random_state=42):
        """
        Initialize the latent space visualizer for a GAN model
        
        Parameters:
        -----------
        gan_model : FashionGAN
            The trained GAN model
        n_samples : int
            Number of samples to generate for visualization
        random_state : int
            Random seed for reproducibility
        """
        self.gan_model = gan_model
        self.n_samples = n_samples
        self.random_state = random_state
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set the random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            
    def generate_latent_samples(self):
        """Generate random samples from the latent space"""
        return torch.randn(self.n_samples, self.gan_model.latent_dim, device=self.device)
    
    def visualize_with_tsne(self, perplexity=30, n_iter=1000):
        """
        Visualize the latent space using t-SNE
        
        Parameters:
        -----------
        perplexity : int
            Perplexity parameter for t-SNE (typical values: 5-50)
        n_iter : int
            Number of iterations for t-SNE optimization
        """
        print("Generating latent samples...")
        z_samples = self.generate_latent_samples().cpu().detach().numpy()
        
        # Get condition from the latest data point
        condition = torch.tensor(self.gan_model.conditions[-1:], 
                                dtype=torch.float32, 
                                device=self.device)
        condition = condition.repeat(self.n_samples, 1)
        
        # Generate samples
        with torch.no_grad():
            gen_samples = self.gan_model.generator(torch.tensor(z_samples, device=self.device), 
                                                 condition).cpu().numpy()
        
        # Apply t-SNE to the latent vectors
        print(f"Applying t-SNE to {self.n_samples} latent vectors...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=self.random_state)
        z_tsne = tsne.fit_transform(z_samples)
        
        # Create a plot
        plt.figure(figsize=(12, 10))
        
        # Calculate a measure of extreme returns in the generated samples
        volatility = np.std(gen_samples, axis=1)
        min_returns = np.min(gen_samples, axis=1)
        
        # Plot with volatility as color
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=volatility, cmap='viridis', 
                             alpha=0.7, s=30, edgecolors='w', linewidths=0.5)
        plt.colorbar(scatter, label='Volatility')
        plt.title('t-SNE Visualization of Latent Space (Colored by Volatility)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Plot with minimum returns as color
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=min_returns, cmap='coolwarm', 
                             alpha=0.7, s=30, edgecolors='w', linewidths=0.5)
        plt.colorbar(scatter, label='Minimum Return')
        plt.title('t-SNE Visualization of Latent Space (Colored by Min Return)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.savefig('latent_space_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return z_samples, z_tsne, gen_samples
    
    def visualize_with_pca(self):
        """Visualize the latent space using PCA"""
        print("Generating latent samples...")
        z_samples = self.generate_latent_samples().cpu().detach().numpy()
        
        # Get condition from the latest data point
        condition = torch.tensor(self.gan_model.conditions[-1:], 
                                dtype=torch.float32, 
                                device=self.device)
        condition = condition.repeat(self.n_samples, 1)
        
        # Generate samples
        with torch.no_grad():
            gen_samples = self.gan_model.generator(torch.tensor(z_samples, device=self.device), 
                                                 condition).cpu().numpy()
        
        # Apply PCA to the latent vectors
        print("Applying PCA to latent vectors...")
        pca = PCA(n_components=3)
        z_pca = pca.fit_transform(z_samples)
        
        # Calculate variance explained
        explained_variance = pca.explained_variance_ratio_
        print(f"Variance explained by first 3 PCs: {explained_variance.sum():.2%}")
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 12))
        
        # Calculate metrics for coloring
        volatility = np.std(gen_samples, axis=1)
        min_returns = np.min(gen_samples, axis=1)
        max_drawdown = np.array([self._calculate_max_drawdown(sample) for sample in gen_samples])
        
        # 3D plot with volatility coloring
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], 
                             c=volatility, cmap='viridis', s=30, alpha=0.7)
        fig.colorbar(scatter1, ax=ax1, label='Volatility')
        ax1.set_title('PCA of Latent Space\n(colored by volatility)')
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax1.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
        
        # 3D plot with min returns coloring
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], 
                             c=min_returns, cmap='coolwarm', s=30, alpha=0.7)
        fig.colorbar(scatter2, ax=ax2, label='Min Return')
        ax2.set_title('PCA of Latent Space\n(colored by minimum return)')
        ax2.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
        
        # 3D plot with max drawdown coloring
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], 
                             c=max_drawdown, cmap='Reds_r', s=30, alpha=0.7)
        fig.colorbar(scatter3, ax=ax3, label='Max Drawdown')
        ax3.set_title('PCA of Latent Space\n(colored by max drawdown)')
        ax3.set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
        ax3.set_zlabel(f'PC3 ({explained_variance[2]:.1%})')
        
        plt.tight_layout()
        plt.savefig('latent_space_pca_3d.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return z_samples, z_pca, gen_samples
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown for a return series"""
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        return np.min(drawdowns)
    
    def latent_space_traversal(self, n_steps=10, dim1=0, dim2=1, scale=3.0):
        """
        Create a grid traversal of the latent space along two dimensions
        
        Parameters:
        -----------
        n_steps : int
            Number of steps in each dimension
        dim1, dim2 : int
            Dimensions to traverse
        scale : float
            Scale factor for the traversal range
        """
        # Create a base random vector
        base_z = torch.zeros(1, self.gan_model.latent_dim, device=self.device)
        
        # Get condition
        condition = torch.tensor(self.gan_model.conditions[-1:], 
                                dtype=torch.float32, 
                                device=self.device)
        
        # Create grid of values for the two dimensions
        linspace = np.linspace(-scale, scale, n_steps)
        grid1, grid2 = np.meshgrid(linspace, linspace)
        
        # Initialize storage for generated samples
        all_samples = np.zeros((n_steps, n_steps, self.gan_model.window_size))
        
        # Generate samples for each point in the grid
        for i in range(n_steps):
            for j in range(n_steps):
                # Set the two dimensions we're varying
                z = base_z.clone()
                z[0, dim1] = grid1[i, j]
                z[0, dim2] = grid2[i, j]
                
                # Generate sample
                with torch.no_grad():
                    gen_sample = self.gan_model.generator(z, condition).cpu().numpy()
                    all_samples[i, j] = gen_sample[0]
        
        # Calculate metrics for each sample
        volatility = np.zeros((n_steps, n_steps))
        min_returns = np.zeros((n_steps, n_steps))
        max_drawdown = np.zeros((n_steps, n_steps))
        
        for i in range(n_steps):
            for j in range(n_steps):
                returns = all_samples[i, j]
                volatility[i, j] = np.std(returns)
                min_returns[i, j] = np.min(returns)
                max_drawdown[i, j] = self._calculate_max_drawdown(returns)
        
        # Plot heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Volatility heatmap
        sns.heatmap(volatility, ax=axes[0], cmap='viridis', annot=False)
        axes[0].set_title(f'Volatility across dimensions {dim1} and {dim2}')
        axes[0].set_xlabel(f'Dimension {dim1} (from {-scale:.1f} to {scale:.1f})')
        axes[0].set_ylabel(f'Dimension {dim2} (from {-scale:.1f} to {scale:.1f})')
        
        # Min returns heatmap
        sns.heatmap(min_returns, ax=axes[1], cmap='coolwarm', annot=False)
        axes[1].set_title(f'Minimum Return across dimensions {dim1} and {dim2}')
        axes[1].set_xlabel(f'Dimension {dim1} (from {-scale:.1f} to {scale:.1f})')
        axes[1].set_ylabel(f'Dimension {dim2} (from {-scale:.1f} to {scale:.1f})')
        
        # Max drawdown heatmap
        sns.heatmap(max_drawdown, ax=axes[2], cmap='Reds_r', annot=False)
        axes[2].set_title(f'Maximum Drawdown across dimensions {dim1} and {dim2}')
        axes[2].set_xlabel(f'Dimension {dim1} (from {-scale:.1f} to {scale:.1f})')
        axes[2].set_ylabel(f'Dimension {dim2} (from {-scale:.1f} to {scale:.1f})')
        
        plt.tight_layout()
        plt.savefig(f'latent_traversal_dim{dim1}_{dim2}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return all_samples, volatility, min_returns, max_drawdown
    
    def latent_component_analysis(self, n_components=10, n_samples=500):
        """
        Analyze how individual components of the latent vector affect the generated outputs
        
        Parameters:
        -----------
        n_components : int
            Number of latent dimensions to analyze
        n_samples : int
            Number of samples per component
        """
        # Select the top components to analyze
        components = list(range(min(n_components, self.gan_model.latent_dim)))
        
        # Get condition
        condition = torch.tensor(self.gan_model.conditions[-1:], 
                                dtype=torch.float32, 
                                device=self.device)
        condition = condition.repeat(n_samples, 1)
        
        # Values to set for each component
        values = np.linspace(-3, 3, n_samples)
        
        # Initialize metrics dataframe
        metrics_df = pd.DataFrame()
        
        # For each component
        for comp in tqdm(components, desc="Analyzing latent components"):
            # Initialize arrays to store metrics
            volatilities = np.zeros(n_samples)
            min_returns_arr = np.zeros(n_samples)
            max_drawdowns = np.zeros(n_samples)
            mean_returns = np.zeros(n_samples)
            
            # For each value
            for i, val in enumerate(values):
                # Create zero vector
                z = torch.zeros(1, self.gan_model.latent_dim, device=self.device)
                
                # Set the component we're analyzing
                z[0, comp] = val
                
                # Generate sample
                with torch.no_grad():
                    gen_sample = self.gan_model.generator(z, condition[0:1]).cpu().numpy()[0]
                
                # Calculate metrics
                volatilities[i] = np.std(gen_sample)
                min_returns_arr[i] = np.min(gen_sample)
                max_drawdowns[i] = self._calculate_max_drawdown(gen_sample)
                mean_returns[i] = np.mean(gen_sample)
            
            # Add to dataframe
            comp_df = pd.DataFrame({
                'component': comp,
                'value': values,
                'volatility': volatilities,
                'min_return': min_returns_arr,
                'max_drawdown': max_drawdowns,
                'mean_return': mean_returns
            })
            metrics_df = pd.concat([metrics_df, comp_df])
        
        # Calculate correlation between component value and metrics
        correlations = pd.DataFrame()
        for comp in components:
            comp_data = metrics_df[metrics_df['component'] == comp]
            corr = {
                'component': comp,
                'volatility_corr': np.corrcoef(comp_data['value'], comp_data['volatility'])[0, 1],
                'min_return_corr': np.corrcoef(comp_data['value'], comp_data['min_return'])[0, 1],
                'max_drawdown_corr': np.corrcoef(comp_data['value'], comp_data['max_drawdown'])[0, 1],
                'mean_return_corr': np.corrcoef(comp_data['value'], comp_data['mean_return'])[0, 1]
            }
            correlations = pd.concat([correlations, pd.DataFrame([corr])])
        
        # Plot correlations
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(correlations['component'], correlations['volatility_corr'])
        plt.title('Correlation between Component Value and Volatility')
        plt.xlabel('Component')
        plt.ylabel('Correlation')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.bar(correlations['component'], correlations['min_return_corr'])
        plt.title('Correlation between Component Value and Min Return')
        plt.xlabel('Component')
        plt.ylabel('Correlation')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.bar(correlations['component'], correlations['max_drawdown_corr'])
        plt.title('Correlation between Component Value and Max Drawdown')
        plt.xlabel('Component')
        plt.ylabel('Correlation')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.bar(correlations['component'], correlations['mean_return_corr'])
        plt.title('Correlation between Component Value and Mean Return')
        plt.xlabel('Component')
        plt.ylabel('Correlation')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('latent_component_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot the effect of the top components
        top_components = correlations.iloc[
            np.argsort(np.abs(correlations['volatility_corr']))[-5:]]['component'].values
        
        plt.figure(figsize=(15, 12))
        
        for i, comp in enumerate(top_components):
            comp_data = metrics_df[metrics_df['component'] == comp]
            
            plt.subplot(5, 4, i*4+1)
            plt.plot(comp_data['value'], comp_data['volatility'])
            plt.title(f'Component {int(comp)}: Volatility')
            plt.grid(alpha=0.3)
            if i == len(top_components)-1:
                plt.xlabel('Component Value')
            
            plt.subplot(5, 4, i*4+2)
            plt.plot(comp_data['value'], comp_data['min_return'])
            plt.title(f'Component {int(comp)}: Min Return')
            plt.grid(alpha=0.3)
            if i == len(top_components)-1:
                plt.xlabel('Component Value')
            
            plt.subplot(5, 4, i*4+3)
            plt.plot(comp_data['value'], comp_data['max_drawdown'])
            plt.title(f'Component {int(comp)}: Max Drawdown')
            plt.grid(alpha=0.3)
            if i == len(top_components)-1:
                plt.xlabel('Component Value')
            
            plt.subplot(5, 4, i*4+4)
            plt.plot(comp_data['value'], comp_data['mean_return'])
            plt.title(f'Component {int(comp)}: Mean Return')
            plt.grid(alpha=0.3)
            if i == len(top_components)-1:
                plt.xlabel('Component Value')
        
        plt.tight_layout()
        plt.savefig('top_components_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df, correlations

# Example usage:
"""
# Assuming you have a trained GAN model
gan = FashionGAN(returns_df, asset_name, latent_dim=200, ...)

# Load the trained model
gan.generator.load_state_dict(torch.load('generator_weights.pt'))
gan.discriminator.load_state_dict(torch.load('discriminator_weights.pt'))

# Create the visualizer
visualizer = LatentSpaceVisualizer(gan, n_samples=2000)

# Use t-SNE to visualize the latent space
z_samples, z_tsne, gen_samples = visualizer.visualize_with_tsne(perplexity=30)

# Use PCA to visualize the latent space in 3D
z_samples, z_pca, gen_samples = visualizer.visualize_with_pca()

# Analyze how traversing specific dimensions affects the output
samples, vol, min_ret, max_dd = visualizer.latent_space_traversal(n_steps=10, dim1=0, dim2=1)

# Analyze how individual latent dimensions affect the output
metrics_df, correlations = visualizer.latent_component_analysis(n_components=10)
"""