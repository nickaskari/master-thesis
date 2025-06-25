import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from dotenv.main import load_dotenv
from torch import autograd  # for computing gradients
from scipy.spatial import distance

load_dotenv(override=True)

class CrazyGAN:
    def __init__(self, returns_df, asset_name, latent_dim=200, window_size=252, quarter_length=200, 
                 batch_size=120, n_epochs=300, lambda_gp=60, lambda_tail=15, lambda_structure=20,
                 lambda_ring=15, num_rings=5):
        """
        Enhanced CGAN for equities with better utilization of PCA-based market regimes.
        
        Parameters:
          - returns_df: Empirical returns DataFrame (or Series) for the asset.
          - asset_name: Name of the asset.
          - latent_dim: Dimensionality of the noise vector.
          - window_size: Size of the rolling window in days.
          - quarter_length: Number of days in the quarter used to compute the condition.
          - batch_size: Batch size for training.
          - n_epochs: Number of epochs for training.
          - lambda_gp: Coefficient for the gradient penalty term.
          - lambda_tail: Coefficient for the tail penalty term.
          - lambda_structure: Coefficient for the structure penalty term.
          - lambda_ring: Coefficient for the new ring structure preservation penalty.
          - num_rings: Number of concentric rings to identify in PCA space.
        """
        self.returns_df = returns_df
        self.asset_name = asset_name
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.quarter_length = quarter_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_gp = lambda_gp
        self.lambda_tail = lambda_tail
        self.lambda_structure = lambda_structure
        self.lambda_ring = lambda_ring
        self.num_rings = num_rings

        # If returns_df is a DataFrame, select the column for the asset.
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]
        else:
            self.returns_series = returns_df

        # Create rolling windows of returns and scale them.
        self.rolling_returns, self.scaler = self.create_rolling_returns(self.returns_series)
        
        # NEW: Create PCA projection of the data for ring structure analysis
        self.pca = PCA(n_components=2)
        self.real_pca = self.pca.fit_transform(self.rolling_returns)
        
        # NEW: Identify ring membership for each window
        self.ring_labels = self.assign_to_rings(self.real_pca, self.num_rings)
        
        # NEW: Calculate distance from center for each window
        self.distances = np.sqrt(self.real_pca[:, 0]**2 + self.real_pca[:, 1]**2)
        
        # NEW: Calculate angle in PCA space
        self.angles = np.arctan2(self.real_pca[:, 1], self.real_pca[:, 0])
        
        # Enhance conditions with PCA-based insights
        self.conditions = self.create_enhanced_conditions(
            self.returns_series, 
            window_size, 
            self.distances, 
            self.angles, 
            self.ring_labels,
            lag_periods=[110, 95, 70]
        )
        
        # Ensure conditions align with rolling returns: discard the first few windows if necessary.
        min_length = min(len(self.rolling_returns), len(self.conditions))
        self.rolling_returns = self.rolling_returns[-min_length:]
        self.conditions = self.conditions[-min_length:]
        
        self.input_shape = (window_size,)
        self.cond_dim = self.conditions.shape[1]
        self.cuda = torch.cuda.is_available()

        # Instantiate Generator and Discriminator with condition inputs.
        self.generator = Generator(self.latent_dim, self.cond_dim, self.input_shape)
        self.discriminator = Discriminator(self.input_shape, self.cond_dim)

        self.dataloader = None
        self.optimizer_G = None
        self.optimizer_D = None
        
        # Store ring statistics for conditioning
        self.ring_stats = self.compute_ring_statistics(self.rolling_returns, self.ring_labels)

    def create_rolling_returns(self, returns_series):
        """
        Create rolling windows of scaled returns.
        """
        window_size = self.window_size
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(returns_series.values.reshape(-1, 1))
        
        rolling_returns = []
        for i in range(len(scaled_returns) - window_size + 1):
            window = scaled_returns[i:i + window_size]
            rolling_returns.append(window.flatten())
        return np.array(rolling_returns), scaler

    def assign_to_rings(self, pca_data, num_rings=5):
        """
        Assign each window to one of the concentric rings based on distance from center.
        """
        distances = np.sqrt(pca_data[:, 0]**2 + pca_data[:, 1]**2)
        
        # Define ring boundaries using percentiles
        percentiles = np.linspace(0, 100, num_rings + 1)
        thresholds = np.percentile(distances, percentiles)
        
        # Assign each point to a ring
        ring_labels = np.zeros(len(distances), dtype=int)
        for i in range(1, len(thresholds)):
            if i == 1:
                # First ring
                ring_labels[distances <= thresholds[i]] = 0
            else:
                # Middle rings
                ring_labels[(distances > thresholds[i-1]) & 
                           (distances <= thresholds[i])] = i-1
        
        return ring_labels

    def compute_ring_statistics(self, rolling_returns, ring_labels):
        """
        Compute statistics for each ring to use for conditioning.
        """
        unique_rings = np.unique(ring_labels)
        ring_stats = {}
        
        for ring in unique_rings:
            ring_windows = rolling_returns[ring_labels == ring]
            if len(ring_windows) > 0:
                # Calculate volatility
                volatility = np.mean([np.std(window) for window in ring_windows])
                
                # Calculate average cumulative return
                cum_returns = np.array([np.sum(window) for window in ring_windows])
                avg_return = np.mean(cum_returns)
                
                # Other statistics can be added here
                ring_stats[ring] = {
                    'volatility': volatility,
                    'avg_return': avg_return,
                    'count': len(ring_windows)
                }
        
        return ring_stats

    def create_enhanced_conditions(self, returns_series, window_size, distances, angles, ring_labels, lag_periods=[252, 150, 63]):
        """
        Create enhanced conditions using both traditional metrics and PCA-based insights.
        """
        conditions = []
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        
        n = len(returns_series)
        max_lag = max(lag_periods)
        
        for i in range(max_lag, n - window_size + 1):
            condition_vector = []
            
            # Traditional lag-based metrics
            for lag in lag_periods:
                window = returns_series.iloc[i - lag:i].to_numpy()
                cum_return = float(np.prod(1 + window) - 1)
                volatility = float(window.std(ddof=1))
                kurtosis = float(pd.Series(window).kurt())
                
                # Calculate drawdown
                window_cum = (1 + window).cumprod()
                running_max = np.maximum.accumulate(window_cum)
                drawdowns = (window_cum - running_max) / running_max
                max_drawdown = float(drawdowns.min())
                
                # Calculate crash duration
                count = 0
                max_count = 0
                for r in window:
                    if r < 0:
                        count += 1
                        max_count = max(max_count, count)
                    else:
                        count = 0
                crash_duration = float(max_count)
                
                condition_vector.extend([
                    cum_return, 
                    volatility, 
                    max_drawdown, 
                    crash_duration
                ])
            
            # Add index for current window
            idx = i - max_lag
            if idx < len(distances):
                # Add PCA-based metrics
                condition_vector.extend([
                    float(distances[idx]),         # Distance from center in PCA space
                    float(np.cos(angles[idx])),    # Cosine of angle (x-component)
                    float(np.sin(angles[idx])),    # Sine of angle (y-component)
                    float(ring_labels[idx])        # Ring membership (0 = inner, higher = outer)
                ])
            else:
                # Handle edge case (should not happen if aligned properly)
                condition_vector.extend([0.0, 0.0, 0.0, 0.0])
                
            conditions.append(condition_vector)
            
        return np.array(conditions, dtype=float)

    def setup(self):
        dataset = TensorDataset(torch.tensor(self.rolling_returns, dtype=torch.float32),
                                torch.tensor(self.conditions, dtype=torch.float32))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def compute_gradient_penalty(self, real_samples, fake_samples, cond):
        """Calculates the gradient penalty loss for WGAN-GP"""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, device=real_samples.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates, cond)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = self.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty

    def compute_tail_penalty(self, real_returns, gen_returns):
        """Penalize differences in both lower and upper tails"""
        real_flat = real_returns.view(-1)
        gen_flat = gen_returns.view(-1)
        
        k = int(0.1 * real_flat.size(0))
        k = max(k, 1)
        
        # Sort for lower and upper tails
        real_sorted, _ = torch.sort(real_flat)
        gen_sorted, _ = torch.sort(gen_flat)
        
        # Lower tail penalty
        real_lower_tail = torch.mean(real_sorted[:k])
        gen_lower_tail = torch.mean(gen_sorted[:k])
        lower_penalty = (gen_lower_tail - real_lower_tail) ** 2
        
        # Upper tail penalty
        real_upper_tail = torch.mean(real_sorted[-k:])
        gen_upper_tail = torch.mean(gen_sorted[-k:])
        upper_penalty = (gen_upper_tail - real_upper_tail) ** 2
        
        return lower_penalty + upper_penalty
    
    def compute_structure_penalty(self, real_returns, gen_returns):
        """Penalty to preserve the ring structure in PCA space"""
        # Use the first 2 PCs 
        real_flat = real_returns.view(real_returns.size(0), -1).cpu().detach().numpy()
        gen_flat = gen_returns.view(gen_returns.size(0), -1).cpu().detach().numpy()
        
        # Use the pre-fit PCA transform
        real_pca = torch.tensor(self.pca.transform(real_flat), device=real_returns.device)
        gen_pca = torch.tensor(self.pca.transform(gen_flat), device=gen_returns.device)
        
        # Calculate distance from origin for each point
        real_dist = torch.sqrt(torch.sum(real_pca**2, dim=1))
        gen_dist = torch.sqrt(torch.sum(gen_pca**2, dim=1))
        
        # Compare distributions of distances (this preserves the ring structure)
        real_dist_sorted, _ = torch.sort(real_dist)
        gen_dist_sorted, _ = torch.sort(gen_dist)
        
        # Calculate MSE between sorted distances to match the ring structure
        dist_penalty = F.mse_loss(gen_dist_sorted, real_dist_sorted)
        
        return dist_penalty

    def compute_ring_penalty(self, real_returns, gen_returns, cond):
        """
        New penalty that enforces the generated returns to match the statistical 
        properties of the ring they're conditioned on.
        """
        batch_size = real_returns.size(0)
        device = real_returns.device
        
        # Extract the ring label from the condition vector
        # Assuming ring label is the last element in the condition
        ring_labels = cond[:, -1].long()
        
        # Get the generated returns
        gen_flat = gen_returns.view(gen_returns.size(0), -1)
        
        # Calculate statistics of generated returns
        gen_volatility = torch.std(gen_returns, dim=1)
        gen_cum_return = torch.sum(gen_returns, dim=1)
        
        # Initialize penalty
        ring_penalty = torch.tensor(0.0, device=device)
        
        # For each unique ring in the batch
        for ring in torch.unique(ring_labels):
            ring_idx = ring_labels == ring
            if torch.sum(ring_idx) > 0:
                # Get the target statistics for this ring
                target_vol = torch.tensor(self.ring_stats[ring.item()]['volatility'], 
                                          device=device)
                target_return = torch.tensor(self.ring_stats[ring.item()]['avg_return'], 
                                             device=device)
                
                # Calculate mean statistics for generated samples in this ring
                ring_vol = gen_volatility[ring_idx].mean()
                ring_ret = gen_cum_return[ring_idx].mean()
                
                # Add penalty for deviations from ring statistics
                vol_penalty = F.mse_loss(ring_vol, target_vol)
                ret_penalty = F.mse_loss(ring_ret, target_return)
                
                ring_penalty += vol_penalty + ret_penalty
                
        return ring_penalty / len(torch.unique(ring_labels))

    def train(self):
        self.setup()
        for epoch in range(self.n_epochs):
            for i, (real_returns, cond) in enumerate(self.dataloader):
                batch_size = real_returns.size(0)
                real_returns = real_returns.to(next(self.discriminator.parameters()).device)
                cond = cond.to(real_returns.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                z = torch.randn(batch_size, self.latent_dim, device=real_returns.device)
                gen_returns = self.generator(z, cond)

                d_real = self.discriminator(real_returns, cond)
                d_fake = self.discriminator(gen_returns.detach(), cond)
                d_loss = -torch.mean(d_real) + torch.mean(d_fake)
                
                gp = self.compute_gradient_penalty(real_returns, gen_returns.detach(), cond)
                d_loss += gp

                d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                if i % 3 == 0:
                    self.optimizer_G.zero_grad()
                    gen_returns = self.generator(z, cond)
                    
                    # Compute penalties
                    tail_penalty = self.compute_tail_penalty(real_returns, gen_returns)
                    structure_penalty = self.compute_structure_penalty(real_returns, gen_returns)
                    ring_penalty = self.compute_ring_penalty(real_returns, gen_returns, cond)
                    
                    # Add penalties to generator loss
                    g_loss = -torch.mean(self.discriminator(gen_returns, cond)) + \
                             self.lambda_tail * tail_penalty + \
                             self.lambda_structure * structure_penalty + \
                             self.lambda_ring * ring_penalty
                             
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                          f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                          f"[Tail: {tail_penalty.item():.4f}] [Structure: {structure_penalty.item():.4f}] "
                          f"[Ring: {ring_penalty.item():.4f}]")

    def generate_scenarios(self, save=True, num_scenarios=50000, target_ring=None,
                          target_volatility=None, target_return=None):
        """
        Generate scenarios with optional targeting of specific market regimes.
        
        Parameters:
        - save: Whether to save the generated scenarios
        - num_scenarios: Number of scenarios to generate
        - target_ring: Specific ring to target (0=inner, higher=outer)
        - target_volatility: Target volatility level
        - target_return: Target cumulative return
        """
        self.generator.eval()
        device = 'cuda' if self.cuda else 'cpu'
        
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim, device=device)
                
                # Base sampling - random conditions
                if target_ring is None and target_volatility is None and target_return is None:
                    indices = np.random.choice(len(self.conditions), size=batch_size, replace=True)
                    cond = torch.tensor(self.conditions[indices], dtype=torch.float32, device=device)
                
                # Targeted sampling - specifically target conditions based on desired market regime
                else:
                    # Start with random conditions as a base
                    indices = np.random.choice(len(self.conditions), size=batch_size, replace=True)
                    cond = self.conditions[indices].copy()
                    
                    # If target ring specified, modify the ring label in the condition
                    if target_ring is not None:
                        # Assuming ring label is the last element
                        cond[:, -1] = target_ring
                        
                        # Also update other PCA-based metrics to be consistent
                        # Get average distance for this ring
                        ring_distances = self.distances[self.ring_labels == target_ring]
                        if len(ring_distances) > 0:
                            avg_distance = np.mean(ring_distances)
                            # Set the distance in the condition
                            cond[:, -4] = avg_distance
                    
                    # If target volatility specified, modify volatility in condition
                    if target_volatility is not None:
                        # Assuming volatility is at indices 1, 5, 9 for different lags
                        cond[:, 1] = target_volatility
                        cond[:, 5] = target_volatility
                        cond[:, 9] = target_volatility
                    
                    # If target return specified, modify returns in condition
                    if target_return is not None:
                        # Assuming returns are at indices 0, 4, 8 for different lags
                        cond[:, 0] = target_return
                        cond[:, 4] = target_return
                        cond[:, 8] = target_return
                    
                    cond = torch.tensor(cond, dtype=torch.float32, device=device)
                
                gen_returns = self.generator(z, cond)
                gen_returns = self.scaler.inverse_transform(gen_returns.cpu().numpy())
                all_generated_returns.append(gen_returns)
                
        all_generated_returns = np.vstack(all_generated_returns)
        
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            
            # Create a more descriptive filename if targeting specific regime
            filename = f'generated_returns_{self.asset_name}'
            if target_ring is not None:
                filename += f'_ring{target_ring}'
            if target_volatility is not None:
                filename += f'_vol{target_volatility:.2f}'
            if target_return is not None:
                filename += f'_ret{target_return:.2f}'
            filename += '_scenarios.pt'
            
            save_path = os.path.join(save_dir, filename)
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated scenarios saved to: {save_path}")
        
        return all_generated_returns


# Enhanced Generator with special handling for market regime conditioning
class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.output_shape = output_shape  # e.g., (252,)
        input_dim = latent_dim + cond_dim  # Concatenate noise and condition.

        # Main network
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(output_shape)))
        )
        
        # Conditioning network - extracts features from condition
        self.condition_network = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2)
        )
        
        # Modulation layers - used to modulate the main network
        self.modulation = nn.Linear(128, 512)

    def forward(self, noise, condition):
        # Process condition through condition network
        cond_features = self.condition_network(condition)
        modulation = torch.sigmoid(self.modulation(cond_features))
        
        # Concatenate noise and condition for main network
        x = torch.cat((noise, condition), dim=1)
        
        # Get intermediate features
        x = self.main_network[0](x)  # First linear
        x = self.main_network[1](x)  # Batch norm
        x = self.main_network[2](x)  # ReLU
        
        # Apply modulation to affect the network based on condition
        x = x * modulation
        
        # Continue through rest of network
        x = self.main_network[3](x)  # Second linear
        x = self.main_network[4](x)  # Batch norm
        x = self.main_network[5](x)  # ReLU
        x = self.main_network[6](x)  # Output linear
        
        return x.view(x.size(0), *self.output_shape)


# Enhanced Discriminator with better condition integration
class Discriminator(nn.Module):
    def __init__(self, input_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape  # e.g., (252,)
        self.cond_dim = cond_dim
        input_dim = int(np.prod(input_shape))
        
        # Process returns
        self.returns_network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # Process conditions
        self.condition_network = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # Combined network after processing returns and conditions separately
        self.combined_network = nn.Sequential(
            nn.Linear(512 + 128, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1)
        )

    def forward(self, returns, condition):
        # Process returns
        x_returns = returns.view(returns.size(0), -1)
        x_returns = self.returns_network(x_returns)
        
        # Process conditions
        x_cond = self.condition_network(condition)
        
        # Concatenate processed returns and conditions
        x_combined = torch.cat((x_returns, x_cond), dim=1)
        
        # Process through combined network
        validity = self.combined_network(x_combined)
        
        return validity