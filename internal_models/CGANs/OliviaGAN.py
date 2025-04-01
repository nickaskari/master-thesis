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

load_dotenv(override=True)

class OliviaGAN:
    # INCREASE LATENT SPACE
    def __init__(self, returns_df, asset_name, latent_dim=200, window_size=252, quarter_length=200, 
                 batch_size=120, n_epochs=600, lambda_gp=60, lambda_tail=40, lambda_structure=30):
        """
        CGAN1: Conditional GAN for equities that conditions on a lagged quarter's cumulative return.
        
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
        self.lambda_nn = 100.0
        self.lambda_outlier = 150.0 

        # If returns_df is a DataFrame, select the column for the asset.
        if isinstance(returns_df, pd.DataFrame):
            self.returns_series = returns_df[asset_name]
        else:
            self.returns_series = returns_df

        # Create rolling windows of returns and scale them.
        self.rolling_returns, self.scaler = self.create_rolling_returns(self.returns_series)
        # Automatically create condition data based on the lagged quarter cumulative returns.
        self.conditions = self.create_lagged_quarter_conditions(self.returns_series, window_size, quarter_length)

        self.conditions = self.create_multi_lag_conditions(self.returns_series, window_size, lag_periods=[251])
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
        
        # PCA and related attributes will be initialized in setup()
        self.pca = None
        self.real_distances_sorted = None
        self.real_distances_tensor = None

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

    def create_lagged_quarter_conditions(self, returns_series, window_size, quarter_length=63):
        """
        Create conditions using the cumulative return over the quarter preceding each rolling window.
        """
        conditions = []
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        
        n = len(returns_series)
        for i in range(quarter_length, n - window_size + 1):
            window = returns_series.iloc[i - quarter_length:i]
            cum_return = np.prod(1 + window) - 1
            volatility = window.std()
            kurtosis = pd.Series(window).kurt()
            conditions.append([cum_return, volatility, kurtosis])       
        return np.array(conditions)
    
    def create_multi_lag_conditions(self, returns_series, window_size, lag_periods=[252, 150, 63]):
        conditions = []
        if not isinstance(returns_series, pd.Series):
            returns_series = pd.Series(returns_series)
        
        n = len(returns_series)
        max_lag = max(lag_periods)
        
        for i in range(max_lag, n - window_size + 1):
            condition_vector = []
            for lag in lag_periods:
                window = returns_series.iloc[i - lag:i].to_numpy()
                cum_return = float(np.prod(1 + window) - 1)
                volatility = float(window.std(ddof=1))
                kurtosis = float(pd.Series(window).kurt())
                threshold = np.percentile(window, 10)
                tail_losses = window[window <= threshold]
                if len(tail_losses) > 0:
                    cvar = float(tail_losses.mean())
                else:
                    cvar = float(threshold)
                window_cum = (1 + window).cumprod()
                running_max = np.maximum.accumulate(window_cum)
                drawdowns = (window_cum - running_max) / running_max
                max_drawdown = float(drawdowns.min())
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
                    kurtosis,
                    max_drawdown, 
                    #crash_duration
                ])
            conditions.append(condition_vector)
            
        return np.array(conditions, dtype=float)

    def setup(self):
        # Original setup code
        dataset = TensorDataset(torch.tensor(self.rolling_returns, dtype=torch.float32),
                                torch.tensor(self.conditions, dtype=torch.float32))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Pre-compute PCA on all training data
        self.pca = PCA(n_components=2)
        self.pca.fit(self.rolling_returns)
        
        # Compute PCA projections and distances
        real_pca = self.pca.transform(self.rolling_returns)
        real_distances = np.sqrt(np.sum(real_pca**2, axis=1))
        self.real_distances_sorted = np.sort(real_distances)
        self.real_distances_tensor = torch.tensor(self.real_distances_sorted, dtype=torch.float32)
        
        # Compute volatility for each window in the original dataset
        # Extract volatility values from conditions (assuming volatility is at index 1 in each condition)
        # This works because your conditions include volatility as one of the features
        volatilities = self.conditions[:, 1]  # Adjust index if needed based on your condition structure
        
        # Create volatility to radius mapping
        # This creates a dict mapping volatility bins to typical radius values
        self.vol_to_radius = {}
        n_bins = 20
        vol_bins = np.linspace(np.min(volatilities), np.max(volatilities), n_bins+1)
        
        for i in range(n_bins):
            vol_min, vol_max = vol_bins[i], vol_bins[i+1]
            mask = (volatilities >= vol_min) & (volatilities < vol_max)
            
            if np.sum(mask) > 0:
                # Get the distances for windows in this volatility bin
                bin_distances = real_distances[mask]
                # Store median and std of distances for this volatility bin
                bin_vol = (vol_min + vol_max) / 2
                self.vol_to_radius[bin_vol] = {
                    'median': np.median(bin_distances),
                    'std': np.std(bin_distances)
                }
        
        if self.cuda:
            self.real_distances_tensor = self.real_distances_tensor.cuda()

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
    
    def compute_structure_penalty(self, gen_returns):
        """Penalty to preserve the ring structure using pre-computed PCA"""
        # Get batch size
        batch_size = gen_returns.size(0)
        
        # Transform generated returns using pre-computed PCA
        gen_flat = gen_returns.view(batch_size, -1).cpu().detach().numpy()
        gen_pca = self.pca.transform(gen_flat)
        
        # Calculate distances from origin
        gen_distances = np.sqrt(np.sum(gen_pca**2, axis=1))
        gen_distances_sorted = np.sort(gen_distances)
        
        # Convert to tensor
        gen_distances_tensor = torch.tensor(gen_distances_sorted, 
                                          dtype=torch.float32, 
                                          device=gen_returns.device)
        
        # Sample points from real distances to match batch size
        indices = np.linspace(0, len(self.real_distances_sorted)-1, batch_size, dtype=int)
        real_sample = self.real_distances_tensor[indices].to(gen_returns.device)
        
        # Calculate MSE
        dist_penalty = F.mse_loss(gen_distances_tensor, real_sample)
        
        return dist_penalty

    def compute_vol_structure_penalty(self, gen_returns, cond):
        """Penalty to preserve the ring structure using volatility as a guide"""
        batch_size = gen_returns.size(0)
        
        # Extract volatility from conditions (adjust index if needed)
        volatilities = cond[:, 1].cpu().detach().numpy()
        
        # Transform generated returns using pre-computed PCA
        gen_flat = gen_returns.view(batch_size, -1).cpu().detach().numpy()
        gen_pca = self.pca.transform(gen_flat)
        
        # Calculate distances from origin
        gen_distances = np.sqrt(np.sum(gen_pca**2, axis=1))
        
        # Calculate target distances based on volatility
        target_distances = np.zeros_like(gen_distances)
        
        for i, vol in enumerate(volatilities):
            # Find closest volatility bin
            closest_vol = min(self.vol_to_radius.keys(), key=lambda x: abs(x - vol))
            
            # Get expected radius and variation
            expected_radius = self.vol_to_radius[closest_vol]['median'] * (vol / closest_vol)**0.5
            radius_std = self.vol_to_radius[closest_vol]['std']
            
            # Set target distance with some noise to avoid collapse
            target_distances[i] = expected_radius
        
        # Convert to tensors
        gen_distances_tensor = torch.tensor(gen_distances, dtype=torch.float32, device=gen_returns.device)
        target_distances_tensor = torch.tensor(target_distances, dtype=torch.float32, device=gen_returns.device)
        
        # Calculate MSE between actual and target distances
        dist_penalty = F.mse_loss(gen_distances_tensor, target_distances_tensor)
        
        return dist_penalty
    
    def compute_nearest_neighbor_penalty(self, gen_returns, max_distance_multiplier=3.0):
        """
        Penalize generated points that are too far from their closest real point in PCA space
        
        Parameters:
        gen_returns: Generated batch of returns
        max_distance_multiplier: How many times the average NN distance is allowed before penalty
        """
        batch_size = gen_returns.size(0)
        
        # Transform generated returns using pre-computed PCA
        gen_flat = gen_returns.view(batch_size, -1).cpu().detach().numpy()
        gen_pca = self.pca.transform(gen_flat)
        
        # Get real data in PCA space (compute once during setup and store for efficiency)
        if not hasattr(self, 'real_pca'):
            self.real_pca = self.pca.transform(self.rolling_returns)
        
        # Calculate nearest neighbor distances for each generated point
        nn_distances = []
        
        for i in range(batch_size):
            # Calculate distance to all real points
            distances = np.sqrt(np.sum((self.real_pca - gen_pca[i])**2, axis=1))
            # Get minimum distance (nearest neighbor)
            min_distance = np.min(distances)
            nn_distances.append(min_distance)
        
        nn_distances = np.array(nn_distances)
        
        # Get average nearest neighbor distance from real data to real data (compute once during setup)
        if not hasattr(self, 'avg_real_nn_distance'):
            real_nn_distances = []
            for i in range(len(self.real_pca)):
                # Calculate distances to all other real points
                distances = np.sqrt(np.sum((self.real_pca - self.real_pca[i])**2, axis=1))
                # Sort distances and take second smallest (first is distance to self = 0)
                sorted_distances = np.sort(distances)
                nearest_neighbor_dist = sorted_distances[1]  # First non-zero distance
                real_nn_distances.append(nearest_neighbor_dist)
            
            self.avg_real_nn_distance = np.mean(real_nn_distances)
            self.std_real_nn_distance = np.std(real_nn_distances)
        
        # Calculate threshold beyond which we apply penalties
        threshold = self.avg_real_nn_distance * max_distance_multiplier
        
        # Only penalize points that are too far away
        excess_distances = np.maximum(0, nn_distances - threshold)
        
        # Convert to tensor for the loss
        excess_distances_tensor = torch.tensor(
            excess_distances, 
            dtype=torch.float32, 
            device=gen_returns.device
        )
        
        # Use mean squared error as the penalty
        penalty = torch.mean(excess_distances_tensor ** 2)
        
        return penalty

    def compute_outlier_boundary_penalty(self, gen_returns, outlier_percentile=90):
        """
        Specifically penalize the most extreme outliers in PCA space
        
        Parameters:
        gen_returns: Generated batch of returns
        outlier_percentile: Percentile threshold to identify outlier points (e.g., 90 = top 10% most extreme)
        """
        batch_size = gen_returns.size(0)
        device = gen_returns.device
        
        # Transform generated returns to 2D PCA space
        gen_flat = gen_returns.view(batch_size, -1).cpu().detach().numpy()
        gen_pca = self.pca.transform(gen_flat)
        
        # Calculate distances from origin
        gen_distances = np.sqrt(gen_pca[:, 0]**2 + gen_pca[:, 1]**2)
        
        # Compute or retrieve the convex hull of real data points (boundary)
        if not hasattr(self, 'real_data_hull'):
            from scipy.spatial import ConvexHull
            real_pca = self.pca.transform(self.rolling_returns)
            
            # Create a slightly expanded hull to allow some flexibility
            # Find the centroid of all points
            centroid = np.mean(real_pca, axis=0)
            
            # Expand points slightly outward from centroid (by 5%)
            expanded_points = centroid + (real_pca - centroid) * 1.05
            
            # Calculate the convex hull of these expanded points
            self.real_data_hull = ConvexHull(expanded_points)
            print("Created convex hull boundary for real data in PCA space")
        
        # Identify the most extreme generated points
        outlier_threshold = np.percentile(gen_distances, outlier_percentile)
        outlier_mask = gen_distances >= outlier_threshold
        outlier_points = gen_pca[outlier_mask]
        
        # If no outliers are identified, return zero penalty
        if len(outlier_points) == 0:
            return torch.tensor(0.0, device=device)
        
        # For each outlier point, calculate distance to the hull boundary
        from scipy.spatial import distance_matrix
        
        # Get the hull vertices in PCA space
        hull_vertices = self.real_data_hull.points[self.real_data_hull.vertices]
        
        # Calculate distances from each outlier to each hull vertex
        dist_matrix = distance_matrix(outlier_points, hull_vertices)
        
        # For each outlier, find the minimum distance to any hull vertex
        min_distances = np.min(dist_matrix, axis=1)
        
        # Convert to tensor and compute mean squared penalty
        distances_tensor = torch.tensor(min_distances, dtype=torch.float32, device=device)
        penalty = torch.mean(distances_tensor ** 2)
        
        # Add diagnostic information
        num_outliers = len(outlier_points)
        avg_distance = np.mean(min_distances)
        max_distance = np.max(min_distances)
        print(f"Outlier penalty: {num_outliers}/{batch_size} points (>{outlier_percentile}%). "
            f"Avg dist: {avg_distance:.4f}, Max dist: {max_distance:.4f}, Penalty: {penalty.item():.4f}")
        
        return penalty

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
                    structure_penalty = self.compute_vol_structure_penalty(gen_returns, cond)
                    outlier_penalty = self.compute_outlier_boundary_penalty(gen_returns)
                    
                    # Add the new nearest neighbor penalty
                    #nn_penalty = self.compute_nearest_neighbor_penalty(gen_returns)
                    
                    # Add all penalties to generator loss
                    g_loss = -torch.mean(self.discriminator(gen_returns, cond)) + \
                            self.lambda_tail * tail_penalty + \
                            self.lambda_structure * structure_penalty 
                            #self.lambda_nn * nn_penalty + \
                            
                            #self.lambda_outlier * outlier_penalty  
                            #self.lambda_structure * structure_penalty + \
                    
                    g_loss.backward()
                    self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{self.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                          f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                          f"[Tail penalty: {tail_penalty.item():.4f}] [Outlier penalty: {structure_penalty.item():.4f}]")

    def generate_scenarios(self, save=True, num_scenarios=50000):
        self.generator.eval()
        device = 'cuda' if self.cuda else 'cpu'
        
        all_generated_returns = []
        batch_size = 1000

        with torch.no_grad():
            for _ in range(num_scenarios // batch_size):
                z = torch.randn(batch_size, self.latent_dim, device=device)
                indices = np.random.choice(len(self.conditions), size=batch_size, replace=True)
                cond = torch.tensor(self.conditions[indices], dtype=torch.float32, device=device)
                gen_returns = self.generator(z, cond)
                gen_returns = self.scaler.inverse_transform(gen_returns.cpu().numpy())
                all_generated_returns.append(gen_returns)
                
        all_generated_returns = np.vstack(all_generated_returns)
        
        if save:
            save_dir = "generated_CGAN_output_test"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'generated_returns_{self.asset_name}_final_scenarios.pt')
            torch.save(torch.tensor(all_generated_returns), save_path)
            print(f"Generated scenarios saved to: {save_path}")
        
        return all_generated_returns


# Conditional Generator: accepts noise and condition input.
class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, output_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.output_shape = output_shape  # e.g., (252,)
        input_dim = latent_dim + cond_dim  # Concatenate noise and condition.

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(output_shape)))
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        out = self.model(x)
        return out.view(out.size(0), *self.output_shape)


# Conditional Discriminator: accepts a return sequence and the condition.
class Discriminator(nn.Module):
    def __init__(self, input_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape  # e.g., (252,)
        self.cond_dim = cond_dim
        input_dim = int(np.prod(input_shape)) + cond_dim  # Concatenate flattened returns and condition.
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1000, 1)
        )

    def forward(self, returns, condition):
        x = returns.view(returns.size(0), -1)
        x = torch.cat((x, condition), dim=1)
        validity = self.model(x)
        return validity