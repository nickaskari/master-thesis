import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_plot(real_returns, generated_returns, title="PCA Plot"):
    """
    Compute PCA on the real_returns (as the base) and then transform generated_returns
    using the same PCA model. If the inputs are 1D, they are reshaped to 2D.
    
    Parameters:
      real_returns: array-like of shape (n_samples,) or (n_samples, n_features)
      generated_returns: array-like of shape (n_samples,) or (n_samples, n_features)
      title: Title for the plot.
      
    Returns:
      explained_variance: tuple (PC1, PC2) if two components exist, 
                          or (PC1, None) if only one component is available.
    """
    real_returns = np.asarray(real_returns)
    generated_returns = np.asarray(generated_returns)
    if real_returns.ndim == 1:
        real_returns = real_returns.reshape(-1, 1)
    if generated_returns.ndim == 1:
        generated_returns = generated_returns.reshape(-1, 1)
    
    n_features = real_returns.shape[1]
    n_components = min(2, n_features)
    pca = PCA(n_components=n_components)
    real_pca = pca.fit_transform(real_returns)
    gen_pca = pca.transform(generated_returns)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1] if n_components>1 else np.zeros_like(real_pca[:,0]), 
                color="blue", label="Empirical BOF", alpha=0.5)
    plt.scatter(gen_pca[:, 0], gen_pca[:, 1] if n_components>1 else np.zeros_like(gen_pca[:,0]),
                color="red", label="Generated BOF", alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2" if n_components>1 else "0 (only 1 feature)")
    plt.title(title)
    plt.legend()
    plt.show()
    
    explained_variance = pca.explained_variance_ratio_
    if n_components == 1:
        return (explained_variance[0], None)
    return explained_variance