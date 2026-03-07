import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import pickle


class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=n_components)
              
    def fit(self, X, y=None, batch_size=1024):
        """Fits the PCA model in batches."""
        n_samples, n_features = X.shape
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Fitting PCA"):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            self.pca.partial_fit(X[start:end])
            
        return self
    
    def transform(self, X):
        """Projects data to the latent space."""
        return self.pca.transform(X)
    
    def inverse_transform(self, Z):
        """Reconstructs data from the latent space."""
        return self.pca.inverse_transform(Z)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod   
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

import numpy as np
import pickle

class HankelDMDModel:
    """
    Latent-space DMD model using Hankel-delay embedding for temporal dynamics.
    
    This model expects pre-computed latent representations (Z).
    Input shape (Fit/Predict): (B, T, latent_dim)
    Output shape (Predict): (B, steps, latent_dim)
    """

    def __init__(
            self, 
            latent_dim: int,
            delays: int, 
            ridge_lambda: float = 1e-4, 
            rho_max: float = 0.999,
        ):
        self.latent_dim = latent_dim
        self.delays = delays
        self.ridge_lambda = ridge_lambda
        self.rho_max = rho_max
        
        self.A = None      # Dynamics matrix (Operator)
        self.mu = None     # Hankel-space mean
        self.std = None    # Hankel-space std

    def _build_hankel_pairs(self, Z_batches):
        """
        Creates time-lagged pairs (X, Y) from latent trajectories.
        Z_batches: Array of shape (B, T, latent_dim)
        """
        X_list, Y_list = [], []
        d = self.delays
        B, T, L = Z_batches.shape

        for b in range(B):
            z_traj = Z_batches[b]
            if T <= d: continue
            
            # Slide window to create Hankel matrix columns
            for t in range(d - 1, T - 1):
                # xt: window of d steps (Hankel column at time t)
                # yt: window shifted by 1 (Hankel column at time t+1)
                xt = z_traj[t - d + 1 : t + 1].flatten()
                yt = z_traj[t - d + 2 : t + 2].flatten()
                X_list.append(xt)
                Y_list.append(yt)

        return np.array(X_list).T, np.array(Y_list).T

    def fit(self, Z):
        """
        Trains the DMD operator A on latent data Z.
        Z: Array of shape (B, T, latent_dim)
        """
        # 1. Build Hankel Pairs from latent trajectories
        X, Y = self._build_hankel_pairs(Z)
        X, Y = X.astype(np.float64), Y.astype(np.float64)

        # 2. Normalize for numerical stability in Ridge Regression
        self.mu = X.mean(axis=1, keepdims=True)
        self.std = X.std(axis=1, keepdims=True) + 1e-6
        
        Xn = (X - self.mu) / self.std
        Yn = (Y - self.mu) / self.std

        # 3. Solve Ridge Regression: A = Yn @ Xn^T @ (Xn @ Xn^T + lambda*I)^-1
        XXt = Xn @ Xn.T
        reg = self.ridge_lambda * np.eye(XXt.shape[0])
        self.A = (Yn @ Xn.T) @ np.linalg.inv(XXt + reg)

        # 4. Spectral radius stabilization (prevents exploding rollouts)
        eigvals = np.linalg.eigvals(self.A)
        rho = np.max(np.abs(eigvals))
        if rho > self.rho_max:
            self.A *= (self.rho_max / rho)
            
        self.A = self.A.astype(np.float32)
        return self

    def predict(self, Z, steps):
        """
        Standard rollout interface.
        Z: Latent context of shape (B, T_context, latent_dim)
        steps: How many future steps to predict.
        Returns: (B, steps, latent_dim)
        """
        B, T_context, L = Z.shape
        d = self.delays
        
        # Flattened normalization constants for broadcasting
        mu_f = self.mu.flatten()
        std_f = self.std.flatten()
        
        all_preds = []

        for b in range(B):
            # 1. Initialize Hankel state from the end of the input sequence
            z_current = Z[b, -d:].flatten()
            
            b_preds = []
            for _ in range(steps):
                # 2. Linear step in Hankel space
                z_next = self.A @ ((z_current - mu_f) / std_f)
                z_next = z_next * std_f + mu_f
                
                # 3. Extract the predicted latent vector (the tail of the new Hankel state)
                new_latent = z_next[-L:]
                b_preds.append(new_latent)
                
                # 4. Update Hankel state for autoregression
                z_current = z_next
                
            all_preds.append(b_preds)

        return np.array(all_preds).astype(np.float32)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)