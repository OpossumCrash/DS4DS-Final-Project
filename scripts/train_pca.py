import matplotlib.pyplot as plt
import numpy as np

import sys
import os

# Adds the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dmd import MyPCA
from src.utils import load_data
from src.viz import plot_weather

RESULTS_DIR = "results/pca/"
import os 
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Load data
    series = load_data(train=True)
    test_series = load_data(train=False)
    B, T, C, H, W = series.shape
        
    Cout = 2 
    t_hist, t_rollout = 48, 48
    
    latent_dim = 128
    pca = MyPCA(n_components=latent_dim)
    pca.fit(series.reshape(B * T, C * H * W), batch_size=1024)
    pca.save("checkpoints/pca.pkl")
    
    # Plot the 8 most important PCA modes as spatial patterns
    plt.figure(figsize=(12, 6))
    for i in range(16):
        mode = pca.pca.components_[i].reshape(C, H, W)
        plt.subplot(4, 4, i + 1)
        mag = np.linalg.norm(mode, axis=0)
        plt.imshow(mag, cmap='viridis')
        plt.title(f"Mode {i+1}")
        plt.axis('off')
    plt.suptitle("Top 8 PCA Modes (Spatial Patterns)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR + "pca_modes.png")
    
    orig = series[0, :t_hist].reshape(t_hist, C, H, W)
    recon = pca.inverse_transform(pca.transform(orig.reshape(t_hist, -1))).reshape(t_hist, C, H, W)
    plot_weather(orig, recon, num_steps=4)
    plt.savefig(RESULTS_DIR + "pca_reconstruction.png")
    