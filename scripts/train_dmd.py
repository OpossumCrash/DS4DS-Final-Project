import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.dmd import HankelDMDModel, MyPCA
from src.utils import load_data
from src.viz import plot_weather

RESULTS_DIR = "results/dmd/"
import os 
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Load data
    train = load_data(train=True)
    test = load_data(train=False)
    B, T, C, H, W = train.shape
        
    Cout = 2 
    t_hist, t_rollout = 1, 1
    
    latent_dim = 128
    pca = MyPCA(n_components=latent_dim)
    pca = pca.load("checkpoints/pca.pkl")
    
    # Convert to latent space for DMD
    transformed_train = pca.transform(train.reshape(-1, C*H*W)).reshape(-1, T, latent_dim)
    transformed_test = pca.transform(test.reshape(-1, C*H*W)).reshape(-1, T, latent_dim)
    
    model = HankelDMDModel(latent_dim, delays=t_hist).fit(transformed_train)
    model.save("checkpoints/dmd.pkl")
    
    # Test model via mse over rollout
    n_steps = 240
    x0 = transformed_test[0:1, :t_hist]
    y_pred = model.predict(x0, n_steps)
    y_true = transformed_test[0:1, t_hist:t_hist+n_steps]
    
    # Transform back to original space for evaluation and visualization
    y_true = pca.inverse_transform(y_true.reshape(-1, latent_dim)).reshape(1, n_steps, C, H, W)
    y_pred = pca.inverse_transform(y_pred.reshape(-1, latent_dim)).reshape(1, n_steps, C, H, W)
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"DMD Test MSE over {n_steps}-step rollout: {mse:.6f}")
    
    # Visualize some rollouts
    plot_weather(y_true[0], y_pred[0], num_steps=4)
    plt.savefig(RESULTS_DIR + "dmd_rollout.png")
    
        
    
    