import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
import os
# Adds the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import make_loader
from src.runner import train_forecaster, evaluate_rollout
from src.viz import plot_weather
from src.utils import load_data
from src.models.fno import LatentFNO
from src.models.dmd import MyPCA

RESULTS_DIR = "results/fno/"
import os 
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data
    train = load_data(train=True)
    test = load_data(train=False)
    
    # 2. Downsample via PCA
    B, T, C, H, W = train.shape
    latent_dim = 128
    pca = MyPCA(n_components=latent_dim)
    pca = pca.load("checkpoints/pca.pkl")
    train = pca.transform(train.reshape(-1, C * H * W)).reshape(-1, T, latent_dim)
    test = pca.transform(test.reshape(-1, C * H * W)).reshape(-1, T, latent_dim)
    
    # Convert to torch tensors and move to device
    train = torch.from_numpy(train).float().to(device)
    test = torch.from_numpy(test).float().to(device)
    
    # Parameters
    Cout = 2 
    t_hist, t_rollout = 1, 1
    start_times = [0.0] * train.shape[0]
    
    # 3. Create Loaders
    ds, train_loader = make_loader(
        train, t_hist, t_rollout, batch_size=32, 
        shuffle=True, dt=1.0, start_times=start_times
    )
    ds_test, test_loader = make_loader(
        test, t_hist, t_rollout, batch_size=64, 
        shuffle=False
    )
    
    # 3. Create Model
    model = LatentFNO(
        latent_dim=latent_dim, t_hist=t_hist, width=64, modes=32, n_layers=4
    ).to(device)
    print(f"TFNO model parameters: {sum(p.numel() for p in model.parameters())}")
    

    # Callback: Handles upsampling cleanly so training loop avoids spatial coupling constraints
    def train_plot_callback(y_true, y_pred, save_path):
        B, T, R = y_true.shape
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        y_true = pca.inverse_transform(y_true.reshape(B * T, R)).reshape(B, T, C, H, W)
        y_pred = pca.inverse_transform(y_pred.reshape(B * T, R)).reshape(B, T, C, H, W)
        plot_weather(y_true[0], y_pred[0], num_steps=5)
        plt.savefig(save_path)
        plt.close()

    # 4. Train
    train_forecaster(
        RESULTS_DIR,
        model, 
        train_loader, 
        test_loader, 
        criterion=F.mse_loss,
        device=device, 
        epochs=10, lr=1e-3, 
        plot_callback=train_plot_callback
    )
    
    # Save model
    torch.save(model.state_dict(), "checkpoints/fno.pth")
    
    # 5. Long Rollout Tracking (e.g. 240 steps)
    y_true_test_240, yhat_test_240 = evaluate_rollout(
        RESULTS_DIR,
        model, 
        train, test, 
        t_hist, total_steps=240, dt=1.0, 
        start_times=start_times, 
        device=device
    )

    # Render a final example for the evaluated Deep Rollout specifically
    try:
        eval_len = yhat_test_240.shape[1]
        y_true_test_240 = y_true_test_240.cpu().numpy()
        yhat_test_240 = yhat_test_240.cpu().numpy()
        y_true_test_240 = pca.inverse_transform(y_true_test_240.reshape(-1, latent_dim)).reshape(1, eval_len, C, H, W)
        yhat_test_240 = pca.inverse_transform(yhat_test_240.reshape(-1, latent_dim)).reshape(1, eval_len, C, H, W)
        plot_weather(y_true_test_240.cpu().numpy(), yhat_test_240.cpu().numpy(), num_steps=5)
        plt.savefig(RESULTS_DIR + "final_test_rollout_reconstruction.png")
        plt.close()
    except Exception as e:
        print(f"Final upsample visualization ignored. Reason: {e}")
        
    
    