import torch
import matplotlib.pyplot as plt

import sys
import os
# Adds the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tfno import TFNO2D, spherical_h1_loss
from src.dataset import make_loader
from src.runner import train_forecaster, evaluate_rollout
from src.viz import plot_weather
from src.utils import load_data, spectral_downsample_2d, spectral_upsample_2d

RESULTS_DIR = "results/tfno/"
import os
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load data
    series = load_data(train=True)
    test_series = load_data(train=False)
    series = torch.from_numpy(series).float().to(device)
    test_series = torch.from_numpy(test_series).float().to(device)
    B, T, C, H, W = series.shape
    
    Cout = 2 
    factor = 4
    t_hist, t_rollout = 1, 1
    dt = 1
    
    # 2. Downsample
    H_2, W_2 = H // factor, W // factor
    series = spectral_downsample_2d(
        series.view(-1, C, H, W), factor=factor
    ).view(B, T, C, H_2, W_2)
    test_series = spectral_downsample_2d(
        test_series.view(-1, C, H, W), factor=factor
    ).view(test_series.shape[0], T, C, H_2, W_2)
    
    start_times = [0.0] * series.shape[0]

    ds, train_loader = make_loader(
        series, t_hist, t_rollout, batch_size=16, 
        shuffle=True, dt=dt, start_times=start_times
    )
    ds_test, test_loader = make_loader(
        test_series, t_hist, t_rollout, batch_size=64, 
        shuffle=False
    )
    
    # 3. Create Model
    model = TFNO2D(
        factor=factor, in_channels=C, out_channels=Cout,
        in_time=t_hist, out_time=t_rollout,
        width=16,
        depth=4,
        modes_h=64,
        modes_w=32,
        factorization="tucker",
        cp_rank=16
    ).to(device)   
    print(f"TFNO model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Callback: Handles upsampling cleanly so training loop avoids spatial coupling constraints
    def train_plot_callback(y_true, y_pred, save_path):
        # View inputs cleanly: (t_rollout, C_out, H_2, W_2) -> (t_rollout, C_out, H, W)
        y0_up = spectral_upsample_2d(
            y_true[0].view(-1, Cout, H_2, W_2), factor=factor
        ).view(-1, Cout, H, W)
        yhat_up = spectral_upsample_2d(
            y_pred[0].view(-1, Cout, H_2, W_2), factor=factor
        ).view(-1, Cout, H, W)
        plot_weather(y0_up.numpy(), yhat_up.numpy())
        plt.savefig(save_path)
        plt.close()

    # 4. Train
    train_forecaster(
        RESULTS_DIR,
        model, 
        train_loader, 
        test_loader, 
        criterion=spherical_h1_loss,
        device=device, 
        epochs=1, lr=1e-3, 
        plot_callback=train_plot_callback
    )
    # Save model
    torch.save(model.state_dict(), "checkpoints/tfno.pth")
    
    # 5. Long Rollout Tracking (e.g. 240 steps)
    y_true_test_240, yhat_test_240 = evaluate_rollout(
        RESULTS_DIR,
        model, 
        series, test_series, 
        t_hist, total_steps=240, dt=dt, 
        start_times=start_times, 
        device=device
    )

    # Render a final example for the evaluated Deep Rollout specifically
    try:
        eval_len = yhat_test_240.shape[1]
        y0_up = spectral_upsample_2d(
            y_true_test_240[0].view(-1, Cout, H_2, W_2), factor=factor
        ).view(eval_len, Cout, H, W)
        yhat_up = spectral_upsample_2d(
            yhat_test_240[0].view(-1, Cout, H_2, W_2), factor=factor
        ).view(eval_len, Cout, H, W)
        plot_weather(y0_up.cpu().numpy(), yhat_up.cpu().numpy())
        plt.savefig(RESULTS_DIR + "final_test_rollout_reconstruction.png")
        plt.close()
    except Exception as e:
        print(f"Final upsample visualization ignored. Reason: {e}")