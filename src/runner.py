from typing import List, Optional, Callable
import io
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from src.dataset import WindowDataset
from src.models.tfno import h1_loss


def train_forecaster(
        save_dir: str,
        model: nn.Module, 
        train_loader, 
        test_loader, 
        criterion: Callable, 
        device: torch.device, 
        epochs: int = 10, lr: float = 1e-3, 
        plot_callback=None
    ):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = criterion
    
    # Extract dt dynamically from dataset
    dt = train_loader.dataset.dt
    saved_frames = []
    train_losses_history = []
    test_losses_history = []
    
    for epoch in range(1, epochs + 1):
        # --- Training Pass ---
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x, y, t_grid in pbar:
            x, y, t_grid = x.to(device), y.to(device), t_grid.to(device)
            optimizer.zero_grad()

            yhat = model(x, t_grid, steps=y.shape[1])  # (B, t_rollout, C_out, H, W)
            loss = criterion(yhat, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            running_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{running_loss / (pbar.n + 1):.6f}"})
        
        train_loss = running_loss / len(train_loader)

        # --- Sound Evaluation Pass on Test Set ---
        model.eval()
        test_loss = 0.0
        last_y, last_yhat = None, None
        
        with torch.no_grad():
            for x, y, t_grid in test_loader:
                x, y, t_grid = x.to(device), y.to(device), t_grid.to(device)
                yhat = model(x, t_grid, steps=y.shape[1], dt=dt)
                loss = criterion(yhat, y) / len(y)  # Average per batch for accurate epoch-level loss
                test_loss += loss.item()
                last_y, last_yhat = y, yhat
                
        test_loss /= len(test_loader)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        train_losses_history.append(train_loss)
        test_losses_history.append(test_loss)
        
        # --- Visualization & Callbacks ---
        if plot_callback and last_y is not None and last_yhat is not None:
            # Create a bytes buffer instead of writing to disk directly
            buf = io.BytesIO()
            # Delegate upsampling & plot_weather back to __main__ callback for clean separation
            plot_callback(last_y.cpu(), last_yhat.cpu(), buf)
            buf.seek(0)
            
            # Load into PIL Image internally and keep in memory
            img = Image.open(buf)
            img.load()  # Ensure data is fully parsed so we don't hold a reference to the buffer
            saved_frames.append(img)
            
            # Overwrite predictions.png every epoch for live monitoring
            img.save(save_dir + "predictions.png")

    # Compile the GIF at the end of training from memory
    if saved_frames:
        try:
            saved_frames[0].save(
                save_dir + "training_progress.gif",
                format="GIF",
                append_images=saved_frames[1:],
                save_all=True,
                duration=400, # 400ms per frame
                loop=0
            )
            print("Successfully compiled training_progress.gif")
        except Exception as e:
            print(f"Warning: Could not compile GIF: {e}")
            
    # Save DataFrame
    df_losses = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": train_losses_history,
        "test_loss": test_losses_history
    })
    df_losses.to_csv(save_dir + "loss_history.csv", index=False)
    print("Saved epoch loss history to loss_history.csv")
    
    # Plot Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_history, label="Train Loss", linewidth=2)
    plt.plot(test_losses_history, label="Test Loss", linewidth=2, linestyle='--')
    plt.title("Training and Test Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir + "loss_curves.png")
    plt.close()
    print("Saved loss curves to loss_curves.png")

def evaluate_rollout(save_dir, model, series_train, series_test, t_hist, total_steps, dt, start_times, device):
    """ Evaluates Train and Test series for a deep rollout and plots MSE comparison. """
    model.eval()
    with torch.no_grad():
        # --- 1. Evaluate on Train Set Sample ---
        x_train = series_train[0:1, :t_hist].to(device)
        t_grid_train = torch.linspace(start_times[0], start_times[0] + (t_hist - 1) * dt, t_hist, dtype=torch.float32).unsqueeze(0).to(device)
        
        yhat_train = model(x_train, t_grid_train, steps=total_steps, dt=dt)
        y_true_train = series_train[0:1, t_hist : t_hist + total_steps].to(device)
        
        # Safeguard sequence length matching in case sequence isn't fully total_steps long
        eval_len_train = min(yhat_train.shape[1], y_true_train.shape[1])
        mse_train = ((yhat_train[:, :eval_len_train] - y_true_train[:, :eval_len_train]) ** 2).mean(dim=0)
        dims_to_reduce = list(range(2, len(mse_train.shape)))  # Reduce over all but batch and time
        mse_train = mse_train.mean(dim=dims_to_reduce).cpu().numpy()
        
        # --- 2. Evaluate on Test Set Sample ---
        x_test = series_test[0:1, :t_hist].to(device)
        t_grid_test = torch.linspace(start_times[0], start_times[0] + (t_hist - 1) * dt, t_hist, dtype=torch.float32).unsqueeze(0).to(device)
        
        yhat_test = model(x_test, t_grid_test, steps=total_steps, dt=dt)
        y_true_test = series_test[0:1, t_hist : t_hist + total_steps].to(device)
        
        eval_len_test = min(yhat_test.shape[1], y_true_test.shape[1])
        mse_test = ((yhat_test[:, :eval_len_test] - y_true_test[:, :eval_len_test]) ** 2).mean(dim=0)
        mse_test = mse_test.mean(dim=dims_to_reduce).cpu().numpy()

    # --- Plot the MSE Comparison Line Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(mse_train, label="Train MSE", linewidth=2)
    plt.plot(mse_test, label="Test MSE", linewidth=2, linestyle='--')
    plt.title(f"MSE per Timestep ({total_steps} step rollout)")
    plt.xlabel("Rollout Timestep")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir + "long_rollout_mse_comparison.png")
    plt.close()
    
    return y_true_test, yhat_test