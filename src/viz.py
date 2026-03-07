import numpy as np
import matplotlib.pyplot as plt


def plot_weather(truth, prediction=None, num_steps=5):
    """
    Plots the ground truth, prediction, and their difference over evenly spaced timesteps.
    
    Args:
        truth: numpy array of shape (T, 2, 128, 256) representing ground truth.
        prediction: Optional numpy array of shape (T, 2, 128, 256) representing the model's output.
        num_steps: Integer specifying how many timesteps to visualize across the entire sequence.
    """
    T = truth.shape[0]
    
    # Select evenly spaced timesteps from 0 to T-1
    timesteps = np.linspace(0, T - 1, num_steps, dtype=int)
    
    # Determine the layout
    rows = 3 if prediction is not None else 1
    fig, axes = plt.subplots(rows, num_steps, figsize=(4 * num_steps, 3 * rows), constrained_layout=True)
    
    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_steps == 1:
        axes = np.expand_dims(axes, axis=1)

    # 2D Earth coordinate translation
    extent = [-180, 180, -90, 90]
    
    # Helper function to compute velocity magnitude
    def get_magnitude(data, t):
        return np.linalg.norm(data[t], axis=0)  # L2 norm across the velocity components (u, v)

    # Calculate global min/max for the velocity magnitude to keep color scales locked
    truth_mag_all = np.linalg.norm(truth, axis=1) 
    vmin, vmax = np.min(truth_mag_all), np.max(truth_mag_all)
    
    if prediction is not None:
        pred_mag_all = np.linalg.norm(prediction, axis=1)
        vmin = min(vmin, np.min(pred_mag_all))
        vmax = max(vmax, np.max(pred_mag_all))
        
        # Calculate global max for the difference (error) plot
        diff_vmax = np.max(np.abs(truth_mag_all - pred_mag_all))

    for i, t in enumerate(timesteps):
        # --- Row 1: Ground Truth ---
        mag_t = get_magnitude(truth, t)
        im_truth = axes[0, i].imshow(mag_t, origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Truth (t={t})")
        
        # --- Row 2 & 3: Prediction and Difference ---
        if prediction is not None:
            # Prediction
            mag_p = get_magnitude(prediction, t)
            im_pred = axes[1, i].imshow(mag_p, origin='lower', extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f"Prediction (t={t})")
            
            # Difference (Absolute Error)
            diff = np.linalg.norm(truth[t] - prediction[t], axis=0) # L2 norm across the velocity components
            # Using 'inferno' or 'magma' helps distinguish the error map from the weather map
            im_diff = axes[2, i].imshow(diff, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=diff_vmax)
            axes[2, i].set_title(f"Absolute Error")
            
    # Format axes (only show coordinates on the outer edges to keep it clean)
    for r in range(rows):
        for c in range(num_steps):
            if r == rows - 1:
                axes[r, c].set_xlabel("Longitude")
            else:
                axes[r, c].set_xticks([])
                
            if c == 0:
                axes[r, c].set_ylabel("Latitude")
            else:
                axes[r, c].set_yticks([])

    # Add colorbars to the right side of each row
    fig.colorbar(im_truth, ax=axes[0, :], aspect=20, label="Velocity Mag")
    if prediction is not None:
        fig.colorbar(im_pred, ax=axes[1, :], aspect=20, label="Velocity Mag")
        fig.colorbar(im_diff, ax=axes[2, :], aspect=20, label="Absolute Error")

    plt.suptitle("Weather Simulation: Ground Truth vs. Prediction", fontsize=16)
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def velocity_to_rgb(u, v, vmax):
    """
    Converts u and v velocity components into an RGB image.
    Angle is mapped to Hue, Magnitude is mapped to Value.
    """
    # Calculate magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    # Angle in radians (-pi to pi), then shift to [0, 1] for HSV hue
    angle = np.arctan2(v, u)
    angle_norm = (angle + np.pi) / (2 * np.pi)
    
    # Normalize magnitude for Value channel (0 to 1)
    mag_norm = np.clip(mag / vmax, 0, 1)
    
    # Create HSV image: 
    # H = Angle, S = 1.0 (constant), V = Normalized Magnitude
    hsv = np.zeros((u.shape[0], u.shape[1], 3))
    hsv[..., 0] = angle_norm
    hsv[..., 1] = 0.9 # High saturation
    hsv[..., 2] = mag_norm
    
    return hsv_to_rgb(hsv)

def plot_weather_colored(truth, prediction=None, num_steps=5):
    """
    Plots weather data where color hue represents direction and brightness represents magnitude.
    """
    T = truth.shape[0]
    timesteps = np.linspace(0, T - 1, num_steps, dtype=int)
    
    rows = 3 if prediction is not None else 1
    fig, axes = plt.subplots(rows, num_steps, figsize=(4 * num_steps, 3 * rows), constrained_layout=True)
    
    if rows == 1: axes = np.expand_dims(axes, axis=0)
    if num_steps == 1: axes = np.expand_dims(axes, axis=1)

    extent = [-180, 180, -90, 90]
    
    # Calculate global vmax for consistent brightness across all plots
    truth_mag_all = np.sqrt(truth[:, 0]**2 + truth[:, 1]**2)
    vmax = np.max(truth_mag_all)
    if prediction is not None:
        pred_mag_all = np.sqrt(prediction[:, 0]**2 + prediction[:, 1]**2)
        vmax = max(vmax, np.max(pred_mag_all))
        diff_vmax = np.max(np.linalg.norm(truth - prediction, axis=1))

    for i, t in enumerate(timesteps):
        # --- Row 1: Ground Truth ---
        rgb_truth = velocity_to_rgb(truth[t, 0], truth[t, 1], vmax)
        axes[0, i].imshow(rgb_truth, origin='lower', extent=extent)
        axes[0, i].set_title(f"Truth (t={t})")
        
        if prediction is not None:
            # --- Row 2: Prediction ---
            rgb_pred = velocity_to_rgb(prediction[t, 0], prediction[t, 1], vmax)
            axes[1, i].imshow(rgb_pred, origin='lower', extent=extent)
            axes[1, i].set_title(f"Prediction (t={t})")
            
            # --- Row 3: Error (Magnitude of Difference) ---
            # Error is still best visualized with a standard heatmap (magnitude)
            diff = np.linalg.norm(truth[t] - prediction[t], axis=0)
            im_diff = axes[2, i].imshow(diff, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=diff_vmax)
            axes[2, i].set_title(f"Vector Error Mag")

    # Formatting
    for r in range(rows):
        for c in range(num_steps):
            if r == rows - 1: axes[r, c].set_xlabel("Longitude")
            else: axes[r, c].set_xticks([])
            if c == 0: axes[r, c].set_ylabel("Latitude")
            else: axes[r, c].set_yticks([])

    # Add a colorbar only for the Error map (since RGB plots don't use standard colorbars)
    if prediction is not None:
        fig.colorbar(im_diff, ax=axes[2, :], aspect=20, label="Error Magnitude")

    plt.suptitle("Weather Simulation: Directional Color Coding", fontsize=16)