import os
import numpy as np
import torch

def load_data(root_dir: str="./data/", train: bool=True):
    """
    Load and concatenate .npy files from the specified directory.
    
    Parameters:
    -----------
    root_dir (str): The root directory containing the 'train' and 'test'
        subdirectories. Default is "./data/". train (bool): If True, load data
    from the 'train' subdirectory; otherwise, load from the 'test'
        subdirectory. Default is True.
    
    Returns:
    --------
    all_data (numpy.ndarray): A concatenated array of all loaded data.
    
    Example usage:
    train_data = load_data(train=True)
    test_data = load_data(train=False)
    
    """
    # Gather all file paths
    if train:
        path = os.path.join(root_dir, "train")
    else:
        path = os.path.join(root_dir, "test")
    file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")]
    print(f"Found {len(file_paths)} files in {'train' if train else 'test'} directory.")
    
    # Load and concatenate data
    data_list = []
    for file_path in file_paths:
        data = np.load(file_path)
        data_list.append(data)
    all_data = np.stack(data_list, axis=0)
    print(f"Loaded data shape: {all_data.shape}")
    return all_data


def spectral_downsample_2d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Periodic in both axes. x: (B,C,H,W) -> (B,C,H/f,W/f)
    """
    B, C, H, W = x.shape
    H2, W2 = H // factor, W // factor
    X = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")        # (B,C,H,Wf)
    Wf2 = W2 // 2 + 1

    X2 = torch.zeros(B, C, H2, Wf2, device=x.device, dtype=X.dtype)

    h_keep = min(H2 // 2, H // 2)
    X2[:, :, :h_keep, :Wf2] = X[:, :, :h_keep, :Wf2]
    if h_keep > 0:
        X2[:, :, -h_keep:, :Wf2] = X[:, :, -h_keep:, :Wf2]

    y = torch.fft.irfft2(X2, s=(H2, W2), dim=(-2, -1), norm="ortho")
    return y


def spectral_upsample_2d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Periodic in both axes. x: (B,C,H2,W2) -> (B,C,H2*factor,W2*factor)
    """
    B, C, H2, W2 = x.shape
    H, W = H2 * factor, W2 * factor
    X2 = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")       # (B,C,H2,W2f)
    W2f = X2.shape[-1]
    Wf = W // 2 + 1

    X = torch.zeros(B, C, H, Wf, device=x.device, dtype=X2.dtype)

    h_keep = min(H2 // 2, H // 2)
    w_keep = min(W2f, Wf)

    X[:, :, :h_keep, :w_keep] = X2[:, :, :h_keep, :w_keep]
    if h_keep > 0:
        X[:, :, -h_keep:, :w_keep] = X2[:, :, -h_keep:, :w_keep]

    y = torch.fft.irfft2(X, s=(H, W), dim=(-2, -1), norm="ortho")
    return y

