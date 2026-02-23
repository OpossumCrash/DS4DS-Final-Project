import os
import numpy as np

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
    all_data = np.concatenate(data_list, axis=0)
    print(f"Loaded data shape: {all_data.shape}")
    return all_data