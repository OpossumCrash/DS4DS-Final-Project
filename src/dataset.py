import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List

class WindowDataset(Dataset):
    def __init__(self, series: torch.Tensor, t_hist: int, t_rollout: int, stride: int = 1, dt: float = 1.0, start_times: Optional[List[float]] = None):
        """
        series: (B, T, C, H, W)
        dt: Timestep length (e.g., 1.0 for days, 1/24 for hours)
        start_times: Absolute starting timestamp for each sequence in the batch B. 
                     If None, defaults to 0.0 for all sequences.
        """
        self.series = series
        self.t_hist = t_hist
        self.t_rollout = t_rollout
        self.dt = dt
        
        B, T = series.shape[0:2]
        self.start_times = start_times if start_times is not None else [0.0] * B

        max_start = T - (t_hist + t_rollout)
        starts = list(range(0, max_start + 1, stride))

        self._index = [(b, s) for b in range(B) for s in starts]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        b, s = self._index[idx]
        x = self.series[b, s : s + self.t_hist]
        y = self.series[b, s + self.t_hist : s + self.t_hist + self.t_rollout]
        
        # Construct the exact time grid for this sequence's window
        t0 = self.start_times[b] + s * self.dt
        t_grid = torch.linspace(t0, t0 + (self.t_hist - 1) * self.dt, self.t_hist, dtype=torch.float32)

        return x, y, t_grid
    
def make_loader(series, t_hist, t_rollout, batch_size=8, stride=1, dt=1.0, start_times=None, shuffle=True):
    ds = WindowDataset(series, t_hist, t_rollout, stride, dt=dt, start_times=start_times)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)