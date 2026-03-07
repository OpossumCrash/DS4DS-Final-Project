import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def compl_mul1d(self, x_ft, w):
        return torch.einsum("bim,iom->bom", x_ft, w)

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            x.shape[0],
            self.weights.shape[1],
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = self.compl_mul1d(x_ft[:, :, :m], self.weights[:, :, :m])
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spec(x) + self.w(x))


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def compl_mul1d(self, x_ft, w):
        return torch.einsum("bim,iom->bom", x_ft, w)

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            x.shape[0],
            self.weights.shape[1],
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = self.compl_mul1d(x_ft[:, :, :m], self.weights[:, :, :m])
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)

class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spec(x) + self.w(x))

class LatentFNO(nn.Module):
    def __init__(self, latent_dim, t_hist, width=48, modes=16, n_layers=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_hist = t_hist
        
        # Input: t_hist context steps + 4 features (sin_day, cos_day, sin_week, space_coord)
        self.in_channels = t_hist + 4

        self.fc0 = nn.Linear(self.in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, 1)

    def forward(self, z, t_grid, steps, dt=1.0):
        """
        Pure Latent Autoregressive Rollout.
        z: Latent context of shape (B, T_in, latent_dim)
        t_grid: Time grid of shape (B, T_in)
        steps: Number of future steps to predict
        Returns: 
            full_pred_latent: (B, steps, latent_dim)
            latent_context: (B, t_hist, latent_dim) - the actual context used
        """
        B, T_total, R = z.shape
        device = z.device
        
        # 1. Prepare Context & Start Time
        # Only use the requested t_hist window
        latent_context = z[:, -self.t_hist:]
        t_start = t_grid[:, -1]
        
        # Static space grid for the latent dimensions
        x_coord = torch.linspace(0.0, 1.0, R, device=device).view(1, R).repeat(B, 1)
        
        predictions_latent = []
        current_latent_seq = latent_context.clone()

        # 2. Autoregressive Loop
        for i in range(steps):
            # Target time index for this step
            tt = (t_start + (i + 1) * dt).view(B, 1)
            
            # Feature engineering (Temporal & Spatial embedding)
            time_feats = torch.stack([
                torch.sin(2.0 * math.pi * tt / 24.0).repeat(1, R),
                torch.cos(2.0 * math.pi * tt / 24.0).repeat(1, R),
                torch.sin(2.0 * math.pi * tt / 168.0).repeat(1, R),
                x_coord,
            ], dim=-1) # (B, R, 4)

            # FNO block (spectral convolution along the latent dimension R)
            # h input shape: (B, R, t_hist + 4)
            h = torch.cat([current_latent_seq.transpose(1, 2), time_feats], dim=-1)
            h = self.fc0(h).transpose(1, 2)
            
            for blk in self.blocks:
                h = blk(h)
            
            h = h.transpose(1, 2)
            h = F.gelu(self.fc1(h))
            dx = self.fc2(h).squeeze(-1) # (B, R)

            # Residual update: z[t+1] = z[t] + delta
            z_next = (current_latent_seq[:, -1, :] + dx).unsqueeze(1)
            predictions_latent.append(z_next)

            # Slide window: pop first, append predicted
            current_latent_seq = torch.cat([current_latent_seq[:, 1:], z_next], dim=1)

        full_pred_latent = torch.cat(predictions_latent, dim=1)
        
        return full_pred_latent