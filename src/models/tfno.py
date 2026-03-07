import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import torch_harmonics as th

def _rfft2(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W) real
    return torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

def _irfft2(xf: torch.Tensor, s: Tuple[int, int]) -> torch.Tensor:
    # xf: (B, C, H, Wf) complex
    return torch.fft.irfft2(xf, s=s, dim=(-2, -1), norm="ortho")

# ---------------------------
# Tensorized weight parameterizations (Using Complex weights)
# ---------------------------
@dataclass
class TuckerRanks:
    rH: int
    rW: int
    rI: int
    rO: int
    rL: int


class JointTuckerWeight(nn.Module):
    """
    Joint Tucker factorization of the full operator weight tensor W:
    Using complex parameters to allow phase shifts in the frequency domain.
    """
    def __init__(
        self,
        L: int,
        cin: int,
        cout: int,
        modes_h: int,
        modes_w: int,
        ranks: TuckerRanks,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.L = L
        dtype = torch.cfloat # CRITICAL: Spectral weights must be complex

        # Core
        self.G = nn.Parameter(init_scale * torch.randn(ranks.rL, ranks.rO, ranks.rI, ranks.rH, ranks.rW, dtype=dtype))

        # Factors: Initialized to N(0,1). The init_scale on G avoids variance collapse across the product.
        self.U_L = nn.Parameter(torch.randn(L, ranks.rL, dtype=dtype))
        self.U_O = nn.Parameter(torch.randn(cout, ranks.rO, dtype=dtype))
        self.U_I = nn.Parameter(torch.randn(cin, ranks.rI, dtype=dtype))
        # allocate 2 * modes_h to cover independent weights for both positive and negative frequencies
        self.U_H = nn.Parameter(torch.randn(modes_h, ranks.rH, dtype=dtype))
        self.U_W = nn.Parameter(torch.randn(modes_w, ranks.rW, dtype=dtype))

    def weight_for_layer(self, l: int) -> torch.Tensor:
        uL = self.U_L[l]
        T = torch.einsum("l,loipq->oipq", uL, self.G)
        A = torch.einsum("oipq,hp->oihq", T, self.U_H)
        B = torch.einsum("oihq,wq->oihw", A, self.U_W)
        C = torch.einsum("ao,oihw->aihw", self.U_O, B)
        W = torch.einsum("bi,aihw->abhw", self.U_I, C)
        return W


class JointCPWeight(nn.Module):
    """
    Joint CP factorization of W.
    Using complex parameters to allow phase shifts in the frequency domain.
    """
    def __init__(
        self,
        L: int,
        cin: int,
        cout: int,
        modes_h: int,
        modes_w: int,
        rank: int,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.L = L
        dtype = torch.cfloat # CRITICAL: Spectral weights must be complex

        self.lam = nn.Parameter(init_scale * torch.randn(rank, dtype=dtype))
        # Factors: Initialized to N(0,1)
        self.U_L = nn.Parameter(torch.randn(L, rank, dtype=dtype))
        self.U_O = nn.Parameter(torch.randn(cout, rank, dtype=dtype))
        self.U_I = nn.Parameter(torch.randn(cin, rank, dtype=dtype))
        self.U_H = nn.Parameter(torch.randn(modes_h, rank, dtype=dtype))
        self.U_W = nn.Parameter(torch.randn(modes_w, rank, dtype=dtype))

    def weight_for_layer(self, l: int) -> torch.Tensor:
        coeff = self.lam * self.U_L[l]
        W = torch.einsum("r,or,ir,hr,wr->oihw", coeff, self.U_O, self.U_I, self.U_H, self.U_W)
        return W

# ---------------------------
# Spectral convolution
# ---------------------------

class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, modes_h: int, modes_w: int, weight_provider: nn.Module, layer_index: int):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.Wprov = weight_provider
        self.l = layer_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        xf = _rfft2(x)
        Wf = xf.shape[-1]

        out_f = torch.zeros(B, self.cout, H, Wf, device=x.device, dtype=xf.dtype)
        w = self.Wprov.weight_for_layer(self.l)

        mh = min(self.modes_h, H)
        mw = min(self.modes_w, Wf)

        # Top frequencies (Positive H modes)
        out_f[:, :, :mh, :mw] = torch.einsum("bihw,oihw->bohw", xf[:, :, :mh, :mw], w[:, :, :mh, :mw])
        
        # Cast the m=0 mode to real to ensure the output is strictly real after iFFT
        if mh > 0 and mw > 0:
            out_f[:, :, 0, :mw] = out_f[:, :, 0, :mw].real
            
        return _irfft2(out_f, s=(H, W))
    
import torch
import torch.nn as nn
import torch_harmonics as th

class SphericalSpectralConv2d(nn.Module):
    def __init__(self, factor: int, cin: int, cout: int, modes_h: int, modes_w: int, weight_provider: nn.Module, layer_index: int):
        super().__init__()
        self.cin = cin
        self.cout = cout
        # In SHT, modes_h and modes_w correspond to l_max and m_max
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.Wprov = weight_provider
        self.l = layer_index

        # SHT setup for 128x256 grid
        # nlat/nlon define the physical grid size
        self.sht = th.RealSHT(nlat=128//factor, nlon=256//factor, norm="ortho")
        self.isht = th.InverseRealSHT(nlat=128//factor, nlon=256//factor, norm="ortho")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, Cin, H, W) -> (B, Cin, 128, 256)
        B, Cin, H, W = x.shape
        
        # 1. Transform to Spherical Harmonic coefficients
        # Output shape: (B, Cin, L, M) where L=128, M=129 (due to real transform)
        xf = self.sht(x)
        L, M = xf.shape[-2], xf.shape[-1]

        # 2. Prepare output tensor in spectral domain
        out_f = torch.zeros(B, self.cout, L, M, device=x.device, dtype=xf.dtype)
        
        # 3. Get weights from your provider
        # Note: Ensure your WeightProvider returns a tensor compatible with (Cin, Cout, L, M)
        w = self.Wprov.weight_for_layer(self.l)

        # 4. Frequency Filtering (Handling "High Frequencies")
        # In SHT, we filter by truncating the degree (l) and order (m).
        # We take the lowest modes (low-pass) but your request to "keep high frequencies" 
        # implies you want to apply the weights across the specified mode range.
        mh = min(self.modes_h, L)
        mw = min(self.modes_w, M)

        # Apply weights to the selected frequency bands
        out_f[:, :, :mh, :mw] = torch.einsum(
            "bihw,oihw->bohw", 
            xf[:, :, :mh, :mw], 
            w[:, :, :mh, :mw]
        )

        # 5. Inverse Transform back to Physical Space (Earth Grid)
        # This automatically handles the "same point" logic at the poles.
        return self.isht(out_f)

# ---------------------------
# Improved FNO backbone block (per Paper Figure 3d)
# ---------------------------

class ChannelLayerNorm(nn.Module):
    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class SpatialMLP(nn.Module):
    def __init__(self, c: int, expansion: float = 0.5, act: nn.Module = nn.GELU()):
        super().__init__()
        hidden = max(1, int(c * expansion))
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TFNOBlock2D(nn.Module):
    """ TFNO Block following the improved backbone in the paper. """
    def __init__(self, factor, width: int, modes_h: int, modes_w: int, weight_provider: nn.Module, layer_index: int, act: nn.Module = nn.GELU()):
        super().__init__()
        self.norm1 = ChannelLayerNorm(width)
        self.spectral = SphericalSpectralConv2d(factor, width, width, modes_h, modes_w, weight_provider, layer_index)
        self.skip = nn.Conv2d(width, width, kernel_size=1)
        self.act = act

        self.norm2 = ChannelLayerNorm(width)
        self.mlp = SpatialMLP(width, expansion=0.5, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.spectral(y)
        y = y + self.skip(x)
        y = self.act(y)

        z = self.norm2(y)
        z = self.mlp(z)
        y = y + z
        return self.act(y)


# ---------------------------
# Main TFNO Model
# ---------------------------
class TFNO2D(nn.Module):
    """
    Tensorized Fourier Neural Operator with Spherical 3D Coordinates.
    """
    def __init__(
        self,
        factor: int,
        in_channels: int,
        out_channels: int,
        in_time: int,
        out_time: int,
        width: int,
        depth: int,
        modes_h: int,
        modes_w: int,
        factorization: Literal["tucker", "cp"] = "tucker",
        tucker_ranks: Optional[TuckerRanks] = None,
        cp_rank: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_time = in_time
        self.default_steps = out_time 

        # UPDATED: lift_in_channels now has +3 for (x, y, z) instead of +2
        lift_in_channels = in_time * (in_channels + 2) + 3
        self.lift = nn.Conv2d(lift_in_channels, width, kernel_size=1)

        if factorization == "tucker":
            ranks = tucker_ranks or TuckerRanks(
                rH=max(4, modes_h // 4), rW=max(4, modes_w // 4),
                rI=max(8, width // 4), rO=max(8, width // 4), rL=max(2, depth // 2),
            )
            self.Wprov = JointTuckerWeight(depth, width, width, modes_h, modes_w, ranks)
        else:
            self.Wprov = JointCPWeight(depth, width, width, modes_h, modes_w, cp_rank or max(16, width // 2))

        self.blocks = nn.ModuleList([
            TFNOBlock2D(factor, width, modes_h, modes_w, self.Wprov, l) for l in range(depth)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, out_channels, kernel_size=1),
        )

    def predict_step(self, x: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # 1. Improved Coordinate Generation
        # Longitude: 0 to 2pi (endpoint=False to avoid duplicating the prime meridian)
        phi_lin = torch.linspace(0, 2 * math.pi, W + 1, device=device, dtype=dtype)[:-1]
        # Latitude: -pi/2 to pi/2
        theta_lin = torch.linspace(-math.pi / 2, math.pi / 2, H, device=device, dtype=dtype)
        
        grid_theta, grid_phi = torch.meshgrid(theta_lin, phi_lin, indexing='ij')

        # 3D Cartesian Coordinates (Unit Sphere Embedding)
        # This is perfect for SHT/TFNO as it provides a smooth, non-singular coordinate basis
        pos_x = torch.cos(grid_theta) * torch.cos(grid_phi)
        pos_y = torch.cos(grid_theta) * torch.sin(grid_phi)
        pos_z = torch.sin(grid_theta)
        pos_3d = torch.stack([pos_x, pos_y, pos_z], dim=0).unsqueeze(0).expand(B, 3, H, W)

        # 2. Optimized Forcing Logic
        # Move forcing constants to a config or init to avoid re-calculating
        t_grid_expand = t_grid.view(B, T, 1, 1, 1).expand(B, T, 1, H, W)
        
        phi_c = (t_grid_expand / 1.0) * 2 * math.pi # time_of_day
        theta_c = torch.sin((t_grid_expand / 365.0) * 2 * math.pi) * 0.4 # time_of_year
        
        # Using broadcasting effectively
        phi_exp = grid_phi.view(1, 1, 1, H, W)
        theta_exp = grid_theta.view(1, 1, 1, H, W)

        # Solar forcing: note the cos(phi - phi_c) handles the daily rotation
        forcing = torch.cos(phi_exp - phi_c) * torch.exp(-(theta_exp - theta_c)**2 / (math.pi/2)**2)

        # 3. Concatenation and Lifting
        # REMOVED: Redundant grid_x/grid_y from previous versions
        # We use (x, t, forcing, pos_3d)
        x_enriched = torch.cat([x, t_grid_expand, forcing], dim=2)  # (B, T, C+1+1+3, H, W)
        x_flat = x_enriched.reshape(B, T * (C + 2), H, W)
        
        # Combined features for the FNO lifting layer
        out = torch.cat([x_flat, pos_3d], dim=1) 
        
        out = self.lift(out)
        for blk in self.blocks:
            out = blk(out)
        out = self.project(out)
        
        return out.reshape(B, 1, self.out_channels, H, W)

    def forward(self, x: torch.Tensor, t_grid: Optional[torch.Tensor] = None, steps: Optional[int] = None, dt: float = 1.0) -> torch.Tensor:
        steps = steps if steps is not None else self.default_steps
        B, T, C, H, W = x.shape
        if t_grid is None:
            t_grid = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, T)
        curr_x, curr_t = x, t_grid
        preds = []
        for _ in range(steps):
            pred = self.predict_step(curr_x, curr_t)
            preds.append(pred)
            if steps > 1:
                curr_x = torch.cat([curr_x[:, 1:], pred], dim=1)
                curr_t = torch.cat([curr_t[:, 1:], curr_t[:, -1:] + dt], dim=1)
        return torch.cat(preds, dim=1)
    
def h1_loss(y_pred, y_true, alpha=0.1):
    """ Standard MSE + Sobolev penalty on gradients. """
    mse = F.mse_loss(y_pred, y_true)
    dy_pred, dx_pred = torch.gradient(y_pred, dim=(-2, -1))
    dy_true, dx_true = torch.gradient(y_true, dim=(-2, -1))
    grad_loss = F.mse_loss(dy_pred, dy_true) + F.mse_loss(dx_pred, dx_true)
    return mse + alpha * grad_loss

def spherical_h1_loss(y_pred, y_true, alpha=0.1):
    """ MSE + Sobolev penalty weighted by cos(lat) for spherical distortion. """
    B, T, C, H, W = y_pred.shape
    device = y_pred.device
    lat = torch.linspace(-math.pi / 2, math.pi / 2, H, device=device)
    weights = torch.cos(lat).view(1, 1, 1, H, 1).expand(B, T, C, H, W)
    mse = (weights * (y_pred - y_true)**2).sum() / weights.sum()
    dy_pred, dx_pred = torch.gradient(y_pred, dim=(-2, -1))
    dy_true, dx_true = torch.gradient(y_true, dim=(-2, -1))
    grad_loss = (weights * (dy_pred - dy_true)**2).mean() + (weights * (dx_pred - dx_true)**2).mean()
    return mse + alpha * grad_loss
