# tlgn.py
# Temporal Latent Gradient Network (TLGN) - minimal PyTorch implementation
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        _, h = self.gru(x)  # [num_layers*(1/2), B, H]
        z1 = h[-1]  # [B, H]
        return z1


class LatentGradient(nn.Module):
    """
    z_{t+1} = z_t + G(z_t, x_t, t)
    Here G is a small MLP; optional control head on x_t only.
    """
    def __init__(self, z_dim: int, x_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + x_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim),
        )
        self.ctrl = nn.Sequential(
            nn.Linear(x_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, z_dim),
        )

    def forward(self, z_t: torch.Tensor, x_t: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # z_t: [B, Z], x_t: [B, X], t_scalar: [B, 1]
        g = self.mlp(torch.cat([z_t, x_t, t_scalar], dim=-1))
        u = self.ctrl(x_t)
        return z_t + g + u


class Decoder(nn.Module):
    def __init__(self, z_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TLGN(nn.Module):
    """
    Minimal TLGN:
      1) Encode past window -> z1
      2) Roll latent with gradient block for h steps
      3) Decode each latent to x_hat
    """
    def __init__(self, x_dim: int, z_dim: int = 128, enc_hidden: int = 128, grad_hidden: int = 128,
                 dec_hidden: int = 128, enc_layers: int = 1):
        super().__init__()
        self.encoder = GRUEncoder(x_dim, enc_hidden, num_layers=enc_layers, bidirectional=False)
        self.proj_z = nn.Linear(self.encoder.out_dim, z_dim)
        self.grad = LatentGradient(z_dim=z_dim, x_dim=x_dim, hidden=grad_hidden)
        self.decoder = Decoder(z_dim=z_dim, out_dim=x_dim, hidden=dec_hidden)

    def forward(self, x_hist: torch.Tensor, x_future_cond: Optional[torch.Tensor] = None,
                horizon: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x_hist: [B, L, D] past observations
          x_future_cond: [B, H, D] optional teacher-forcing inputs for t-conditional part of G
          horizon: prediction length H
        Returns:
          x_hat: [B, H, D]
          z_traj: [B, H+1, Z] (including initial z1)
        """
        B, L, D = x_hist.shape
        z = self.proj_z(self.encoder(x_hist))  # [B, Z]
        z_traj = [z]
        preds = []

        # if future cond not provided, repeat last historic step
        if x_future_cond is None:
            last = x_hist[:, -1, :]  # [B, D]
            x_future_cond = last.unsqueeze(1).repeat(1, horizon, 1)

        for t in range(horizon):
            # scalar time feature: normalized step index
            t_feat = torch.full((B, 1), float(t + 1), device=x_hist.device) / float(horizon)
            z = self.grad(z, x_future_cond[:, t, :], t_feat)
            z_traj.append(z)
            preds.append(self.decoder(z))

        x_hat = torch.stack(preds, dim=1)  # [B, H, D]
        z_traj = torch.stack(z_traj, dim=1)  # [B, H+1, Z]
        return x_hat, z_traj

    @staticmethod
    def smoothness_reg(z_traj: torch.Tensor) -> torch.Tensor:
        # second-order smoothness on latent: ||z_{t+1} - 2 z_t + z_{t-1}||^2
        if z_traj.size(1) < 3:
            return torch.tensor(0.0, device=z_traj.device)
        z_tm1 = z_traj[:, :-2, :]
        z_t = z_traj[:, 1:-1, :]
        z_tp1 = z_traj[:, 2:, :]
        diff2 = z_tp1 - 2.0 * z_t + z_tm1
        return (diff2.pow(2).sum(dim=-1)).mean()
