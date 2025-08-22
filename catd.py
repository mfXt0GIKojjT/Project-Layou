# catd.py
# Causal-Aware Temporal Denoising (CATD) regularizers/utilities
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class CausalKernel(nn.Module):
    """Learnable discrete delay kernel kappa over [0..tau_max], softmax-normalized."""
    def __init__(self, tau_max: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(tau_max + 1))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)  # [tau_max+1]


def apply_causal_kernel(u: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """
    Convolve along time with learned kernel (causal).
    u: [B, H, M]  (covariates across horizon H)
    kappa: [K] where K = tau_max+1
    Returns: [B, H, M]
    """
    B, H, M = u.shape
    K = kappa.numel()
    device = u.device
    out = torch.zeros_like(u)
    for t in range(H):
        wsum = 0.0
        acc = torch.zeros((B, M), device=device)
        for tau in range(K):
            idx = t - tau
            if idx < 0:
                continue
            acc = acc + kappa[tau] * u[:, idx, :]
            wsum += float(kappa[tau].item())
        if wsum > 0:
            out[:, t, :] = acc
    return out


def total_variation_2nd(x_hat: torch.Tensor) -> torch.Tensor:
    """Second-order total variation on outputs: ||x_{t+1} - 2x_t + x_{t-1}||^2."""
    if x_hat.size(1) < 3:
        return torch.tensor(0.0, device=x_hat.device)
    x_tm1 = x_hat[:, :-2, :]
    x_t = x_hat[:, 1:-1, :]
    x_tp1 = x_hat[:, 2:, :]
    diff2 = x_tp1 - 2.0 * x_t + x_tm1
    return (diff2.pow(2).sum(dim=-1)).mean()


def monotonic_penalty(x_hat: torch.Tensor, v: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Encourage <x_{t+1}-x_t, v> >= 0 when mask=1. If v=None, defaults to +1 for each dim.
    x_hat: [B, H, D]
    v: [D]
    mask: [B, H-1] boolean/float in {0,1}
    """
    B, H, D = x_hat.shape
    if H < 2:
        return torch.tensor(0.0, device=x_hat.device)
    delta = x_hat[:, 1:, :] - x_hat[:, :-1, :]  # [B, H-1, D]
    if v is None:
        v = torch.ones(D, device=x_hat.device)
    proj = (delta * v.view(1, 1, -1)).sum(dim=-1)  # [B, H-1]
    viol = torch.clamp(-proj, min=0.0)
    if mask is not None:
        viol = viol * mask
    return viol.mean()


def entropy_reg(kappa: torch.Tensor) -> torch.Tensor:
    """Entropy regularization to sharpen the kernel."""
    eps = 1e-12
    return (kappa * (kappa.add(eps).log())).sum()
