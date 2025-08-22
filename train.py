# train.py
# Minimal training script wiring TLGN + CATD + metrics
from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tlgn import TLGN
from catd import total_variation_2nd, monotonic_penalty, CausalKernel, apply_causal_kernel
from metrics import classification_metrics, confusion
from visualization import plot_curves, plot_confusion


class ToySeqDataset(Dataset):
    """
    A toy dataset of sequences for next-step classification/regression demo.
    Replace with real data loading (e.g., MIMIC-III-derived features).
    """
    def __init__(self, num: int = 2000, L: int = 24, H: int = 6, D: int = 8, num_classes: int = 3, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.X_hist = rng.normal(size=(num, L, D)).astype(np.float32)
        self.X_future = rng.normal(size=(num, H, D)).astype(np.float32)
        # simple label from sign of a linear combination of last future step
        w = rng.normal(size=(D,))
        logits = self.X_future[:, -1, :].dot(w)
        y = np.digitize(logits, np.quantile(logits, [1/3, 2/3]))
        self.y = y.astype(np.int64)
        self.num_classes = num_classes
        self.H = H

    def __len__(self): return self.X_hist.shape[0]
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X_hist[idx]),
                torch.from_numpy(self.X_future[idx]),
                torch.tensor(self.y[idx], dtype=torch.long))


@dataclass
class Config:
    x_dim: int = 8
    z_dim: int = 128
    enc_hidden: int = 128
    grad_hidden: int = 128
    dec_hidden: int = 128
    enc_layers: int = 1
    horizon: int = 6
    batch_size: int = 64
    epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 1e-4
    tv_weight: float = 1.0
    mono_weight: float = 0.1
    tau_max: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "experiments/results"


def train_one_epoch(model: TLGN, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    cfg: Config, kappa: CausalKernel) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc = 0.0, 0.0
    for x_hist, x_future, y in loader:
        x_hist = x_hist.to(cfg.device)  # [B,L,D]
        x_future = x_future.to(cfg.device)  # [B,H,D]
        y = y.to(cfg.device)  # [B]

        # Forward (predict future H steps)
        x_hat, z_traj = model(x_hist, x_future_cond=x_future, horizon=cfg.horizon)  # [B,H,D]

        # Simple classification head: average of predicted horizon -> logits
        pooled = x_hat.mean(dim=1)  # [B,D]
        logits = nn.Linear(pooled.size(-1), int(y.max().item()) + 1).to(cfg.device)(pooled)  # ephemeral head
        cls_loss = ce(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()

        # CATD-style regularizers
        tv = total_variation_2nd(x_hat)
        mono = monotonic_penalty(x_hat)  # optional mask/vector could be added

        loss = cls_loss + cfg.tv_weight * tv + cfg.mono_weight * mono

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_hist.size(0)
        total_acc += acc * x_hist.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model: TLGN, loader: DataLoader, cfg: Config) -> Tuple[float, Dict[str, float]]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_y, all_pred, all_proba = [], [], []
    for x_hist, x_future, y in loader:
        x_hist = x_hist.to(cfg.device)
        x_future = x_future.to(cfg.device)
        y = y.to(cfg.device)

        x_hat, z_traj = model(x_hist, x_future_cond=x_future, horizon=cfg.horizon)
        pooled = x_hat.mean(dim=1)
        head = nn.Linear(pooled.size(-1), int(max(3, y.max().item() + 1))).to(cfg.device)
        logits = head(pooled)
        loss = ce(logits, y)

        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)
        total_loss += loss.item() * x_hist.size(0)

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(prob.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_proba = np.concatenate(all_proba)
    mets = classification_metrics(all_y, all_pred, all_proba)
    mets["loss"] = total_loss / len(loader.dataset)
    return total_loss / len(loader.dataset), mets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--out_dir", type=str, default=Config.out_dir)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, out_dir=args.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Data
    ds_train = ToySeqDataset(num=1200)
    ds_val = ToySeqDataset(num=300, seed=7)
    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

    # Model + optimizer
    model = TLGN(x_dim=cfg.x_dim, z_dim=cfg.z_dim, enc_hidden=cfg.enc_hidden,
                 grad_hidden=cfg.grad_hidden, dec_hidden=cfg.dec_hidden,
                 enc_layers=cfg.enc_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    kappa = CausalKernel(cfg.tau_max).to(cfg.device)  # optional use for covariates

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best = float("inf")
    for ep in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, cfg, kappa)
        val_loss, mets = evaluate(model, val_loader, cfg)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(mets.get("accuracy", 0.0))

        print(f"Epoch {ep+1}/{cfg.epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={mets.get('accuracy', 0.0):.4f} "
              f"| val_f1={mets.get('f1', 0.0):.4f} val_auc={mets.get('auc', float('nan'))}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_model.pt"))

    # plots
    plot_curves(history, cfg.out_dir)

    # confusion matrix on val
    # quick pass to gather labels/preds again
    _, mets = evaluate(model, val_loader, cfg)
    # for confusion matrix we need the actual arrays; recompute here for clarity
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_hist, x_future, y in val_loader:
            x_hist = x_hist.to(cfg.device)
            x_future = x_future.to(cfg.device)
            y_true.append(y.numpy())
            x_hat, _ = model(x_hist, x_future_cond=x_future, horizon=cfg.horizon)
            pooled = x_hat.mean(dim=1)
            head = nn.Linear(pooled.size(-1), int(max(3, y.max().item() + 1))).to(cfg.device)
            logits = head(pooled)
            y_pred.append(logits.argmax(dim=-1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    cm = confusion(y_true, y_pred)
    classes = [f"C{i}" for i in range(cm.shape[0])]
    plot_confusion(cm, classes, cfg.out_dir)

    print("Done. Artifacts saved to:", cfg.out_dir)


if __name__ == "__main
1111