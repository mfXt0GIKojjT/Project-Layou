# visualization.py
# Simple plotting helpers for training curves and confusion matrix
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_curves(history: Dict[str, list], save_dir: str, fname: str = "training_curves.png") -> str:
    """
    history: keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc' -> list of floats
    """
    os.makedirs(save_dir, exist_ok=True)

    # loss
    plt.figure()
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    loss_path = os.path.join(save_dir, fname.replace(".png", "_loss.png"))
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # accuracy (optional)
    if "train_acc" in history or "val_acc" in history:
        plt.figure()
        if "train_acc" in history:
            plt.plot(history["train_acc"], label="train_acc")
        if "val_acc" in history:
            plt.plot(history["val_acc"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        acc_path = os.path.join(save_dir, fname.replace(".png", "_acc.png"))
        plt.savefig(acc_path, bbox_inches="tight")
        plt.close()
    else:
        acc_path = ""

    return loss_path


def plot_confusion(cm: np.ndarray, class_names: list, save_dir: str, fname: str = "confusion_matrix.png") -> str:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    out_path = os.path.join(save_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
1111