"""Matplotlib plots shared by training, evaluation, tuning and comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training_curves(history: list[dict], path, title: str = ""):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set(xlabel="epoch", ylabel="top-1 accuracy", title="Accuracy", ylim=(0, 1))
    for a in ax:
        a.legend()
    fig.suptitle(title)
    return _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, path, title: str = "Confusion matrix (row-normalised)"):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set(xlabel="predicted class id", ylabel="true class id", title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_confused_pairs(pairs: list[dict], path, k: int = 12):
    pairs = pairs[:k][::-1]
    if not pairs:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(pairs) + 1))
    ax.barh([f"{p['true']} → {p['predicted']}" for p in pairs], [p["count"] for p in pairs], color="#c0504d")
    ax.set(xlabel="# test images", title="Most confused species pairs (true → predicted)")
    return _save(fig, path)


def plot_per_class_accuracy(per_class: list[dict], path, k: int = 15):
    worst, best = per_class[:k], per_class[-k:]
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.32 * k + 1.5))
    ax[0].barh([d["class"] for d in worst][::-1], [d["accuracy"] for d in worst][::-1], color="#c0504d")
    ax[0].set(title=f"{k} hardest classes", xlabel="accuracy", xlim=(0, 1))
    ax[1].barh([d["class"] for d in best], [d["accuracy"] for d in best], color="#4f81bd")
    ax[1].set(title=f"{k} easiest classes", xlabel="accuracy", xlim=(0, 1))
    return _save(fig, path)


def plot_model_comparison(rows: list[dict], path, metrics=("accuracy", "f1_macro", "top5_accuracy")):
    names = [r["run"] for r in rows]
    x = np.arange(len(names))
    w = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(names)), 4.5))
    for i, m in enumerate(metrics):
        vals = [r.get(m, np.nan) for r in rows]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set(ylim=(0, 1.05), ylabel="score (test set)", title="Model comparison")
    ax.legend()
    return _save(fig, path)


def plot_val_curves_overlay(histories: dict[str, list[dict]], path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, h in histories.items():
        ax.plot([r["epoch"] for r in h], [r["val_acc"] for r in h], label=name)
    ax.set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy per run", ylim=(0, 1))
    ax.legend(fontsize=8)
    return _save(fig, path)
