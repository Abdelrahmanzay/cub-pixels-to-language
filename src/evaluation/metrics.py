"""Classification metrics.

* accuracy        fraction of images whose top-1 prediction is correct
* top5_accuracy   correct class is among the 5 most likely (useful for fine-grained
                  classes where the top-2/3 are often very similar species)
* precision/recall/F1 (macro)  per-class scores averaged with equal weight per class,
                  so rare/hard classes count as much as easy ones
* balanced_accuracy  mean per-class recall
* confusion matrix + the most confused class pairs
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


@torch.no_grad()
def predict_loader(model, loader, device, amp: bool = True):
    model.eval()
    logits_all, labels_all, idx_all = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        labels_all.append(y)
        idx_all.append(idx)
    return torch.cat(logits_all), torch.cat(labels_all), torch.cat(idx_all)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    probs = logits.softmax(1)
    preds = probs.argmax(1).numpy()
    y = labels.numpy()
    top5 = probs.topk(min(5, num_classes), dim=1).indices.numpy()
    p, r, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "top5_accuracy": float(np.mean([y[i] in top5[i] for i in range(len(y))])),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
    }


def confusion(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> np.ndarray:
    return confusion_matrix(labels.numpy(), logits.argmax(1).numpy(), labels=list(range(num_classes)))


def most_confused_pairs(cm: np.ndarray, class_names: list[str], k: int = 15) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        out.append({"true": class_names[i], "predicted": class_names[j], "count": int(off[i, j]),
                    "true_class_support": int(cm[i].sum())})
    return out


def per_class_accuracy(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    support = cm.sum(1)
    acc = np.divide(np.diag(cm), support, out=np.zeros(len(support)), where=support > 0)
    return sorted([{"class": c, "accuracy": float(a), "support": int(s)}
                   for c, a, s in zip(class_names, acc, support)], key=lambda d: d["accuracy"])
