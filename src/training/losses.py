"""Loss functions.

* ``ce``      cross-entropy (with optional label smoothing). Works with hard labels and with the
              soft labels produced by Mixup/CutMix.
* ``cosine``  cosine loss + small cross-entropy term (Barz & Denzler, "Deep Learning on Small
              Datasets using the Cosine Loss", WACV 2020). The network output z is L2-normalised
              and pulled towards the one-hot vector of the true class:

                  L = (1 - cos(z, e_y)) + lambda * CE(W * z/|z|, y)

              Because only the DIRECTION of z matters, the loss cannot be lowered by simply making
              logits larger (which is how cross-entropy overfits small datasets). The paper reports
              large gains when training from scratch on small fine-grained datasets such as CUB.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _to_probs(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    if target.dtype in (torch.int64, torch.int32, torch.long):
        return F.one_hot(target, num_classes).float()
    return target.float()


class CosineXentLoss(nn.Module):
    """``logits`` are the scaled cosine scores returned by a cosine-head Classifier
    (``scale * z/|z|``); ``aux`` is the classifier's extra linear layer for the CE term."""

    def __init__(self, aux: nn.Module, scale: float, xent_weight: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.aux = aux
        self.scale = scale
        self.xent_weight = xent_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = logits.float() / self.scale  # unit-length embedding
        t = _to_probs(target, z.size(1))
        cos = (z * F.normalize(t, dim=1)).sum(1)
        loss = (1 - cos).mean()
        if self.xent_weight > 0:
            loss = loss + self.xent_weight * self.ce(self.aux(z), t if target.dtype.is_floating_point else target)
        return loss


def build_criterion(cfg: dict, model) -> nn.Module:
    tcfg = cfg["train"]
    name = tcfg.get("loss", "ce")
    ls = float(tcfg.get("label_smoothing", 0.0))
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if name == "cosine":
        return CosineXentLoss(model.aux, model.cosine_scale, float(tcfg.get("cosine_xent_weight", 0.1)), ls)
    raise ValueError(f"Unknown loss {name}")
