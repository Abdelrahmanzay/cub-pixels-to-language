"""Image classifier = any encoder + global average pooling + linear head."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.models.encoders import BaseEncoder, build_encoder


class Classifier(nn.Module):
    def __init__(self, encoder: BaseEncoder, num_classes: int, dropout: float = 0.2,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(encoder.feature_dim, num_classes))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:  # linear probe: keep BN statistics fixed too
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.encoder(x)))

    def param_groups(self, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
        """Head gets ``lr``; a pretrained backbone gets ``lr * backbone_lr_mult``
        (discriminative fine-tuning). A from-scratch backbone uses the full LR."""
        mult = backbone_lr_mult if self.encoder.is_pretrained else 1.0
        groups = [{"params": [p for p in self.head.parameters()], "lr": lr,
                   "weight_decay": weight_decay, "name": "head"}]
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        if enc:
            groups.append({"params": enc, "lr": lr * mult, "weight_decay": weight_decay,
                           "name": "encoder"})
        return groups


def build_classifier(cfg: dict, num_classes: int) -> Classifier:
    m = cfg["model"]
    enc = build_encoder(m["encoder"], pretrained=m.get("pretrained", False), **(m.get("encoder_kwargs") or {}))
    return Classifier(enc, num_classes, dropout=m.get("dropout", 0.2),
                      freeze_encoder=m.get("freeze_encoder", False))


def save_checkpoint(path: str | Path, model: Classifier, cfg: dict, class_names: list[str], extra: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "class_names": class_names, **extra}, path)


def load_classifier(path: str | Path, device: torch.device | str = "cpu") -> tuple[Classifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg_no_download = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    model = build_classifier(cfg_no_download, len(ckpt["class_names"]))
    model.encoder.is_pretrained = cfg["model"].get("pretrained", False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
