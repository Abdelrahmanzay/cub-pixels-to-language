"""Image classifier = any encoder + global average pooling + linear head."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoders import BaseEncoder, build_encoder


class Classifier(nn.Module):
    def __init__(self, encoder: BaseEncoder, num_classes: int, dropout: float = 0.2,
                 freeze_encoder: bool = False, head: str = "linear", cosine_scale: float = 16.0):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(encoder.feature_dim, num_classes))
        self.head_type = head
        self.cosine_scale = cosine_scale
        if head == "cosine":  # output = scale * z/|z| ; extra layer for the auxiliary CE term
            self.aux = nn.Linear(num_classes, num_classes)
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
        z = self.head(self.pool(self.encoder(x)))
        if self.head_type == "cosine":
            # scaled cosine similarity to every class's one-hot vector; argmax = predicted class,
            # and softmax gives usable confidences for top-k, TTA and Grad-CAM
            return self.cosine_scale * F.normalize(z.float(), dim=1)
        return z

    def param_groups(self, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
        """Head gets ``lr``; a pretrained backbone gets ``lr * backbone_lr_mult``
        (discriminative fine-tuning). A from-scratch backbone uses the full LR."""
        mult = backbone_lr_mult if self.encoder.is_pretrained else 1.0
        head_params = list(self.head.parameters()) + (list(self.aux.parameters()) if hasattr(self, "aux") else [])
        groups = [{"params": head_params, "lr": lr, "weight_decay": weight_decay, "name": "head"}]
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        if enc:
            groups.append({"params": enc, "lr": lr * mult, "weight_decay": weight_decay,
                           "name": "encoder"})
        return groups


def build_classifier(cfg: dict, num_classes: int) -> Classifier:
    m = cfg["model"]
    enc = build_encoder(m["encoder"], pretrained=m.get("pretrained", False), **(m.get("encoder_kwargs") or {}))
    head = "cosine" if cfg.get("train", {}).get("loss", "ce") == "cosine" else "linear"
    return Classifier(enc, num_classes, dropout=m.get("dropout", 0.2),
                      freeze_encoder=m.get("freeze_encoder", False), head=head,
                      cosine_scale=float(m.get("cosine_scale", 16.0)))


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
