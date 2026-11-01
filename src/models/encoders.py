"""Visual encoders.

Every encoder implements the same small interface (``BaseEncoder``) so the
classifier, the colorization U-Net (Milestone 2) and the caption decoder
(Milestone 2) can use ANY encoder without changes:

* ``forward_features(x) -> list[Tensor]``  multi-scale feature maps, shallow -> deep
  (the U-Net uses all of them as skip connections, the captioner uses the last one)
* ``forward(x) -> Tensor``                 the deepest feature map  (B, C, h, w)
* ``out_channels: list[int]``              channels of every returned feature map
* ``feature_dim: int``                     channels of the deepest map
* ``gradcam_layer: nn.Module``             layer whose activations Grad-CAM uses
* ``backbone_parameters()``                parameters that get the (smaller) backbone LR

New encoders are added with ``@register_encoder("name")`` (registry / factory
pattern) and selected from the config with ``model.encoder: name``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn

ENCODERS: dict[str, Callable[..., "BaseEncoder"]] = {}


def register_encoder(name: str):
    def deco(fn):
        ENCODERS[name] = fn
        return fn
    return deco


def build_encoder(name: str, pretrained: bool = False, **kwargs) -> "BaseEncoder":
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    return ENCODERS[name](pretrained=pretrained, **kwargs)


class BaseEncoder(nn.Module, ABC):
    out_channels: list[int]
    is_pretrained: bool = False

    @property
    def feature_dim(self) -> int:
        return self.out_channels[-1]

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[-1]

    @property
    @abstractmethod
    def gradcam_layer(self) -> nn.Module:
        ...

    def backbone_parameters(self):
        return self.parameters()


# --------------------------------------------------------------------------- #
# From-scratch CNN (our own architecture, randomly initialised)
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """3x3 -> 3x3 residual block with BatchNorm (He et al., 2016)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class PlainBlock(nn.Module):
    """Same as ResidualBlock but without the skip connection (ablation)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class ScratchCNN(BaseEncoder):
    """Configurable CNN trained from scratch.

    stem (conv 3x3 s2 + conv 3x3 + maxpool) -> 4 stages of blocks.
    Each stage halves the resolution (except the first) and widens the channels.
    Input 224 -> feature maps at 56, 28, 14, 7.
    """

    def __init__(self, widths=(64, 128, 256, 512), depths=(2, 2, 2, 2),
                 residual: bool = True, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        if pretrained:
            raise ValueError("ScratchCNN has no pretrained weights (that is the point).")
        block = ResidualBlock if residual else PlainBlock
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0] // 2, 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(widths[0] // 2, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        stages, c_in = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depths)):
            stride = 1 if i == 0 else 2
            blocks = [block(c_in, w, stride)] + [block(w, w, 1) for _ in range(d - 1)]
            stages.append(nn.Sequential(*blocks))
            c_in = w
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(widths)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # zero-init the last BN of each residual branch: blocks start as identity
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward_features(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats

    @property
    def gradcam_layer(self):
        return self.stages[-1]


@register_encoder("scratch_cnn")
def _scratch(pretrained=False, **kw):
    return ScratchCNN(pretrained=pretrained, **kw)


@register_encoder("scratch_cnn_small")
def _scratch_small(pretrained=False, **kw):
    return ScratchCNN(widths=(32, 64, 128, 256), depths=(1, 1, 1, 1), pretrained=pretrained, **kw)


@register_encoder("scratch_cnn_plain")
def _scratch_plain(pretrained=False, **kw):
    return ScratchCNN(residual=False, pretrained=pretrained, **kw)


# --------------------------------------------------------------------------- #
# Pretrained torchvision backbones (transfer learning)
# --------------------------------------------------------------------------- #
class ResNetEncoder(BaseEncoder):
    def __init__(self, arch: str = "resnet50", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        weights = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
                   "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
                   "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[arch] if pretrained else None
        net = getattr(tvm, arch)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        exp = 4 if arch == "resnet50" else 1
        self.out_channels = [64 * exp, 128 * exp, 256 * exp, 512 * exp]
        self.is_pretrained = pretrained

    def forward_features(self, x):
        x = self.stem(x)
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        return feats

    @property
    def gradcam_layer(self):
        return self.stages[-1]


class EfficientNetEncoder(BaseEncoder):
    """EfficientNet-B0. ``features`` has 9 blocks; we expose the outputs at strides 4, 8, 16, 32."""

    TAPS = (2, 3, 5, 8)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = tvm.efficientnet_b0(weights=weights).features
        self.out_channels = [24, 40, 112, 1280]
        self.is_pretrained = pretrained

    def forward_features(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.TAPS:
                feats.append(x)
        return feats

    @property
    def gradcam_layer(self):
        return self.features[-1]


class ConvNeXtEncoder(BaseEncoder):
    """ConvNeXt-Tiny. ``features`` alternates downsample / stage: stages at indices 1, 3, 5, 7."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = tvm.convnext_tiny(weights=weights).features
        self.out_channels = [96, 192, 384, 768]
        self.is_pretrained = pretrained

    def forward_features(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in (1, 3, 5, 7):
                feats.append(x)
        return feats

    @property
    def gradcam_layer(self):
        return self.features[7]


@register_encoder("resnet18")
def _r18(pretrained=True, **kw):
    return ResNetEncoder("resnet18", pretrained)


@register_encoder("resnet50")
def _r50(pretrained=True, **kw):
    return ResNetEncoder("resnet50", pretrained)


@register_encoder("efficientnet_b0")
def _effb0(pretrained=True, **kw):
    return EfficientNetEncoder(pretrained)


@register_encoder("convnext_tiny")
def _cnxt(pretrained=True, **kw):
    return ConvNeXtEncoder(pretrained)
