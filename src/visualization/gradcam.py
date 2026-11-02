"""Grad-CAM (Selvaraju et al., 2017).

For a target class c, Grad-CAM weights each channel k of a conv layer's
activation map A^k by the spatially averaged gradient of the class score y^c:

    alpha_k^c = mean_{i,j} dy^c / dA^k_ij
    L^c       = ReLU( sum_k alpha_k^c * A^k )

then upsamples L^c to the image size. Bright regions are the pixels that most
increased the score of class c.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data.transforms import denormalize  # noqa: E402


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _inp, out):
        self.activations = out.detach()

    def _save_grad(self, _m, _gin, gout):
        self.gradients = gout[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | None = None):
        """x: (B,3,H,W). Returns (cams (B,H,W) in [0,1], logits)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)  # needed when the encoder is frozen
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.view(cam.size(0), -1)
        mn, mx = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - mn) / (mx - mn + 1e-8)).view_as(cam)
        return cam.detach().cpu(), logits.detach().cpu()

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def gradcam_grid(images: torch.Tensor, labels: list[int], model_outputs: list[dict],
                 class_names: list[str], mean, std, path, title: str):
    """images: (N,3,H,W) normalised. model_outputs: one dict per model with keys
    name, cams (N,H,W), preds (N,), conf (N,)."""
    n, m = images.size(0), len(model_outputs)
    fig, axes = plt.subplots(n, m + 1, figsize=(2.6 * (m + 1), 2.6 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        img = denormalize(images[i], mean, std).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"true: {class_names[labels[i]][:24]}", fontsize=7)
        for j, mo in enumerate(model_outputs):
            ok = mo["preds"][i] == labels[i]
            cam = mo["cams"][i]
            if cam.shape != img.shape[:2]:  # models with different input sizes (e.g. 224 vs 384 px)
                cam = F.interpolate(cam[None, None], size=img.shape[:2], mode="bilinear",
                                    align_corners=False)[0, 0]
            axes[i, j + 1].imshow(overlay(img, cam.numpy()))
            axes[i, j + 1].set_title(f"{mo['name']}\npred: {class_names[mo['preds'][i]][:22]} "
                                     f"({mo['conf'][i]:.2f})", fontsize=7, color="green" if ok else "red")
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
