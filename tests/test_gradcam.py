"""Grad-CAM smoke tests on a tiny, randomly initialized CPU classifier."""
import pytest
import torch

from src.models.classifier import build_classifier
from src.visualization.gradcam import GradCAM


@pytest.mark.parametrize("freeze_encoder", [False, True])
@pytest.mark.parametrize("targets", [None, [0, 2]])
def test_gradcam_shape_range_and_hook_cleanup(freeze_encoder, targets):
    # A fixed seed keeps these tests reproducible without downloading weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = build_classifier(
            {"model": {"encoder": "scratch_cnn_small", "pretrained": False,
                       "freeze_encoder": freeze_encoder}},
            num_classes=3,
        )
        x = torch.randn(2, 3, 64, 64)

    layer = model.encoder.gradcam_layer
    forward_hooks = set(layer._forward_hooks)
    backward_hooks = set(layer._backward_hooks)
    cam = GradCAM(model, layer)
    try:
        assert len(layer._forward_hooks) == len(forward_hooks) + 1
        assert len(layer._backward_hooks) == len(backward_hooks) + 1
        class_idx = None if targets is None else torch.tensor(targets)
        cams, logits = cam(x, class_idx)

        assert cams.shape == (2, 64, 64)
        assert logits.shape == (2, 3)
        assert torch.isfinite(cams).all()
        assert torch.all((cams >= 0) & (cams <= 1))
        assert not cams.requires_grad
        assert not logits.requires_grad
        assert cams.device.type == "cpu"
    finally:
        cam.remove()

    assert set(layer._forward_hooks) == forward_hooks
    assert set(layer._backward_hooks) == backward_hooks
    cam.remove()  # Cleanup is safe to repeat.
