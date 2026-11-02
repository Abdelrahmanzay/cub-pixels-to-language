"""Model contracts using synthetic CPU inputs; never download pretrained weights."""
import pytest
import torch
from torch import nn

from src.models.classifier import Classifier, build_classifier, load_classifier, save_checkpoint
from src.models.encoders import ENCODERS, ScratchCNN, build_encoder


@pytest.fixture(autouse=True)
def reproducible_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            yield
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_encoder_feature_contract(name):
    encoder = build_encoder(name, pretrained=False).eval()
    with torch.inference_mode():
        features = encoder.forward_features(torch.randn(2, 3, 224, 224))

    assert len(features) == len(encoder.out_channels) == 4
    assert [feature.shape[1] for feature in features] == encoder.out_channels
    assert [feature.shape[0] for feature in features] == [2] * 4
    assert [feature.shape[-2:] for feature in features] == [
        (56, 56), (28, 28), (14, 14), (7, 7)
    ]
    assert encoder.feature_dim == encoder.out_channels[-1]
    assert isinstance(encoder.gradcam_layer, nn.Module)
    assert not encoder.is_pretrained


def test_unknown_encoder_raises():
    with pytest.raises(KeyError, match="Unknown encoder"):
        build_encoder("unknown_encoder", pretrained=False)


def test_scratch_rejects_pretrained_weights():
    with pytest.raises(ValueError, match="no pretrained weights"):
        ScratchCNN(pretrained=True)


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_classifier_output_shape(name):
    model = Classifier(build_encoder(name, pretrained=False), num_classes=200).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 200)
    assert torch.isfinite(output).all()


def test_frozen_encoder_stays_in_eval_and_only_head_gets_gradients():
    model = Classifier(build_encoder("scratch_cnn_small", pretrained=False), 200,
                       freeze_encoder=True)
    model.eval()
    model.train()
    assert model.training and model.head.training
    assert all(not module.training for module in model.encoder.modules())
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        f"head.{name}" for name, _ in model.head.named_parameters()
    }

    batch_norm = next(m for m in model.encoder.modules() if isinstance(m, nn.BatchNorm2d))
    running_mean = batch_norm.running_mean.clone()
    running_var = batch_norm.running_var.clone()
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.head.parameters())
    torch.testing.assert_close(batch_norm.running_mean, running_mean, rtol=0, atol=0)
    torch.testing.assert_close(batch_norm.running_var, running_var, rtol=0, atol=0)
    assert [g["name"] for g in model.param_groups(0.01, 0.1, 0.001)] == ["head"]


@pytest.mark.parametrize("is_pretrained", [False, True])
def test_param_groups_apply_backbone_multiplier_only_to_pretrained(is_pretrained):
    encoder = build_encoder("scratch_cnn_small", pretrained=False)
    # Exercise optimizer policy independently of fetching actual pretrained weights.
    encoder.is_pretrained = is_pretrained
    model = Classifier(encoder, 200)
    groups = model.param_groups(lr=0.01, backbone_lr_mult=0.1, weight_decay=0.001)
    by_name = {group["name"]: group for group in groups}
    assert set(by_name) == {"head", "encoder"}
    assert by_name["head"]["lr"] == pytest.approx(0.01)
    assert by_name["encoder"]["lr"] == pytest.approx(0.001 if is_pretrained else 0.01)
    assert all(group["weight_decay"] == 0.001 for group in groups)
    assert {id(p) for p in by_name["head"]["params"]} == {id(p) for p in model.head.parameters()}
    assert {id(p) for p in by_name["encoder"]["params"]} == {id(p) for p in encoder.parameters()}
    parameter_ids = [id(p) for group in groups for p in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))


@pytest.mark.parametrize("freeze_encoder", [False, True])
@pytest.mark.parametrize("loss", ["ce", "cosine"])
def test_checkpoint_round_trip_preserves_outputs(tmp_path, freeze_encoder, loss):
    config = {"model": {"encoder": "scratch_cnn_small", "pretrained": False,
                        "freeze_encoder": freeze_encoder, "dropout": 0.3},
              "train": {"loss": loss}}
    class_names = [f"bird_{i}" for i in range(200)]
    model = build_classifier(config, num_classes=200).eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        expected = model(inputs)
    path = tmp_path / "nested" / "classifier.pt"
    save_checkpoint(path, model, config, class_names, extra={"epoch": 3})
    restored, checkpoint = load_classifier(path, device="cpu")
    with torch.inference_mode():
        actual = restored(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not restored.training
    assert restored.freeze_encoder == freeze_encoder
    assert restored.head_type == ("cosine" if loss == "cosine" else "linear")
    assert checkpoint["config"] == config
    assert checkpoint["class_names"] == class_names
    assert checkpoint["epoch"] == 3
