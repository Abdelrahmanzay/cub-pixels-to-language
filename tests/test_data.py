"""Transforms and dataset wiring; uses in-memory/temp images, no prepared dataset needed."""
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.datasets import CubClassificationDataset
from src.data.transforms import build_train_transform

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@pytest.fixture
def random_image():
    return Image.new("RGB", (300, 200), color=(120, 80, 40))


@pytest.mark.parametrize("level", ["none", "light", "standard", "strong"])
def test_build_train_transform_returns_float_tensor_of_expected_shape(level, random_image):
    transform = build_train_transform(level, size=64, mean=MEAN, std=STD)
    out = transform(random_image)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (3, 64, 64)


def test_build_train_transform_unknown_level_raises():
    with pytest.raises(ValueError):
        build_train_transform("not-a-level", size=64, mean=MEAN, std=STD)


def test_cub_classification_dataset_returns_tensor_label_and_index(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (50, 50), color=(255, 0, 0)).save(img_dir / "0.jpg")
    Image.new("RGB", (50, 50), color=(0, 255, 0)).save(img_dir / "1.jpg")

    df = pd.DataFrame({
        "path": ["images/0.jpg", "images/1.jpg"],
        "label": [3, 7],
    })
    transform = build_train_transform("none", size=32, mean=MEAN, std=STD)
    dataset = CubClassificationDataset(df, root=tmp_path, transform=transform)

    tensor, label, index = dataset[1]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert label == 7
    assert index == 1
    assert len(dataset) == 2
