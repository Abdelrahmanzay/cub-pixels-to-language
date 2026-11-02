"""Hand-computed classification examples; no dataset or trained weights needed."""
import numpy as np
import pytest
import torch

from src.evaluation.metrics import (classification_metrics, confusion,
                                    most_confused_pairs, per_class_accuracy)


def test_perfect_logits():
    labels = torch.arange(6)
    metrics = classification_metrics(10 * torch.eye(6), labels, num_classes=6)

    for value in metrics.values():
        assert value == pytest.approx(1.0)


def test_known_accuracy_and_top5():
    # Two correct top-1 predictions. Class 2 is third; class 5 is sixth.
    logits = torch.tensor([[6., 5., 4., 3., 2., 1.],
                           [5., 6., 4., 3., 2., 1.],
                           [6., 5., 4., 3., 2., 1.],
                           [6., 5., 4., 3., 2., 1.]])
    labels = torch.tensor([0, 1, 2, 5])
    metrics = classification_metrics(logits, labels, num_classes=6)

    assert metrics["accuracy"] == pytest.approx(2 / 4)
    assert metrics["top5_accuracy"] == pytest.approx(3 / 4)


def test_macro_metrics_on_imbalanced_labels():
    # Supports are 3 and 1; recalls are 2/3 and 1, precisions 1 and 1/2.
    logits = torch.tensor([[3., 1.], [3., 1.], [1., 3.], [1., 3.]])
    labels = torch.tensor([0, 0, 0, 1])
    metrics = classification_metrics(logits, labels, num_classes=2)

    assert metrics["accuracy"] == pytest.approx(3 / 4)
    assert metrics["precision_macro"] == pytest.approx(3 / 4)
    assert metrics["recall_macro"] == pytest.approx(5 / 6)
    assert metrics["f1_macro"] == pytest.approx((4 / 5 + 2 / 3) / 2)
    assert metrics["balanced_accuracy"] == pytest.approx(5 / 6)
    # With fewer than five classes, top-5 considers every class.
    assert metrics["top5_accuracy"] == pytest.approx(1.0)


def test_confusion_uses_true_rows_and_predicted_columns():
    logits = torch.tensor([[3., 1., 0.], [3., 1., 0.], [0., 3., 1.]])
    labels = torch.tensor([0, 1, 1])

    np.testing.assert_array_equal(
        confusion(logits, labels, num_classes=3),
        [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
    )


def test_most_confused_pairs_are_directional_and_ignore_diagonal():
    cm = np.array([[20, 4, 1], [2, 10, 3], [0, 1, 8]])
    original = cm.copy()

    assert most_confused_pairs(cm, ["A", "B", "C"], k=2) == [
        {"true": "A", "predicted": "B", "count": 4, "true_class_support": 25},
        {"true": "B", "predicted": "C", "count": 3, "true_class_support": 15},
    ]
    np.testing.assert_array_equal(cm, original)


def test_most_confused_pairs_omits_zero_counts():
    assert most_confused_pairs(np.diag([3, 2, 1]), ["A", "B", "C"]) == []


def test_per_class_accuracy_is_sorted_ascending():
    cm = np.array([[8, 2, 0, 0], [3, 1, 0, 0], [0, 1, 3, 0], [0, 0, 0, 0]])

    assert per_class_accuracy(cm, ["A", "B", "C", "absent"]) == [
        {"class": "absent", "accuracy": 0.0, "support": 0},
        {"class": "B", "accuracy": 0.25, "support": 4},
        {"class": "C", "accuracy": 0.75, "support": 4},
        {"class": "A", "accuracy": 0.8, "support": 10},
    ]
