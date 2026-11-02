# Classification evaluation and Grad-CAM

`src/evaluation/metrics.py` evaluates logits against integer species labels.
Softmax converts logits to probabilities; the largest value gives the top-1
prediction. Scores below range from 0 to 1, with 1 best (multiply by 100 for a
percentage). The confusion matrix contains counts instead.

## Accuracy and top-5

- **Accuracy** (`accuracy`): correct top-1 predictions divided by the number of
  images. A score of 0.80 means the predicted species was correct for 80% of
  images. Each image has equal weight, so larger classes contribute more.
  See [scikit-learn's accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
- **Top-5 accuracy** (`top5_accuracy`): fraction of images whose true species is
  among the five highest scoring classes. It measures whether the correct bird
  species appears in a short candidate list, useful for similar-looking species.
  It does not mean the first choice was correct. Our implementation uses
  `min(5, num_classes)`, so tests with at most five classes always score 1 here.
  See [top-k accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html).

## Precision, recall, F1, and balanced accuracy

For one class, treat that species as positive and all other species as negative.
TP means correctly predicted members, FP means other species predicted as this
one, and FN means members predicted as another species.

| Metric | Per-class definition | What a high score means |
|---|---|---|
| Precision | `TP / (TP + FP)` | Predictions of this species are usually correct. |
| Recall | `TP / (TP + FN)` | Most actual members of this species are found. |
| F1 | `2 * precision * recall / (precision + recall)` | Both precision and recall are high. |

`precision_macro`, `recall_macro`, and `f1_macro` are the unweighted means of
the respective per-class scores. Macro F1 averages individual F1 values; it is
not the F1 calculated from macro precision and macro recall. The implementation
uses `zero_division=0`, assigning zero when a denominator is zero.
See [the precision/recall/F-score API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).

**Balanced accuracy** (`balanced_accuracy`) is the mean recall across the true
classes. For example, 0.80 means the average species has 80% of its images
classified correctly. It equals macro recall when both average over the same
classes, as on the full CUB test set.
See [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

### Why macro averaging matters for CUB

The test set is imbalanced: each species has 11 to 30 images. Ordinary accuracy
therefore gives a 30-image species about `30 / 11 = 2.73` times the weight of an
11-image species. Macro scores give every species equal weight, exposing weak
performance on smaller classes that overall accuracy can hide. Report both:
accuracy describes performance per image; macro scores describe performance
per species. Top-5 is also averaged per image and does not correct imbalance.

A tiny worked example uses true labels `[A, A, A, B]` and predictions
`[A, A, B, B]`. Accuracy is `3/4`. A has precision `1`, recall `2/3`, F1 `4/5`;
B has precision `1/2`, recall `1`, F1 `2/3`. Thus macro precision is `3/4`,
macro recall and balanced accuracy are `5/6`, and macro F1 is `11/15`.
These values are checked in `tests/test_metrics.py`.

Implementation detail for small subsets: macro precision/recall/F1 use
scikit-learn's default union of observed true and predicted labels; classes
absent from both are omitted. Balanced accuracy averages classes with true
support. A predicted-only class can therefore make these averages differ.

## Confusion matrix and per-class diagnostics

`confusion()` builds a `num_classes x num_classes` matrix, including absent
classes. Entry `[i, j]` counts images of true species `i` predicted as species
`j`: rows are truth, columns are predictions. Diagonal counts are correct;
off-diagonal counts identify mistakes. Row sums are true class supports.
See [the confusion matrix definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).

`most_confused_pairs()` ranks nonzero off-diagonal entries by count and reports
the true class support. A-to-B and B-to-A are separate errors, not a combined
pair. Counts should be read alongside support because larger classes have more
opportunities for errors. `per_class_accuracy()` divides each diagonal count by
its row sum (per-class recall), then sorts from worst to best. For zero-support
rows it reports zero as a convention; that is not evidence of poor performance
on an unseen class.

## How Grad-CAM works

[Grad-CAM (Selvaraju et al.)](https://arxiv.org/abs/1610.02391) explains a selected
class score using activations in a convolutional layer:

1. Run a forward pass and retain the layer's feature maps `A[k]`.
2. Backpropagate the selected class logit `y[c]` to those feature maps.
3. Average each channel's gradients over spatial positions:
   `alpha[k] = mean_ij(d y[c] / d A[k,i,j])`.
4. Form `L = ReLU(sum_k(alpha[k] * A[k]))`. ReLU retains positive evidence
   for that class.
5. Upsample the coarse map to the input image dimensions for an overlay.

Bright areas indicate regions contributing positive evidence for the target
class. Inspect both correct and incorrect predictions for bird features or
background shortcuts. This is a coarse explanation, not a segmentation mask or
proof of a causal relationship.

Our `GradCAM` uses `model.encoder.gradcam_layer`, forward/backward hooks, and
bilinear upsampling. It normalizes each image's map independently to `[0, 1]`
with an epsilon to handle constant maps. A constant map becomes all zeros.
Heatmap intensity is not a class probability and should not be compared as
absolute confidence across images. The default target is each image's predicted
class; supplying `class_idx` instead explains those chosen classes.

The model runs in evaluation mode, but autograd stays enabled. The input is
cloned with `requires_grad=True`, allowing Grad-CAM even with a frozen encoder.
The returned maps and logits are detached CPU tensors. Always remove hooks,
including when an explanation fails:

```python
from src.visualization.gradcam import GradCAM

cam = GradCAM(model, model.encoder.gradcam_layer)
try:
    cams, logits = cam(images)
finally:
    cam.remove()
```

## Verification without the dataset

```bash
python -m pytest tests/test_metrics.py tests/test_gradcam.py -q
```

The tests use hand-built logits/confusion matrices and a randomly initialized
`scratch_cnn_small` classifier on `(2, 3, 64, 64)` inputs. Grad-CAM tests cover
predicted and explicit targets, frozen and trainable encoders, finite maps of
shape `(2, 64, 64)` in `[0, 1]`, and removal of both hooks. No dataset or
pretrained weight downloads are needed.
