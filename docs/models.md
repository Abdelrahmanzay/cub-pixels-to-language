# Models and the shared visual encoder

The classifier combines a visual encoder with global average pooling, dropout,
and a linear layer that produces 200 class scores. The encoder learns spatial
features; the head turns them into bird-species predictions. The implementation
lives in `src/models/encoders.py` and `src/models/classifier.py`.

## Our from-scratch residual CNN

`scratch_cnn` starts with random weights rather than ImageNet weights. Its stem
uses a 3x3 stride-2 convolution (3 to 32 channels), BatchNorm and ReLU, followed
by a second 3x3 convolution (32 to 64 channels), BatchNorm and ReLU, then a
stride-2 max pool. A 224x224 image therefore becomes a 64x56x56 feature map.

Four stages each contain two residual blocks. Their outputs are:

| Stage | Channels | Spatial size for a 224x224 input |
|---|---:|---|
| 1 | 64 | 56x56 |
| 2 | 128 | 28x28 |
| 3 | 256 | 14x14 |
| 4 | 512 | 7x7 |

A block applies 3x3 convolution, BatchNorm, ReLU, another 3x3 convolution and
BatchNorm. It adds a shortcut to this result before the final ReLU. The shortcut
is an identity when shapes match; otherwise a stride-2 1x1 convolution and
BatchNorm match the new resolution and channel count. Stages 2-4 downsample in
their first block.

Residual connections give information and gradients a shorter route through the
network. Each block can learn a correction to its input instead of having to
reconstruct useful features from nothing. This helps optimize a deeper network.

Convolutions use Kaiming initialization. The final BatchNorm scale in each
residual branch starts at zero, so that branch initially contributes zero and
the block starts as `ReLU(shortcut(x))`. For matching shapes and nonnegative
inputs this is an identity mapping. A downsampling block starts with its learned
projection shortcut instead; calling every block an exact identity would be
inaccurate. The zero scale is trainable, allowing the residual branch to become
active during training.

`scratch_cnn_small` uses the same design with widths 32, 64, 128 and 256 and one
block per stage. It provides a cheaper option for smoke tests and experiments.
Both variants reject `pretrained=True` because no pretrained weights are provided.

## No-skip ablation

`scratch_cnn_plain` keeps the default stem, stage widths, depth and output sizes,
but replaces residual blocks with two convolution/BatchNorm/ReLU sequences and
removes the shortcut addition. Its BatchNorm scales start at one rather than
zero: zeroing a plain branch would suppress its only signal path.

Comparing it with `scratch_cnn` investigates the effect of residual connections
under otherwise similar settings. Parameter counts are slightly different
because projection shortcuts are also removed, so it is not an exactly
parameter-matched experiment.

## Torchvision backbones

The wrappers retain spatial feature extraction and discard each model's original
ImageNet classifier. They can use pretrained weights for transfer learning, or
random initialization with `pretrained=False`. Tests and the summary script
always use random initialization to avoid downloads.

### ResNet-50 and ResNet-18

ResNet-50 uses a 7x7 stride-2 convolution and max-pooling stem followed by four
residual stages. Its bottleneck blocks use 1x1 convolutions to narrow and expand
channels around a 3x3 spatial convolution. The returned channels are 256, 512,
1024 and 2048. The pretrained wrapper selects torchvision's ImageNet-1K V2 weights.

The registry also includes ResNet-18, a shallower network with basic two-3x3
residual blocks. Its stage channels are 64, 128, 256 and 512, and its pretrained
option uses ImageNet-1K V1 weights. Both return maps at strides 4, 8, 16 and 32.

### EfficientNet-B0

EfficientNet-B0 uses inverted bottlenecks: it expands channels, processes spatial
patterns using inexpensive depthwise convolutions, then projects channels down.
Squeeze-and-excitation learns channel weights from pooled image information.
The EfficientNet family balances depth, width and input resolution through
compound scaling; B0 is its baseline size.

Our wrapper collects torchvision feature blocks 2, 3, 5 and 8, returning 24, 40,
112 and 1280 channels. These are selected multi-scale outputs rather than four
identical kinds of internal block. The last includes the final 1x1 expansion.
The pretrained option uses ImageNet-1K V1 weights.

### ConvNeXt-Tiny

ConvNeXt-Tiny modernizes convolutional residual networks using 7x7 depthwise
convolutions, LayerNorm and GELU. Pointwise layers mix and expand channels inside
the residual blocks. A stride-4 patchifying stem creates the first spatial grid;
downsampling layers separate four stages with 96, 192, 384 and 768 channels.
Our wrapper returns torchvision feature blocks 1, 3, 5 and 7. Its pretrained
option uses ImageNet-1K V1 weights.

## Linear probe versus full fine-tuning

A linear probe sets `freeze_encoder=True`. Encoder parameters do not require
gradients; only the classifier head is trained. The overridden `Classifier.train`
also puts the encoder back in evaluation mode after `model.train()`. This matters:
freezing parameters alone does not stop BatchNorm running statistics changing,
or disable stochastic encoder layers. Head dropout remains active during training.

Full fine-tuning leaves the encoder trainable. `param_groups` gives the head the
base learning rate. If `encoder.is_pretrained` is true, the encoder receives
`lr * backbone_lr_mult`, allowing gentler changes to pretrained features.
A randomly initialized encoder receives the full base rate. A frozen encoder
is omitted from the optimizer groups entirely.

The default head produces logits, not probabilities; softmax is applied when
probabilities are needed. The existing optional cosine head normalizes its
200-dimensional output and scales it, and creates an auxiliary layer for the
training loss. It is separate from the default linear-probe setup.

## BaseEncoder interface and reuse in Milestone 2

| Interface | Meaning |
|---|---|
| `forward_features(x)` | Four tensors ordered from shallow to deep, each shaped `(batch, channels, height, width)` |
| `forward(x)` | The deepest feature map, not a pooled vector |
| `out_channels` | Channel counts of the four returned tensors |
| `feature_dim` | `out_channels[-1]`, the deepest map's channel count |
| `gradcam_layer` | An `nn.Module` whose spatial activations support Grad-CAM |
| `is_pretrained` | Whether this encoder was initialized from pretrained weights |
| `backbone_parameters()` | Encoder parameters, available for optimizer construction |

For a 224x224 input the four grids are 56x56, 28x28, 14x14 and 7x7. Early grids
retain fine spatial details; deeper grids represent more abstract patterns.
The planned colorization U-Net needs these multi-scale tensors for skip
connections that join encoder features to matching decoder resolutions.
The planned caption decoder needs the final spatial grid so it can use different
image locations rather than only a global average. These are downstream uses
of the interface; the decoders are Milestone 2 work.

The registry/factory (`ENCODERS`, `register_encoder`, `build_encoder`) selects
implementations by name. A shared interface lets the classifier and future
decoders swap backbones without rewriting their feature-extraction logic.
An unknown encoder name raises `KeyError`.

## Validation and measurements

Run `python -m pytest tests/test_models.py -q` from the repository root. Tests
check every registered encoder's feature grids and classifier output, invalid
configuration errors, frozen-encoder behavior including BatchNorm statistics,
optimizer parameter ownership and learning rates, and exact prediction
preservation after saving/loading checkpoints. Checkpoint tests cover linear
and cosine heads with both frozen and unfrozen encoders. Synthetic tensors and
temporary checkpoint files avoid any dataset download.

The pretrained learning-rate policy is tested by setting `is_pretrained` on a
randomly initialized small encoder; this isolates the policy without fetching
weights and does not claim to test ImageNet weight loading.

Run `python scripts/model_summary.py` to regenerate [model_summary.md](model_summary.md).
It counts encoder parameters without the classification head, and reports the
median CPU forward latency after warm-up. Evaluation mode does not freeze
parameters, so the unfrozen summary reports equal total and trainable counts.
