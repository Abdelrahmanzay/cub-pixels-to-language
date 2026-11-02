# Encoder model summary

Generated with `python scripts/model_summary.py`.

Environment: Windows 10, AMD64; CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; Python 3.11.6, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu.

Method: CPU, 1 intra-op thread(s), eval mode, inference mode, one synthetic RGB image of shape `(1, 3, 224, 224)`, 3 warm-up passes and the median of 10 timed `forward_features` passes.

Every encoder uses `pretrained=False`; no dataset or weights are downloaded. Initialization, image generation and disk I/O are excluded from timing. Latency depends on the machine and system load; it is not training time.

Counts cover the encoder only, without a classifier head. Eval/inference mode does not change `requires_grad`, so all parameters remain trainable here. A frozen linear probe would instead have zero trainable encoder parameters.

| Name | Total params | Trainable params | Channels per stage | CPU forward (ms/image) |
|---|---:|---:|---|---:|
| convnext_tiny | 27,818,592 | 27,818,592 | 96, 192, 384, 768 | 385.393 |
| efficientnet_b0 | 4,007,548 | 4,007,548 | 24, 40, 112, 1280 | 77.694 |
| resnet18 | 11,176,512 | 11,176,512 | 64, 128, 256, 512 | 177.919 |
| resnet50 | 23,508,032 | 23,508,032 | 256, 512, 1024, 2048 | 369.170 |
| scratch_cnn | 11,186,464 | 11,186,464 | 64, 128, 256, 512 | 180.251 |
| scratch_cnn_plain | 11,012,640 | 11,012,640 | 64, 128, 256, 512 | 175.302 |
| scratch_cnn_small | 1,230,608 | 1,230,608 | 32, 64, 128, 256 | 42.499 |

All four maps have spatial sizes 56x56, 28x28, 14x14 and 7x7 for this input. See [models.md](models.md) for architecture and interface explanations.
