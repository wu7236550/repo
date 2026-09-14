# Training Configurations

Reproduces the experimental settings reported in Section 4.2 and Table 2 of the
manuscript "A Lightweight RT-DETR for Road Defect Detection".

## Environment

| Item | Value |
|------|-------|
| OS | CentOS 7.9 |
| CPU | Intel Xeon Silver 4210R |
| GPU | NVIDIA GeForce RTX 3080 (10 GB) |
| Python / PyTorch | PyTorch 2.4.0 + CUDA 11.8 |
| Inference acceleration | TensorRT 8.6 |

## Model

| Item | Value |
|------|-------|
| Baseline | RT-DETR-R18 |
| Decoder layers | 6 |
| Object queries | 300 |
| Input size | 416×416 (letterbox-padded from 1920×1080) |

## Hyperparameters (Table 2)

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0001 |
| Image size | 416×416 |
| Optimizer | AdamW |
| Batch size | 16 |
| Epochs | 240 |

## Augmentation (Section 4.1)

- Random horizontal flipping, probability 0.5
- Random photometric perturbation (brightness ±10%, contrast ±5%)
- Random cropping and scaling (0.8–1.2×)
- Mosaic augmentation, probability 0.2

## Proposed modules (final model)

The final model configuration is:
`ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_LRPB-AIFI.yaml`

1. **MSPC** — Multi-Scale Partial Convolution with Cross-Stage
   Partial connections (backbone).
2. **MCAF** — Multi-scale Context-Anchored Fusion: the reparameterized NCSPELAN4 neck
   with Context Anchor Attention.
3. **LRPB-AIFI** — Learnable Relative Position Bias for Attention-based Intra-scale Feature Interaction.
