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
`ultralytics/cfg/models/rt-detr/rtdetr-CSP-PMSFA_RepNCSPELAN4_CAA_TransformerEncoderLayer_DPB.yaml`

1. **CSP-PMSFA** — Partial Multi-Scale Feature Aggregation with Cross-Stage
   Partial connections (backbone).
2. **RepNCSPELAN4_CAA** — Reparameterized NCSPELAN4 neck with Context Anchor
   Attention.
3. **DPB-AIFI** — Dual-Path Bottleneck AIFI encoder enhancement.
