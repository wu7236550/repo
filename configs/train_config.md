# Training Configurations

Reproduces the experimental settings reported in Section 4.2 and Table 2 of the manuscript
"Road Surface Defect Detection Based on a Lightweight RT-DETR with Multi-Scale Partial
Convolution and Context-Aware Position Modeling".

## Environment

| Item | Value |
|------|-------|
| OS | CentOS 7.9 |
| CPU | Intel Xeon Silver 4210R |
| GPU | NVIDIA GeForce RTX 3080 (10 GB) |
| Python / PyTorch | PyTorch 2.4.0 + CUDA 11.8 |
| Inference acceleration | TensorRT 8.6 |
| FLOPs profiler | fvcore (`FlopCountAnalysis`) |

## Model

| Item | Value |
|------|-------|
| Baseline | RT-DETR-R18 |
| Decoder layers | 3 (the official RT-DETR-R18 setting: `RTDETRDecoder [nc, 256, 300, 4, 8, 3]`) |
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
| Random seeds (core experiments) | 0–4 |

## Augmentation (Section 4.1)

- Random horizontal flipping, probability 0.5
- Random photometric perturbation (brightness ±10 %, contrast ±5 %)
- Random cropping and scaling (0.8–1.2×)
- Mosaic augmentation, probability 0.2

Augmentation is applied to the training split only.

## The eight ablation configurations (Table 3)

All configurations live in `src/ultralytics/cfg/models/rt-detr/` and are trained with the
same schedule, giving a complete 2³ factorial design.

| No. | MSPC | MCAF | LRPB-AIFI | Configuration file |
|-----|-----------|------------------|----------|--------------------|
| 1 | ✗ | ✗ | ✗ | `rtdetr-r18.yaml` |
| 2 | ✓ | ✗ | ✗ | `rtdetr-MSPC.yaml` |
| 3 | ✗ | ✓ | ✗ | `rtdetr-MCAF.yaml` |
| 4 | ✗ | ✗ | ✓ | `rtdetr-LRPB-AIFI.yaml` |
| 5 | ✓ | ✓ | ✗ | `rtdetr-MSPC_MCAF.yaml` |
| 6 | ✓ | ✗ | ✓ | `rtdetr-MSPC_LRPB-AIFI.yaml` |
| 7 | ✗ | ✓ | ✓ | `rtdetr-MCAF_LRPB-AIFI.yaml` |
| 8 | ✓ | ✓ | ✓ | `rtdetr-MSPC_MCAF_LRPB-AIFI.yaml` |

## Proposed modules

1. **MSPC** — lightweight backbone that combines multi-scale partial convolution
   (3×3/5×5/7×7 kernels applied to a 50 % channel split) with a cross-stage-partial (CSP)
   structure. Implemented as `MSPC` in `src/ultralytics/nn/extra_modules/block.py`.
2. **MCAF** — the re-parameterised RepNCSPELAN4 block of YOLOv9 followed by
   context anchor attention (CAA) from PKINet. Implemented as `MCAF` in
   `src/ultralytics/nn/extra_modules/block.py`.
3. **LRPB-AIFI** — replaces the AIFI encoder. A learnable relative position bias (LRPB) takes the place of the fixed sinusoidal encoding:
   an MLP maps the relative coordinate offsets (Δx, Δy) to per-head scalar attention biases. The
   bias tables are generated at runtime for the actual feature-map size, so 416×416
   (S5 = 13×13), 640×640 (S5 = 20×20) and non-square inputs are all supported.
   Implemented as `LRPB_AIFI` / `LRPB_Attention` / `LRPB` in
   `src/ultralytics/nn/extra_modules/transformer.py`.

> Note on terminology: `LRPB` stands for *learnable relative position bias*, following
> CrossFormer++ [18]. It is a learnable continuous parameterisation of relative position
> bias and is **not** a content-dependent dynamic mechanism, because the MLP receives only
> the relative coordinate offsets.
