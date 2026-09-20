# Training Configurations

Reproduces the experimental settings reported in Section 4.2 and Table 4 of the manuscript
*Lightweight RT-DETR for Mixed-Surface Crack and Pothole Detection*.

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX A6000 (48 GB), one device |
| PyTorch / CUDA | PyTorch 2.1 + CUDA 11.8 |
| Inference acceleration | TensorRT (FPS and export path only; training is plain PyTorch) |
| Complexity profiler | `fvcore` 0.1.5 (Table 2) and the ultralytics `model_info` THOP convention |

The manuscript states the single-GPU training and timing environment of Section 4.2; it does
not report the host OS or CPU, so those are not asserted here.

## Model

| Item | Value |
|------|-------|
| Baseline | RT-DETR-R18 |
| Decoder layers | 3 (the official RT-DETR-R18 setting: `RTDETRDecoder [nc, 256, 300, 4, 8, 3]`) |
| Object queries | 300 |
| Input size | 416×416 (letterbox, no distortion) |
| Loss | `RTDETRDetectionLoss`, NMS-free (the deformable decoder selects slots; no NMS) |

## Hyperparameters (Table 4)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Base learning rate | 1e-4 |
| LR schedule | cosine to 1e-6 over 240 epochs |
| Weight decay | 1e-4 |
| Warm-up | 5 epochs, linear |
| Batch size | 16 |
| Epochs | 240 |
| Input size | 416×416 (letterbox, no distortion) |
| Loss weights | cls 1.0, bbox 5.0, giou 2.0 |
| Gradient clip | max-norm 0.1 |
| EMA | decay 0.9999 |
| Eval confidence | 0.01 (recall sweep); operating point 0.5 for the qualitative figures |
| Random seeds | 42, 123, 2024, 7, 31415 |

These values are the ones set in `src/train.py` and `src/ultralytics/cfg/default.yaml`. Two
implementation details are worth stating explicitly, because the argument names do not mean
what they appear to mean:

* **Loss weights.** `RT-DETR` does not use the `box`/`cls`/`dfl` gains of the YOLO training
  path. `RTDETRDetectionModel.init_criterion` builds
  `RTDETRDetectionLoss(nc, use_vfl=True)`, whose gains come from `DETRLoss`'s default
  `loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, ...}` — exactly the cls 1.0 / bbox 5.0 /
  giou 2.0 of Table 4. The `box: 7.5`, `cls: 0.5`, `dfl: 1.5` entries in `default.yaml` are
  therefore inert for this model and are left at the library default.
* **Warm-up.** In this fork the warm-up loop compares `args.warmup_epochs` against the
  *global iteration index* (`src/ultralytics/engine/trainer.py`, `_do_train`), so the value
  is an iteration budget rather than an epoch count. The released value `warmup_epochs: 2000`
  is about 5 epochs on the released training split (6,162 images / batch 16 ≈ 385 iterations
  per epoch), which is the "5 epochs, linear" of Table 4.

## Augmentation (Table 4)

- Random horizontal flipping, probability 0.5
- HSV jitter (0.015, 0.7, 0.4) — hue, saturation, value
- Mosaic, closed over the last 15 epochs

These values are the ones set in `src/train.py` (`fliplr=0.5`, `hsv_h=0.015`, `hsv_s=0.7`,
`hsv_v=0.4`, `mosaic=0.2`, `close_mosaic=15`).

Two points are documented rather than resolved, because Table 4 does not settle them:

* Table 4 names mosaic and its closing schedule but **not** its probability. The released
  script keeps the value used for the reported runs (`mosaic=0.2`).
* The scaling and translation jitter that the released script also applies
  (`scale=0.2`, `translate=0.1`) are not listed in Table 4. They are retained unchanged so
  that the released recipe stays identical to the one that produced the reported numbers;
  see `../docs/manuscript_alignment.md`.

Augmentation is applied to the training split only; validation and test images are never
augmented.

## The eight factorial ablation configurations (Table 5(a))

All configurations live in `src/ultralytics/cfg/models/rt-detr/` and are trained with the
same schedule, seed set and split, giving a complete 2³ factorial design.

| No. | MSPC | MCAF | LRPB | Configuration file |
|-----|-----------|------------------|----------|--------------------|
| 1 | ✗ | ✗ | ✗ | `rtdetr-r18.yaml` |
| 2 | ✓ | ✗ | ✗ | `rtdetr-MSPC.yaml` |
| 3 | ✗ | ✓ | ✗ | `rtdetr-MCAF.yaml` |
| 4 | ✗ | ✗ | ✓ | `rtdetr-LRPB-AIFI.yaml` |
| 5 | ✓ | ✓ | ✗ | `rtdetr-MSPC_MCAF.yaml` |
| 6 | ✓ | ✗ | ✓ | `rtdetr-MSPC_LRPB-AIFI.yaml` |
| 7 | ✗ | ✓ | ✓ | `rtdetr-MCAF_LRPB-AIFI.yaml` |
| 8 | ✓ | ✓ | ✓ | `rtdetr-MSPC_MCAF_LRPB-AIFI.yaml` |

The five control configurations that separate the CAA contribution from the fusion-block
replacement, and the four positional-encoding schemes, live in the `controls/`
subdirectory; see `../docs/reproducibility.md`, Section 4.

## Proposed modules

1. **MSPC** — Multi-Scale Partial Convolution backbone. A full-width 3×3 convolution is
   followed by a channel split, a depthwise 5×5 convolution over the complete C/2 slice and a
   depthwise 7×7 convolution over a C/4 tap copied out of that slice; the branches
   concatenate to 5C/4 channels and a 1×1 convolution projects back to C with a residual
   connection. Implemented as `MSPC` (built on the `MSPConv` unit) in
   `src/ultralytics/nn/extra_modules/block.py`. The released backbone replaces stages 2–4,
   leaving stage 1 unchanged.
2. **MCAF** — Multi-scale Context-Anchored Fusion. The re-parameterised RepNCSPELAN4 block
   of YOLOv9 followed by context anchor attention (CAA) from PKINet. Implemented as `MCAF`
   in `src/ultralytics/nn/extra_modules/block.py` (CAA in `attention.py`).
3. **LRPB-AIFI** — Learnable Relative Position Bias for Attention-based Intra-scale Feature
   Interaction. It replaces the AIFI encoder: the fixed sinusoidal encoding is dropped and a
   learnable relative position bias is added to the attention logits, produced by a small MLP
   that maps the relative coordinate offsets (dx, dy) to per-head scalars. The bias tables are
   generated at runtime for the actual feature-map size, so 416×416 (S5 = 13×13), 640×640
   (S5 = 20×20) and non-square inputs are all supported. Implemented as `LRPB_AIFI` /
   `LRPB_Attention` / `LRPB` in `src/ultralytics/nn/extra_modules/transformer.py`.

> Note on terminology: `LRPB` stands for *learnable relative position bias*, following
> CrossFormer++ [18]. It is a learnable continuous parameterisation of relative position
> bias and is **not** a content-dependent dynamic mechanism, because the MLP receives only
> the relative coordinate offsets.
