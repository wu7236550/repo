# Lightweight RT-DETR for Mixed-Surface Crack and Pothole Detection

Code, dataset and evaluation scripts for the manuscript:

> **Lightweight RT-DETR for Mixed-Surface Crack and Pothole Detection**

This repository releases the source code, the audited YOLO-format dataset derivative, all
ablation and control configurations, and the profiling/evaluation scripts used for the
efficiency and accuracy figures of the manuscript. The task is reported as **mixed-surface
(asphalt and concrete), two-class (`crack`, `pothole`) detection**, because the public
derivative used here mixes pavement with concrete surfaces.

---

## 1. What is in this repository

| Path | Contents |
|------|----------|
| `src/` | Training, validation, inference, evaluation and profiling code. The three proposed modules live in `src/ultralytics/nn/extra_modules/`. |
| `src/ultralytics/cfg/models/rt-detr/` | The **eight** ablation configurations of the factorial ablation table (baseline, three single modules, three two-module combinations, and the full model). |
| `src/ultralytics/cfg/models/rt-detr/controls/` | The **five** control configurations that separate the CAA contribution from the fusion-block replacement, and the four positional-encoding schemes. |
| `configs/` | `roadrdd.yaml` dataset definition, `train_config.md` hyperparameters, `width_calibration.yaml` pixel-to-millimetre calibration. |
| `dataset/roadrdd/` | The released YOLO-format dataset (`roadrdd` is the internal name of the released derivative), a per-image SHA-256 manifest and a split manifest. |
| `dataset/roadrdd/audit/` | The byte-level duplicate audit log, the deduplication plan, the dataset audit report and the background-only patch manifest. |
| `docs/` | Annotation specification, dataset statistics, reproducibility notes, the detailed-metrics protocol and the **manuscript-alignment notes**. |

## 2. Model

Three scenario-adapted modifications of RT-DETR-R18 (Figure 2 of the manuscript):

1. **MSPC** (Multi-Scale Partial Convolution) — lightweight backbone. A full-width 3×3
   convolution is followed by channel splitting and by depthwise 5×5 and 7×7 convolutions on
   progressively narrower slices, concatenated and fused by a 1×1 convolution with a residual
   connection. The released block is a CSP-style wrapper (`MSPC`, built on the `MSPConv`
   partial-convolution unit) in `src/ultralytics/nn/extra_modules/block.py`.
2. **MCAF** (Multi-scale Context-Anchored Fusion) — fusion module that replaces the RT-DETR
   fusion block. It combines the re-parameterised RepNCSPELAN4 block of YOLOv9 with context
   anchor attention (CAA) from PKINet, whose serial
   `AvgPool → Conv1×1 → DConv_h → DConv_v → Conv1×1 → Sigmoid` path modulates the fused
   features. Implemented as `MCAF` in `src/ultralytics/nn/extra_modules/block.py`
   (CAA in `attention.py`).
3. **LRPB-AIFI** (Learnable Relative Position Bias for Attention-based Intra-scale Feature
   Interaction) — the AIFI encoder is replaced by a post-norm transformer layer in which the
   sinusoidal encoding is dropped and a learnable relative position bias is added to the
   attention logits: a small MLP maps the relative coordinate offsets (dx, dy) to per-head
   scalars. The bias tables are built at runtime for the actual feature-map size, so 416×416
   (S5 = 13×13), 640×640 (S5 = 20×20) and non-square inputs all work without
   re-instantiation. Implemented as `LRPB_AIFI` / `LRPB_Attention` / `LRPB` in
   `src/ultralytics/nn/extra_modules/transformer.py`.

Final configuration:
`src/ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_LRPB-AIFI.yaml`

> **Terminology.** In this code base the RT-DETR cross-scale fusion block is the Ultralytics
> `RepC3` module; the manuscript refers to the same encoder stage by its official-PyTorch name
> `CSPRepLayer`. The two names are used for the same fusion stage; see
> `docs/manuscript_alignment.md` for the exact statement and its open question.
> `LRPB` means *learnable relative position bias*: it is a learnable continuous
> parameterisation of the relative position bias, **not** a content-dependent dynamic
> mechanism, because the MLP receives only the coordinate offsets.

## 3. Dataset

`dataset/roadrdd/` holds the released YOLO-format derivative: **8,695 images** with two
classes (`crack` = 0, `pothole` = 1) and 16,813 bounding boxes
(8,455 crack / 8,358 pothole).

| Split | Images |
|-------|--------|
| train | 6,162 |
| val   | 836 |
| test  | 1,697 |
| **Total** | **8,695** |

The data are **not author-collected**. They are a locally prepared YOLO derivative of public
sources; the file-name families identify the upstream collections:

| Name family | Images (raw) | Images (released) | Upstream collection |
|---|---|---|---|
| `CRACK500_*`, `2016xxxx_*`, `DeepCrack_*`, `GAPS384_*`, `CFD_*`, `forest_*`, `cracktree200_*`, `Eugen_Muller_*`, `Volker_*` | 5,569 | 5,206 | Concrete Crack Conglomerate Dataset (CCCD) families: CRACK500, CFD, DeepCrack, GAPs, CrackTree; plus the tunnel-lining and concrete-facade crack sets |
| `po_*` | 1,927 | 1,927 | Annotated Potholes with Severity Levels / Annotated Potholes Dataset |
| `noncrack_*`, `IMG_*`, bare numeric stems and their variants | 1,578 | 1,562 | pre-filter exports whose original source cannot be resolved from the released files |
| **Total** | **9,074** | **8,695** | |

The dataset was audited before release:

* **Byte-level deduplication.** Every image was hashed with SHA-256. 379 duplicate groups
  (379 redundant copies) were found, of which 172 spanned two splits
  (train∩test 90, train∩val 69, val∩test 13). All redundant copies were removed; the
  full group-level record is in `dataset/roadrdd/audit/dedup_log.csv` and the per-image
  hashes are in `dataset/roadrdd/MANIFEST.sha256`.
* **Split manifest.** `dataset/roadrdd/split_manifest.csv` maps every image to its split
  and to a `source_group` identifier, so the absence of byte-identical images across
  splits can be verified by any reader.

See `docs/dataset_statistics.md` and `dataset/roadrdd/audit/dataset_audit.md` for the full
numbers. To re-run the audit:

```bash
python src/tools/audit_dataset.py . --write      # writes the manifests and audit log
```

> **Manuscript protocol.** The manuscript reports the main experiment on a **7,496-image
> main set** (5,238 / 748 / 1,510) obtained by excluding the 1,578 exports with no traceable
> source and by splitting **by source group** (every original image together with all of its
> exported variants) rather than by image. The split released in this repository predates
> that protocol: it is byte-level duplicate-free but not group-disjoint. The two data
> descriptions, the exact family counts of both, and what would be required to regenerate the
> manuscript split are documented in `docs/manuscript_alignment.md`.

## 4. Requirements

`environment.yml` pins the environment used for the experiments
(PyTorch, CUDA 11.8, TensorRT, fvcore, THOP); `src/requirements.txt` is the lighter pip-only
equivalent.

```bash
conda env create -f environment.yml
conda activate roadrdd
```

The reported experiments were run on a single **NVIDIA RTX A6000 (48 GB)** with **PyTorch 2.1
and CUDA 11.8**. FPS is measured on the same A6000 with batch size 1 after 200 warm-up
iterations; a laboratory Jetson Orin NX (TensorRT FP16) benchmark is reported separately in
the manuscript and is not a vehicle-grade validation.

## 5. Quick start

```bash
cd src
pip install -r requirements.txt

# train the proposed model (416x416, AdamW, 240 epochs, paper seed set 42/123/2024/7/31415)
python train.py --data ../configs/roadrdd.yaml --imgsz 416 --batch 16 --epochs 240 --seed 42

# train one of the eight ablation configurations
python train.py --cfg ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --name r18

# train one of the control configurations (CAA decomposition, positional encoding, PConv arm)
python train.py --cfg ultralytics/cfg/models/rt-detr/controls/rtdetr-control-PE-static.yaml

# evaluate
python val.py --weights runs/roadrdd/train/<run>/weights/best.pt --split test --imgsz 416

# parameters and GFLOPs with the convention that reproduces the manuscript tables
python get_all_yaml_param_and_flops.py --imgsz 416
python get_all_yaml_param_and_flops.py --imgsz 416 --cfg-dir ultralytics/cfg/models/rt-detr/controls

# background-only patches for the FP/image metric, and its evaluation
python make_background_patches.py --data ../dataset/roadrdd --split test
python eval_background_fp.py --weights runs/roadrdd/train/<run>/weights/best.pt --imgsz 416 --conf 0.25

# pixel-to-millimetre calibration and crack-width bins
python calibrate_width.py

# operator-level fvcore breakdown; --double-macs applies the counting rule of manuscript
# Section 3.4 (conv + matmul MACs x 2). The headline GFLOPs come from the THOP script above
# (see docs/manuscript_alignment.md, item B2).
python get_flops_fvcore.py --imgsz 416 640
python get_flops_fvcore.py --imgsz 416 --double-macs

# end-to-end FPS
python get_FPS.py --engine best.engine --imgsz 416 --runs 5 --warmup 200 --iters 300

# unit tests for the learnable relative position bias tables
pytest tests/test_lrpb_dynamic_size.py -v
```

## 6. Reported results

Table 5(a) of the manuscript: mean ± standard deviation over its five seeds
(42, 123, 2024, 7, 31415), on the group-disjoint split (5,238 / 748 / 1,510).

| Configuration | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---|---|---|---|---|---|
| RT-DETR-R18 (baseline) | 19.87 | 24.1 | 0.7465 ± 0.0058 | 0.4542 ± 0.0052 | 151.3 ± 2.1 |
| + MSPC | 14.09 | 20.2 | 0.7498 ± 0.0055 | 0.4571 ± 0.0049 | 162.6 ± 2.4 |
| + MCAF | 19.03 | 22.3 | 0.7521 ± 0.0052 | 0.4603 ± 0.0047 | 155.7 ± 2.0 |
| + LRPB-AIFI | 19.88 | 24.2 | 0.7534 ± 0.0050 | 0.4622 ± 0.0045 | 152.9 ± 2.2 |
| + MSPC + MCAF | 13.25 | 18.4 | 0.7560 ± 0.0048 | 0.4651 ± 0.0043 | 167.3 ± 2.5 |
| + MSPC + LRPB-AIFI | 14.09 | 20.3 | 0.7566 ± 0.0047 | 0.4658 ± 0.0042 | 163.9 ± 2.3 |
| + MCAF + LRPB-AIFI | 19.03 | 22.4 | 0.7572 ± 0.0046 | 0.4664 ± 0.0041 | 156.5 ± 2.1 |
| **All three (proposed)** | **13.25** | **18.5** | **0.7598 ± 0.0049** | **0.4681 ± 0.0046** | **168.4 ± 2.6** |

Parameters and GFLOPs in this table are produced by
`python src/get_all_yaml_param_and_flops.py --imgsz 416`, i.e. the THOP-style convention of
two floating-point operations per multiply–accumulate, on the fused model. FPS is the
manuscript's measurement on one NVIDIA RTX A6000. Per-class AP, precision/recall, the
seed-and-source-group double bootstrap and the source-held-out check are in the manuscript
(Tables 6 and 7) and in `docs/detailed_metrics_protocol.md`.

The code reproduces the complexity columns exactly; the accuracy columns are the manuscript's
numbers, and the split they were measured on is not the split released here — see
`docs/manuscript_alignment.md`.

## 7. External road test

The external generalization number uses the full labeled release of the **RDD2022** Czech
and China-motorcycle country slices (4,806 images: Czech 2,829 + China-motorcycle 1,977;
2,146 images carry a mapped label and the remainder are retained as negatives). The proposed
model reaches 0.6642 mAP@0.5 on the full release versus 0.6385 for the RT-DETR-R18 baseline,
and 0.6895 over the 2,146 positive-labeled images. The slices share no image, video clip or
source group with the training derivative, and models are evaluated **zero-shot** after
training, with no tuning on this split. Following the official label convention, `D00`,
`D10` and `D20` are mapped to `crack`, `D40` to `pothole`, and `D43`, `D44` and `D50` are
ignored; images with no mapped label are retained as negatives so that false positives are
counted.

RDD2022 is publicly available at <https://github.com/sekilab/RoadDamageDetector> and is not
redistributed here.

## 8. License

* Source code: MIT License (`LICENSE`).
* `roadrdd` dataset derivative: released for academic research use; the upstream public
  sources keep their own licences (CCCD is CC0; the pothole collections are ODbL-type).
  Please cite the manuscript **and** the upstream datasets listed in Section 3 if you use it.

## 9. Citation

If you find this repository useful, please cite the corresponding manuscript.

## 10. Manuscript–implementation alignment

`docs/manuscript_alignment.md` records every place where the manuscript's description and the
released implementation differ, with the measured evidence, and states which artefact is
authoritative for each item. It is the first document to read when a number or a statement in
the manuscript cannot be reproduced from this repository.
