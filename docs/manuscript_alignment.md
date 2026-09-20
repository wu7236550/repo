# Manuscript-Implementation Alignment

This file records every place where the manuscript *Lightweight RT-DETR for Mixed-Surface Crack
and Pothole Detection* and this repository differ. It is the first document to read when a
number or a statement in the manuscript cannot be reproduced from the repository.

**Authority rule used here.** The manuscript is the authority for what the repository must
*describe*; the code is the authority for what *produced* the reported numbers. Both
requirements are satisfiable for most items, and those were fixed directly (Part A). Where
they cannot both hold, the item is listed with its measured evidence and its handling
(Part B) rather than silently changed. Ambiguities that the manuscript leaves open are
recorded in Part C and left at the value used for the reported runs.

All measurements below were taken from this repository's own code and data; the commands are
given in Part D.

---

## Part A - items changed to match the manuscript

| # | Item | Manuscript | Repository before | Change |
|---|---|---|---|---|
| A1 | Title and task framing | *Lightweight RT-DETR for Mixed-Surface Crack and Pothole Detection*; the task is framed as **mixed-surface** (asphalt and concrete) because the public derivative mixes surfaces | `README.md` and `configs/train_config.md` carried the earlier title *Road Surface Defect Detection Based on a Lightweight RT-DETR with Multi-Scale Partial Convolution and Context-Aware Position Modeling* and framed the task as road-surface inspection | `README.md`, `configs/train_config.md` retitled and reframed; the mixed-surface statement and its reason added |
| A2 | Dataset composition | Three components in Table 3(a): CCCD-like crack families 5,569 exports, `po_*` pothole family 1,927, untraceable families 1,578 excluded; **main set 7,496** = 5,238 / 748 / 1,510 with 14,560 boxes (7,642 crack, 6,918 pothole) | `README.md`, `docs/dataset_statistics.md`, `docs/annotation_spec.md` described only the released derivative (8,695 images, 16,813 boxes) and never mentioned the exclusion or the main set | Component table, main-set table and per-split box counts added; the relationship between 8,695 released and 7,496 main-set images stated explicitly in `README.md` (Section 3), `docs/dataset_statistics.md`, `docs/annotation_spec.md` and `docs/reproducibility.md` (Sections 1 and 1.1) |
| A3 | Execution environment | Single NVIDIA RTX A6000 (48 GB), PyTorch 2.1, CUDA 11.8 (Section 4.2) | `README.md`, `configs/train_config.md`, `docs/reproducibility.md` stated CentOS 7.9 + Intel Xeon Silver 4210R + **RTX 3080 (10 GB)** + PyTorch 2.4.0 + TensorRT 8.6; `environment.yml` and `src/requirements.txt` pinned PyTorch 2.4.0 / torchvision 0.19.0 | GPU, PyTorch and CUDA corrected everywhere; the host OS and CPU are no longer asserted, because the manuscript does not report them; `environment.yml` and `src/requirements.txt` moved to `pytorch=2.1.0` / `torchvision=0.16.0` with a comment recording that the previous pin was 2.4.0 |
| A4 | Learning-rate schedule | AdamW, base LR 1e-4, **cosine to 1e-6** over 240 epochs (Table 4) | `src/train.py` never set `cos_lr`, and `default.yaml` had `cos_lr: False` with `lrf: 1.0`; the resulting schedule was a **constant** LR of 1e-4 | `cos_lr: True`, `lrf: 0.01` added to `src/train.py` and `src/ultralytics/cfg/default.yaml` |
| A5 | Augmentation | H-flip p=0.5; **HSV jitter (0.015, 0.7, 0.4)**; mosaic closed over the last 15 epochs (Table 4) | `src/train.py` used `hsv_s=0.05`, `hsv_v=0.10` and `close_mosaic=0` (mosaic active for the whole schedule) | `hsv_s=0.7`, `hsv_v=0.4`, `close_mosaic=15` in `src/train.py` and `src/ultralytics/cfg/default.yaml`; `configs/train_config.md` restated |
| A6 | Random seeds | 42, 123, 2024, 7, 31415, reported as mean ± standard deviation (Table 4) | `src/train.py` defaulted to `--seed 0` and its help text said "paper uses seeds 0-4" | Default seed is now 42 and the five-seed set is stated in `src/train.py` and `configs/train_config.md` |
| A7 | Loss weights | cls 1.0, bbox 5.0, giou 2.0 (Table 4) | `default.yaml` carries `box: 7.5`, `cls: 0.5`, `dfl: 1.5`, which reads as a contradiction | Verified that RT-DETR does **not** use those gains: `RTDETRDetectionModel.init_criterion` builds `RTDETRDetectionLoss(nc, use_vfl=True)`, whose gains are `DETRLoss`'s defaults `{'class': 1, 'bbox': 5, 'giou': 2, ...}` — identical to Table 4. The values were left unchanged and annotated as inert in `default.yaml` |
| A8 | Warm-up | 5 epochs, linear (Table 4) | `default.yaml` has `warmup_epochs: 2000`, which reads as a 2,000-epoch warm-up | Verified that in this fork the warm-up loop compares `args.warmup_epochs` with the **global iteration index** (`trainer.py`, `_do_train`: `nw = self.args.warmup_epochs`, `if ni <= nw`), so the value is an iteration budget: 2,000 iterations is about 5.2 epochs on the released training split (6,162 / 16 ≈ 385 iterations per epoch). Left unchanged and annotated in `default.yaml` and `configs/train_config.md` |
| A9 | Complexity counting rule | Section 3.4: counted with fvcore, convolution and matrix-multiplication MACs multiplied by two, batch-norm / activations / softmax / LRPB-MLP reported separately | `src/get_flops_fvcore.py` reported only fvcore's plain total, whose convention is one FLOP per MAC | `--double-macs` added, implementing the stated rule (conv + linear + matmul + bmm MACs × 2) alongside the plain total |
| A10 | External road test | Zero-shot on the full RDD2022 release restricted to the Czech and China-motorcycle slices, with the D00/D10/D20 → crack, D40 → pothole, D43/D44/D50 ignore mapping | The external test was not described anywhere in the repository | `README.md`, Section 7 added; `docs/annotation_spec.md` records the mapping and its being external to the training set |
| A11 | Annotation provenance | Section 4.1: the data are **not author-collected** but "a locally prepared YOLO derivative of public sources"; Section 5 records that crack subtype and pothole severity are collapsed | `docs/annotation_spec.md` attributed the labels to an author-side campaign with **LabelImg** and to a manual "unified standard", and cited "Section 4.1 of the manuscript" for it | Section rewritten: labels inherit the upstream datasets' semantics, subtypes and severity are collapsed, and the mask-to-box / label-collapse rules are what the (unreleased) provenance manifest is meant to record |
| A12 | Documentation cross-references | Table 4 = training configuration; Table 5(a) = factorial ablation; Table 5(b) = per-component sub-ablation; Tables 6 and 7 = per-class and source-held-out results | `configs/train_config.md` cited "Table 2" for hyperparameters and "Table 3" for the ablation; `docs/detailed_metrics_protocol.md` cited "Table 4" for the detailed metrics and pointed at the wrong section of `reproducibility.md` | All references corrected in `README.md`, `configs/train_config.md` and `docs/detailed_metrics_protocol.md` |

Verified after the changes: `src/get_all_yaml_param_and_flops.py --imgsz 416` still reproduces
the manuscript's complexity table exactly (19.87 M / 24.1 G baseline, 13.25 M / 18.5 G
proposed), and `rtdetr-MSPC.yaml` measures 14.09 M, matching Table 5(a) row 2.

---

## Part B - essential differences, listed rather than "fixed"

These are places where the repository cannot be made to match the manuscript by editing it,
because the manuscript describes something the release does not contain. None of them was
papered over.

### B1. The released split is not the manuscript's split

* **Manuscript:** images are grouped by source (every original image plus all of its exported
  variants); whole groups are assigned at about 70/10/20, giving 5,238 / 748 / 1,510 after the
  untraceable families are excluded.
* **Release:** `dataset/roadrdd/split_manifest.csv` is **byte-level** duplicate-free only.
  Measured here: of 5,875 distinct `source_group` values, **895 still appear in more than one
  split**; the byte-level audit removed 379 duplicate groups of which 172 spanned two splits.
* **Consequence:** the released test split (1,697 images) is a superset of the manuscript's
  (1,510) and is not group-disjoint. Any metric recomputed from the release is therefore
  computed on a different, larger test set.
* **Handling:** documented in `README.md` (Section 3), `docs/dataset_statistics.md`,
  `docs/annotation_spec.md` and `docs/reproducibility.md` (Sections 1.1 and 2). The grouping
  key is already in the manifest, so a reader can produce a group-disjoint re-split
  deterministically — but not the *manuscript's* split, because the manuscript does not state
  the sampling seed or the exact group-to-split assignment.

### B2. The manuscript's complexity numbers are not reproduced by the profiler it names

* **Manuscript Section 3.4:** fvcore 0.1.5, one `1x3x416x416` input, convolution and
  matrix-multiplication MACs multiplied by two; Table 2 gives 18.5 G (theoretical) and 18.9 G
  (measured) at 416×416 for the proposed model.
* **Measured here at 416×416:** fvcore's plain total for the baseline is **13.23 G** and the
  Section 3.4 rule (conv + matmul MACs × 2) gives **26.33 G**. Neither is the manuscript's
  24.1 G. The ultralytics `model_info` (THOP) path gives **24.1 G** for the baseline and
  **18.5 G** for the proposed model — exactly Table 5(a) and Table 2's theoretical column.
* **Consequence:** the numbers are reproducible, but through the THOP convention
  (`src/get_all_yaml_param_and_flops.py`), not through fvcore. The likely reason is that the
  two profilers count different operator sets and that fvcore cannot account for every
  operator of the custom deformable attention.
* **Handling:** `src/get_flops_fvcore.py` keeps its docstring explicit that it is a
  diagnostic; `docs/reproducibility.md` (Section 5) states which script reproduces which
  column, and reports all three measured values so that a reader can see the difference
  rather than discover it. No number in the repository was adjusted.

### B3. The FasterNet PConv arm of Table 5(b) is not released and is not reproducible

* **Manuscript Table 5(b):** a "FasterNet PConv variant (matched replacement)" at
  **14.62 M / 20.9 G / 0.7471 mAP@0.5**, against MSPC at 14.09 M / 20.2 G / 0.7498.
* **Release:** the eight factorial configurations and five controls contain no PConv
  backbone arm, and no such configuration exists anywhere in this repository's history.
* **Measured here:** the building blocks are present (`Partial_conv3`, `Faster_Block`,
  `C2f_Faster`). Substituting `C2f_Faster` for `MSPC` in `rtdetr-MSPC.yaml` — the same width,
  depth, shortcut and I/O shape, i.e. exactly the "matched replacement" reading — gives
  **13.16 M** (fused) / 13.37 M (unfused), not 14.62 M. A sweep of every registered block
  class as the inner unit of the same `C2f` wrapper found no configuration at 14.62 M either.
* **Handling:** no configuration file was invented. The gap is recorded in
  `docs/reproducibility.md` (Section 4, "Not released") and here; the authors need to supply
  the arm's configuration for that row to be checkable.

### B4. `RepC3` versus `CSPRepLayer`

* **Manuscript Section 2** names the RT-DETR cross-scale fusion unit by its official-PyTorch
  name, `CSPRepLayer`; the manuscript itself states that "the source figure retains the RepC3
  label; implementation equivalence with CSPRepLayer requires confirmation and is not
  asserted here".
* **Release:** the fusion stage is the ultralytics module `RepC3`
  (`src/ultralytics/nn/modules/block.py`), used in every configuration file and in the
  released weights' state dicts.
* **Handling:** not renamed. The two labels denote the same stage in this release, and the
  equivalence question the manuscript leaves open is restated in `README.md` (Section 2) so a
  reader does not mistake the naming difference for a structural one.

### B5. MSPC stage mapping, and the stride-2 shortcut of Table 1

* **Manuscript Section 3.2 / Table 1:** MSPC replaces the *residual bottleneck blocks* in
  stages 2–4 of the backbone, stage 1 unchanged; Table 1 shows a C = 128 stride-2 block whose
  shortcut is average-pool stride 2 followed by a 1×1 convolution; the C = 256 and C = 512
  blocks "scale all channel groups by 2x and 4x".
* **Release:** `rtdetr-MSPC.yaml` replaces the ResNet-style `Blocks` with a YOLO-style
  `Conv` + `MSPC` backbone at widths 128 / 256 / 384 (the last stage repeated three times),
  where `MSPC` is a `C2f` CSP wrapper around `MSPConv`. `MSPConv.forward` is
  `conv4(cat[...]) + x` at stride 1 — it contains **no downsampling shortcut and no
  average-pool path**; all downsampling is done by the separate `Conv, [C, 3, 2]` entries in
  the YAML. There is no stride-2 MSPC block in the release at all.
* **Measured:** the released MSPC arm is 14.09 M (fused) / 20.2 G, which is Table 5(a) row 2 —
  so the released YAML is the configuration that produced the reported numbers.
* **Handling:** the YAML was not rewritten, because changing it would invalidate the
  complexity and accuracy tables. The manuscript's own Figure 2 caption flags the stage-by-stage
  correspondence as unresolved, and its Section 5 states that "the source labels RepC3 in
  Figure 1 and the four MSPC units in Figure 2 are not verified against the CSPRepLayer and
  stages-2-4 configuration described here. Figure 3 omits the downsampling shortcut." Those
  diffs are therefore already acknowledged in the manuscript; `README.md` (Section 2) and
  `configs/train_config.md` now describe the released structure (CSP wrapper, stages 2–4,
  full-width 3×3 + depthwise 5×5 + 7×7 tap) rather than Table 1's stride-2 variant.

### B6. Letterboxing versus the previously claimed re-export

* **Manuscript Table 4:** input 416×416, "letterbox, no distortion".
* **Release documentation (before):** `docs/reproducibility.md` claimed that *no* letterboxing
  is applied because the released images are already square 416×416 re-exports of the original
  frames — which additionally implies a resize, not a letterbox.
* **Measured:** a random sample of 400 of the 8,695 released images is 416×416 in every case,
  so ultralytics' letterbox is a no-op on the release (it adds no padding band). The original
  aspect ratio of the field frames is not recoverable, because the export stripped the EXIF
  metadata.
* **Handling:** `docs/reproducibility.md` (Section 3) now states both facts — the manuscript's
  letterbox setting and the measurement that makes it a no-op on this release — instead of
  claiming there is no letterbox step.

---

## Part C - ambiguities left at the released values

Recorded so that a reader can tell a deliberate choice from an oversight. None of these is a
contradiction; the manuscript is simply silent.

1. **Mosaic probability.** Table 4 names mosaic and its closing schedule (last 15 epochs) but
   not its probability. The released script keeps `mosaic=0.2`, the value the earlier
   manuscript documented for the reported runs.
2. **Scaling and translation jitter.** Table 4 lists H-flip, HSV jitter and mosaic. The
   released script additionally applies `scale=0.2` and `translate=0.1`. They are retained, so
   that the released recipe stays identical to the one that produced the reported numbers.
3. **`patience` and AMP.** `src/train.py` runs without early stopping (`patience=0`) and
   `default.yaml` has `amp: False`. Table 4 does not discuss either.
4. **MCAF depth.** Table 3 states "the repeated number N of RepNBottleneck units is 2". The
   released configurations pass `MCAF, [256, 128, 64, 1]`, i.e. `c5 = 1` to each of the two
   chained `RepNCSP` stages, so the block contains two RepNBottleneck-bearing stages. That is
   the reading under which the manuscript's sentence holds; the alternative reading (two
   RepNBottleneck units inside each stage) is not what the released files do.
5. **Per-class AP and the qualitative figures.** The manuscript reports per-class AP (crack
   0.736, pothole 0.784), a rounding artefact in the mAP average, and Figure 8 as illustrative
   only. These live in the manuscript and in `docs/detailed_metrics_protocol.md`; the
   repository ships no figures and makes no claim about them.

---

## Part D - how each measurement was obtained

All commands are run from the repository root unless stated otherwise, with the environment of
`environment.yml`.

```bash
# Parameter and GFLOPs values (post-fuse, THOP 2x MACs) at 416x416
cd src && python get_all_yaml_param_and_flops.py --imgsz 416

# The same values at 640x640, to check the (416/640)^2 scaling equivalence
cd src && python get_all_yaml_param_and_flops.py --imgsz 640

# fvcore plain total and the Section 3.4 counting rule (conv + matmul MACs x 2)
cd src && python get_flops_fvcore.py --imgsz 416 --double-macs

# Pre-fuse versus post-fuse parameter counts (one configuration per process)
cd src && python get_all_yaml_param_and_flops.py --imgsz 416 --cfg <path-to-yaml>
```

* **Released image count and dimensions:** walk `dataset/roadrdd/images/`, count the image
  files (8,695) and sample 400 of them for their size (416×416 in all 400 cases).
* **Duplicate and group counts:** `dataset/roadrdd/audit/dedup_log.csv` (379 groups, 172
  cross-split) and `dataset/roadrdd/split_manifest.csv` (895 of 5,875 `source_group` values in
  more than one split).
* **Box counts:** the released labels give 8,455 crack and 8,358 pothole (16,813) over 8,695
  images; the manuscript's main-set figures (7,642 / 6,918 = 14,560 over 7,496) are quoted
  from Tables 3(a) and 3(b), because the exclusion list is not recorded in the release.
* **RDD2022 counts and mapping:** quoted from the manuscript and Supplementary File S1; the
  external dataset is not redistributed here.

Note on the two parameter counts: the training/validation model reports the **unfused** count
(14,304,296 for the MSPC arm) and inference re-parameterisation reduces it (14,093,080). The
manuscript's 14.09 M is the post-fuse value, which is why the tables here quote the fused
count. `get_all_yaml_param_and_flops.py` reports both.
