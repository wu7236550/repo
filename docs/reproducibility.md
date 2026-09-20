# Reproducibility Notes

This document records exactly what the repository releases, what it does **not** release, and
the limitations that a reader should be aware of. It is meant to be read together with
`manuscript_alignment.md`, `detailed_metrics_protocol.md`,
`dataset/roadrdd/audit/dataset_audit.md` and `../configs/train_config.md`.

## 1. Dataset provenance

`dataset/roadrdd/` is a mixed-surface defect detection set assembled from public crack and
pothole sources. The manuscript (Section 4.1) states that the data are **not author-collected**
but a locally prepared YOLO derivative of those public sources. The file-name prefixes indicate
the following composition, counted before and after byte-level deduplication:

| Grouping key in file name | Images (raw) | Images (released) | Origin |
|---|---|---|---|
| `CRACK500_*` and `2016xxxx_*` sequences | 4,129 | 3,990 | CRACK500 |
| `po_*` | 1,927 | 1,927 | Annotated Potholes with Severity Levels (pothole class) |
| `img_*` / `IMG_*` | 783 | 767 | untraceable family, mixed |
| `DeepCrack_*` | 485 | 382 | DeepCrack |
| `GAPS384_*` | 433 | 433 | GAPS384 |
| `CFD_*` / `forest_*` | 286 | 240 | CrackForest (CFD) |
| `noncrack_*` | 244 | 244 | untraceable family, background-only frames |
| `CrackTree*` | 187 | 112 | CrackTree |
| `Eugen_Muller_*` | 47 | 47 | tunnel-lining concrete cracks |
| `Volker_*` | 2 | 2 | concrete-facade cracks |
| bare numeric stems | 551 | 551 | untraceable family, pothole class |
| **Total** | **9,074** | **8,695** | |

The two columns differ by exactly the 379 byte-identical copies removed in the audit
(Section 2), so the table is consistent with `audit/dataset_audit.md` in both stages.

This table is derived from file-name prefixes only. It can be recomputed directly from
`dataset/roadrdd/images/` and `dataset/roadrdd/MANIFEST.sha256`.

### 1.1 Relationship to the manuscript's main set

The manuscript groups the same eleven prefixes into three components and reports a **main set**
of 7,496 images after excluding the untraceable families (Table 3(a)):

| Component | Orig. exports | Excl. | Main | Train | Val | Test | Boxes |
|---|---|---|---|---|---|---|---|
| CCCD-like crack families (`CRACK500_*`, `2016xxxx_*`, `DeepCrack_*`, `GAPS384_*`, `CFD_*`, `forest_*`, `CrackTree*`, `Eugen_Muller_*`, `Volker_*`) | 5,569 | 0 | 5,569 | 3,898 | 557 | 1,114 | 7,642 |
| `po_*` pothole family | 1,927 | 0 | 1,927 | 1,340 | 191 | 396 | 6,918 |
| untraceable name families (`img_*`/`IMG_*`, `noncrack_*`, bare numeric stems) | 1,578 | 1,578 | 0 | — | — | — | 0 |
| **Total main set** | **9,074** | **1,578** | **7,496** | **5,238** | **748** | **1,510** | **14,560** |

The manuscript's per-class box counts on that main set are 7,642 crack and 6,918 pothole
(Table 3(b)). The released derivative instead keeps all 8,695 images, which is why its total of
16,813 boxes (8,455 crack / 8,358 pothole) is larger than the 14,560 of the main set: the extra
1,562 released images are the untraceable exports that the manuscript excludes.

**The two splits are not the same split.** The manuscript assigns whole source groups
(every original image plus all of its exported variants) to train/validation/test at about
70/10/20, whereas the released `split_manifest.csv` is byte-level duplicate-free but assigns
images individually. Section 2 below quantifies the resulting difference, and
`manuscript_alignment.md` records what would be needed to regenerate the manuscript split.

### 1.2 Origin of the groups that carry no public dataset name

Two of the rows above were previously recorded as "residual, to be confirmed by the authors".
They are now resolved:

* **`Eugen_Muller_*` and `Volker_*` are published datasets, not field imagery.** Both names
  are the prefixes of crack datasets that circulate in the crack-segmentation benchmark
  collections, where `Eugen Müller` is the tunnel-lining set and `Volker` is the concrete
  facade set. Three independent properties of the released files agree with that
  identification: the prefix is the dataset name itself; the naming convention
  `<contributor>_<image>_<xmin>_<ymin>_<xmax>_<ymax>` is that of a crop with its source
  rectangle, and every instance is a crack (class 0); and the images show the matching
  surface, i.e. light, thin-cracked tunnel-lining concrete and a vertical crack on a
  concrete facade. They must therefore be attributed as public data and not counted as
  self-collected acquisitions. The released counts (47 and 2) are smaller than the sizes
  reported for those collections because a group here is defined by file-name prefix after
  byte-level deduplication, and because the different releases of each dataset differ in
  size.

  The identification is corroborated externally. Independent benchmark tables that use these
  two names report the same surfaces and comparable sizes — `Volker`, 990 images of concrete
  facades, and `Eugen Muller`, 55 images of concrete tunnel lining — and the
  crack-segmentation literature cites them to Pak and Kim (2021) and to Ham et al. (2021)
  respectively. Both datasets are redistributed in public crack-segmentation collections such
  as CrackSeg9k (Kulkarni et al., 2022, arXiv:2208.13054) and the Concrete Crack Conglomerate
  Dataset (Virginia Tech, 2021). The manuscript attributes the two groups accordingly (its
  references [40] and [41]) and cites both sources. In the manuscript they belong to the
  CCCD-like crack component of Table 3(a) above.
* **The bare numeric stems are a pothole-class group of unrecorded provenance.** All 2,252
  annotated instances of this group are class 1 (pothole), at a median normalised box size
  of 0.142 x 0.106, and the images show paved surfaces with litter rather than road crack
  frames. The upstream release of this group is not named anywhere in the repository, and it
  cannot be recovered from the released files: the dataset export stripped all EXIF metadata
  and resampled every image to 416x416. This is why the manuscript classifies the bare
  numeric stems, together with `img_*`/`IMG_*` and `noncrack_*`, as an **untraceable name
  family** and excludes them from the main experiment (1,578 exports in Table 3(a)). They
  must not be described as a public dataset, and they must not be described as a verified
  road-crack acquisition either.

## 2. Dataset audit

The release applies a **byte-level** audit:

* SHA-256 is computed for every image; identical hashes are treated as duplicates.
* 379 duplicate groups were found among the 9,074 pre-audit images; 379 redundant copies were
  removed, leaving **8,695** unique images.
* 172 of those groups spanned two splits (train∩test 90, train∩val 69, val∩test 13). For every
  such group the copy was retained in the higher-priority split (train > val > test) so that
  the test and validation sets no longer contain images that also appear in training.
* The per-image hash list is `dataset/roadrdd/MANIFEST.sha256`; the group-level record is
  `audit/dedup_log.csv`; the list of removed files is `audit/dedup_plan.txt`.

### Residual limitation (important)

Deduplication is **byte-level only**. Grouping images by `source_group` (file name with the
Roboflow `.rf.<hash>` suffix removed) shows that of 5,875 distinct source groups, **895 still
have representatives in more than one split**. In other words, different augmentations or
exports of the *same* source photograph can still be found in both training and test even
though no two files are byte-identical.

The manuscript avoids this by splitting **by source group** rather than by image, and it
reports the consequence explicitly (Section 4.1): no claim of a fully leakage-free split
should be made, and the authors additionally flag a residual same-pothole multi-view risk that
filename grouping cannot merge (about 86 of the 717 pothole base identifiers, about 136 test
images) together with a sensitivity analysis that removes those clusters.

Consequences for this release:

* No claim of a fully leakage-free split should be made for this release.
* The 895 multi-split groups make the released split **not** the manuscript split. Readers who
  want the manuscript protocol must re-split by `source_group`: the grouping key is already
  provided in `split_manifest.csv`, so such a re-split can be produced deterministically once
  the untraceable families are excluded. The released manifest does not record which images
  belong to the untraceable families, so that exclusion list has to be derived from the file
  names (Section 1 above).

## 3. Model and resolution

* Decoder: `RTDETRDecoder [nc, 256, 300, 4, 8, 3]` — 300 object queries and **3 decoder
  layers**, i.e. the official RT-DETR-R18 setting.
* Classes: `nc: 2` (crack, pothole). The configuration files set `nc: 2` directly; the value
  is also overridden by `configs/roadrdd.yaml` at training time.
* Input resolution: 416×416, which the manuscript Table 4 describes as "letterbox, no
  distortion". The released files are already square: a random sample of 400 released images
  is 416×416 in every case, so the standard ultralytics letterbox is a no-op on them (it adds
  no padding band). The acquisition aspect ratio of the original field frames is not
  recoverable from the release, because the export stripped the EXIF metadata and resampled
  every image to 416×416.
* The LRPB-AIFI encoder originally pinned its position-bias tables to a 20×20 token grid, which is
  the S5 size at 640×640 and is incompatible with the 13×13 grid produced at 416×416. The
  tables are now generated at runtime from the actual feature-map size
  (`LRPB_Attention._get_bias_tables`), so both resolutions and non-square inputs work. Behaviour
  at 640×640 is unchanged; `src/tests/test_lrpb_dynamic_size.py` covers 416×416, 640×640 and
  non-square inputs.

## 4. Ablation and control configurations

The eight configurations of the factorial ablation table (manuscript Table 5(a)) are the full
2^3 design over MSPC, MCAF and LRPB-AIFI, and live in `src/ultralytics/cfg/models/rt-detr/`.
Five further configurations isolate the two effects that the ablation table can only report
jointly, and live in the `controls/` subdirectory of the same folder:

| Configuration | Isolates |
|---|---|
| `controls/rtdetr-control-CAA-on-RepC3.yaml` | the context anchor attention alone: the original RepC3 fusion block with CAA inserted after it |
| `controls/rtdetr-control-RepNCSPELAN4-noCAA.yaml` | the block replacement alone: RepNCSPELAN4 in the fusion stage without CAA |
| `controls/rtdetr-control-PE-sinusoidal.yaml` | 2D sin-cos absolute positional encoding (the RT-DETR baseline scheme) |
| `controls/rtdetr-control-PE-none.yaml` | no positional encoding |
| `controls/rtdetr-control-PE-static.yaml` | a static relative-position-bias lookup table on the attention logits |

The two CAA controls decompose `rtdetr-MCAF.yaml`, which applies both changes together. The
decomposition is exact at the parameter level: with four fusion blocks, the block
replacement contributes -1,392,640 parameters and the CAA contributes +550,912, and
19,874,328 - 1,392,640 + 550,912 = 19,032,600, which is the parameter count of
`rtdetr-MCAF.yaml`.

The positional-encoding group is provided by the `PE_AIFI` module
(`src/ultralytics/nn/extra_modules/transformer.py`), whose `pos_mode` argument selects the
scheme; the fourth arm of that group, the learnable relative position bias, is the released
`rtdetr-LRPB-AIFI.yaml`, which uses `LRPB_AIFI`. The three `controls/rtdetr-control-PE-*.yaml`
files differ only in that argument, so the four arms can be compared directly. At the 416×416
training resolution the S5 grid is 13×13, so the static table has 8 × 25 × 25 = 5,000
entries; that is why the static arm has exactly 5,000 parameters more than the two
parameter-free arms, and why the LRPB arm has 824 parameters (0.8 k) more than the baseline.

`PE_AIFI(pos_mode='sinusoidal')` reproduces the baseline `AIFI` exactly: with the same
weights loaded into both and the same input, the outputs are bit-identical, so the
sinusoidal arm is the baseline mechanism rather than a re-implementation of it.

```
cd src
python get_all_yaml_param_and_flops.py --imgsz 416
python get_all_yaml_param_and_flops.py --imgsz 416 --cfg-dir ultralytics/cfg/models/rt-detr/controls
```

The first command profiles the eight ablation configurations and the second the five
controls. The default `--cfg-dir` does not recurse, so the two sets stay separate.

> **Not released.** Table 5(b) of the manuscript also reports a **FasterNet PConv** backbone
> arm (14.62 M / 20.9 G / 0.7471 mAP@0.5) as a matched replacement for MSPC. No configuration
> file for that arm is part of this release. The building blocks it would use are present
> (`Partial_conv3`, `Faster_Block`, `C2f_Faster` in `src/ultralytics/nn/extra_modules/block.py`),
> but substituting `C2f_Faster` for `MSPC` in `rtdetr-MSPC.yaml` — the same width, depth,
> shortcut and I/O shape — measures 13.16 M, not 14.62 M, so that arm cannot be reproduced
> from the released files. See `manuscript_alignment.md`, item A8.

## 5. Measurement protocols

* **Parameters and GFLOPs (the manuscript tables).** The numbers reported in Tables 2 and
  5(a) are reproduced by `src/get_all_yaml_param_and_flops.py --imgsz 416`: the ultralytics
  `model_info` (THOP) convention of **two floating-point operations per multiply–accumulate**,
  batch size 1, on the **fused** model, where "fused" means the inference-time
  re-parameterisation that folds RepConv and RepC3 into single convolutions. At 416×416 the
  script reports 19.87 M / 24.1 G for the RT-DETR-R18 baseline and 13.25 M / 18.5 G for the
  proposed model, i.e. the values in the manuscript. Because both the parameters and the
  FLOPs of a convolution scale with the square of the resolution, profiling at 640×640 and
  multiplying by (416/640)² = 0.4225 gives the same numbers as profiling directly at 416×416;
  `--imgsz 416` and `--imgsz 640` are therefore consistent, and the relative reductions are
  resolution-invariant.
* **`src/get_flops_fvcore.py` is a diagnostic, not the source of the headline numbers.**
  fvcore's `FlopCountAnalysis` counts one multiply–accumulate as one FLOP, so its totals are
  roughly half of the THOP convention, and it reports the per-operator breakdown. Its
  `--double-macs` flag implements the counting rule stated in manuscript Section 3.4
  (convolution and matrix-multiplication MACs multiplied by two, with batch-norm, activations,
  softmax and the LRPB MLP reported separately). Measured on the baseline at 416×416, fvcore
  gives 13.23 G as a plain total and 26.33 G under the ×2 rule, against 24.1 G from the THOP
  path. The two profilers count different operator sets and fvcore cannot see every operator
  of the custom deformable attention, so the fvcore figures are reported only for the
  operator-level breakdown. See `manuscript_alignment.md`, item A7.
* **FPS** — the manuscript measures one NVIDIA RTX A6000, batch size 1, after 200 warm-up
  iterations, with no NMS term for the RT-DETR family (its native decoder is NMS-free).
  `src/get_FPS.py` performs the end-to-end timing (image decode, letterbox resize,
  normalisation, forward, post-processing). A separate laboratory Jetson Orin NX
  (TensorRT FP16) benchmark is reported in the manuscript and is **not** a vehicle-grade
  validation.
* **Detailed module-attribution metrics** — background-only patches for FP/image, the
  crack-width bins and the remaining per-class, per-scale and strict-localisation metrics are
  specified in `detailed_metrics_protocol.md`, together with the scripts that produce them.

## 6. What is not released

* **Trained weights, training logs and prediction files.** The `.gitignore` excludes
  `*.pt`, `*.pth` and `runs/`, so no checkpoints or logs are distributed with this repository.
  The manuscript's Tables 3(a), 4.2 and 6 are described there as reconstructed illustrations,
  and the per-seed logs, checkpoint hashes and raw predictions remain to be supplied.
* **The group-disjoint split manifest** (`split_manifest_group_disjoint.csv`, 7,496 rows) and
  the provenance manifest that the manuscript's Supplementary File S1 specifies. The released
  manifest is the byte-level one described in Section 2.
* **The FasterNet PConv matched-replacement configuration** of Table 5(b) (Section 4 above).
* **The acquisition geometry** consumed by `configs/width_calibration.yaml`, and **a
  per-instance crack-width table**. These are the only two items whose absence prevents a
  reported number from being recomputed, and both are physical inputs that the released
  images cannot supply: the dataset export stripped the EXIF metadata. See Sections 2.2 and
  2.3 of `detailed_metrics_protocol.md`.

The scripts required to regenerate all of these from the released dataset and configurations
are included, but the artefacts themselves are not part of this release.

## 7. Environment

The manuscript's experiments were run on a single **NVIDIA RTX A6000 (48 GB)** with
**PyTorch 2.1** and **CUDA 11.8** (Section 4.2). `environment.yml` pins the environment used
for the experiments; `src/requirements.txt` is the lighter pip-only equivalent.
Both also list `timm`, `efficientnet_pytorch`, `einops`, `dill`, `PyWavelets` and `seaborn`:
these are imported unconditionally by the vendored third-party modules under
`src/ultralytics/nn/`, so they are required merely to `import ultralytics`, even though the
paper's three modules do not use them.
