# Reproducibility Notes

This document records exactly what the repository releases, what it does **not** release, and
the limitations that a reader should be aware of. It is meant to be read together with
`detailed_metrics_protocol.md`, `dataset/roadrdd/audit/dataset_audit.md` and
`../configs/train_config.md`.

## 1. Dataset provenance

`dataset/roadrdd/` is a mixed road-defect detection set assembled from field imagery plus
several public crack-detection repositories. The file-name prefixes indicate the following
composition, counted before and after byte-level deduplication:

| Grouping key in file name | Images (raw) | Images (released) | Origin |
|---|---|---|---|
| `CRACK500_*` and `2016xxxx_*` sequences | 4,129 | 3,990 | CRACK500 |
| `po_*` | 1,927 | 1,927 | field imagery, pothole class |
| `img_*` / `IMG_*` | 783 | 767 | field imagery, mixed |
| `DeepCrack_*` | 485 | 382 | DeepCrack |
| `GAPS384_*` | 433 | 433 | GAPS384 |
| `CFD_*` / `forest_*` | 286 | 240 | CrackForest (CFD) |
| `noncrack_*` | 244 | 244 | background-only frames |
| `CrackTree*` | 187 | 112 | CrackTree |
| `Eugen_Muller_*` | 47 | 47 | Eugen Müller, tunnel lining |
| `Volker_*` | 2 | 2 | Volker, concrete facades |
| bare numeric stems | 551 | 551 | field imagery, pothole class |
| **Total** | **9,074** | **8,695** | |

The two columns differ by exactly the 379 byte-identical copies removed in the audit
(Section 2), so the table is consistent with `audit/dataset_audit.md` in both stages.

This table is derived from file-name prefixes only. It is provided so that readers can see
that the set is a mixture of public datasets and field imagery rather than a single
self-collected corpus, and it can be recomputed directly from `dataset/roadrdd/images/` and
`dataset/roadrdd/MANIFEST.sha256`. Table 1a of the manuscript reports this same prefix-based
decomposition.

### Origin of the groups that carry no public dataset name

Two of the rows above were previously recorded as "residual, to be confirmed by the
authors". They are now resolved:

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
  Dataset (Virginia Tech, 2021). The manuscript attributes the two groups accordingly and
  cites both sources.
* **The bare numeric stems are a pothole-class group of unrecorded provenance.** All 2,252
  annotated instances of this group are class 1 (pothole), at a median normalised box size
  of 0.142 x 0.106, and the images show paved surfaces with litter rather than road crack
  frames. The upstream release of this group is not named anywhere in the repository, and it
  cannot be recovered from the released files: the dataset export stripped all EXIF metadata
  and resampled every image to 416x416. Until the authors name it, this group should not be
  described as a public dataset, and it should not be described as a road-crack acquisition
  either.


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

Consequences:

* No claim of a fully leakage-free split should be made for this release.
* A stricter split would assign every `source_group` to exactly one split. The grouping key
  is already provided in `split_manifest.csv`, so such a re-split can be produced
  deterministically.

## 3. Model and resolution

* Decoder: `RTDETRDecoder [nc, 256, 300, 4, 8, 3]` — 300 object queries and **3 decoder
  layers**, i.e. the official RT-DETR-R18 setting.
* Classes: `nc: 2` (crack, pothole). The configuration files set `nc: 2` directly; the value
  is also overridden by `configs/roadrdd.yaml` at training time.
* Input resolution: 416×416. The released images are already square 416×416 re-exports, so no
  letterboxing is applied at the training resolution; the 1,920×1,080 source frames of the
  field acquisitions were resampled to that size during the dataset export. A scan of the
  border rows of a 300-image sample found no constant or black band, i.e. the release carries
  no letterbox padding.
* The LRPB-AIFI encoder originally pinned its position-bias tables to a 20×20 token grid, which is
  the S5 size at 640×640 and is incompatible with the 13×13 grid produced at 416×416. The
  tables are now generated at runtime from the actual feature-map size
  (`LRPB_Attention._get_bias_tables`), so both resolutions and non-square inputs work. Behaviour
  at 640×640 is unchanged; `src/tests/test_lrpb_dynamic_size.py` covers 416×416, 640×640 and
  non-square inputs.

## 4. Ablation control configurations

The eight configurations of the ablation table are the full 2^3 factorial design over
MSPC, MCAF and LRPB-AIFI, and live in `src/ultralytics/cfg/models/rt-detr/`. Five further
configurations isolate the two effects that the ablation table can only report jointly, and
live in the `controls/` subdirectory of the same folder:

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

## 5. Measurement protocols

* **GFLOPs** — the ultralytics `model_info` (THOP) convention of **two floating-point
  operations per multiply–accumulate**, batch size 1, on the fused model. Run
  `src/get_all_yaml_param_and_flops.py --imgsz <N>`; it profiles every ablation configuration
  at the requested resolution (default 640×640) and prints both parameter counts and GFLOPs.
  The figures quoted in the manuscript are profiled at 640×640 and converted to the target
  resolution by the square of the resolution ratio (416×416 for the main results), so the
  relative reductions are resolution-invariant. This is the convention used by the RT-DETR
  and YOLO literature.
* **`src/get_flops_fvcore.py` is a diagnostic only.** fvcore's `FlopCountAnalysis` counts one
  multiply–accumulate as one FLOP, so its absolute values are roughly half those of the
  `model_info` convention, and it reports per-operator detail rather than the headline number.
  It is included so that readers can inspect the operator-level breakdown; its totals are
  **not** the values reported in the manuscript.
* **FPS** — TensorRT 8.6 FP16, batch size 1, end-to-end (image decode, letterbox resize,
  normalisation, engine forward, post-processing). 200 warm-up iterations, then 5 runs of 300
  measured iterations, `cudaDeviceSynchronize` around each timed segment, GPU clock locked
  with `nvidia-smi -lgc`. Run `src/get_FPS.py`.
* **Detailed module-attribution metrics** — background-only patches for FP/image, the
  crack-width bins and the remaining per-class, per-scale and strict-localisation metrics are
  specified in `detailed_metrics_protocol.md`, together with the scripts that produce them.

## 6. What is not released

* **Trained weights, training logs and prediction files.** The `.gitignore` excludes
  `*.pt`, `*.pth` and `runs/`, so no checkpoints or logs are distributed with this repository.
* **Per-run raw outputs** of the five random-seed experiments.
* **The acquisition geometry** consumed by `configs/width_calibration.yaml`, and **a
  per-instance crack-width table**. These are the only two items whose absence prevents a
  reported number from being recomputed, and both are physical inputs that the released
  images cannot supply: the dataset export stripped the EXIF metadata. See Sections 2.2 and
  2.3 of `detailed_metrics_protocol.md`.

The scripts required to regenerate all of these from the released dataset and configurations
are included, but the artefacts themselves are not part of this release.

## 7. Environment

`environment.yml` pins the exact package versions used for the experiments (PyTorch 2.4.0,
CUDA 11.8, TensorRT 8.6, fvcore). `src/requirements.txt` is the lighter pip-only equivalent.
Both also list `timm`, `efficientnet_pytorch`, `einops`, `dill`, `PyWavelets` and `seaborn`:
these are imported unconditionally by the vendored third-party modules under
`src/ultralytics/nn/`, so they are required merely to `import ultralytics`, even though the
paper's three modules do not use them.
