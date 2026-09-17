# Reproducibility Notes

This document records exactly what the repository releases, what it does **not** release, and
the limitations that a reader should be aware of. It is meant to be read together with
`dataset/roadrdd/audit/dataset_audit.md` and `../configs/train_config.md`.

## 1. Dataset provenance

`dataset/roadrdd/` is a mixed road-defect detection set assembled from a vehicle-mounted
camera plus several public crack-detection repositories. The file-name prefixes indicate the
following composition of the released 8,695 images:

| Grouping key in file name | Images | Notes |
|---|---|---|
| `CRACK500_*` and `2016xxxx_*` sequences | 4,129 | CRACK500-style imagery |
| `po_*` | 1,927 | pothole-origin imagery |
| `img_*` / `IMG_*` | 783 | mixed origin |
| `DeepCrack_*` | 485 | DeepCrack |
| `GAPS384_*` | 433 | GAPS384 |
| `CFD_*` / `forest_*` | 286 | CrackForest (CFD) |
| `noncrack_*` | 244 | background-only frames |
| `CrackTree*` | 187 | CrackTree |
| other prefixes / bare numeric stems | 221 | residual, to be confirmed by the authors |

This table is derived from file-name prefixes only. It is provided so that readers can see
that the set is a mixture of public datasets and field imagery rather than a single
self-collected corpus, and it should be replaced by an authoritative per-source breakdown
before publication.

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
* Input resolution: 416×416, letterbox-padded from 1920×1080 source frames.
* The LRPB encoder originally pinned its position-bias tables to a 20×20 token grid, which is
  the S5 size at 640×640 and is incompatible with the 13×13 grid produced at 416×416. The
  tables are now generated at runtime from the actual feature-map size
  (`LRPB_Attention._get_bias_tables`), so both resolutions and non-square inputs work. Behaviour
  at 640×640 is unchanged; `src/tests/test_lrpb_dynamic_size.py` covers 416×416, 640×640 and
  non-square inputs.

## 4. Measurement protocols

* **GFLOPs** — fvcore `FlopCountAnalysis`, batch size 1, at the stated resolution, with no
  area-scaling extrapolation. Run `src/get_flops_fvcore.py`; it profiles every ablation
  configuration at 416×416 and 640×640 and writes an operator-level breakdown CSV.
* **FPS** — TensorRT 8.6 FP16, batch size 1, end-to-end (image decode, letterbox resize,
  normalisation, engine forward, post-processing). 200 warm-up iterations, then 5 runs of 300
  measured iterations, `cudaDeviceSynchronize` around each timed segment, GPU clock locked
  with `nvidia-smi -lgc`. Run `src/get_FPS.py`.

## 5. What is not released

* **Trained weights, training logs and prediction files.** The `.gitignore` excludes
  `*.pt`, `*.pth` and `runs/`, so no checkpoints or logs are distributed with this repository.
* **Per-run raw outputs** of the five random-seed experiments.

The scripts required to regenerate all of these from the released dataset and configurations
are included, but the artefacts themselves are not part of this release.

## 6. Environment

`environment.yml` pins the exact package versions used for the experiments (PyTorch 2.4.0,
CUDA 11.8, TensorRT 8.6, fvcore). `src/requirements.txt` is the lighter pip-only equivalent.
