# Reproducibility guide

This repository is a **code + reproducible-pipeline** release. Derived benchmark
images are reconstructed from official public sources rather than
redistributed, because some upstream collections are licensed for non-commercial
research only.

## 1. Environment

- Paper hardware: one NVIDIA RTX A6000 (48 GB); edge benchmark on Jetson Orin
  Nano 8 GB (JetPack 6.0, TensorRT 8.6).
- Software: PyTorch 2.1 / CUDA 11.8. A CPU-only install runs the code and the
  parameter/FLOP profiler, but not the reported GPU FPS.

```bash
cd src
pip install -r requirements.txt
# or: conda env create -f ../environment.yml
```

## 2. Build the in-domain benchmark (RoadRDD)

1. Download each source from its official location (versions, URLs, licenses in
   `dataset/roadrdd/SOURCES.md`) and place the raw aggregate under `raw/` as
   `images/<split>/...` with sibling `labels/<split>/...` YOLO files.
2. Byte-level audit / exact-copy collapse:

   ```bash
   python tools/audit_dataset.py --root raw --write
   ```
3. Deterministic group-disjoint re-split (fixed seed = 2024):

   ```bash
   python tools/group_disjoint_resplit.py --in raw --out dataset/roadrdd
   python tools/group_disjoint_resplit.py --in raw --out dataset/roadrdd --apply
   ```

Outputs: `MANIFEST.sha256`, `split_manifest.csv`, `audit/resplit_report.md`,
`audit/byte_collapse_log.csv`, `audit/group_assignment.csv`.

Expected (Section 4.1.1): **8,426** images - 5,898 / 842 / 1,686; 16,312 boxes
(8,498 crack / 7,814 pothole); 15 background-only frames; zero byte-identical
files and zero source photographs shared across splits.

## 3. Build the external RDD2022 partition

Download RDD2022 from <https://github.com/sekilab/RoadDamageDetector>; select
the four-country 10,000-image partition (1,750 / 250 / 500 per country; fixed
released seed); merge the three crack categories to `crack`, map potholes to
`pothole`, ignore repair/blur markings. Expected: 20,373 boxes.

## 4. Train and evaluate

```bash
python train.py --data ../configs/roadrdd.yaml --imgsz 416 --batch 16 --epochs 240 --seed 42
# repeat for seeds 42, 123, 2024, 7, 31415 and average
python train.py --cfg ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml
python val.py --weights runs/roadrdd/train/<run>/weights/best.pt --split test --imgsz 416
```

Recipe: AdamW, 240 epochs, batch 16, lr 1e-4 -> 1e-6 cosine, weight decay 1e-4,
~5-epoch warm-up, gradient clip 0.1, EMA 0.9999, loss gains cls/L1/GIoU =
1/5/2; horizontal flip p = 0.5, HSV (0.015, 0.7, 0.4), mosaic p = 0.2 disabled
for the last 15 epochs, scale/translation 0.2/0.1.

## 5. Complexity and speed

```bash
python tools/profile_model.py                 # params + FLOPs (+ CPU FPS)
python tools/profile_model.py --device cuda   # reproduce A6000 FPS
python tools/profile_model.py --all           # every rt-detr configuration
```

Logs go to `logs/profiler/`. Parameters and GFLOPs are device-independent; FPS
is measured on the target device (batch 1, 200 warm-up, median over 1,000
frames, including letterbox, normalization, forward and post-processing).

## 6. Fine-grained metrics

Per-class AP, AP_small / AP_medium / AP_large (COCO area thresholds 32² and
96²) and AP75 are produced by the standard validator on the test split
(1,686 images; scale mix 18% / 37% / 45%). Background behavior:

- FP per background patch: four fixed-grid 128-pixel candidate patches per test
  image, kept only if they do not intersect any annotation dilated by 16 px;
  detections above 0.25 are counted (5,214 patches over the 1,686 test images).
- FP per image on the 15 background-only frames.

## 7. Design-choice controls (Table 5)

Structural controls ship as whole configurations under
`src/ultralytics/cfg/models/rt-detr/controls/`: `RepNCSPELAN4-noCAA` (neck block
without attention), `CAA-on-RepC3` (attention on the original neck), and the
`PE-none / PE-sinusoidal / PE-static` positional-encoding alternatives.

The remaining within-module sweeps change a single default constant; edit, then
train the full model exactly as in Section 4.

| Control | File / line | Change |
|---|---|---|
| CAA kernel K = 7 or 15 | `nn/extra_modules/attention.py`, `CAA.__init__` | `h_kernel_size` / `v_kernel_size` 11 -> 7 or 15 |
| LRPB hidden D/32 | `nn/extra_modules/transformer.py`, `LRPB_Attention.__init__` (`self.pos = LRPB(self.dim // 4, ...)`) | `self.dim // 4` -> `self.dim // 8` |
| MSPC kernel pair 3×3/5×5 | `nn/extra_modules/block.py`, `MSPConv.__init__` | `Conv(..., k=5)` -> `k=3` and `k=7` -> `k=5` |
| Fixed-quarter PConv | `nn/extra_modules/block.py`, `MSPConv` | replace the cascaded 1/2–1/4 splits with a FasterNet-style fixed one-quarter partial convolution |

## 8. Tests

```bash
pytest tests/test_lrpb_dynamic_size.py -v
```

Verifies LRPB tables for 416×416, 640×640 and non-square inputs.

## 9. Scope

Per the Data Availability Statement, the repository provides the source manifest
(per-source counts, versions, licenses), the byte-level audit and
group-disjoint re-splitting scripts, the training and evaluation code and the
profiler logs. RDD2022 comes from its official release; in-domain images are
reconstructed from the sources in `SOURCES.md`.
