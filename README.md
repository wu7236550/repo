# Road Surface Defect Detection Based on a Lightweight RT-DETR

Code and dataset repository for the manuscript:

> **Road Surface Defect Detection Based on a Lightweight RT-DETR with Multi-Scale Partial
> Convolution and Context-Aware Position Modeling** (submitted to *Sensors*, MDPI)

The repository releases the complete source code, the audited dataset, all ablation
configurations and the evaluation scripts needed to reproduce the efficiency and accuracy
figures reported in the paper.

---

## 1. What is in this repository

| Path | Contents |
|------|----------|
| `src/` | Training, validation, inference and profiling code. The three proposed modules live in `src/ultralytics/nn/extra_modules/`. |
| `src/ultralytics/cfg/models/rt-detr/` | The **eight** ablation configurations of Table 3 (baseline, three single modules, three two-module combinations, and the full model). |
| `configs/` | `roadrdd.yaml` dataset definition and `train_config.md` hyperparameters. |
| `dataset/roadrdd/` | The audited road-defect dataset with YOLO-format annotations, a per-image SHA-256 manifest and a split manifest. |
| `dataset/roadrdd/audit/` | The byte-level duplicate audit log, the deduplication plan and the dataset audit report. |
| `docs/` | Annotation specification, dataset statistics and reproducibility notes. |

## 2. Model

Three scenario-adapted modifications of RT-DETR-R18:

1. **MSPC** — lightweight backbone replacing the residual blocks; multi-scale partial
   convolution (3×3/5×5/7×7 kernels over a 50 % channel split) combined with a
   cross-stage-partial structure.
2. **MCAF** — fusion module replacing RepC3; the re-parameterised
   RepNCSPELAN4 block of YOLOv9 followed by context anchor attention (CAA) from PKINet,
   whose serial `AvgPool → Conv1×1 → DConv_h → DConv_v → Conv1×1 → Sigmoid` path
   strengthens slender, continuous crack features.
3. **LRPB-AIFI** — the AIFI encoder is replaced by a learnable continuous relative position
   bias (LRPB): a small MLP maps relative coordinate offsets (Δx, Δy) to per-head scalar
   attention biases. The bias tables are built at runtime for the actual feature-map size,
   so 416×416 (S5 = 13×13), 640×640 (S5 = 20×20) and non-square inputs all work without
   re-instantiation.

Final configuration:
`src/ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_LRPB-AIFI.yaml`

## 3. Dataset

`dataset/roadrdd/` holds the audited road-defect dataset: **8,695 images** with two
classes (`crack`, `pothole`) and YOLO-format bounding boxes.

| Split | Images |
|-------|--------|
| train | 6,162 |
| val   | 836 |
| test  | 1,697 |
| **Total** | **8,695** |

The dataset was audited before release:

* **Byte-level deduplication.** Every image was hashed with SHA-256. 379 duplicate groups
  (379 redundant copies) were found, of which 172 spanned two splits
  (train∩test 90, train∩val 69, val∩test 13). All redundant copies were removed; the
  full group-level record is in `dataset/roadrdd/audit/dedup_log.csv` and the per-image
  hashes are in `dataset/roadrdd/MANIFEST.sha256`.
* **Split manifest.** `dataset/roadrdd/split_manifest.csv` maps every image to its split
  and to a `source_group` identifier, so the absence of byte-identical images across
  splits can be verified by any reader.
* **Instance counts** after deduplication: 8,455 crack, 8,358 pothole (16,813 boxes).

See `docs/dataset_statistics.md` and `dataset/roadrdd/audit/dataset_audit.md` for the full
numbers. To re-run the audit:

```bash
python src/tools/audit_dataset.py . --write      # writes the manifests and audit log
```

> **Known limitation.** Deduplication removes byte-identical images only. Grouping images by
> `source_group` (the Roboflow `.rf.<hash>` suffix stripped) shows that some augmentations of
> the same source photograph still land in different splits; `docs/reproducibility.md`
> documents this explicitly rather than claiming a fully leakage-free split.

## 4. Requirements

`environment.yml` pins the environment used for the experiments
(PyTorch 2.4.0, CUDA 11.8, TensorRT 8.6, fvcore); `src/requirements.txt` is the lighter
pip-only equivalent.

```bash
conda env create -f environment.yml
conda activate roadrdd
```

The reported experiments were run on CentOS 7.9 with an Intel Xeon Silver 4210R and a single
NVIDIA GeForce RTX 3080 (10 GB).

## 5. Quick start

```bash
cd src
pip install -r requirements.txt

# train the proposed model (416x416, AdamW, 240 epochs)
python train.py --data ../configs/roadrdd.yaml --imgsz 416 --batch 16 --epochs 240

# train one of the eight ablation configurations
python train.py --cfg ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --name r18

# evaluate
python val.py --weights runs/roadrdd/train/<run>/weights/best.pt --split test --imgsz 416

# GFLOPs with fvcore (the tool used throughout the paper)
python get_flops_fvcore.py --imgsz 416 640

# end-to-end FPS with TensorRT FP16
python get_FPS.py --engine best.engine --imgsz 416 --runs 5 --warmup 200 --iters 300

# unit tests for the learnable relative position bias tables
pytest tests/test_lrpb_dynamic_size.py -v
```

## 6. Cross-dataset transfer

The RDD2022 dataset used for the cross-dataset transfer experiment (Section 4.9) is publicly
available at <https://github.com/sekilab/RoadDamageDetector>. It is not redistributed here.

## 7. License

* Source code: MIT License (`LICENSE`).
* `roadrdd` dataset: released for academic research use; please cite the manuscript.

## 8. Citation

If you find this repository useful, please cite the corresponding manuscript.
