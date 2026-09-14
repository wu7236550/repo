# Lightweight RT-DETR for Road Defect Detection

Official code and dataset repository for the manuscript:

**"A Lightweight RT-DETR for Road Defect Detection"** (submitted to *Sensors*).

This repository contains:

- `src/` — source code for training, validation, inference and evaluation,
  including the three proposed modules (MSPC, MCAF, LRPB-AIFI)
  and the final model configuration under
  `src/ultralytics/cfg/models/rt-detr/`.
- `configs/` — dataset configuration (`roadrdd.yaml`) and the training
  hyperparameters (`train_config.md`).
- `dataset/roadrdd/` — the self-built road-defect dataset (9074 images with
  YOLO-format annotations).
- `docs/` — annotation specification and dataset statistics.

## Requirements

See `src/requirements.txt`. The experiments were run with PyTorch 2.4.0 +
CUDA 11.8 on a single NVIDIA RTX 3080 (10 GB).

## Quick start

```bash
cd src
pip install -r requirements.txt
python train.py --data ../configs/roadrdd.yaml \
    --cfg ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_LRPB-AIFI.yaml \
    --imgsz 416 --batch 16 --epochs 240
```

## Cross-dataset validation

The RDD2022 dataset used for cross-dataset validation is publicly available at
https://github.com/sekilab/RoadDamageDetector.

## License

- Source code: MIT License (`LICENSE`).
- roadrdd dataset: academic research use; please cite the manuscript.

## Citation

If you find this repository useful, please cite the corresponding manuscript.
