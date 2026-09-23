# RoadRDD annotation specification

## Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | crack | Cracks on pavement, structural concrete and other surfaces (all native crack subtypes collapsed) |
| 1 | pothole | Potholes (severity levels collapsed) |

## Format

Annotations use the YOLO detection format. Each image has a companion `.txt`
file with the same basename; each line encodes one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Coordinates are **normalized to [0, 1]** relative to image width/height.
- `<x_center>`, `<y_center>`: box centre; `<width>`, `<height>`: box size.

Images with no annotated defect have an empty label file; **17** such
background-only frames are retained.

## Provenance

The benchmark is not the result of an author-run annotation campaign; it is a
detection derivative of public sources (see
`../dataset/roadrdd/SOURCES.md`). Crack labels are derived from CRACK500,
DeepCrack, GAPs, CrackForest, CrackTree, structural concrete crack images and a
small composite group of other public pavement-crack photographs
(pixel masks converted to boxes, subtypes collapsed to
`crack`). Pothole labels come from annotated pothole collections (severity
levels collapsed to `pothole`). Validation and test images are never augmented.

For the external RDD2022 experiment, the official `D00`, `D10` and `D20`
classes are mapped to `crack` and `D40` to `pothole`; repair/blur markings are
ignored.

## Splits

The released split is **group-disjoint**:

- no two byte-identical images appear in different splits (verified by
  SHA-256), and
- no source photograph - identified by stripping the Roboflow `.rf.<hash>`
  augmentation suffix so all variants share a `source_group` id - contributes to
  more than one split.

The split is recorded in `split_manifest.csv` (image, split, source group,
hash) and produced deterministically by
`../src/tools/group_disjoint_resplit.py` (fixed seed = 2024).

## Statistics

| Split | Images |
|-------|-------:|
| train | 6,087 |
| val   | 869 |
| test  | 1,739 |
| **Total** | **8,695** |

| Class | Instances |
|-------|----------:|
| crack | 8,439 |
| pothole | 8,358 |
| **Total** | **16,797** |

17 background-only frames.

## License

The derivative is released for research use; the upstream sources keep their
own licenses, some of which are non-commercial research only. Cite the
manuscript and the upstream sources (see `SOURCES.md`).
