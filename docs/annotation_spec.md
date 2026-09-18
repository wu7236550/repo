# roadrdd Annotation Specification

## Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | crack | Longitudinal, transverse and alligator cracks on the road surface |
| 1 | pothole | Pavement potholes |

## Format

Annotations follow the YOLO detection format. Each image has a companion `.txt` file with
the same basename. Each line encodes one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Coordinates are **normalized** to `[0, 1]` relative to image width/height.
- `<x_center>`, `<y_center>`: centre of the bounding box.
- `<width>`, `<height>`: box dimensions.

Images without any annotated defect have an empty label file. 17 such background-only
images are retained in the release.

## Annotation protocol (Section 4.1 of the manuscript)

- Annotation tool: LabelImg.
- Annotation disagreements were resolved against a single unified standard.
- Low-quality samples with ambiguous defects were removed.

## Data splits

The split is **duplicate-free at the byte level**: no two byte-identical images appear in
different splits. The split is recorded in `split_manifest.csv`, which lists, for every
image, its split assignment, a `source_group` identifier and its SHA-256 hash.

The split is **not** leakage-free at the source-image level. Grouping images by
`source_group` shows that 895 of the 5,875 source groups still have representatives in more
than one split, so no claim of a fully leakage-free split should be made for this release;
see `reproducibility.md`, Section 2, for the residual limitation.

`source_group` is obtained by stripping the Roboflow `.rf.<hash>` augmentation suffix from
the file name, so that all augmented variants of one source photograph share a group id.
Readers can therefore verify the split independently.

## Dataset statistics

Statistics for the released (byte-level deduplicated) dataset:

| Split | Images |
|-------|--------|
| train | 6,162 |
| val   | 836 |
| test  | 1,697 |
| **Total** | **8,695** |

| Class | Instances |
|-------|-----------|
| crack | 8,455 |
| pothole | 8,358 |
| **Total** | **16,813** |

The pre-audit release contained 9,074 images. The byte-level audit found 379 duplicate
groups (379 redundant copies), of which 172 spanned two splits; all redundant copies were
removed. See `audit/dataset_audit.md` for the full audit and
`../docs/reproducibility.md` for the residual limitations.

## Dataset license

This dataset is released for academic research use. If you use `roadrdd` in your work, please
cite the corresponding manuscript.
