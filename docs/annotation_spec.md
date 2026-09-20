# roadrdd Annotation Specification

## Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | crack | Cracks on pavement and on concrete surfaces (subtypes collapsed) |
| 1 | pothole | Potholes (severity levels collapsed) |

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

## Provenance of the labels (Section 4.1 of the manuscript)

The manuscript does **not** describe an author-side annotation campaign. Section 4.1 states
that the data are not author-collected but "a locally prepared YOLO derivative of public
sources", so the released labels inherit the annotation semantics of the upstream datasets:

- Crack annotations come from the Concrete Crack Conglomerate Dataset (CCCD) families —
  CRACK500, CFD, DeepCrack, GAPs, CrackTree — plus the tunnel-lining and concrete-facade
  crack sets. All crack subtypes are collapsed into the single class `crack`.
- Pothole annotations come from the annotated pothole collections, whose severity levels are
  collapsed into the single class `pothole`.
- The mask-to-box / label-collapse rules that produced each label are what the provenance
  manifest of Supplementary File S1 is meant to record; that manifest is not part of this
  release (see `reproducibility.md`, Section 6).

Section 4.1 also states that the untraceable name families are excluded from the main
experiment, and that validation and test images are never augmented.

Related but separate: for the **external** RDD2022 benchmark the manuscript maps the official
`D00`, `D10` and `D20` classes to `crack` and `D40` to `pothole`, and ignores `D43`, `D44`
and `D50`.

## Data splits

The split is **duplicate-free at the byte level**: no two byte-identical images appear in
different splits. The split is recorded in `split_manifest.csv`, which lists, for every
image, its split assignment, a `source_group` identifier and its SHA-256 hash.

The split is **not** leakage-free at the source-image level. Grouping images by
`source_group` shows that 895 of the 5,875 source groups still have representatives in more
than one split, so no claim of a fully leakage-free split should be made for this release.
The manuscript instead publishes a split that is disjoint at the level of whole source
groups, and reports the residual same-pothole multi-view risk that filename grouping cannot
merge; see `reproducibility.md`, Sections 1.1 and 2.

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
removed. The manuscript's main set is the 7,496-image subset obtained after also excluding
the untraceable families (14,560 boxes); see `dataset_statistics.md`. See
`audit/dataset_audit.md` for the full audit and `reproducibility.md` for the residual
limitations.

## Dataset license

This dataset is released for academic research use; the upstream public sources keep their own
licences (CCCD is CC0; the pothole collections are ODbL-type). If you use `roadrdd` in your
work, please cite the corresponding manuscript **and** the upstream datasets listed in
`reproducibility.md`, Section 1.
