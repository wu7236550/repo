# roadrdd Annotation Specification

## Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | crack | Longitudinal, transverse, and alligator cracks on the road surface |
| 1 | pothole | Pavement potholes |

## Format

Annotations follow the YOLO detection format. Each image has a companion `.txt`
file with the same basename. Each line encodes one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Coordinates are **normalized** to `[0, 1]` relative to image width/height.
- `<x_center>`, `<y_center>`: center of the bounding box.
- `<width>`, `<height>`: box dimensions.

## Annotation protocol (Section 4.1 of the manuscript)

- Annotation tool: LabelImg.
- Annotation disagreements resolved with a unified standard.
- Low-quality samples with ambiguous defects were removed.

## Data splits

Random split at 7:1:2 (train : val : test), with the guarantee that images from
the same road segment never appear in both the training and test sets.

## Dataset statistics

| Split | Images |
|-------|--------|
| train | 6351 |
| val   | 908  |
| test  | 1815 |
| Total | 9074 |

## Dataset license

This dataset is released for academic research use. If you use roadrdd in your
work, please cite the corresponding manuscript.
