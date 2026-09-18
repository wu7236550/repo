# Detailed-Metrics Protocol

This document specifies how the module-attribution metrics of the detailed-metrics table
(Table 4 of the manuscript) are produced, and states exactly which parts of them can and
cannot be reproduced from this repository. It is meant to be read together with
`reproducibility.md` and `../dataset/roadrdd/audit/dataset_audit.md`.

Everything described here is executed by scripts in `../src/`:

| Metric | Script | Released input |
|---|---|---|
| FP/image on background-only patches | `make_background_patches.py`, `eval_background_fp.py` | `../dataset/roadrdd/audit/background_patches.csv` |
| Crack-width bins | `calibrate_width.py` | `../configs/width_calibration.yaml` |
| Per-class AP, AP_small / AP_medium / AP_large, AP75 | the standard validator, see Section 4 | released labels |

---

## 1. Background-only patches and FP/image

### 1.1 Definition of a background patch

A **background patch** is a square crop of a released image that does not intersect any
annotated bounding box, dilated by a margin of 16 pixels. The dilation is applied so that
a patch is rejected not only when it contains an annotation but also when it comes close
to one; without it, half of a crack could lie just outside the crop and produce a
legitimate detection that the metric would count as a false positive.

### 1.2 Sampling rule

Candidates are enumerated on a regular grid with a stride of 64 pixels over every image of
the test split, at a patch side of 128 pixels on the 416x416 network input. The metric is a
**road-surface** false-positive rate, so the three groups whose surface is concrete rather
than road are skipped by default: `noncrack_*` (concrete-wall non-crack frames),
`Eugen_Muller_*` (tunnel lining) and `Volker_*` (concrete facade). On the released data this
leaves 1,643 of the 1,697 test images and yields 12,011 eligible patches in the full test
split, 11,483 after the exclusion.

The 120 patches are then selected by ranking the candidates with the SHA-256 digest of
`(seed, image, x, y)` and taking the first 120 in that order. Ranking by a digest rather
than by a pseudo-random generator makes the selection independent of the Python and NumPy
versions: the same command produces the same manifest everywhere. The seed is `0`.

```
cd src
python make_background_patches.py --data ../dataset/roadrdd --split test \
    --patch 128 --stride 64 --margin 16 --n 120 --seed 0 \
    --exclude-prefix noncrack_,Eugen_Muller_,Volker_
```

The committed manifest is `dataset/roadrdd/audit/background_patches.csv` and lists, for
every selected patch, the source image, its split and source group, the crop origin, the
patch size and the SHA-256 of the source image. Re-running the command above reproduces
this file byte for byte. Pass `--exclude-prefix ''` to sample the whole test split
instead.

### 1.3 Why patches rather than whole images

A whole-image criterion cannot supply 120 samples: only **17 of the 8,695 released images
have no annotation at all** (2 in the CRACK500 group, 8 among the numeric stems, 7 in the
noncrack group), and only 10 of those are road-surface images. The defect-free area of the
released set therefore has to be sampled at the patch level, which is why the metric in the
manuscript is stated per patch.

### 1.4 Metric

Every patch is known to contain no annotated defect, so each predicted box above the
confidence threshold counts as a false positive. FP/image is the mean number of such boxes
per patch:

```
cd src
python eval_background_fp.py --weights <weights> --imgsz 416 --conf 0.25
```

The confidence threshold is a free parameter of the protocol and is reported with the
result; the manuscript uses 0.25. The script prints the aggregate value and a
per-source-group breakdown.

### 1.5 Composition of the released selection

The 120 selected patches come from 112 distinct test images and seven road-surface source
groups: CRACK500 (including the `2016xxxx_*` sequence) 63, bare numeric stems 19, `po_` 18,
GAPS384 8, DeepCrack 6, `img_*` / `IMG_*` 5, CFD 1. No patch comes from `noncrack_*`,
`Eugen_Muller_*` or `Volker_*`. The selection is a uniform draw over eligible windows, so
the composition follows the composition of the test split rather than being equalised
across sources.

---

## 2. Crack-width bins

### 2.1 Definition

The three crack-width rows of the detailed-metrics table group the annotated instances of
the test set by the physical width of the crack, with bin edges at 3 mm and 10 mm:

* `width < 3 mm`
* `3 mm <= width < 10 mm`
* `width >= 10 mm`

Recall is reported per bin.

### 2.2 Pixel-to-millimetre calibration

The bins are applied to a width in pixels and converted with a ground sampling distance.
`configs/width_calibration.yaml` supports two equivalent routes:

* **Route A - acquisition geometry.** For a camera whose optical axis is perpendicular to
  the road surface at the working distance,
  `mm_per_pixel = mounting_height_mm / focal_length_px`, with
  `focal_length_px = focal_length_mm / pixel_pitch_mm`. This is the route that matches the
  vehicle-mounted acquisition.
* **Route B - direct measurement.** Photograph a target of known physical size at the
  working distance and record how many pixels it spans:
  `mm_per_pixel = reference_mm / reference_pixels`.

```
cd src
python calibrate_width.py
```

The script prints the resolved calibration, the bin edges expressed in pixels, and, given
a label directory, the population of each bin. For example, a ground sampling distance of
1.25 mm per pixel puts the 3 mm and 10 mm edges at 2.4 and 8.0 pixels respectively.

The two numbers of the chosen route are **not** stored in this repository. They are
physical properties of the acquisition, and the released images cannot supply them: the
dataset export stripped all EXIF metadata and resampled every image to 416x416. The script
exits with an explicit message listing the fields to fill until one route is complete.

### 2.3 Limitation: the released labels carry bounding boxes only

The three width-binned rows are the one part of the detailed-metrics table that **cannot be
reproduced from the released labels**, and the calibration constant is not the only reason.

* The released annotations are bounding boxes. For a crack, the box width is the extent of
  the annotated region along one image axis, not the crack opening width, and the two are
  unrelated: the median crack box in the test split is 161 x 160 pixels, i.e. roughly
  square, whereas a crack opening is measured perpendicular to the crack path.
* Reproducing these three rows therefore requires one width per annotated instance,
  measured from a segmentation mask or manually, released as a side table.

The interface for that side table is implemented: a CSV with a `width_px` column holding
the crack width of each instance in pixels at the 416x416 input resolution.

```
cd src
python calibrate_width.py --widths ../dataset/roadrdd/audit/crack_widths.csv
```

Until such a table is released, these three rows should be read as the authors'
measurements under the stated protocol rather than as quantities a reader can recompute.

### 2.4 Limitation: the calibration covers one acquisition geometry only

A single ground sampling distance describes one camera at one working distance. Of the nine
source groups in the test split, only the field-imagery groups were acquired with the
vehicle-mounted camera described in the manuscript; the public crack datasets were
acquired with smartphone, handheld and industrial cameras whose scale is not recorded here.
`applies_to_source_groups` in `configs/width_calibration.yaml` names the groups the
calibration is valid for, and `calibrate_width.py` reports how many annotated boxes fall
inside and outside that portion. On the test split, only about a quarter of the annotated
boxes belong to a group with a recorded acquisition geometry, so a width-binned statistic
computed over the whole test split mixes physically different scales.

---

## 3. What this means for Table 4

| Column | Reproducible from the release |
|---|---|
| FP/image | **Yes**, given model weights: the protocol, the 120 patches and the evaluation script are all released |
| AP_crack, AP_pothole | Yes, from the released labels with the standard validator |
| AP_small, AP_medium, AP_large | Yes, given the area thresholds of Section 4 |
| AP75 | Yes |
| Recall by crack-width bin | **No** - requires the per-instance width table of Section 2.3 |

---

## 4. The remaining columns

`AP_crack`, `AP_pothole`, `AP75` and the size-binned APs are the standard COCO-style
metrics computed on the test split at the 416x416 input resolution:

* `AP_small`, `AP_medium`, `AP_large` use the COCO area thresholds of 32^2 and 96^2 pixels
  at 416x416, i.e. boxes below 1,024, between 1,024 and 9,216, and above 9,216 square
  pixels respectively.
* `AP75` is the mean average precision at an IoU threshold of 0.75.
* All per-class values follow the same protocol as the aggregate mAP reported in the
  ablation table, evaluated on the 1,697-image test split of the released manifest.

---

## 5. Summary of what is not released

1. **The acquisition geometry** needed by `configs/width_calibration.yaml` (Section 2.2).
2. **A per-instance crack-width table** (Section 2.3).
3. **The five random seeds' per-run logs** - see Section 5 of `reproducibility.md`.
4. **Trained weights.**

Items 1 and 2 are the only ones that block the reproduction of a reported number; both are
covered by the scripts released here and both require physical input that the repository
cannot contain.
