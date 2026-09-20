# roadrdd Dataset Statistics

Summary of the released `roadrdd` dataset — the mixed-surface crack and pothole derivative
described in Section 4.1 of the manuscript *Lightweight RT-DETR for Mixed-Surface Crack and
Pothole Detection*. The authoritative, machine-generated record is
`../dataset/roadrdd/audit/dataset_audit.md`; this page is a human-readable digest.

## What the release contains

`dataset/roadrdd/` ships all **8,695** de-duplicated images: the manuscript's 7,496-image main
set **plus** the 1,562 (of 1,578 pre-audit) exports whose source cannot be traced. Splitting
the two apart is a property of the file names, not of the manifest; see
`reproducibility.md`, Section 1.

## Splits of this release (after byte-level deduplication)

| Split | Images | Share |
|-------|--------|-------|
| train | 6,162 | 70.9 % |
| val   | 836   | 9.6 % |
| test  | 1,697 | 19.5 % |
| **Total** | **8,695** | 100 % |

Target ratio 7 : 1 : 2.

## Instances in this release

| Class | Label | Instances | Images containing it |
|-------|-------|-----------|----------------------|
| crack | `crack` (0) | 8,455 | — |
| pothole | `pothole` (1) | 8,358 | — |
| **Total** | | **16,813** | |

There are 17 background-only images (empty label files). Exact per-split instance counts and
per-image box counts can be recomputed from `../dataset/roadrdd/labels/` or read from
`../dataset/roadrdd/split_manifest.csv` together with the label files.

## The manuscript's main set

The manuscript excludes the untraceable families and reports a **7,496-image** main set with
**14,560** boxes (Table 3(a) and Table 3(b)):

| Split | Images | crack boxes | pothole boxes | Total boxes |
|-------|--------|-------------|---------------|-------------|
| train | 5,238 | 5,358 | 4,852 | 10,210 |
| val   | 748   | 764   | 692   | 1,456   |
| test  | 1,510 | 1,520 | 1,374 | 2,894   |
| **Total** | **7,496** | **7,642** | **6,918** | **14,560** |

| Component | Orig. exports | Excluded | Main set |
|---|---|---|---|
| CCCD-like crack families | 5,569 | 0 | 5,569 |
| `po_*` pothole family | 1,927 | 0 | 1,927 |
| untraceable name families (`img_*`/`IMG_*`, `noncrack_*`, bare numeric stems) | 1,578 | 1,578 | 0 |
| **Total** | **9,074** | **1,578** | **7,496** |

The manuscript assigns whole source groups to the three splits, at about 70/10/20; the split
shipped here is byte-level duplicate-free but assigns images individually, so the two splits
differ. See `reproducibility.md`, Section 2.

## Audit trail

| File | Contents |
|------|----------|
| `../dataset/roadrdd/MANIFEST.sha256` | SHA-256 of every released image |
| `../dataset/roadrdd/split_manifest.csv` | image → split → `source_group` → SHA-256 |
| `../dataset/roadrdd/audit/dedup_log.csv` | one row per duplicate group: kept copy and removed copies |
| `../dataset/roadrdd/audit/dedup_plan.txt` | flat list of the removed image/label paths |
| `../dataset/roadrdd/audit/dataset_audit.md` | machine-generated audit report |

## Before / after

| Item | Pre-audit | Released |
|------|-----------|----------|
| Images | 9,074 | 8,695 |
| train / val / test | 6,351 / 908 / 1,815 | 6,162 / 836 / 1,697 |
| Byte-level duplicate groups | 379 (172 cross-split) | 0 |
| Class instances | 9,146 crack, 8,430 pothole | 8,455 crack, 8,358 pothole |

Regenerate with:

```bash
python src/tools/audit_dataset.py . --write
```

See `reproducibility.md` for the residual source-group limitation.
