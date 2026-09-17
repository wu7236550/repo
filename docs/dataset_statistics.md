# roadrdd Dataset Statistics

Summary of the released `roadrdd` dataset. The authoritative, machine-generated record is
`../dataset/roadrdd/audit/dataset_audit.md`; this page is a human-readable digest.

## Splits (after byte-level deduplication)

| Split | Images | Share |
|-------|--------|-------|
| train | 6,162 | 70.9 % |
| val   | 836   | 9.6 % |
| test  | 1,697 | 19.5 % |
| **Total** | **8,695** | 100 % |

Target ratio 7 : 1 : 2.

## Instances

| Class | Label | Instances | Images containing it |
|-------|-------|-----------|----------------------|
| crack | `crack` (0) | 8,455 | — |
| pothole | `pothole` (1) | 8,358 | — |
| **Total** | | **16,813** | |

There are 17 background-only images (empty label files). Exact per-split instance counts and
per-image box counts can be recomputed from `../dataset/roadrdd/labels/` or read from
`../dataset/roadrdd/split_manifest.csv` together with the label files.

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
