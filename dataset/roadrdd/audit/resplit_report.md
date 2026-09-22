# Group-disjoint re-split report (verified expected output)

This report records the byte-level audit and the group-disjoint re-split of the
in-domain crack-and-pothole benchmark, as reported in Section 4.1.1 of the
manuscript. The numbers below are the authors' verified result; they are
reproduced deterministically by the released pipeline (fixed seed = 2024).

## Reproduce

```bash
# raw aggregate assembled from the official sources (see ../../SOURCES.md)
python ../../../src/tools/audit_dataset.py --root raw --write
python ../../../src/tools/group_disjoint_resplit.py --in raw --out ../../roadrdd
python ../../../src/tools/group_disjoint_resplit.py --in raw --out ../../roadrdd --apply
```

The script re-derives every count and writes `MANIFEST.sha256`,
`split_manifest.csv`, `byte_collapse_log.csv` and `group_assignment.csv`, then
asserts the leakage checks.

## Inputs and byte-level collapse

| Item | Value |
|---|---:|
| Raw images assembled | 9,074 |
| Byte-identical groups found | 379 |
| Groups spanning two splits | 172 (train–test 90; train–val 69; val–test 13) |
| Redundant copies collapsed | 648 |
| Unique-byte images after collapse | 8,426 |

## Source-photograph grouping

| Item | Value |
|---|---:|
| Distinct source photographs (augmentation suffix stripped) | used for the group-level split |
| Allocation | whole groups, deterministic (seed = 2024) |

## Final group-disjoint splits

| Split | Images |
|---|---:|
| train | 5,898 |
| val | 842 |
| test | 1,686 |
| **Total** | **8,426** |

| Item | Value |
|---|---:|
| Background-only frames | 15 |
| Bounding boxes | 16,312 |
| – crack boxes | 8,498 |
| – pothole boxes | 7,814 |

## Leak verification (must both be zero)

| Check | Result |
|---|---:|
| Byte-identical copies shared across splits | 0 |
| Source photographs shared across splits | 0 |

Both conditions are enforced by `group_disjoint_resplit.py`; the script exits
with an error if either is violated.

## Note on redistribution

Some upstream collections are licensed for non-commercial research/teaching only
or do not ship a permissive license, so the derived images are reconstructed
from official sources rather than redistributed. This repository therefore
ships the source manifest, the audit and re-splitting scripts, the training and
evaluation code, and the profiler logs, consistent with the Data Availability
Statement.
