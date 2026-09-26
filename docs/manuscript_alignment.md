# Manuscript Alignment Notes

This repository accompanies *A Lightweight Real-Time Detection Transformer with
Multi-Scale Partial Convolution and Context-Anchored Fusion for Road Crack and
Pothole Detection*.

## Scope of the release

The manuscript reports a two-class crack-and-pothole study based on a group-disjoint
8,695-image in-domain benchmark and a source-disjoint RDD2022 external experiment.
The repository provides the implementation, configurations, source provenance, audit
and reconstruction pipeline, reproducibility settings, and profiler logs required by
the Data Availability Statement.

## Consistency points

- RoadRDD contains 6,087 training, 869 validation, and 1,739 test images after the
  byte-level collapse and group-disjoint re-split.
- Training uses `warmup_epochs: 2000` as an iteration budget. With 6,087 training
  images and batch size 16, this is approximately 5.3 epochs and is described in the
  manuscript as an approximately five-epoch linear warm-up.
- The reported model modifications are MSPC, MCAF, and LRPB-AIFI. The complete
  2^3 factorial configurations and their controls are retained under
  `src/ultralytics/cfg/models/rt-detr/`.
- The retained RT-DETR decoder uses a 256-channel model dimension, 300 object queries, three decoder feature levels, four sampling points per head, eight attention heads, and three decoder layers.
- Derived images are intentionally not redistributed. Their sources, versions, and
  licence terms are recorded in `dataset/roadrdd/SOURCES.md`, and the supplied scripts
  rebuild the auditable group-disjoint split from legally obtained source data.

The reported accuracy values are manuscript measurements. The repository publishes the
configuration and code needed to reproduce the protocol, while source-image availability
continues to depend on the original data providers and their licences.
