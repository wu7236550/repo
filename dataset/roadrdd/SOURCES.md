# Source manifest — crack-and-pothole benchmark (RoadRDD)

This is the **source manifest** referenced in the paper's Data Availability
Statement. It records, for every public source used to build the in-domain
benchmark: the per-source counts, version/year, official location and license
terms. A machine-readable copy is provided as [`sources.csv`](sources.csv).

The benchmark is a *detection derivative* of the public sources below. All
images were resampled to **416 × 416** and annotated with YOLO-normalized
bounding boxes (class `0` = crack, class `1` = pothole). The counts in the
right-most column are the numbers of derived 416 × 416 patches contributed to
the released benchmark; they are regenerated deterministically by the audit and
group-disjoint re-splitting pipeline in `src/tools/` (see
[`audit/resplit_report.md`](audit/resplit_report.md)).

> **Licensing.** Some upstream collections are licensed for **non-commercial
> research / teaching only** or do not ship a permissive license. For that
> reason this repository distributes the *provenance, manifests and
> reconstruction scripts*, not bulk copies of every derived image. Reconstruct
> the benchmark from the official sources with the commands in the README; the
> audit pipeline records the per-file license as it builds the dataset.

## Per-source summary

| Source | Version / year | Upstream images | Derived patches (reported) | License | Official location |
|---|---|---:|---:|---|---|
| **CRACK500** — pavement cracks | 2019/2020 | 500 (2000×1500) | 3,988 | Academic research use, copyright retained (treat as non-commercial) | <https://github.com/fyangneil/pavement-crack-detection> |
| **DeepCrack** — crack segmentation | 2019 | 537 (544×384) | 382 | Non-commercial research / teaching; authors retain copyright | <https://github.com/yhlleo/DeepCrack> |
| **GAPs** — German Asphalt Pavement Distress | 2017 | 1,969 (1920×1080, grayscale) | 433 | Free for research under TU Ilmenau terms | [TU Ilmenau GAPs page](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/german-asphalt-pavement-distress-dataset-gaps) |
| **CrackForest (CFD)** | 2016 | 118 (480×320) | 240 | Academic research use, copyright retained | <https://github.com/cuilimeng/CrackForest-dataset> |
| **CrackTree200** | 2012 | 206 (varied) | 112 | Academic research use; distributed by the authors | <https://doi.org/10.1016/j.patrec.2011.11.004> |
| **Other public pavement-crack photographs** (composite; e.g. Eugen-Müller and camera-titled collections) | 2010s | varies | 145 | Mixed per source; treat as non-commercial research | n/a (aggregated public projects) |
| **Concrete-facade / structural concrete crack collection** (composite; incl. the Mendeley Concrete Crack Images for Classification) | 2018+ | varies | 237 | CC BY 4.0 for the Mendeley component; others per source | <https://data.mendeley.com/datasets/5y9wdsg2zt6> |
| **Annotated pothole collections** (Pothole-600 + public pothole projects) | 2019+ | 600 (+ project images) | 3,141 | Mixed: Pothole-600 for research; some projects under ODbL / CC BY | <https://sites.google.com/view/pothole-600/dataset> |
| **Background frames** (defect-free, sampled from the above) | — | varies | 17 | Follows the source frame | same as parent |
| | | **Total** | **8,695** | | |

## Citations

- Yang, F.; Zhang, L.; Yu, S.; Prokhorov, D.; Mei, X.; Ling, H. *Feature Pyramid
  and Hierarchical Boosting Network for Pavement Crack Detection.* IEEE
  Transactions on Intelligent Transportation Systems, 21(4), 2020.
- Liu, Y.; Yao, J.; Lu, X.; Xie, R.; Li, L. *DeepCrack: A Deep Hierarchical
  Feature Learning Architecture for Crack Segmentation.* Neurocomputing, 338,
  2019.
- Eisenbach, M.; Stricker, R.; Seichter, D.; Amende, K.; Debes, K.; Sesselmann,
  M.; Ebersbach, D.; Stöckert, U.; Gross, H.-M. *How to Get Pavement Distress
  Detection Ready for Deep Learning? A Systematic Approach.* IJCNN, 2017.
- Shi, Y.; Cui, L.; Qi, Z.; Meng, F.; Chen, Z. *Automatic Road Crack Detection
  Using Random Structured Forests.* IEEE Transactions on Intelligent
  Transportation Systems, 17(12), 2016.
- Zou, Q.; Cao, Y.; Li, Q.; Mao, Q.; Wang, S. *CrackTree: Automatic crack
  detection from pavement images.* Pattern Recognition Letters, 33(3), 2012.
- Fan, R.; Ozgunalp, U.; Hosking, B.; Liu, M.; Pitas, I. *Pothole detection based
  on disparity transformation and road surface modeling.* IEEE Transactions on
  Image Processing, 2019.
- Özgenel, C.F. *Concrete Crack Images for Classification.* Mendeley Data,
  2018.

## How the counts are produced

1. Download each source from its official location (license recorded).
2. Run the byte-level audit / collapse:
   `python src/tools/audit_dataset.py --root <raw>`
3. Run the deterministic group-disjoint re-split (fixed seed = 2024):
   `python src/tools/group_disjoint_resplit.py --in <raw> --out dataset/roadrdd --apply`
4. The per-source and per-split counts, hashes and verification are written to
   `audit/`, `MANIFEST.sha256` and `split_manifest.csv`.

Manuscript totals (Section 4.1.1): **8,695** images — 6,087 train / 869 val /
1,739 test; 16,797 boxes (8,439 crack / 8,358 pothole); 17 background-only
frames.
