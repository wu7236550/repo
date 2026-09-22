"""False positives per image on the released background-only patches.

Every patch listed in the manifest contains no annotated defect, so each predicted box
above the confidence threshold is a false positive. The reported metric is the mean
number of such boxes per patch, together with a per-source-group breakdown.

The manifest is produced by make_background_patches.py; the protocol and the meaning of
the threshold are documented in docs/reproducibility.md (Section 6).

Usage:
    python eval_background_fp.py --weights runs/detect/train/weights/best.pt \
        --imgsz 416 --conf 0.25
"""
import argparse
import csv
import os
import warnings

warnings.filterwarnings('ignore')

from PIL import Image  # noqa: E402
from ultralytics import RTDETR  # noqa: E402


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(os.path.join(here, '..', 'dataset', 'roadrdd'))
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--weights', required=True, help='model weights used for the evaluation')
    p.add_argument('--data', default=default_data, help='dataset root')
    p.add_argument('--patches', default=None,
                   help='patch manifest (default: <data>/audit/background_patches.csv)')
    p.add_argument('--imgsz', type=int, default=416, help='network input resolution')
    p.add_argument('--conf', type=float, default=0.25,
                   help='confidence threshold above which a box counts as a detection')
    p.add_argument('--device', default='cpu', help="inference device, e.g. 'cpu' or '0'")
    return p.parse_args()


def main():
    args = parse_args()
    patches = args.patches or os.path.join(args.data, 'audit', 'background_patches.csv')
    with open(patches, newline='') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit('empty patch manifest: %s' % patches)

    model = RTDETR(args.weights)
    total = 0
    per_group = {}
    for rec in rows:
        image = Image.open(os.path.join(args.data, rec['image']))
        x, y, size = int(rec['x']), int(rec['y']), int(rec['size'])
        crop = image.crop((x, y, x + size, y + size))
        result = model.predict(crop, imgsz=args.imgsz, conf=args.conf,
                               device=args.device, verbose=False)[0]
        n = 0 if result.boxes is None else len(result.boxes)
        total += n
        key = rec['source_group'].split('_')[0]
        hits, count = per_group.get(key, (0, 0))
        per_group[key] = (hits + n, count + 1)

    print('patches      : %d' % len(rows))
    print('conf threshold: %.2f' % args.conf)
    print('FP/image     : %.4f  (%d detections over %d patches)' % (total / len(rows), total, len(rows)))
    print()
    print('%-16s %6s %8s %10s' % ('source group', 'n', 'FP', 'FP/image'))
    for key, (hits, count) in sorted(per_group.items(), key=lambda kv: -kv[1][1]):
        print('%-16s %6d %8d %10.4f' % (key, count, hits, hits / count))


if __name__ == '__main__':
    main()
