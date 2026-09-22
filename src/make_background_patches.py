"""Deterministic background-patch manifest for the false-positive metric.

Reproduces the background protocol of Section 4.2 of the manuscript:

    "on each test image four candidate 128-pixel patches sit at fixed grid
     positions, a patch is kept only if it does not intersect any annotation
     dilated by 16 pixels, and detections above a confidence of 0.25 are
     counted; this yields 5,214 patches over the 1,686 test images."

For every image of the chosen split, FOUR 128 x 128 candidates are placed at a
fixed 2 x 2 grid of positions. A candidate is kept only when it does not
intersect any annotated box dilated by ``--margin`` pixels. Every kept patch is
written to the manifest (there is no random sub-selection), so the manifest and
its patch count are deterministic and independent of the Python/NumPy versions.
The reported benchmark result is 5,214 kept patches over the 1,686 test images.

Use ``eval_background_fp.py`` to count detections (false positives) on these
patches.

Usage:
    python make_background_patches.py --data ../dataset/roadrdd --split test
"""
import argparse
import csv
import os

IMAGE_SIZE = 416
PATCH = 128
MARGIN = 16


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(os.path.join(here, '..', 'dataset', 'roadrdd'))
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', default=default_data,
                   help='dataset root holding images/, labels/ and split_manifest.csv')
    p.add_argument('--out', default=None,
                   help='output CSV (default: <data>/audit/background_patches.csv)')
    p.add_argument('--split', default='test',
                   help="split to sample ('test', 'val', 'train' or 'all')")
    p.add_argument('--patch', type=int, default=PATCH)
    p.add_argument('--margin', type=int, default=MARGIN,
                   help='reject a patch if it comes this close to an annotated box')
    p.add_argument('--positions', default=None,
                   help='optional "x1,y1;x2,y2;..." fixed top-left corners; '
                        'default is a balanced 2x2 grid')
    return p.parse_args()


def default_positions(image_size, patch):
    """Balanced fixed 2x2 grid of four patch top-left corners."""
    leftover = image_size - patch            # 288 for 416/128
    gap = leftover // 3                     # ~96 between patch origins
    first = (image_size - (patch + gap)) // 2
    starts = [first, first + gap]
    return [(x, y) for y in starts for x in starts]


def read_boxes(label_path, image_size):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = (float(v) for v in parts)
            boxes.append((xc * image_size, yc * image_size,
                          w * image_size, h * image_size))
    return boxes


def overlaps(box, x, y, size, margin):
    xc, yc, w, h = box
    x1 = xc - w / 2.0 - margin
    x2 = xc + w / 2.0 + margin
    y1 = yc - h / 2.0 - margin
    y2 = yc + h / 2.0 + margin
    return not (x2 <= x or x1 >= x + size or y2 <= y or y1 >= y + size)


def main():
    args = parse_args()
    manifest = os.path.join(args.data, 'split_manifest.csv')
    if not os.path.exists(manifest):
        raise SystemExit('split manifest not found: %s' % manifest)
    out = args.out or os.path.join(args.data, 'audit', 'background_patches.csv')

    with open(manifest, newline='') as fh:
        records = list(csv.DictReader(fh))
    if args.split != 'all':
        records = [r for r in records if r['split'] == args.split]

    if args.positions:
        positions = [tuple(int(v) for v in pair.split(','))
                     for pair in args.positions.split(';')]
    else:
        positions = default_positions(IMAGE_SIZE, args.patch)
    if len(positions) != 4:
        raise SystemExit('the protocol places exactly four patches per image')

    n_candidates = 0
    rows = []
    pid = 0
    for rec in records:
        label = os.path.join(
            args.data,
            rec['image'].replace('images/', 'labels/').rsplit('.', 1)[0] + '.txt')
        boxes = read_boxes(label, IMAGE_SIZE)
        for (x, y) in positions:
            n_candidates += 1
            if any(overlaps(b, x, y, args.patch, args.margin) for b in boxes):
                continue
            rows.append([pid, rec['image'], rec['split'], rec['source_group'],
                         x, y, args.patch, rec.get('sha256', '')])
            pid += 1

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['patch_id', 'image', 'split', 'source_group',
                         'x', 'y', 'size', 'sha256'])
        writer.writerows(rows)

    print('split=%s images=%d fixed positions=%d candidates=%d kept=%d '
          'patch=%dx%d margin=%d'
          % (args.split, len(records), len(positions), n_candidates, len(rows),
             args.patch, args.patch, args.margin))
    print('written: %s' % out)


if __name__ == '__main__':
    main()
