"""Deterministic selection of the background-only patches used for the FP/image metric.

A background patch is a P x P crop of a released image that does not intersect any
annotated bounding box, dilated by ``--margin`` pixels. Candidates are enumerated on a
regular grid with the given stride and ranked by the SHA-256 digest of
(seed, image, x, y); the first ``--n`` candidates in that order are selected.

Ranking by a digest rather than by a pseudo-random generator makes the selection
independent of the Python and NumPy versions: the same manifest is produced on any
platform. The script verifies that none of the selected patches overlaps an annotated
box and refuses to write the manifest if the check fails.

The metric is defined for road-surface imagery, so the three groups whose surface is
not a road are excluded by default: ``noncrack_*`` (concrete-wall non-crack frames),
``Eugen_Muller_*`` (tunnel lining) and ``Volker_*`` (concrete facade).

Usage:
    python make_background_patches.py --data ../dataset/roadrdd \
        --out ../dataset/roadrdd/audit/background_patches.csv
"""
import argparse
import csv
import hashlib
import os

IMAGE_SIZE = 416


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(os.path.join(here, '..', 'dataset', 'roadrdd'))
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', default=default_data,
                   help='dataset root holding images/, labels/ and split_manifest.csv')
    p.add_argument('--out', default=None,
                   help='output CSV (default: <data>/audit/background_patches.csv)')
    p.add_argument('--split', default='test',
                   help="split to sample from ('test', 'val', 'train' or 'all')")
    p.add_argument('--patch', type=int, default=128, help='patch side in pixels')
    p.add_argument('--stride', type=int, default=64, help='grid stride in pixels')
    p.add_argument('--margin', type=int, default=16,
                   help='a patch is rejected if it comes this close to an annotated box')
    p.add_argument('--n', type=int, default=120, help='number of patches to select')
    p.add_argument('--seed', type=int, default=0, help='seed mixed into the ranking digest')
    p.add_argument('--exclude-prefix', default='noncrack_,Eugen_Muller_,Volker_',
                   help='comma-separated file-name prefixes to skip; the default drops the '
                        'groups whose surface is concrete rather than road')
    p.add_argument('--save-crops', default=None,
                   help='optional directory in which to also write the selected crops')
    return p.parse_args()


def read_boxes(label_path, image_size):
    """Return the annotated boxes of one image in pixel coordinates."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = (float(v) for v in parts)
            boxes.append((xc * image_size, yc * image_size, w * image_size, h * image_size))
    return boxes


def overlaps(box, patch, margin):
    """True if the axis-aligned box, dilated by ``margin``, intersects the patch."""
    xc, yc, w, h = box
    x1 = xc - w / 2.0 - margin
    x2 = xc + w / 2.0 + margin
    y1 = yc - h / 2.0 - margin
    y2 = yc + h / 2.0 + margin
    px, py, ps = patch
    return not (x2 <= px or x1 >= px + ps or y2 <= py or y1 >= py + ps)


def rank_key(seed, image, x, y):
    return hashlib.sha256(('%d|%s|%d|%d' % (seed, image, x, y)).encode('utf-8')).hexdigest()


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
    n_in_split = len(records)
    skip = tuple(s for s in (p.strip() for p in args.exclude_prefix.split(',')) if s)
    if skip:
        records = [r for r in records
                   if not os.path.basename(r['image']).startswith(skip)]

    limit = IMAGE_SIZE - args.patch
    if limit < 0:
        raise SystemExit('patch size %d exceeds the %d-pixel images' % (args.patch, IMAGE_SIZE))
    grid = list(range(0, limit + 1, args.stride))

    candidates = []
    for rec in records:
        label = os.path.join(args.data,
                             rec['image'].replace('images/', 'labels/').rsplit('.', 1)[0] + '.txt')
        boxes = read_boxes(label, IMAGE_SIZE)
        for y in grid:
            for x in grid:
                patch = (x, y, args.patch)
                if any(overlaps(b, patch, args.margin) for b in boxes):
                    continue
                candidates.append((rank_key(args.seed, rec['image'], x, y), rec, x, y))

    candidates.sort(key=lambda c: c[0])
    if len(candidates) < args.n:
        raise SystemExit('only %d eligible patches found, %d requested'
                         % (len(candidates), args.n))
    selected = candidates[:args.n]

    # the selection must be verifiable, so re-check every patch before writing
    for _, rec, x, y in selected:
        label = os.path.join(args.data,
                             rec['image'].replace('images/', 'labels/').rsplit('.', 1)[0] + '.txt')
        boxes = read_boxes(label, IMAGE_SIZE)
        if any(overlaps(b, (x, y, args.patch), args.margin) for b in boxes):
            raise SystemExit('selected patch %s (%d, %d) overlaps an annotation' % (rec['image'], x, y))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['patch_id', 'image', 'split', 'source_group', 'x', 'y', 'size', 'sha256'])
        for i, (_, rec, x, y) in enumerate(selected):
            writer.writerow([i, rec['image'], rec['split'], rec['source_group'], x, y,
                             args.patch, rec['sha256']])

    if args.save_crops:
        from PIL import Image
        os.makedirs(args.save_crops, exist_ok=True)
        for i, (_, rec, x, y) in enumerate(selected):
            src = Image.open(os.path.join(args.data, rec['image']))
            src.crop((x, y, x + args.patch, y + args.patch)).save(
                os.path.join(args.save_crops, '%03d.jpg' % i), quality=95)

    groups = {}
    for _, rec, _, _ in selected:
        key = rec['source_group'].split('_')[0]
        groups[key] = groups.get(key, 0) + 1
    print('split=%s  images=%d (%d after group exclusion)  candidates=%d  selected=%d  '
          'patch=%dx%d  stride=%d  margin=%d  seed=%d'
          % (args.split, n_in_split, len(records), len(candidates), len(selected),
             args.patch, args.patch, args.stride, args.margin, args.seed))
    print('excluded file-name prefixes: %s' % (', '.join(skip) if skip else '(none)'))
    print('source-group composition of the selection (top 12):')
    for key, count in sorted(groups.items(), key=lambda kv: -kv[1])[:12]:
        print('   %-14s %3d' % (key, count))
    print('written: %s' % out)


if __name__ == '__main__':
    main()
