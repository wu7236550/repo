"""Pixel-to-millimetre calibration and crack-width binning.

Resolves the ground sampling distance from configs/width_calibration.yaml, prints the
bin edges in pixels, and optionally bins the ground-truth box widths of a label
directory into the three width classes used in the detailed-metrics table.

Usage:
    python calibrate_width.py
    python calibrate_width.py --labels ../dataset/roadrdd/labels/test --split test
"""
import argparse
import csv
import os
import sys


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, '..'))
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--calibration', default=os.path.join(root, 'configs', 'width_calibration.yaml'),
                   help='calibration configuration file')
    p.add_argument('--data', default=os.path.join(root, 'dataset', 'roadrdd'), help='dataset root')
    p.add_argument('--labels', default=None,
                   help='label directory to bin; defaults to <data>/labels/test')
    p.add_argument('--widths', default=None,
                   help='alternative per-instance width table (CSV with a width_px column); '
                        'required to reproduce the width-binned rows of the detailed-metrics '
                        'table, because the released labels carry bounding boxes only')
    p.add_argument('--split', default='test', help='split used for the source-group lookup')
    p.add_argument('--imgsz', type=int, default=416, help='network input resolution')
    return p.parse_args()


def bin_label(bins, index):
    """Human-readable label of the index-th width bin."""
    low = 'any' if index == 0 else '>= %.1f mm' % bins[index - 1]
    high = '' if index == len(bins) else ' and < %.1f mm' % bins[index]
    return low + high


def load_config(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit('PyYAML is required to read %s' % path)
    if not os.path.exists(path):
        raise SystemExit('calibration file not found: %s' % path)
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_mm_per_pixel(cfg):
    geometry = cfg.get('geometry') or {}
    direct = cfg.get('direct') or {}
    height = geometry.get('mounting_height_mm')
    focal_px = geometry.get('focal_length_px')
    ref_mm = direct.get('reference_mm')
    ref_px = direct.get('reference_pixels')

    filled = [bool(height and focal_px), bool(ref_mm and ref_px)]
    if sum(filled) != 1:
        print('the calibration is not resolved.', file=sys.stderr)
        print('', file=sys.stderr)
        print('Fill in exactly one route in %s:' % CALIBRATION_PATH, file=sys.stderr)
        print('  Route A (vehicle-mounted geometry):', file=sys.stderr)
        print('      geometry.mounting_height_mm   <- camera height above the road, in mm', file=sys.stderr)
        print('      geometry.focal_length_px      <- focal_length_mm / pixel_pitch_mm', file=sys.stderr)
        print('  Route B (direct measurement at the working distance):', file=sys.stderr)
        print('      direct.reference_mm           <- physical size of the reference target, in mm', file=sys.stderr)
        print('      direct.reference_pixels       <- number of pixels the target spans', file=sys.stderr)
        print('', file=sys.stderr)
        print('These are physical properties of the acquisition and cannot be derived from',
              file=sys.stderr)
        print('the released images, whose EXIF metadata was stripped by the dataset export.',
              file=sys.stderr)
        raise SystemExit(2)
    if filled[0]:
        return height / focal_px, 'geometry: %.1f mm / %.1f px' % (height, focal_px)
    return ref_mm / ref_px, 'direct: %.2f mm / %.1f px' % (ref_mm, ref_px)


def main():
    global CALIBRATION_PATH
    args = parse_args()
    CALIBRATION_PATH = args.calibration
    cfg = load_config(args.calibration)
    mm_per_pixel, source = resolve_mm_per_pixel(cfg)
    bins = cfg.get('width_bins_mm') or [3.0, 10.0]
    groups = cfg.get('applies_to_source_groups') or []

    print('calibration file : %s' % args.calibration)
    print('mm per pixel     : %.5f  (%s)' % (mm_per_pixel, source))
    print('pixel per mm     : %.4f' % (1.0 / mm_per_pixel))
    print('bin edges        : %s mm  ->  %s px'
          % (', '.join('%.1f' % b for b in bins),
             ', '.join('%.1f' % (b / mm_per_pixel) for b in bins)))
    print('calibrated groups: %s' % (', '.join(groups) if groups else 'all source groups'))
    print('  note: a box of width W pixels is binned as <b0, b0..b1 or >b1, where b0 and b1 '
          'are the bin edges above.')

    labels = args.labels or os.path.join(args.data, 'labels', args.split)
    if args.widths:
        if not os.path.exists(args.widths):
            raise SystemExit('width table not found: %s' % args.widths)
        counts = [0] * (len(bins) + 1)
        with open(args.widths, newline='') as fh:
            for rec in csv.DictReader(fh):
                width_mm = float(rec['width_px']) * mm_per_pixel
                counts[sum(1 for b in bins if width_mm >= b)] += 1
        print('\nwidth-bin population of %s (%d instances):' % (args.widths, sum(counts)))
        for i in range(len(bins) + 1):
            print('   %-22s %8d' % (bin_label(bins, i), counts[i]))
        return

    if not os.path.isdir(labels):
        print('\nlabel directory not found, skipping the width-bin population: %s' % labels)
        return

    groups_of = {}
    manifest = os.path.join(args.data, 'split_manifest.csv')
    if os.path.exists(manifest):
        with open(manifest, newline='') as fh:
            for rec in csv.DictReader(fh):
                groups_of[rec['image'].rsplit('/', 1)[-1].rsplit('.', 1)[0]] = rec['source_group']

    inside = [0] * (len(bins) + 1)
    outside = [0] * (len(bins) + 1)
    for name in sorted(os.listdir(labels)):
        if not name.endswith('.txt'):
            continue
        stem = name[:-4]
        group = groups_of.get(stem, '')
        calibrated = (not groups) or any(group.startswith(g) for g in groups)
        target = inside if calibrated else outside
        with open(os.path.join(labels, name)) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 5:
                    continue
                width_px = float(parts[3]) * args.imgsz
                width_mm = width_px * mm_per_pixel
                index = sum(1 for b in bins if width_mm >= b)
                target[index] += 1

    def edges(i):
        return bin_label(bins, i)

    total = sum(inside) + sum(outside)
    print('\nwidth-bin population of %s (%d boxes):' % (labels, total))
    print('%-22s %10s %10s' % ('bin', 'calibrated', 'uncalibrated'))
    for i in range(len(bins) + 1):
        print('%-22s %10d %10d' % (edges(i), inside[i], outside[i]))
    if total:
        print('\nshare inside a calibrated acquisition geometry: %.1f%%'
              % (100.0 * sum(inside) / total))


if __name__ == '__main__':
    main()
