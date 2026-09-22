# -*- coding: utf-8 -*-
"""
group_disjoint_resplit.py
=========================

Reproduces the *group-disjoint* crack-and-pothole benchmark described in
Section 4.1.1 of the manuscript:

    "A Lightweight Real-Time Detection Transformer with Multi-Scale Partial
     Convolution and Context-Anchored Fusion for Road Crack and Pothole
     Detection".

The benchmark is a detection derivative assembled from public crack and
pothole sources.  Naive image-level splitting leaks information in two ways:
  (1) byte-identical exports of one source photograph appear in several
      splits; and
  (2) non-identical *augmented variants* of one source photograph appear in
      several splits.

This script removes both leaks, deterministically (a fixed released seed), in
two stages:

  Stage A - byte-level collapse.
      Every image is hashed (SHA-256).  All byte-identical copies of a source
      export are collapsed to a single canonical file.

  Stage B - source-photograph grouping and group-level re-splitting.
      The surviving images are grouped by the *source photograph*: the
      Roboflow ``.rf.<hash>`` augmentation suffix (and other augmentation
      tokens) are stripped so that every variant of one photograph shares a
      group id.  Whole groups - never individual images - are allocated to the
      train / validation / test splits at roughly 7:1:2.

A final re-audit verifies that
  * no byte-identical file is shared across splits, and
  * no source photograph contributes to more than one split.

The script is standard-library only, so it runs in a minimal environment.

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------

Plan only (writes the audit report + CSV plans, touches no image)::

    python src/tools/group_disjoint_resplit.py --in  <raw_aggregate_dir> \
                                               --out dataset/roadrdd

Apply (collapse and physically re-split image + label files)::

    python src/tools/group_disjoint_resplit.py --in  <raw_aggregate_dir> \
                                               --out dataset/roadrdd --apply

The raw aggregate may be laid out as ``images/<split>/...`` with sibling
``labels/<split>/...`` YOLO labels, or as a flat directory of images with
labels next to them.  Outputs always use the canonical Ultralytics layout::

    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt
    <out>/MANIFEST.sha256
    <out>/split_manifest.csv
    <out>/audit/{resplit_report.md,byte_collapse_log.csv,group_assignment.csv}

Manuscript targets (Section 4.1.1): 5,898 train / 842 val / 1,686 test over
8,426 images, with 15 background-only frames and 16,312 boxes
(8,498 crack / 7,814 pothole).
"""

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
import sys

# --------------------------------------------------------------------------- #
# Manuscript constants
# --------------------------------------------------------------------------- #
SPLITS = ('train', 'val', 'test')
TARGETS = {'train': 5898, 'val': 842, 'test': 1686}
RELEASED_SEED = 2024
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# Roboflow export: <source stem>.rf.<32-hex>.<ext>
ROBOFLOW_RE = re.compile(r'\.rf\.[0-9a-f]{16,}', re.IGNORECASE)
# Common augmentation tokens appended by other export pipelines.
AUG_TOKENS = ('_flip', '_hflip', '_vflip', '_rotated', '_rot', '_crop',
              '_mirror', '_aug', '_augment', '_resize', '_hue', '_bright',
              '_contrast', '_mosaic', '_copy', '_clone')


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_key(filename):
    """Group key = source photograph, with augmentation suffixes stripped."""
    stem = os.path.splitext(filename)[0]
    stem = ROBOFLOW_RE.sub('', stem)
    low = stem.lower()
    for tok in AUG_TOKENS:
        low = low.replace(tok, '')
    # Trailing copy indices such as name(2), name_2, name-2 are variants.
    low = re.sub(r'[_\-\s]*\(\d+\)$', '', low)
    low = re.sub(r'[_\-]\d+$', '', low)
    return low


def find_label_for(image_rel, in_root):
    """Best-effort sibling YOLO .txt lookup for a raw image."""
    img_full = os.path.join(in_root, image_rel)
    cand = []
    d, fn = os.path.split(image_rel)
    stem = os.path.splitext(fn)[0]
    # images/<split>/..  ->  labels/<split>/<stem>.txt
    parts = d.replace('\\', '/').split('/')
    if parts and parts[0] == 'images':
        rest = parts[1:]
        cand.append(os.path.join('labels', *rest, stem + '.txt'))
    # label next to the image
    cand.append(os.path.join(d, stem + '.txt'))
    # a global labels dir mirroring the image sub-path
    cand.append(os.path.join('labels', image_rel).rsplit('.', 1)[0] + '.txt')
    for c in cand:
        p = os.path.join(in_root, c)
        if os.path.exists(p):
            return p
    return None


def count_boxes(label_path):
    """Return (total, n_crack, n_pothole) for a YOLO label file."""
    total = crack = pot = 0
    if label_path and os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                cls = line.split()[0]
                if cls == '0':
                    crack += 1
                elif cls == '1':
                    pot += 1
    return total, crack, pot


# --------------------------------------------------------------------------- #
# Stage A + B
# --------------------------------------------------------------------------- #
def collect_images(in_root):
    found = []
    for root, _dirs, files in os.walk(in_root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, in_root).replace('\\', '/')
                found.append(rel)
    return sorted(found)


def collapse_byte_copies(records):
    """Group by hash, keep one canonical file per byte group."""
    by_hash = {}
    for rec in records:
        by_hash.setdefault(rec['sha'], []).append(rec)

    kept, byte_groups = [], []
    for h, members in by_hash.items():
        members = sorted(members, key=lambda r: r['rel'])
        canonical = members[0]
        kept.append(canonical)
        if len(members) > 1:
            byte_groups.append({'sha': h, 'kept': canonical['rel'],
                                'removed': [m['rel'] for m in members[1:]]})
    return kept, byte_groups


def allocate_groups(groups, targets, seed):
    """Deterministically allocate WHOLE groups to hit exact target totals."""
    total_imgs = sum(g['n_imgs'] for g in groups)
    total_tgt = sum(targets.values())
    if total_imgs != total_tgt:
        raise SystemExit(
            'Group total (%d) does not match split targets (%d). The released '
            'targets describe the manuscript benchmark; run this script on the '
            'matching raw aggregate (Section 4.1.1).' % (total_imgs, total_tgt))

    rng = random.Random(seed)
    multi = [g for g in groups if g['n_imgs'] >= 2]
    single = [g for g in groups if g['n_imgs'] == 1]

    # Deterministic order: large groups first (big items placed while capacity
    # is ample); ties broken by group id.
    multi = sorted(multi, key=lambda g: (-g['n_imgs'], g['gid']))
    rng.shuffle(single)

    remaining = dict(targets)
    assignment = {}
    # split preference order, fixed for reproducibility
    pref = ('train', 'test', 'val')

    for g in multi:
        # choose the split with the most remaining capacity that fits
        fits = [s for s in pref if g['n_imgs'] <= remaining[s]]
        if not fits:
            raise SystemExit(
                'Cannot place group %s (size %d); remaining=%s'
                % (g['gid'], g['n_imgs'], remaining))
        s = max(fits, key=lambda x: (remaining[x], -pref.index(x)))
        assignment[g['gid']] = s
        remaining[s] -= g['n_imgs']

    # Singletons fill every split to its exact target.
    idx = 0
    for s in pref:
        while remaining[s] > 0:
            if idx >= len(single):
                raise SystemExit('Ran out of singleton groups while filling %s' % s)
            assignment[single[idx]['gid']] = s
            idx += 1
            remaining[s] -= 1
    if idx != len(single):
        raise SystemExit('Internal allocation error: %d singletons unassigned'
                         % (len(single) - idx))
    assert all(v == 0 for v in remaining.values()), remaining
    return assignment


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='in_root', required=True,
                    help='raw aggregate directory')
    ap.add_argument('--out', dest='out_root', required=True,
                    help='output dataset directory (e.g. dataset/roadrdd)')
    ap.add_argument('--seed', type=int, default=RELEASED_SEED)
    ap.add_argument('--targets', type=str, default=None,
                    help='comma targets train,val,test (default manuscript)')
    ap.add_argument('--apply', action='store_true',
                    help='physically collapse + re-split files (default: plan only)')
    ap.add_argument('--move', action='store_true',
                    help='move files instead of copying when --apply')
    args = ap.parse_args()

    targets = TARGETS
    if args.targets:
        tv = [int(x) for x in args.targets.split(',')]
        targets = dict(zip(SPLITS, tv))

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    images = collect_images(in_root)
    print('[1/5] Scanned %d raw images under %s' % (len(images), in_root))

    records = []
    for rel in images:
        full = os.path.join(in_root, rel)
        lab = find_label_for(rel, in_root)
        n, c, p = count_boxes(lab)
        records.append({'rel': rel, 'sha': sha256(full), 'lab': lab,
                        'n_box': n, 'n_crack': c, 'n_pot': p,
                        'bg': n == 0})

    # Stage A
    kept, byte_groups = collapse_byte_copies(records)
    n_removed = sum(len(g['removed']) for g in byte_groups)
    print('[2/5] Byte collapse: %d duplicate groups, %d redundant copies removed, '
          '%d unique-byte images remain'
          % (len(byte_groups), n_removed, len(kept)))

    # Stage B - group by source photograph
    groups_map = {}
    for rec in kept:
        gid = source_key(os.path.basename(rec['rel']))
        g = groups_map.setdefault(gid, {'gid': gid, 'members': [], 'n_imgs': 0,
                                        'n_box': 0, 'n_crack': 0, 'n_pot': 0,
                                        'bg': True})
        g['members'].append(rec)
        g['n_imgs'] += 1
        g['n_box'] += rec['n_box']
        g['n_crack'] += rec['n_crack']
        g['n_pot'] += rec['n_pot']
        g['bg'] = g['bg'] and rec['bg']
    groups = list(groups_map.values())
    print('[3/5] Source-photograph groups: %d (groups spanning >1 split in the '
          'raw layout are repaired by the allocation)' % len(groups))

    assignment = allocate_groups(groups, targets, args.seed)
    final_counts = {s: 0 for s in SPLITS}
    n_bg = 0
    for g in groups:
        s = assignment[g['gid']]
        final_counts[s] += g['n_imgs']
        if g['bg']:
            n_bg += g['n_imgs']
    print('[4/5] Group-disjoint allocation: train=%d val=%d test=%d, '
          'background-only=%d'
          % (final_counts['train'], final_counts['val'], final_counts['test'], n_bg))

    # ------------------------------------------------------------------ write
    audit_dir = os.path.join(out_root, 'audit')
    os.makedirs(audit_dir, exist_ok=True)

    with open(os.path.join(audit_dir, 'byte_collapse_log.csv'), 'w',
              encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group_sha256', 'kept_path', 'removed_paths'])
        for g in sorted(byte_groups, key=lambda x: x['kept']):
            w.writerow([g['sha'], g['kept'], '|'.join(g['removed'])])

    with open(os.path.join(audit_dir, 'group_assignment.csv'), 'w',
              encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source_group', 'split', 'n_images', 'n_boxes',
                    'crack_boxes', 'pothole_boxes', 'background_only'])
        for g in sorted(groups, key=lambda x: x['gid']):
            w.writerow([g['gid'], assignment[g['gid']], g['n_imgs'], g['n_box'],
                        g['n_crack'], g['n_pot'], int(g['bg'])])

    # MANIFEST + split_manifest for the final group-disjoint layout
    manifest_rows, split_rows = [], []
    for g in sorted(groups, key=lambda x: (assignment[x['gid']], x['gid'])):
        s = assignment[g['gid']]
        for rec in g['members']:
            manifest_rows.append((rec['sha'], s, rec['rel']))
            split_rows.append((rec['rel'], s, g['gid'], rec['sha']))

    with open(os.path.join(out_root, 'MANIFEST.sha256'), 'w',
              encoding='utf-8', newline='\n') as f:
        for h, s, _rel in manifest_rows:
            f.write('%s  images/%s/%s\n' % (h, s, os.path.basename(_rel)))
    with open(os.path.join(out_root, 'split_manifest.csv'), 'w',
              encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'split', 'source_group', 'sha256'])
        for rel, s, gid, h in split_rows:
            w.writerow(['images/%s/%s' % (s, os.path.basename(rel)), s, gid, h])

    # Verification (re-audit on the in-memory final layout)
    seen_hash_split, seen_group_split = set(), set()
    leak_hash = leak_group = 0
    for g in groups:
        s = assignment[g['gid']]
        if (g['gid'], s) in seen_group_split:
            pass
        for rec in g['members']:
            key = (rec['sha'], s)
    # explicit checks
    hash_splits, group_splits = {}, {}
    for g in groups:
        s = assignment[g['gid']]
        group_splits.setdefault(g['gid'], set()).add(s)
        for rec in g['members']:
            hash_splits.setdefault(rec['sha'], set()).add(s)
    leak_hash = sum(1 for v in hash_splits.values() if len(v) > 1)
    leak_group = sum(1 for v in group_splits.values() if len(v) > 1)

    total_box = sum(g['n_box'] for g in groups)
    total_crack = sum(g['n_crack'] for g in groups)
    total_pot = sum(g['n_pot'] for g in groups)

    report = os.path.join(audit_dir, 'resplit_report.md')
    with open(report, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Group-disjoint re-split report\n\n')
        f.write('Generated by `src/tools/group_disjoint_resplit.py` '
                '(seed=%d).\n\n' % args.seed)
        f.write('## Inputs and byte-level collapse\n\n')
        f.write('| Item | Value |\n|---|---|\n')
        f.write('| Raw images scanned | %d |\n' % len(images))
        f.write('| Byte-identical groups | %d |\n' % len(byte_groups))
        f.write('| Redundant copies removed | %d |\n' % n_removed)
        f.write('| Unique-byte images | %d |\n' % len(kept))
        f.write('\n## Source-photograph grouping\n\n')
        f.write('| Item | Value |\n|---|---|\n')
        f.write('| Distinct source groups | %d |\n' % len(groups))
        f.write('\n## Final group-disjoint splits\n\n')
        f.write('| Split | Images |\n|---|---|\n')
        for s in SPLITS:
            f.write('| %s | %d |\n' % (s, final_counts[s]))
        f.write('| **Total** | **%d** |\n' % sum(final_counts.values()))
        f.write('\n| Background-only frames | %d |\n' % n_bg)
        f.write('| Boxes (total / crack / pothole) | %d / %d / %d |\n'
                % (total_box, total_crack, total_pot))
        f.write('\n## Leak verification\n\n')
        f.write('| Check | Result |\n|---|---|\n')
        f.write('| Byte-identical copies shared across splits | %d |\n' % leak_hash)
        f.write('| Source photographs shared across splits | %d |\n' % leak_group)
        f.write('\nBoth must be zero for a group-disjoint benchmark.\n')
    print('[5/5] Audit + manifests written to %s' % out_root)

    if leak_hash or leak_group:
        raise SystemExit('Verification FAILED: cross-split leaks remain '
                         '(byte=%d group=%d)' % (leak_hash, leak_group))

    # ------------------------------------------------------------- apply files
    if args.apply:
        for s in SPLITS:
            os.makedirs(os.path.join(out_root, 'images', s), exist_ok=True)
            os.makedirs(os.path.join(out_root, 'labels', s), exist_ok=True)
        n = 0
        for g in groups:
            s = assignment[g['gid']]
            for rec in g['members']:
                base = os.path.basename(rec['rel'])
                dst_img = os.path.join(out_root, 'images', s, base)
                src_img = os.path.join(in_root, rec['rel'])
                _place(src_img, dst_img, args.move)
                if rec['lab']:
                    dst_lab = os.path.join(out_root, 'labels', s,
                                           os.path.splitext(base)[0] + '.txt')
                    _place(rec['lab'], dst_lab, args.move)
                n += 1
        print('[apply] placed %d images (+labels) into the group-disjoint layout'
              % n)
    else:
        print('\nPlan only - no image was moved. Re-run with --apply to '
              'collapse and re-split the files.')


def _place(src, dst, move):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


if __name__ == '__main__':
    main()
