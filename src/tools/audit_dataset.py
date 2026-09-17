# -*- coding: utf-8 -*-
import io, os, sys, csv, hashlib, collections, shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO = os.path.abspath(sys.argv[1])
DRY = '--write' not in sys.argv
DO_DELETE = '--delete' in sys.argv
DS = os.path.join(REPO, 'dataset', 'roadrdd')
SPLITS = ('train', 'val', 'test')
PRIO = {'train': 0, 'val': 1, 'test': 2}


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stem(fn):
    base = os.path.splitext(fn)[0]
    if '.rf.' in base:
        base = base.split('.rf.')[0]
    return base


items = []
for s in SPLITS:
    d = os.path.join(DS, 'images', s)
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        items.append({'split': s, 'fn': fn, 'path': os.path.join(d, fn)})

print('Scanned %d images (train=%d val=%d test=%d)' % (
    len(items),
    sum(1 for i in items if i['split'] == 'train'),
    sum(1 for i in items if i['split'] == 'val'),
    sum(1 for i in items if i['split'] == 'test')))

for it in items:
    it['sha'] = sha256(it['path'])
    it['stem'] = stem(it['fn'])

byh = collections.defaultdict(list)
for it in items:
    byh[it['sha']].append(it)

groups = {h: v for h, v in byh.items() if len(v) > 1}
print('Byte-level duplicate groups: %d, covering %d files' % (len(groups), sum(len(v) for v in groups.values())))

removed, kept = [], []
for h, members in groups.items():
    members_sorted = sorted(members, key=lambda x: (PRIO[x['split']], x['fn']))
    kept.append(members_sorted[0])
    removed.extend(members_sorted[1:])

print('Keeping %d, removing %d redundant copies' % (len(kept), len(removed)))
cross = collections.Counter()
for h, members in groups.items():
    ss = tuple(sorted({m['split'] for m in members}))
    cross[ss] += 1
for k, v in sorted(cross.items()):
    print('   group %-24s %d' % ('+'.join(k), v))

n_after = len(items) - len(removed)
per_after = collections.Counter(i['split'] for i in items if i not in removed)
print('After de-duplication: %d images (train=%d val=%d test=%d)' % (
    n_after, per_after['train'], per_after['val'], per_after['test']))

if not DO_DELETE:
    print('\n[PLAN] no file deleted (pass --delete to remove them). Removal list: dataset/roadrdd/audit/dedup_plan.txt')

os.makedirs(os.path.join(DS, 'audit'), exist_ok=True)
with open(os.path.join(DS, 'audit', 'dedup_plan.txt'), 'w', encoding='utf-8', newline='\n') as f:
    for it in sorted(removed, key=lambda x: (PRIO[x['split']], x['fn'])):
        f.write('dataset/roadrdd/images/%s/%s\n' % (it['split'], it['fn']))
        f.write('dataset/roadrdd/labels/%s/%s\n' % (it['split'], os.path.splitext(it['fn'])[0] + '.txt'))

if DO_DELETE:
    n = 0
    for it in removed:
        os.remove(it['path'])
        lp = os.path.join(DS, 'labels', it['split'], os.path.splitext(it['fn'])[0] + '.txt')
        if os.path.exists(lp):
            os.remove(lp)
        n += 1
    print('[OK] removed %d redundant copies and their labels' % n)

survivors = [i for i in items if i not in removed]

with open(os.path.join(DS, 'MANIFEST.sha256'), 'w', encoding='utf-8', newline='\n') as f:
    for it in sorted(survivors, key=lambda x: (x['split'], x['fn'])):
        f.write('%s  images/%s/%s\n' % (it['sha'], it['split'], it['fn']))

with open(os.path.join(DS, 'split_manifest.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image', 'split', 'source_group', 'sha256'])
    for it in sorted(survivors, key=lambda x: (x['split'], x['fn'])):
        w.writerow(['images/%s/%s' % (it['split'], it['fn']), it['split'], it['stem'], it['sha']])

with open(os.path.join(DS, 'audit', 'dedup_log.csv'), 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group_id', 'sha256', 'kept_path', 'removed_paths', 'splits_involved'])
    for gi, (h, members) in enumerate(sorted(groups.items()), 1):
        ms = sorted(members, key=lambda x: (PRIO[x['split']], x['fn']))
        w.writerow([gi, h,
                    'images/%s/%s' % (ms[0]['split'], ms[0]['fn']),
                    '|'.join('images/%s/%s' % (m['split'], m['fn']) for m in ms[1:]),
                    '+'.join(sorted({m['split'] for m in members}))])

by_stem = collections.defaultdict(set)
for it in survivors:
    by_stem[it['stem']].add(it['split'])
resid = {k: v for k, v in by_stem.items() if len(v) > 1}

inst = collections.Counter()
for it in survivors:
    lp = os.path.join(DS, 'labels', it['split'], os.path.splitext(it['fn'])[0] + '.txt')
    if not os.path.exists(lp):
        continue
    for line in open(lp, encoding='utf-8', errors='ignore'):
        if line.strip():
            inst[line.split()[0]] += 1

with open(os.path.join(DS, 'audit', 'dataset_audit.md'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('# roadrdd dataset audit report\n\n')
    f.write('Generated by `src/tools/audit_dataset.py` (SHA-256 byte-level audit).\n\n')
    f.write('## 1. Byte-level duplicate audit\n\n')
    f.write('| Item | Value |\n|---|---|\n')
    f.write('| Images scanned | %d |\n' % len(items))
    f.write('| Duplicate groups (identical SHA-256) | %d |\n' % len(groups))
    f.write('| Redundant copies removed | %d |\n' % len(removed))
    f.write('| Images after deduplication | %d |\n' % n_after)
    f.write('\nCross-split duplicate groups:\n\n| Splits involved | Groups |\n|---|---|\n')
    for k, v in sorted(cross.items()):
        f.write('| %s | %d |\n' % ('+'.join(k), v))
    f.write('\n## 2. Split composition after deduplication\n\n')
    f.write('| Split | Images |\n|---|---|\n')
    for s in SPLITS:
        f.write('| %s | %d |\n' % (s, per_after[s]))
    f.write('| **Total** | **%d** |\n' % n_after)
    f.write('\n## 3. Instance counts after deduplication\n\n')
    f.write('| Class | Label | Instances |\n|---|---|---|\n')
    f.write('| Crack | crack (0) | %d |\n' % inst.get('0', 0))
    f.write('| Pothole | pothole (1) | %d |\n' % inst.get('1', 0))
    f.write('| **Total** | | **%d** |\n' % sum(inst.values()))
    f.write('\n## 4. Residual source-image grouping\n\n')
    f.write('`source_group` in `split_manifest.csv` strips the Roboflow `.rf.<hash>` suffix, '
            'so all augmentations of one source image share the same group id.\n\n')
    f.write('| Item | Value |\n|---|---|\n')
    f.write('| Distinct source groups | %d |\n' % len(by_stem))
    f.write('| Source groups still spanning >1 split | %d |\n' % len(resid))
    f.write('\nSee `dedup_log.csv` for the full group-level record and `MANIFEST.sha256` for '
            'per-image hashes.\n')

print('\n[OK] manifests written (after de-duplication: %d images)' % n_after)
print('     MANIFEST.sha256 / split_manifest.csv / audit/dedup_log.csv / audit/dataset_audit.md / audit/dedup_plan.txt')
print('     source groups still spanning splits: %d / %d' % (len(resid), len(by_stem)))
print('     instances: crack=%d pothole=%d total=%d' % (inst.get('0', 0), inst.get('1', 0), sum(inst.values())))
