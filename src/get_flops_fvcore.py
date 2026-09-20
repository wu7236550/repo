# -*- coding: utf-8 -*-
"""Operator-level complexity profiler (fvcore diagnostic).

Manuscript Section 3.4 states that parameters and FLOPs are counted with fvcore on a single
1x3xHxW tensor, counting convolution and matrix-multiplication MACs multiplied by two, with
batch normalization, activations, the softmax of self-attention and the LRPB MLP evaluations
reported separately.

`--double-macs` implements exactly that stated rule: it sums the MACs fvcore attributes to
`conv`, `linear`, `matmul` and `bmm` and multiplies by two. Two caveats are worth recording,
because they are why this script is a diagnostic rather than the source of the manuscript
tables:

* fvcore counts one multiply-accumulate as one FLOP, so its plain total is about half of the
  THOP-style convention used for the headline numbers.
* The reported headline GFLOPs of Tables 2 and 5(a) are reproduced by
  `get_all_yaml_param_and_flops.py` (the ultralytics `model_info` THOP path, two operations
  per MAC, on the fused model). The two profilers do not agree exactly, because they count
  different operator sets and fvcore cannot see every operator of the custom deformable
  attention in the same way. See `../docs/manuscript_alignment.md`.

Run from inside <repo>/src.
"""
import argparse
import csv
import os

import torch

CFGS = [
    ('r18 (baseline)', 'rtdetr-r18.yaml'),
    ('+ MSPC', 'rtdetr-MSPC.yaml'),
    ('+ MCAF', 'rtdetr-MCAF.yaml'),
    ('+ LRPB-AIFI', 'rtdetr-LRPB-AIFI.yaml'),
    ('+ MSPC + MCAF', 'rtdetr-MSPC_MCAF.yaml'),
    ('+ MSPC + LRPB-AIFI', 'rtdetr-MSPC_LRPB-AIFI.yaml'),
    ('+ MCAF + LRPB-AIFI', 'rtdetr-MCAF_LRPB-AIFI.yaml'),
    ('+ all three (proposed)', 'rtdetr-MSPC_MCAF_LRPB-AIFI.yaml'),
]
CFG_DIR = 'ultralytics/cfg/models/rt-detr'
NC = 2

# Operators that carry multiply-accumulates, i.e. the "conv + matrix multiplication" set of
# manuscript Section 3.4.  `linear`, `matmul` and `bmm` are the matrix-multiplication forms;
# `conv` covers convolution, including the depthwise and grouped variants.
MAC_OPS = ('conv', 'linear', 'matmul', 'bmm')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--imgsz', type=int, nargs='+', default=[416, 640], help='resolutions to profile')
    p.add_argument('--cfg', type=str, default=None, help='profile a single config instead of all')
    p.add_argument('--out', type=str, default='flops_fvcore.csv')
    p.add_argument('--ops-out', type=str, default='flops_ops_breakdown.csv')
    p.add_argument('--double-macs', action='store_true',
                   help='also report conv + matmul MACs multiplied by two, the counting rule '
                        'stated in manuscript Section 3.4')
    return p.parse_args()


def build(cfg_path):
    from ultralytics import RTDETR
    model = RTDETR(cfg_path)
    model.model.nc = NC
    model.model.eval()
    return model.model


def profile(model, size):
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    x = torch.zeros(1, 3, size, size)
    fca = FlopCountAnalysis(model, x)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    total = fca.total()
    return total / 1e9, fca, flop_count_table(fca)


def double_macs_gflops(fca):
    """conv + matrix-multiplication MACs x 2, per manuscript Section 3.4."""
    by_op = fca.by_operator()
    macs = sum(v for k, v in by_op.items() if k in MAC_OPS)
    return 2.0 * macs / 1e9


def main():
    opt = parse_args()
    cfgs = [(f, s) for f, s in CFGS if opt.cfg is None or s == os.path.basename(opt.cfg)]
    if not cfgs:
        cfgs = [('custom', opt.cfg)]

    rows, op_rows = [], []
    for label, yaml_name in cfgs:
        cfg = opt.cfg if opt.cfg else os.path.join(CFG_DIR, yaml_name)
        if not os.path.exists(cfg):
            print('SKIP (not found): %s' % cfg)
            continue
        model = build(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        row = {'config': label, 'cfg': cfg, 'params_M': n_params / 1e6}
        for size in opt.imgsz:
            g, fca, table = profile(model, size)
            row['gflops_%d' % size] = g
            if opt.double_macs:
                row['gflops_double_macs_%d' % size] = double_macs_gflops(fca)
            print('%-34s %4dx%-4d  %8.2f GFLOPs (fvcore total)   %6.2f M params'
                  % (label, size, size, g, row['params_M']))
            if opt.double_macs:
                print('%-34s %4dx%-4d  %8.2f GFLOPs (conv + matmul MACs x 2)'
                      % ('', size, size, row['gflops_double_macs_%d' % size]))
            for name, flops in fca.by_operator().items():
                op_rows.append({'config': label, 'imgsz': size, 'operator': name, 'gflops': flops / 1e9})
        if len(opt.imgsz) == 2:
            row['ratio_%d_over_%d' % (opt.imgsz[1], opt.imgsz[0])] = \
                row['gflops_%d' % opt.imgsz[1]] / row['gflops_%d' % opt.imgsz[0]]
        rows.append(row)

    keys = sorted({k for r in rows for k in r})
    with open(opt.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(opt.ops_out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'imgsz', 'operator', 'gflops'])
        w.writeheader()
        w.writerows(op_rows)
    print('\nwrote %s and %s' % (opt.out, opt.ops_out))


if __name__ == '__main__':
    main()
