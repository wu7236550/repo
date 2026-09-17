# -*- coding: utf-8 -*-
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--imgsz', type=int, nargs='+', default=[416, 640], help='resolutions to profile')
    p.add_argument('--cfg', type=str, default=None, help='profile a single config instead of all')
    p.add_argument('--out', type=str, default='flops_fvcore.csv')
    p.add_argument('--ops-out', type=str, default='flops_ops_breakdown.csv')
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
            print('%-34s %4dx%-4d  %8.2f GFLOPs   %6.2f M params' % (label, size, size, g, row['params_M']))
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
