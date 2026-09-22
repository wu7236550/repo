# -*- coding: utf-8 -*-
"""
profile_model.py
================

Reproduces the model-complexity and runtime profiling reported in the paper
(Section 4.2, Tables 3-5): trainable parameters, GFLOPs at 416 x 416 and
inference speed.

Headline parameters and GFLOPs use the same convention as the manuscript - the
Ultralytics ``model_info`` (THOP) count of two floating-point operations per
multiply-accumulate, on the fused inference graph. A raw ``thop`` pass is also
recorded as a cross-check. Throughput (FPS) is hardware dependent: the paper
numbers were measured on an NVIDIA RTX A6000 (CUDA 11.8). On a CPU host this
script records a clearly-labelled CPU figure; reproduce the GPU figure with
``--device cuda``.

Usage
------
    python tools/profile_model.py                    # full + baseline
    python tools/profile_model.py --device cuda      # reproduce GPU FPS
    python tools/profile_model.py --all              # every rt-detr cfg
    python tools/profile_model.py --out logs/profiler
"""

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

CFG_DIR = os.path.join(_SRC, 'ultralytics', 'cfg', 'models', 'rt-detr')

DEFAULT_MODELS = [
    ('baseline_RT-DETR-R18', 'rtdetr-r18.yaml'),
    ('full_MSPC-MCAF-LRPB', 'rtdetr-MSPC_MCAF_LRPB-AIFI.yaml'),
]


def all_model_configs():
    return [(fn.replace('rtdetr-', '').replace('.yaml', ''), fn)
            for fn in sorted(os.listdir(CFG_DIR)) if fn.endswith('.yaml')]


def headline_params_flops(cfg_file, imgsz):
    """Fused model_info (THOP) figures - the manuscript convention."""
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import model_info
    m = RTDETR(cfg_file)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            m.fuse()
        except Exception:
            pass
        n_layers, n_params, n_grads, flops = model_info(m.model, imgsz=imgsz)
    return n_layers, n_params, n_grads, flops, m


def thop_crosscheck(cfg_file, imgsz, device):
    from ultralytics.nn.tasks import DetectionModel
    try:
        from thop import profile
        model = DetectionModel(cfg_file, nc=2, verbose=False).to(device).eval()
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with contextlib.redirect_stdout(io.StringIO()):
            macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs
    except Exception:
        return None


def measure_latency(rtmodel, imgsz, device, runs):
    model = rtmodel.model.to(device).eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(min(10, runs)):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
    return dt * 1e3, 1.0 / dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default=os.path.join(_SRC, '..', 'logs', 'profiler'))
    args = ap.parse_args()

    models = all_model_configs() if args.all else DEFAULT_MODELS
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    summary = []

    for tag, fn in models:
        cfg_path = os.path.join(CFG_DIR, fn)
        if not os.path.exists(cfg_path):
            print('!! missing config %s' % cfg_path)
            continue
        print('Profiling %s ...' % tag)
        n_layers, n_params, n_grads, flops_info, rtmodel = headline_params_flops(
            cfg_path, args.imgsz)
        macs = thop_crosscheck(cfg_path, args.imgsz, device)
        thop_gflops = (2 * macs / 1e9) if macs else float('nan')
        lat, fps = measure_latency(rtmodel, args.imgsz, device, args.runs)
        summary.append((tag, n_params, flops_info, fps))

        log_path = os.path.join(out_dir, 'profile_%s.log' % tag)
        rel_cfg = os.path.relpath(cfg_path, os.path.dirname(_SRC)).replace('\\', '/')
        with open(log_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('Model profile: %s\n' % tag)
            f.write('Config: %s\n' % rel_cfg)
            f.write('Input: 1 x 3 x %d x %d\n' % (args.imgsz, args.imgsz))
            f.write('Device: %s\n' % device)
            f.write('PyTorch: %s\n' % torch.__version__)
            f.write('Timed runs: %d\n\n' % args.runs)
            f.write('--- Headline (Ultralytics model_info / THOP, fused graph) ---\n')
            f.write('Layers      : %d\n' % n_layers)
            f.write('Parameters  : %d (%.3f M)\n' % (n_params, n_params / 1e6))
            f.write('FLOPs       : %.2f G\n' % flops_info)
            f.write('\n--- Cross-check (raw thop on the unfused graph) ---\n')
            f.write('FLOPs       : %.2f G\n' % thop_gflops)
            f.write('\n--- Runtime (model forward only) ---\n')
            f.write('Latency     : %.3f ms/frame\n' % lat)
            f.write('Throughput  : %.2f FPS (%s)\n' % (fps, device))
            f.write('\nNote: the manuscript FPS is an end-to-end A6000 measurement '
                    '(batch 1, 200 warm-up, median over 1,000 frames); reproduce '
                    'with --device cuda. Parameters and GFLOPs are device-independent.\n')
        print('   params=%.3fM  FLOPs(model_info)=%.2fG  thop=%.2fG  %s-FPS=%.2f'
              % (n_params / 1e6, flops_info, thop_gflops, device, fps))

    summ_path = os.path.join(out_dir, 'profiler_summary.txt')
    with open(summ_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# Profiler summary (input 416x416, device=%s)\n' % device)
        f.write('# Headline GFLOPs use Ultralytics model_info (THOP), fused.\n')
        for tag, p, g, fps in summary:
            f.write('%-28s params=%7.3fM  GFLOPs=%6.2f  FPS=%7.1f\n'
                    % (tag, p / 1e6, g, fps))
    print('\nSummary ->', summ_path)


if __name__ == '__main__':
    main()
