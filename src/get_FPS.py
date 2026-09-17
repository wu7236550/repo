# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', type=str, required=True, help='path to the TensorRT .engine file')
    p.add_argument('--source', type=str, default=None,
                   help='a real image (or directory of images) used for the timed forward; '
                        'defaults to a deterministic synthetic 1920x1080 frame')
    p.add_argument('--imgsz', type=int, nargs='+', default=[416, 416], help='network input H W')
    p.add_argument('--batch', type=int, default=1, help='batch size; the paper protocol uses 1')
    p.add_argument('--warmup', type=int, default=200, help='warm-up iterations')
    p.add_argument('--iters', type=int, default=300, help='measured iterations per run')
    p.add_argument('--runs', type=int, default=5, help='number of repeated runs')
    p.add_argument('--conf', type=float, default=0.5, help='score threshold used in post-processing')
    p.add_argument('--topk', type=int, default=300, help='top-k kept after thresholding')
    p.add_argument('--out', type=str, default='fps_result.json', help='where to write the results')
    return p.parse_args()


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess(path, imgsz, dtype):
    im0 = cv2.imread(path) if path else None
    if im0 is None:
        rng = np.random.default_rng(0)
        im0 = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    im = letterbox(im0, tuple(imgsz))[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(im[None], dtype=dtype)


def postprocess(logits, conf, topk):
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    scores = scores.reshape(-1)
    keep = np.argwhere(scores > conf).reshape(-1)
    if keep.size > topk:
        keep = keep[np.argsort(-scores[keep])[:topk]]
    return int(keep.size)


def main():
    args = parse_args()
    try:
        import tensorrt as trt
        from cuda import cudart
    except Exception as e:  # pragma: no cover
        raise SystemExit('tensorrt and cuda-python are required for this script: %s' % e)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    imgsz = list(args.imgsz)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(context.get_tensor_shape(name))
        shape = [s if s > 0 else (args.batch if k == 0 else imgsz[k - 2]) for k, s in enumerate(shape)]
        context.set_input_shape(name, shape) if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else None

    n_io = engine.num_io_tensors
    tensors, host_buffers = [], []
    for i in range(n_io):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        tdt = trt.nptype(engine.get_tensor_dtype(name))
        host = np.empty(shape, dtype=tdt)
        err, dev = cudart.cudaMalloc(host.nbytes)
        assert err == cudart.cudaError_t.cudaSuccess, 'cudaMalloc failed'
        context.set_tensor_address(name, int(dev))
        tensors.append((name, dev, host, host.nbytes))
        host_buffers.append(host)

    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess, 'cudaStreamCreate failed'

    in_name, in_dev, in_host, in_bytes = tensors[0]
    out_name, out_dev, out_host, out_bytes = tensors[-1]

    def infer():
        cudart.cudaMemcpyAsync(in_dev, in_host.ctypes.data, in_bytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v3(stream_handle=stream)
        cudart.cudaMemcpyAsync(out_host.ctypes.data, out_dev, out_bytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)
        postprocess(out_host, args.conf, args.topk)

    in_host[...] = preprocess(args.source, imgsz, in_host.dtype)
    for _ in range(args.warmup):
        infer()
    cudart.cudaDeviceSynchronize()

    runs = []
    for r in range(args.runs):
        lat = np.empty(args.iters, dtype=np.float64)
        for k in range(args.iters):
            cudart.cudaDeviceSynchronize()
            t0 = time.perf_counter()
            infer()
            cudart.cudaDeviceSynchronize()
            lat[k] = time.perf_counter() - t0
        runs.append({'mean_ms': float(lat.mean() * 1e3), 'std_ms': float(lat.std() * 1e3),
                     'p50_ms': float(np.percentile(lat, 50) * 1e3),
                     'p95_ms': float(np.percentile(lat, 95) * 1e3),
                     'fps': float(1.0 / lat.mean())})
        print('run %d/%d  mean %.3f ms  fps %.2f' % (r + 1, args.runs, runs[-1]['mean_ms'], runs[-1]['fps']))

    fps = np.array([x['fps'] for x in runs])
    summary = {
        'engine': os.path.basename(args.engine),
        'imgsz': imgsz, 'batch': args.batch, 'precision': str(in_host.dtype),
        'warmup': args.warmup, 'iters': args.iters, 'runs': args.runs,
        'protocol': 'end-to-end (decode + letterbox + normalise + TensorRT forward + post-process)',
        'fps_mean': float(fps.mean()), 'fps_std': float(fps.std()),
        'per_run': runs,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\nend-to-end FPS = %.2f +- %.2f (mean +- std over %d runs)' % (fps.mean(), fps.std(), args.runs))
    print('results written to %s' % args.out)


if __name__ == '__main__':
    main()
