# -*- coding: utf-8 -*-
import argparse
import os
import warnings

import numpy as np
from prettytable import PrettyTable

warnings.filterwarnings('ignore')

from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    return '%.1f' % (os.stat(path).st_size / 1024 / 1024)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True, help='path to best.pt')
    p.add_argument('--data', type=str, default='../configs/roadrdd.yaml')
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--imgsz', type=int, default=416, help='evaluation image size (paper: 416)')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--project', type=str, default='runs/roadrdd/val')
    p.add_argument('--name', type=str, default=None)
    return p.parse_args()


def main():
    opt = parse_args()
    model = RTDETR(opt.weights)
    result = model.val(data=opt.data,
                       split=opt.split,
                       imgsz=opt.imgsz,
                       batch=opt.batch,
                       device=opt.device,
                       project=opt.project,
                       name=opt.name,
                       )

    if model.task == 'detect':
        length = result.box.p.size
        model_names = list(result.names.values())
        pre = result.speed['preprocess']
        inf = result.speed['inference']
        post = result.speed['postprocess']
        total = pre + inf + post

        _, n_p, _, flops = model_info(model.model)

        info = PrettyTable()
        info.title = 'Model Info'
        info.field_names = ['GFLOPs', 'Parameters', 'Pre-process/img', 'Inference/img',
                            'Post-process/img', 'FPS (e2e)', 'FPS (forward)', 'Weight size']
        info.add_row(['%.1f' % flops, '{:,}'.format(n_p),
                      '%.6fs' % (pre / 1000), '%.6fs' % (inf / 1000), '%.6fs' % (post / 1000),
                      '%.2f' % (1000 / total), '%.2f' % (1000 / inf), '%sMB' % get_weight_size(opt.weights)])
        print(info)

        met = PrettyTable()
        met.title = 'Detection Metrics (%s split)' % opt.split
        met.field_names = ['Class', 'Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.75', 'mAP@0.5:0.95']
        for idx in range(length):
            met.add_row([model_names[idx],
                         '%.4f' % result.box.p[idx],
                         '%.4f' % result.box.r[idx],
                         '%.4f' % result.box.f1[idx],
                         '%.4f' % result.box.ap50[idx],
                         '%.4f' % result.box.all_ap[idx, 5],
                         '%.4f' % result.box.ap[idx]])
        met.add_row(['all',
                     '%.4f' % result.results_dict['metrics/precision(B)'],
                     '%.4f' % result.results_dict['metrics/recall(B)'],
                     '%.4f' % np.mean(result.box.f1[:length]),
                     '%.4f' % result.results_dict['metrics/mAP50(B)'],
                     '%.4f' % np.mean(result.box.all_ap[:length, 5]),
                     '%.4f' % result.results_dict['metrics/mAP50-95(B)']])
        print(met)

        with open(result.save_dir / 'paper_data.txt', 'w+', errors='ignore', encoding='utf-8') as f:
            f.write(str(info) + '\n' + str(met))
        print('results saved to %s/paper_data.txt' % result.save_dir)


if __name__ == '__main__':
    main()
