# -*- coding: utf-8 -*-
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

FINAL_CFG = ('ultralytics/cfg/models/rt-detr/'
             'rtdetr-MSPC_MCAF_LRPB-AIFI.yaml')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default=FINAL_CFG, help='model configuration YAML')
    p.add_argument('--data', type=str, default='../configs/roadrdd.yaml', help='dataset YAML')
    p.add_argument('--weights', type=str, default='', help='optional pretrained weights')
    p.add_argument('--imgsz', type=int, default=416, help='training image size (paper: 416)')
    p.add_argument('--epochs', type=int, default=240)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8,
                   help='set to 0 if the dataloader hangs (common on Windows)')
    p.add_argument('--seed', type=int, default=0, help='random seed (paper uses seeds 0-4)')
    p.add_argument('--device', type=str, default='0', help="e.g. '0' or '0,1' or 'cpu'")
    p.add_argument('--project', type=str, default='runs/roadrdd/train')
    p.add_argument('--name', type=str, default=None)
    return p.parse_args()


def main():
    opt = parse_args()
    name = opt.name or opt.cfg.split('/')[-1].replace('.yaml', '')
    model = RTDETR(opt.cfg)
    if opt.weights:
        model.load(opt.weights)
    model.train(
        data=opt.data,
        cache=False,
        imgsz=opt.imgsz,
        epochs=opt.epochs,
        batch=opt.batch,
        workers=opt.workers,
        seed=opt.seed,
        deterministic=True,
        device=opt.device,
        optimizer='AdamW',
        lr0=0.0001,
        patience=0,
        project=opt.project,
        name=name,
        fliplr=0.5,
        hsv_v=0.10,
        hsv_s=0.05,
        scale=0.2,
        mosaic=0.2,
        close_mosaic=0,
    )


if __name__ == '__main__':
    main()
