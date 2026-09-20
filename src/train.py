# -*- coding: utf-8 -*-
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR

FINAL_CFG = ('ultralytics/cfg/models/rt-detr/'
             'rtdetr-MSPC_MCAF_LRPB-AIFI.yaml')

# Manuscript Table 4: five fixed seeds, reported as mean +/- standard deviation.
PAPER_SEEDS = (42, 123, 2024, 7, 31415)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default=FINAL_CFG, help='model configuration YAML')
    p.add_argument('--data', type=str, default='../configs/roadrdd.yaml', help='dataset YAML')
    p.add_argument('--weights', type=str, default='', help='optional pretrained weights')
    p.add_argument('--imgsz', type=int, default=416, help='training image size (Table 4: 416)')
    p.add_argument('--epochs', type=int, default=240)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=8,
                   help='set to 0 if the dataloader hangs (common on Windows)')
    p.add_argument('--seed', type=int, default=PAPER_SEEDS[0],
                   help='random seed (Table 4 trains every key configuration with '
                        + '/'.join(str(s) for s in PAPER_SEEDS) + ')')
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
        lrf=0.01,            # cosine decays 1e-4 -> 1e-6 over the 240 epochs (Table 4)
        cos_lr=True,         # Table 4: "cosine to 1e-6 over 240 epochs"
        momentum=0.9,
        weight_decay=0.0001,
        warmup_epochs=2000,  # iteration budget in this fork; about 5 epochs (Table 4)
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=0,
        project=opt.project,
        name=name,
        # Table 4 augmentation.  Note that the RT-DETR criterion takes its loss gains
        # (`class`/`bbox`/`giou` = 1/5/2) from ultralytics.models.utils.loss.DETRLoss,
        # not from the box/cls/dfl arguments of the YOLO training path.
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.2,
        mosaic=0.2,
        close_mosaic=15,     # Table 4: mosaic closed over the last 15 epochs
    )


if __name__ == '__main__':
    main()
