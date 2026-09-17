# -*- coding: utf-8 -*-
import argparse
import warnings

warnings.filterwarnings('ignore')

from ultralytics import RTDETR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='runs/roadrdd/train/exp/weights/best.pt')
    p.add_argument('--source', type=str, default='../dataset/roadrdd/images/test',
                   help='image, directory or video to run inference on')
    p.add_argument('--imgsz', type=int, default=416, help='inference image size (paper: 416)')
    p.add_argument('--conf', type=float, default=0.5,
                   help='confidence threshold; the paper reports visualisations at 0.5')
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--project', type=str, default='runs/roadrdd/detect')
    p.add_argument('--name', type=str, default='exp')
    return p.parse_args()


if __name__ == '__main__':
    opt = parse_args()
    model = RTDETR(opt.weights)
    model.predict(source=opt.source,
                  imgsz=opt.imgsz,
                  conf=opt.conf,
                  device=opt.device,
                  project=opt.project,
                  name=opt.name,
                  save=True,
                  )
