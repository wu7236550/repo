import warnings
warnings.filterwarnings('ignore')
import argparse
import glob
import torch, tqdm
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def parse_args():
    p = argparse.ArgumentParser(
        description='Parameters and GFLOPs for every configuration under '
                    'ultralytics/cfg/models/rt-detr, using the ultralytics model_info (THOP) '
                    'convention of two floating-point operations per multiply-accumulate.')
    p.add_argument('--cfg-dir', type=str, default='ultralytics/cfg/models/rt-detr',
                   help='directory holding the model YAML files')
    p.add_argument('--imgsz', type=int, default=640,
                   help='profiling resolution; 640 is the value quoted in the manuscript')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    flops_dict = {}
    for yaml_path in tqdm.tqdm(sorted(glob.glob(f'{args.cfg_dir}/*.yaml'))):
        if 'DCN' in yaml_path:
            continue
        try:
            model = RTDETR(yaml_path)
            model.fuse()
            n_l, n_p, n_g, flops = model_info(model.model, imgsz=args.imgsz)
            flops_dict[yaml_path] = [flops, n_p]
        except Exception:
            continue

    sorted_items = sorted(flops_dict.items(), key=lambda x: x[1][0])
    print(f'# imgsz={args.imgsz}')
    for key, value in sorted_items:
        print(f"{key}: {value[0]:.2f} GFLOPs {value[1]:,} Params")
