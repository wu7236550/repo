import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/path/to/RTDETR/ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_LRPB-AIFI.yaml')
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=1,
                workers=0,
                patience=0,
                project='runs/train',
                name='rtdetr-MSPC_MCAF_LRPB-AIFI',
                )