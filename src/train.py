import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_AIFI_DyT.yaml')
    model.train(data='dataset/roadrdd/data.yaml',
                cache=False,
                imgsz=640,
                epochs=240,
                batch=16,
                workers=16,
                patience=0,
                project='runs/roadrdd/train',
                name='rtdetr-MSPC_MCAF_AIFI_DyT',
                )
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-MSPC_MCAF_AIFI_EDFFN.yaml')
    model.train(data='dataset/roadrdd/data.yaml',
                cache=False,
                imgsz=640,
                epochs=240,
                batch=16,
                workers=16,
                patience=0,
                project='runs/roadrdd/train',
                name='rtdetr-MSPC_MCAF_AIFI_EDFFN',
                )