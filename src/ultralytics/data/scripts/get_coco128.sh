#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)

d='../datasets'
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco128.zip'
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

wait
