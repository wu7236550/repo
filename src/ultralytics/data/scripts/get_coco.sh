#!/bin/bash
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Download COCO 2017 dataset http://cocodataset.org

if [ "$#" -gt 0 ]; then
  for opt in "$@"; do
    case "${opt}" in
    --train) train=true ;;
    --val) val=true ;;
    --test) test=true ;;
    --segments) segments=true ;;
    --sama) sama=true ;;
    esac
  done
else
  train=true
  val=true
  test=false
  segments=false
  sama=false
fi

d='../datasets'
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
if [ "$segments" == "true" ]; then
  f='coco2017labels-segments.zip'
elif [ "$sama" == "true" ]; then
  f='coco2017labels-segments-sama.zip' # 199 MB https://www.sama.com/sama-coco-dataset/
else
  f='coco2017labels.zip'
fi
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

d='../datasets/coco/images'
url=http://images.cocodataset.org/zips/
if [ "$train" == "true" ]; then
  f='train2017.zip'
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
if [ "$val" == "true" ]; then
  f='val2017.zip'
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
if [ "$test" == "true" ]; then
  f='test2017.zip'
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &
fi
wait
