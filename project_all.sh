#!/bin/bash

IMG_DIR="/mnt/data/feature_extraction/data/Sorted_all/images_clahe_crop/*"
NETWORK="/mnt/data/feature_extraction/featmodels/stylegan3/training-runs/00017-stylegan2-myxo1-256x256-gpus1-batch16-gamma10/network-snapshot-001000.pkl"
degree=4
for FILE in $IMG_DIR;
do
  IMG_FILE=${FILE##*/};
  IMG_NAME=${IMG_FILE%%.*};
  echo $FILE;
  echo $IMG_NAME;
  python projector_myxonet.py --outdir=/mnt/data/feature_extraction/data/Sorted_all/images_clahe_crop_resnet_feats/$IMG_NAME --network=$NETWORK --target=$FILE --save-video=False &
  [ `expr $i % $degree` -eq 0 ] && wait
done