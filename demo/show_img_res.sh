#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
CHECKPOINT=$2
OUT=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/top_down_img_demo.py $CONFIG $CHECKPOINT --img-root /root/data/coco/val2017/ \
       --json-file /root/data/coco/annotations/person_keypoints_val2017.json --out-img-root $OUT
