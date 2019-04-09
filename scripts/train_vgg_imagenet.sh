#!/bin/sh

cd ../exper/


THRESHOLD=0.6


python train_frame.py --arch=vgg_v1 --epoch=100 --lr=0.001 --batch_size=16 --dataset=imagenet  \
	--disp_interval=400 \
	--num_classes=14 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
    --resume=False \
    --sample=0 \

