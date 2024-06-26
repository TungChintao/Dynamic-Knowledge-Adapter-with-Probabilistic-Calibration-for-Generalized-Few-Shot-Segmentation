#!/bin/sh
PARTITION=Segmentation

GPU_ID=3
dataset=oem # pascal coco
#exp_name=split0

file_root=pretrain
arch=PSPNet
net=efficientnet # vgg resnet50

exp_dir=exp/${file_root}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
#config=config/${dataset}/${dataset}_${exp_name}_${net}_base.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
#cp train.sh train.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m train \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log
