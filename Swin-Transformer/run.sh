#!/bin/sh

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

set -e

USAGE="usage: bash run.sh [tiny|giant] [amp|msamp]"

if [ "$#" -ne 2 ]; then
  echo $USAGE
  exit 1
fi

DATA_PATH=../../ImageNet
GPU_NUM=8
MASTER_PORT=12345

model=$1
amp_type=$2

if [ "$model" == "tiny" -a "$amp_type" == "amp" ]; then
    echo "run tiny Swin-Transformer with AMP"
    python -m torch.distributed.launch \
        --nproc_per_node $GPU_NUM \
        --master_port $MASTER_PORT  \
        ../third_party/Swin-Transformer/main.py \
        --cfg ../third_party/Swin-Transformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
        --data-path $DATA_PATH \
        --batch-size 128 \
        --output output
elif [ "$model" == "tiny" -a "$amp_type" == "msamp" ]; then
    echo "run tiny Swin-Transformer with MS-AMP"
    python -m torch.distributed.launch \
        --nproc_per_node $GPU_NUM \
        --master_port $MASTER_PORT  \
        ../third_party/Swin-Transformer/main.py \
        --cfg ../third_party/Swin-Transformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
        --data-path $DATA_PATH \
        --batch-size 128 \
        --output output_msamp \
        --enable-msamp \
        --msamp-opt-level O2
elif [ "$model" == "giant" -a "$amp_type" == "amp" ]; then
    echo "run giant Swin-Transformer with AMP"
    python -m torch.distributed.launch \
        --nproc_per_node $GPU_NUM \
        --master_port $MASTER_PORT  \
        ../third_party/Swin-Transformer/main.py \
        --cfg ../third_party/Swin-Transformer/configs/swin/swin_giant_patch4_window7_224.yaml \
        --data-path $DATA_PATH \
        -output output_giant \
        --batch-size 16
elif [ "$model" == "giant" -a "$amp_type" == "msamp" ]; then
    echo "run giant Swin-Transformer with MS-AMP"
    python -m torch.distributed.launch \
        --nproc_per_node $GPU_NUM \
        --master_port $MASTER_PORT  \
        ../third_party/Swin-Transformer/main.py \
        --cfg ../third_party/Swin-Transformer/configs/swin/swin_giant_patch4_window7_224.yaml \
        --data-path $DATA_PATH \
        --batch-size 16 \
        --output output_giant_msamp \
        --enable-msamp \
        --msamp-opt-level O2
else
    echo $USAGE
    exit 1
fi
