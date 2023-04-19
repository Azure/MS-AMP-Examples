#!/bin/sh

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

set -e

USAGE="usage: bash run.sh [small|large] [amp|msamp]"

if [ "$#" -ne 2 ]; then
  echo $USAGE
  exit 1
fi

DATA_PATH=../../ImageNet
GPU_NUM=8

model=$1
amp_type=$2

if [ "$model" == "small" -a "$amp_type" == "amp" ]; then
    echo "run small DeiT with AMP"
    python -m torch.distributed.launch \
       --nproc_per_node=$GPU_NUM \
       --use_env \
       ../third_party/deit/main.py \
       --model deit_small_patch16_224 \
       --batch-size 128 \
       --data-path $DATA_PATH \
       --output_dir output \
       --eval-dist \
       --no-model-ema
elif [ "$model" == "small" -a "$amp_type" == "msamp" ]; then
    echo "run small DeiT with MS-AMP"
    python -m torch.distributed.launch \
       --nproc_per_node=$GPU_NUM \
       --use_env ../third_party/deit/main.py \
       --model deit_small_patch16_224 \
       --batch-size 128 \
       --data-path $DATA_PATH \
       --output_dir output_msamp \
       --eval-dist \
       --no-model-ema \
       --enable-msamp \
       --msamp-opt-level O2
elif [ "$model" == "large" -a "$amp_type" == "amp" ]; then
    echo "run large DeiT with AMP"
    python -m torch.distributed.launch \
       --nproc_per_node=$GPU_NUM \
       --use_env \
       ../third_party/deit/main.py \
       --model deit_large_patch16_224 \
       --batch-size 64 \
       --data-path $DATA_PATH \
       --output_dir output_large \
       --eval-dist \
       --no-model-ema
elif [ "$model" == "large" -a "$amp_type" == "msamp" ]; then
    echo "run large DeiT with MS-AMP"
    python -m torch.distributed.launch \
       --nproc_per_node=$GPU_NUM \
       --use_env \
       ../third_party/deit/main.py \
       --model deit_large_patch16_224 \
       --batch-size 64 \
       --data-path $DATA_PATH \
       --output_dir output_large_msamp \
       --eval-dist \
       --no-model-ema \
       --enable-msamp \
       --msamp-opt-level O2
else
    echo $USAGE
    exit 1
fi
