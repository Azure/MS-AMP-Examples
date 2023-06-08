#!/bin/sh

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

set -e

USAGE="usage: bash run.sh [amp|msamp]"

if [ "$#" -ne 1 ]; then
  echo $USAGE
  exit 1
fi

DATA_PATH=$PWD/data-bin/wikitext-103
GPU_NUM=4
amp_type=$1
fairseq_train=`which fairseq-hydra-train`

if [ "$amp_type" = "amp" ]; then
    echo "run RoBERTa base with AMP"
    SAVE_PATH=$PWD/checkpoints/roberta_amp/

    python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node=$GPU_NUM \
        $fairseq_train \
        --config-dir ../third_party/fairseq/examples/roberta/config/pretraining \
        --config-name base  \
        task.data=$DATA_PATH \
        checkpoint.save_dir=$SAVE_PATH \
        dataset.skip_invalid_size_inputs_valid_test=True \
        dataset.batch_size=64 \
        optimization.update_freq=[8] \
        common.fp16=False \
        common.amp=True \
        checkpoint.save_interval_updates=500 \
        common.log_interval=20 \
        dataset.validate_interval_updates=500 \
        distributed_training.ddp_backend=c10d

elif [ "$amp_type" = "msamp" ]; then
    echo "run RoBERTa base with MS-AMP"
    SAVE_PATH=$PWD/checkpoints/roberta_msamp/
    python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node=$GPU_NUM \
        $fairseq_train \
        --config-dir ../third_party/fairseq/examples/roberta/config/pretraining \
        --config-name base  \
        task.data=$DATA_PATH \
        checkpoint.save_dir=$SAVE_PATH \
        dataset.skip_invalid_size_inputs_valid_test=True \
        dataset.batch_size=64 \
        optimization.update_freq=[8] \
        common.fp16=False \
        common.amp=True \
        checkpoint.save_interval_updates=500 \
        common.log_interval=20 \
        dataset.validate_interval_updates=500 \
        common.msamp=True \
        common.msamp_opt_level=O2 \
        distributed_training.ddp_backend=c10d
else
    echo $USAGE
    exit 1
fi
