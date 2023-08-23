#!/bin/bash

# Runs the "345M" parameter model

set -e

USAGE="usage: bash pretrain_345m.sh [fp16|msamp]"

if [ "$#" -ne 1 ]; then
  echo $USAGE
  exit 1
fi

FP_TYPE=$1
VOCAB_FILE=$PWD/data/gpt2-vocab.json
MERGE_FILE=$PWD/data/gpt2-merges.txt
DATA_PATH=$PWD/data/wikipedia_text_document
BS=4
GLOBAL_BS=8
CLIP_GRAD=1.0
LOG_INTERVAL=100

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $BS \
    --global-batch-size $GLOBAL_BS \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad $CLIP_GRAD \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval $LOG_INTERVAL \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

config_json="./ds_config.json"
cat <<EOT >$config_json
{
    "train_micro_batch_size_per_gpu": $BS,
    "train_batch_size": $GLOBAL_BS,
    "gradient_clipping": $CLIP_GRAD,
    "fp16": {
        "enabled": true
    },
    "steps_per_print": 100
}
EOT

export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ "$FP_TYPE" = "fp16" ]; then
    echo "run 345M gpt3 with fp16"
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_345m_fp16
    torchrun ../third_party/Megatron-DeepSpeed/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH

elif [ "$FP_TYPE" = "msamp" ]; then
    echo "run 345M gpt3 with MS-AMP"

    DEEPSPEED_ARGS=" \
        --deepspeed \
        --deepspeed_config ${config_json} \
    "
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_345m_msamp

    torchrun ../third_party/Megatron-DeepSpeed/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --msamp \
        --msamp-opt-level O2 \
        $DEEPSPEED_ARGS
else
    echo $USAGE
    exit 1
fi
