#!/bin/bash

# Runs the "345M" parameter model

set -e

USAGE="usage: bash pretrain_345m_megatron.sh [fp16|msamp]"

if [ "$#" -ne 1 ]; then
  echo $USAGE
  exit 1
fi

FP_TYPE=$1

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE=$PWD/data/gpt2-vocab.json
MERGE_FILE=$PWD/data/gpt2-merges.txt
DATA_PATH=$PWD/data/wikipedia_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10
"

if [ "$FP_TYPE" = "fp16" ]; then
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_345m_fp16
    torchrun $DISTRIBUTED_ARGS ../third_party/Megatron-LM/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH

elif [ "$FP_TYPE" = "msamp" ]; then
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_345m_msamp
    torchrun $DISTRIBUTED_ARGS ../third_party/Megatron-LM/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --msamp \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
else
    echo $USAGE
    exit 1
fi
