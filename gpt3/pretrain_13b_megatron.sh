#!/bin/bash

# Runs the "13B" parameter model
# GPT-13B: 40 layers, 5120 hidden size, 40 attention heads

set -e

USAGE="usage: bash pretrain_13b_megatron.sh [bf16|te|msamp|fp4]"

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
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --distributed-backend nccl \
    --no-query-key-layer-scaling \
    --seed 43 \
    --num-layers 40 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-samples 146484375 \
    --lr-decay-samples 131835938 \
    --lr-warmup-samples 4096000 \
    --lr 2.0e-4 \
    --min-lr 2.0e-5 \
    --lr-decay-style cosine \
    --micro-batch-size 1 \
    --global-batch-size 1280 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.0099 \
    --num-workers 1 \
    --bf16 \
    --sequence-parallel \
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
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 200 \
    --eval-iters 7
"

if [ "$FP_TYPE" = "bf16" ]; then
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_13b_bf16
    torchrun $DISTRIBUTED_ARGS ../third_party/Megatron-LM/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
elif [ "$FP_TYPE" = "te" ]; then
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_13b_te
    torchrun $DISTRIBUTED_ARGS ../third_party/Megatron-LM/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --fp8-hybrid \
        --transformer-impl transformer_engine \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
elif [ "$FP_TYPE" = "msamp" ]; then
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_13b_msamp
    torchrun $DISTRIBUTED_ARGS ../third_party/Megatron-LM/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --fp8-hybrid \
        --transformer-impl transformer_engine \
        --msamp \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
elif [ "$FP_TYPE" = "fp4" ]; then
    CHECKPOINT_PATH=$PWD/checkpoints/gpt_13b_fp4
    export USE_W_SIMU_FP4=1
    export USE_W_DIFFERENTIABLE_GRADIENT_ESTIMATOR=1
    export USE_A_SIMU_FP4=1
    torchrun $DISTRIBUTED_ARGS ../third_party/Megatron-LM/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --fp8-hybrid \
        --transformer-impl transformer_engine \
        --msamp \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH
else
    echo $USAGE
    exit 1
fi
