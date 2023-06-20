#! /bin/bash

# Runs the "13B" parameter model
# GPT-13B: 40 layers, 5120 hidden size, 40 attention heads

set -e

NODE_RANK=0
NNODES=1
GPUS_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=6001

BS=4 
PP=1  #pipline parallel
TP=2  #tensor parallel
CLIP_GRAD=1.0
GLOBAL_BATCH_SIZE=1280
LOG_INTERVAL=1

ZERO_STAGE=1
config_json="./ds_config.json"

cat <<EOT >$config_json
{
  "train_micro_batch_size_per_gpu": $BS,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": $CLIP_GRAD,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true
  },
  "steps_per_print": $LOG_INTERVAL
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

VOCAB_FILE=$PWD/data/gpt2-vocab.json
MERGE_FILE=$PWD/data/gpt2-merges.txt
DATA_PATH=$PWD/data/wikipedia_text_document
CHECKPOINT_PATH=$PWD/checkpoints/gpt_13b
DATA_PATH=$PWD/data/wikipedia_text_document
DATASET="1.0 ${DATA_PATH}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

WANDB_DISABLE=1 python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  ../third_party/Megatron-DeepSpeed/pretrain_gpt.py \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
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
  --micro-batch-size $BS \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --clip-grad $CLIP_GRAD \
  --weight-decay 0.1 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.0099 \
  --log-interval $LOG_INTERVAL \
  --eval-iters 7 \
  --eval-interval 200 \
  --save-interval 2000 \
  --save $CHECKPOINT_PATH \
  --num-workers 1 \
  --data-path $DATASET \
  --vocab-file $VOCAB_FILE \
  --merge-file $MERGE_FILE \
  --split 949,50,1 \
  --fp16 \
  --checkpoint-activations \
  $DEEPSPEED_ARGS