DATA_PATH=../../ImageNet

python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --use_env \
       ../third_party/deit/main.py \
       --model deit_large_patch16_224 \
       --batch-size 64 \
       --data-path $DATA_PATH \
       --output_dir output_large_msamp \
       --no-model-ema \
       --enable-msamp \
       --msamp-opt-level O2