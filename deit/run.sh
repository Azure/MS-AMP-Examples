DATA_PATH=../../ImageNet

export CUDA_VISIBLE_DEVICES=4,5,6,7
rm -rf output

python -m torch.distributed.launch \
       --nproc_per_node=4 \
       --use_env \
       ../third_party/deit/main.py \
       --model deit_small_patch16_224 \
       --batch-size 256 \
       --data-path $DATA_PATH \
       --output_dir output \
       --no-model-ema