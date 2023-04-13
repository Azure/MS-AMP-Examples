python -m torch.distributed.launch \
       --nproc_per_node=8 \
       --use_env main.py \
       --model deit_small_patch16_224 \
       --batch-size 256 \
       --data-path /dev/shm/ImageNet/ \
       --output_dir output
