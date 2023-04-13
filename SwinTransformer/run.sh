DATA_PATH=../../ImageNet

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_port 12345  \
    main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
    --data-path $DATA_PATH \
    --batch-size 128