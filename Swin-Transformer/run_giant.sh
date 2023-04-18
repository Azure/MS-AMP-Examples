DATA_PATH=../../ImageNet

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --master_port 12345  \
    ../third_party/Swin-Transformer/main.py \
    --cfg ../third_party/Swin-Transformer/configs/swin/swin_giant_patch4_window7_224.yaml \
    --data-path $DATA_PATH \
    -output output_giant \
    --batch-size 16