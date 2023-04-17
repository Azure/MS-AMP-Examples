DATA_PATH=../../ImageNet
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 12345  \
    ../third_party/Swin-Transformer/main.py \
    --cfg ../third_party/Swin-Transformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
    --data-path $DATA_PATH \
    --batch-size 256