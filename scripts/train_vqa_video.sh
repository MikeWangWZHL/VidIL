echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

CONFIG_NAME="<name of yaml file in 'configs/' >" # e.g., train_blip_video_vqa_msrvtt

### train
python -m torch.distributed.run --nproc_per_node=2 train_vqa_video.py \
--config configs/${CONFIG_NAME}.yaml \
--output_dir output/${CONFIG_NAME} \
# --evaluate

