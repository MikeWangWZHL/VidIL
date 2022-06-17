echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

CONFIG_NAME=$1 # e.g., train_blip_video_captioning_msrvtt

python -m torch.distributed.run --nproc_per_node=$N_GPU train_caption_video.py \
--config configs/${CONFIG_NAME}.yaml \
--output_dir output/${CONFIG_NAME} \
# --evaluate
