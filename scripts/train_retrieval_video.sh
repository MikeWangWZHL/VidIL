echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"


CONFIG_NAME="<name of yaml file in 'configs/' >" # e.g., train_blip_video_retrieval_msrvtt_gt
# NOTE: if using pseudo labels, please set up field named "train_ann_jsonl" in the corresponding yaml file. 

## pseudo msrvtt adding frame captions
python -m torch.distributed.run --nproc_per_node=$N_GPU train_retrieval_video.py \
--config configs/${CONFIG_NAME}.yaml \
--output_dir output/${CONFIG_NAME} \
# --evaluate