if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

CKPT="<absolute path to './ckpt'>"
DATASETS="<absolute path to './shared_datasets'>"


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$CKPT,dst=/src/ckpt,type=bind,readonly \
    --mount src=$DATASETS,dst=/src/shared_datasets,type=bind,readonly \
    -w /src mikewangwzhl/vidil
