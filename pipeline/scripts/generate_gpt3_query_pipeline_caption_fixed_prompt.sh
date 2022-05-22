### device config ###
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

### config ### 
DATASET=$1
SPLIT=$2
# DATASET="msrvtt"
# DATASET="youcook2"
# DATASET="vatex"
# DATASET="webvid"
OUTPUT_ROOT_DIR=$3
SHARED_DATASETS="shared_datasets"
echo "running pipeline on dataset: $DATASET,$SPLIT"
echo "output root dir: $OUTPUT_ROOT_DIR"
echo "shared_datasets dir: $SHARED_DATASETS"

PROMPT_TASK="caption"
PROMPT_PREFIX="pipeline/prompt_prefix/myprompts.txt"

OUTPUT_DIR="$OUTPUT_ROOT_DIR/${DATASET}_${SPLIT}" # path to unique directory that will store all intermidiate and final results
CONFIG="configs/pipeline_config/pipeline_config_${DATASET}_${SPLIT}.yaml"
VISUAL_TOKENIZATION_ENCODER="clip" # "blip", "florence" 

VISUAL_TOKENIZATION_OUTPUT_DIR="$OUTPUT_DIR/visual_tokenization_$VISUAL_TOKENIZATION_ENCODER"
FRAME_CAPTION_OUTPUT_DIR="$OUTPUT_DIR/frame_caption"
PROMPT_OUTPUT_DIR="$OUTPUT_DIR/input_prompts"

mkdir -p $OUTPUT_DIR
mkdir -p $VISUAL_TOKENIZATION_OUTPUT_DIR
mkdir -p $FRAME_CAPTION_OUTPUT_DIR
mkdir -p $PROMPT_OUTPUT_DIR

# run visual tokenization 
if test -f "$VISUAL_TOKENIZATION_OUTPUT_DIR/visual_tokens.json"; then
    echo "visual tokens exist"
else
    echo "run visual tokenization..."
    python -m torch.distributed.run --nproc_per_node=$N_GPU run_visual_tokenization.py \
    --config $CONFIG \
    --output_dir $VISUAL_TOKENIZATION_OUTPUT_DIR \
    --encoder_version $VISUAL_TOKENIZATION_ENCODER
fi

# run capfilt
if test -f "$FRAME_CAPTION_OUTPUT_DIR/video_text_CapFilt.json"; then
    echo "frame captions exist"
else
    echo "run frame captioning..."
    python -m torch.distributed.run --nproc_per_node=$N_GPU run_video_CapFilt.py \
    --config $CONFIG \
    --output_dir $FRAME_CAPTION_OUTPUT_DIR
fi

# generate prompt
OUTPUT_NAME="gpt3_queries_${DATASET}_${SPLIT}_${PROMPT_TASK}_${VISUAL_TOKENIZATION_ENCODER}.jsonl"

python generate_prompts_fixed_prefix.py \
--prompt_task $PROMPT_TASK \
--visual_tokens_dir $VISUAL_TOKENIZATION_OUTPUT_DIR \
--frame_captions_dir $FRAME_CAPTION_OUTPUT_DIR \
--prompt_prefix $PROMPT_PREFIX \
--output_dir $PROMPT_OUTPUT_DIR \
--add_objects \
--add_events \
--add_attributes \
--output_name $OUTPUT_NAME \
--caption_all_video