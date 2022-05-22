### device config ###
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

### general config ###
DATASET=$1
DATASET="vlep"
SPLIT=$2
OUTPUT_ROOT_DIR=$3
SHARED_DATASETS="shared_datasets"

echo "running pipeline on dataset: ${DATASET}_${SPLIT}"
echo "output root dir: $OUTPUT_ROOT_DIR"
echo "shared_datasets dir: $SHARED_DATASETS"

PROMPT_TASK="vlep"
VISUAL_TOKENIZATION_ENCODER="clip" # "blip"
PROMPT_TEMPLATE="temporal_natural" # "static", "temporal_index"
ASR_PATH="$SHARED_DATASETS/VLEP/ann/videoid_2_subtitle_vlep.json"
EXAMPLEID_2_EVENTS="$SHARED_DATASETS/VLEP/ann/example_id_2_events_answer_${SPLIT}.json"

### config for randomly generating in-context examples ###
INSTRUCTION_LINE="Predict what is more likely to happen next based on the frame captions and dialogue. Example:"
SHOT=$4
RANDOM_SEED=$5

TRAINSET_JSON_ANN="${SHARED_DATASETS}/VLEP/ann/example_id_2_events_answer_train.json" # vlep train
TARGET_JSON_ANN="${SHARED_DATASETS}/VLEP/ann/example_id_2_events_answer_${SPLIT}.json"


## path to visual tokens/frame caption dir containing results for training set videos: this can be found in the outpur dir by command such as: https://github.com/MikeWangWZHL/Knowledge_Enhanced_Video_Pretraining#11-run-pipeline-for-msrvtt 
## below are the examples from CLIP encoder results 
TRAIN_DATASET_VISUAL_TOKENIZATION_DIR="${OUTPUT_ROOT_DIR}/${DATASET}_train/visual_tokenization_${VISUAL_TOKENIZATION_ENCODER}"
TRAIN_DATASET_FRAME_CAPTION_DIR="${OUTPUT_ROOT_DIR}/${DATASET}_train/frame_caption"




### config for output query lines ### 

OUTPUT_DIR="$OUTPUT_ROOT_DIR/${DATASET}_${SPLIT}" # path to unique directory that will store all intermidiate and final results
CONFIG="configs/pipeline_config/pipeline_config_${DATASET}_${SPLIT}.yaml"

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
OUTPUT_NAME="gpt3_queries_${PROMPT_TASK}_${SPLIT}_${VISUAL_TOKENIZATION_ENCODER}_shot_${SHOT}_seed_${RANDOM_SEED}_multichoice.jsonl"

python generate_prompts_random_prefix_vlep_multichoice.py \
--prompt_task $PROMPT_TASK \
--visual_tokens_dir $VISUAL_TOKENIZATION_OUTPUT_DIR \
--frame_captions_dir $FRAME_CAPTION_OUTPUT_DIR \
--output_dir $PROMPT_OUTPUT_DIR \
--output_name $OUTPUT_NAME \
--caption_all_video \
--prompt_temporal_template $PROMPT_TEMPLATE \
--trainset_json_ann $TRAINSET_JSON_ANN \
--target_json_ann $TARGET_JSON_ANN \
--train_dataset_visual_tokens_dir $TRAIN_DATASET_VISUAL_TOKENIZATION_DIR \
--train_dataset_frame_captions_dir $TRAIN_DATASET_FRAME_CAPTION_DIR \
--instruction_line "$INSTRUCTION_LINE" \
--seed $RANDOM_SEED \
--shot $SHOT \
--add_ASR \
--asr_path $ASR_PATH \
--no-add_objects \
--no-add_events \
--no-add_attributes
