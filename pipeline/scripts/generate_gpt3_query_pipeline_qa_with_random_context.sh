### device config ###
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "number of GPU: $N_GPU"

### config ### 
DATASET=$1
SPLIT=$2
# DATASET="msrvtt"
# DATASET="msvd"
OUTPUT_ROOT_DIR=$3
SHARED_DATASETS="shared_datasets"
echo "running pipeline on dataset: ${DATASET}_${SPLIT}"
echo "output root dir: $OUTPUT_ROOT_DIR"
echo "shared_datasets dir: $SHARED_DATASETS"

PROMPT_TASK="qa"
VISUAL_TOKENIZATION_ENCODER="clip" # "blip", "florence" 
PROMPT_TEMPLATE="temporal_natural" # "static", "temporal_index"


### config for randomly generating in-context examples ###
INSTRUCTION_LINE="Answer the question based on the objects, events, attributes and frame captions. Example:"
SHOT=$4
RANDOM_SEED=$5

## path to training set ann, in order to make sure that we pick the video examples only in training set, here are the options for each dataset
if [ $DATASET == "msvd" ]
then
    TRAINSET_JSON_ANN="${SHARED_DATASETS}/msvd_qa/ann/videoid_2_question_answer_msvd_train.json" # msvd qa
fi

if [ $DATASET == "msrvtt" ]
then
    TRAINSET_JSON_ANN="${SHARED_DATASETS}/MSRVTT_qa/ann/videoid_2_question_answer_msrvtt_train.json" # msrvtt qa
fi

## path to visual tokens/frame caption dir containing results for training set videos: this can be found in the outpur dir by command such as: https://github.com/MikeWangWZHL/Knowledge_Enhanced_Video_Pretraining#11-run-pipeline-for-msrvtt 
TRAIN_DATASET_VISUAL_TOKENIZATION_DIR="${OUTPUT_ROOT_DIR}/${DATASET}_train/visual_tokenization_${VISUAL_TOKENIZATION_ENCODER}"
TRAIN_DATASET_FRAME_CAPTION_DIR="${OUTPUT_ROOT_DIR}/${DATASET}_train/frame_caption"


### load question answer ann ###
if [ $DATASET == "msvd" ]
then
    QUESTION_ANSWER_DICT="$SHARED_DATASETS/msvd_qa/ann/videoid_2_question_answer_msvd_full.json"
fi

if [ $DATASET == "msrvtt" ]
then
    QUESTION_ANSWER_DICT="$SHARED_DATASETS/MSRVTT_qa/ann/videoid_2_question_answer_msrvtt_full.json"
fi

### set up output dirs ###
OUTPUT_DIR="$OUTPUT_ROOT_DIR/${DATASET}_${SPLIT}" # same dir as caption task, visual tokens and caption can be reused
CONFIG="configs/pipeline_config/pipeline_config_${DATASET}_${SPLIT}.yaml"


VISUAL_TOKENIZATION_OUTPUT_DIR="$OUTPUT_DIR/visual_tokenization_$VISUAL_TOKENIZATION_ENCODER"
FRAME_CAPTION_OUTPUT_DIR="$OUTPUT_DIR/frame_caption"
PROMPT_OUTPUT_DIR="$OUTPUT_DIR/input_prompts_qa"

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
OUTPUT_NAME="gpt3_queries_${DATASET}_${SPLIT}_${PROMPT_TASK}_${VISUAL_TOKENIZATION_ENCODER}_shot_${SHOT}_seed_${RANDOM_SEED}.jsonl"

python generate_prompts_random_prefix.py \
--prompt_task $PROMPT_TASK \
--visual_tokens_dir $VISUAL_TOKENIZATION_OUTPUT_DIR \
--frame_captions_dir $FRAME_CAPTION_OUTPUT_DIR \
--output_dir $PROMPT_OUTPUT_DIR \
--add_objects \
--add_events \
--add_attributes \
--output_name $OUTPUT_NAME \
--caption_all_video \
--prompt_temporal_template $PROMPT_TEMPLATE \
--question_answer_path $QUESTION_ANSWER_DICT \
--trainset_json_ann $TRAINSET_JSON_ANN \
--train_dataset_visual_tokens_dir $TRAIN_DATASET_VISUAL_TOKENIZATION_DIR \
--train_dataset_frame_captions_dir $TRAIN_DATASET_FRAME_CAPTION_DIR \
--instruction_line "$INSTRUCTION_LINE" \
--seed $RANDOM_SEED \
--shot $SHOT \