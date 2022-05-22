DATASET="<dataset>" # see below for options
INPUT_NAME="<input_query_jsonl_name>"
GPT3_RESPONSE="<path_to_gpt_response>/${INPUT_NAME}.jsonl"
IDX2IDS="<path_to_input_root>/${INPUT_NAME}__idx_2_videoid.json"
OUTPUT_DIR="<path_to_input_root>/evaluation"


##### example with files in "output_example/" #####

# DATASET="msrvtt"
# INPUT_NAME="temp_0.0_msrvtt_caption_with_in_context_selection_clip_shot_10_seed_42_N_5"
# GPT3_RESPONSE="output_example/msrvtt/gpt3_response/${INPUT_NAME}.jsonl"
# IDX2IDS="output_example/msrvtt/input_prompts/${INPUT_NAME}__idx_2_videoid.json"
# OUTPUT_DIR="output_example/msrvtt/evaluation"

###################################################


if [ $DATASET == "msrvtt" ]
then
    GT="shared_datasets/MSRVTT_caption/ann/test_caption.jsonl"
fi

if [ $DATASET == "youcook2" ]
then
    GT="shared_datasets/Youcook2/value_ann/yc2c/videoid_2_text_yc2c_val.jsonl"
fi

if [ $DATASET == "vatex" ]
then
    GT="shared_datasets/Vatex/value_ann/vatex_en_c/videoid_2_text_vatex_en_c_test_public.jsonl"
fi


mkdir -p $OUTPUT_DIR

python3 eval_video_captioning_results.py \
--pred ${GPT3_RESPONSE} \
--pred_idx_2_vid ${IDX2IDS} \
--gt ${GT} \
--output_dir ${OUTPUT_DIR} \
--input_format gpt3_response