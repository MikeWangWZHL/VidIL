
DATASET="<dataset>" # see below for options
INPUT_NAME="<input_query_jsonl_name>"
GPT3_RESPONSE="<path_to_gpt_response>/${INPUT_NAME}.jsonl"
IDX2IDS="<path_to_input_root>/${INPUT_NAME}__idx_2_videoid.json"


##### example with files in "output_example/" #####

# DATASET="msvd"
# INPUT_NAME="temp_0.0_gpt3_queries_msvd_qa_clip_shot_5_seed_42"
# GPT3_RESPONSE="output_example/msvd_test/gpt_response/${INPUT_NAME}.jsonl"
# IDX2IDS="output_example/msvd_test/input_prompts/${INPUT_NAME}__idx_2_videoid.json"

###################################################


if [ $DATASET == "msrvtt" ]
then
    python3 eval_video_qa_result.py \
    --pred_jsonl ${GPT3_RESPONSE} \
    --idx_2_video_id  ${IDX2IDS} \
    --gt shared_datasets/MSRVTT_qa/ann/test.jsonl \
    --answer_list_json shared_datasets/MSRVTT_qa/ann/test_answer_list.json \
    --setting "generation_gpt3_raw"
fi

if [ $DATASET == "msvd" ]
then
    python3 eval_video_qa_result.py \
    --pred_jsonl ${GPT3_RESPONSE} \
    --idx_2_video_id  ${IDX2IDS} \
    --gt shared_datasets/msvd_qa/ann/test.jsonl \
    --answer_list_json shared_datasets/msvd_qa/ann/test_answer_list.json \
    --setting "generation_gpt3_raw"
fi