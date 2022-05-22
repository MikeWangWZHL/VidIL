
INPUT_JSON="<input_result_json>" # e.g., 

if [ $DATASET == "msrvtt" ]
then
    python3 eval_video_qa_result.py \
    --pred ${INPUT_JSON} \
    --gt shared_datasets/MSRVTT_qa/ann/test.jsonl \
    --setting "ranking"
fi

if [ $DATASET == "msvd" ]
then
    python3 eval_video_qa_result.py \
    --pred ${INPUT_JSON} \
    --gt shared_datasets/msvd_qa/ann/test.jsonl \
    --setting "ranking"
fi