DATASET="msrvtt" # "youcook2", "vatex"
RESULT_FILE="<path_to_result_json_file>"
OUTPUT_DIR="<path_to_output_dir>"


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
--pred ${RESULT_FILE} \
--gt ${GT} \
--output_dir ${OUTPUT_DIR} \
--input_format result_file