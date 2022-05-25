# VidIL: [Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners](https://arxiv.org/abs/2205.10747)

<img src="vidIL.gif" width="1200">

---

## Download Datasets & Checkpoints
- Download dataset annotations zip from [here](https://uofi.box.com/s/wwh4phgetakycvzeyxoih5qupevzj9xe). Then unzip the downloaded datasets under `shared_datasets/`. The resulting shared_dataset folder structure is expected to be:
    ```
    shared_datasets
    ├── README.md
    ├── MSRVTT_caption
    ├── MSRVTT_qa
    ...
    ```
    Then, please refer to [Dataset Instruction](shared_datasets/README.md) for downloading and processing raw videos.

- Download BLIP checkpoints:
    ```
    bash download_blip_checkpoints.sh
    ```

- Download Input & Output Examples zip from [here](https://uofi.box.com/s/vsnh9l5qn5p08spoftgs8anabmbll8ky). Unzip the folders under `output_example/`, the resulting `output_example/` folder structure is expected to be:
    ```
    output_example
    ├── msrvtt
    ├── msvd_test
    ├── vlep_test
    └── README.md
    ```

## Set Up Environment

- launch the docker environment:
    - (1) set up variable "CKPT" and "DATASETS" as commented in `run_docker_vidil.sh`
    - (2) run docker image 
        ```
        bash run_docker_vidil.sh
        ```

- set up GPU devices:
within the docker image, set up the following environment variables to config GPT devices
    ```
    export N_GPU=<num of gpus>
    export CUDA_VISIBLE_DEVICES=<0,1,2...>
    ```

---

## Generate Video Representation & GPT-3 Prompt 
The following scripts runs the pipeline which, (1) generates frame captions; (2) generates visual tokens (3) generates few-shot prompt readily for GPT-3. The output folder have the following structure:
```
    {dataset_split}
    ├── frame_caption
    │   ├── config.yaml  # config for frame captioning
    │   ├── video_text_Cap.json  # frame captions w/o filtering
    │   ├── video_text_CapFilt.json  # frame captions w/ filtering
    ├── input_prompts 
    │   ├── {output_name}.jsonl  # config for frame captioning
    │   ├── {output_name}__idx_2_videoid.json  # line idx to video id
    │   ├── {output_name}__chosen_samples.json  # chosen examples in the support
    │   ... 
    ├── visual_tokenization_{encoder_name}           
    │   ├── config.yaml  # config for visual tokenization
    │   └── visual_tokens.json  # raw visual tokens of each frame
    └──
```

All scripts should be run at `/src` dir, namely, the root directory after running the docker image. The following are examples for running the pipeline with in-context example selection for some datasets. For additional notes on running pipeline scripts, please refer to [Pipeline Instruction](pipeline/README.md).


### Standalone Pipeline for Frame Captioning and Visaul Tokenization
Since we need to sample few-shot support set from training sets, for each dataset, at the first time running the pipeline, we need to do frame captioning and visual tokenization on the training set. 

For `<dataset> in ["msrvtt","youcook2","vatex","msvd","vlep"]`:
```
bash pipeline/scripts/run_frame_captioning_and_visual_tokenization.sh <dataset> train <output_root>
```
An example of the frame caption and visual token dir can be found at: `output_example/msrvtt/frame_caption` , `output_example/msrvtt/visual_tokenization_clip`

### Video Captioning
For `<dataset> in ["msrvtt","youcook2","vatex"]`:
- (1) Run the [Standalone Frame Captioning and Visaul Tokenization pipieline](#Standalone-Pipeline-for-Frame-Captioning-and-Visaul-Tokenization) for the chosen `<dataset>`

- (2) Run pipeline for generating video captioning prompts for `<dataset>` `<split> in ["train","val","test"]`
    - w/o ASR: 

    ```
    bash pipeline/scripts/generate_gpt3_query_pipeline_caption_with_in_context_selection.sh <dataset> <split> <output_root> 10 42 5 caption
    ```
    - w/ ASR:

    ```
    bash pipeline/scripts/generate_gpt3_query_pipeline_caption_with_in_context_selection_with_asr.sh <dataset> <split> <output_root> 10 42 5 caption_asr
    ```
    An example of the output prompt jsonl can be found at `output_example/msrvtt/input_prompts/temp_0.0_msrvtt_caption_with_in_context_selection_clip_shot_10_seed_42_N_5.jsonl`.

### Video Question Answering
For `<dataset> in ["msrvtt","msvd"]`:
- (1) Run the [Standalone Frame Captioning and Visaul Tokenization pipieline](#Standalone-Pipeline-for-Frame-Captioning-and-Visaul-Tokenization) for the chosen `<dataset>`

- (2) Run pipeline for generating video question answering prompts for `<dataset>` `<split> in ["train","val","test"]`

    ```
    bash pipeline/scripts/generate_gpt3_query_pipeline_qa_with_in_context_selection.sh <dataset> <split> <output_root> 5 42 5 question
    ```
    
    An example of the output prompt jsonl can be found at `output_example/msvd_test/input_prompts/temp_0.0_gpt3_queries_msvd_qa_clip_shot_5_seed_42.jsonl`.

### Video-Language Event Prediction (VLEP)
- (1) Run the [Standalone Frame Captioning and Visaul Tokenization pipieline](#Standalone-Pipeline-for-Frame-Captioning-and-Visaul-Tokenization) for the chosen `vlep`

- (2) Run pipeline for generating vlep prompts

    ```
        bash pipeline/scripts/generate_gpt3_query_pipeline_vlep_with_random_context_asr_multichoice.sh <dataset> <split> <output_root> 10 42
    ```
    An example of the output prompt jsonl can be found at `output_example/vlep_test/input_prompts/temp_0.0_vlep_test_clip_shot_10_seed_42_multichoice.jsonl`.


### Semi-Supervised Text-Video Retrieval
For semi-supervised setting, we first generate pseudo label on the training set, we then train BLIP on the pseudo labeled dataset for retrieval.
- (1) Generate pseudo labeled training set annotation json: suppose we have the raw gpt3 response stored at `<gpt3_response_dir>`, the input_prompt dir is at `<input_prompts_dir>`, run:

    ```
        python utils_gpt3/process_gpt3_response.py --gpt3_response_dir <gpt3_response_dir> --input_prompts_dir <input_prompts_dir> --output_dir <processed_response_dir>
        python utils_gpt3/gpt3_response_to_jsonl.py --dataset <dataset_name> --gpt3_processed_dir <processed_response_dir> --output_dir <pseudo_label_ann_output_dir>
    ```

    An example of the  `<gpt3_response_dir>`, `<input_prompts_dir>`, `<processed_response_dir>` and `pseudo_label_ann_output_dir` can be found at `output_example/msrvtt/gpt3_response`, `output_example/msrvtt/input_prompts`, `output_example/msrvtt/processed_response_dir`, `output_example/msrvtt/pseudo_label_ann`.

- (2) Finetune pretrained BLIP from pseudo labeled data:
For `<dataset> in ["msrvtt","vatex"]`, set the value of the field named `train_ann_jsonl` in `configs/train_blip_video_retrieval_<dataset>_pseudo.yaml` to be the path to the output jsonl from step one in `<pseudo_label_ann_output_dir>`. Then run:

    ```
    bash scripts/train_caption_video.sh train_blip_video_retrieval_<dataset>_pseudo
    ```


## Evaluation
Scripts for evaluating generation results from GPT-3:
- Video Captioning: please refer to the example written in the script for more details about the required inputs

    ```
    bash scripts/evaluation/eval_caption_from_gpt3_response.sh
    ```

- Question Answering: please refer to the example written in the script for more details about the required inputs

    ```
    bash scripts/evaluation/eval_qa_from_gpt3_response.sh
    ```

- VLEP:
    - (1) get the processed gpt3 response; an example of the: `<gpt3_response_dir>`, `<input_prompts_dir>` and `<processed_response_dir>` can be found at: `output_example/vlep_test/gpt3_response`, `output_example/vlep_test/input_prompts`,  `output_example/vlep_test/gpt3_response_processed`

        ```
            python utils_gpt3/process_gpt3_response.py --gpt3_response_dir <gpt3_response_dir> --input_prompts_dir <input_prompts_dir> --output_dir <processed_response_dir>
        ```
    - (2) run the following script to generate the output in the official format for [CodaLab submission](https://github.com/jayleicn/VideoLanguageFuturePred/blob/main/standalone_eval/README.md#codalab-submission); an example of the output jsonl can be found at `output_example/vlep_test/evaluation/temp_0.0_vlep_test_clip_shot_10_seed_42_multichoice_eval.jsonl`

        ```
            python eval_vlep.py --gpt3_processed_response <processed_response_json> --output_path <output_jsonl_path>
        ```


# Citation
```
```

# Acknowledgement
The implementation of VidIL relies on resources from 
[BLIP](https://github.com/salesforce/BLIP),
[ALPRO](https://github.com/salesforce/ALPRO),
[transformers](https://github.com/huggingface/transformers). We thank the original authors for their open-sourced code and encourage users to cite their works when applicable.