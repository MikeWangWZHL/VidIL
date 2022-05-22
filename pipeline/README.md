# Additional Pipeline Scripts Instruction

## Explaination of Positional Arguments for Pipeline Scripts
Note that not all pipeline scripts containing all of the following positional arguments, please check the script of interest for more detail. The supported positional arguments are:
```
DATASET=$1          # target dataset, e.g., "msrvtt", "msvd", "youcook2", "vatex"
SPLIT=$2            # dataset split, e.g., "train", "val", "test"              
OUTPUT_ROOT_DIR=$3  # path to the output_root, e.g., "output_root_example"
SHOT=$4             # number of shots in the support set, e.g., 10
RANDOM_SEED=$5      # random seed, e.g., 42 
N=$6                # number of selected in-context example, e.g., 5
COMPARING_TARGET=$7 # which part of text representation is used for computing similarity for in-context example selection, e.g., "question", "caption"
```
To reproduce the results in the paper, for captioning, set RANDOM_SEED=40,41,42, SHOT=10, N=5; for qa, set RANDOM_SEED=42,43,44, SHOT=5, N=5; for VLEP, set RANDOM_SEED=42, SHOT=10. We didn't use in-context selection for VLEP, the result is already significantly higher than SOTA.


## Detailed control on generated prompt
Please refer to the argparse arguments definition in the `generate_prompts*.py` file called in any pipeline script of interest for detailed control on how the generated prompt looks like. Such as, whether or not adding frame captions, objects, events, attributes, etc.  

Note that for ecah outputed directory from the pipeline, e.g., `output_example/msvd_test`, if the frame captioning and visual tokenization configuration is not be changed, the frame captions and visual tokens can be reused for generating prompt with different configurations.

## Using Manually Written In-context Examples
By default, the in-context examples are randomly sampled from training set, one can also specify a particular prompt prefic containing manually written prompts. Suppose one has created a prompt at `pipeline/prompt_prefix/myprompt.txt`, to use this prompt:
- for captioning, specify `PROMPT_PREFIX="pipeline/prompt_prefix/myprompt.txt"` in `pipeline/generate_gpt3_query_pipeline_caption_fixed_prompt.sh` then at `/src` (root dir running the docker image), run:
```
bash pipeline/generate_gpt3_query_pipeline_caption_fixed_prompt.sh <dataset_name> <output_dir> shared_datasets
```
- for qa, `PROMPT_PREFIX="pipeline/prompt_prefix/myprompt.txt"` in `pipeline/generate_gpt3_query_pipeline_qa_fixed_prompt.sh` then at `/src` (root dir running the docker image), run:
```
bash pipeline/generate_gpt3_query_pipeline_qa_fixed_prompt.sh <dataset_name> <output_dir> shared_datasets
```