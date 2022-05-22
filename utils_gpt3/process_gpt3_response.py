import json
import os
from collections import defaultdict
import argparse
from glob import glob

def load_jsonl(path):
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

### processing gpt3 response ###
def clean_text(text, filter_short_response = False):
    text_ = text.strip()
    if '\n\nObjects:' in text_:
        text = text_.split('\n\nObjects:')[0]
    elif '\n\nFrame' in text_:
        text = text_.split('\n\nFrame')[0]

    if filter_short_response:
        tokens = text.split(' ')
        token_counts = defaultdict(int)
        for tok in tokens:
            token_counts[tok] += 1
        for key,value in token_counts.items():
            if value >= 8 and key not in ['.', ',' ,'a', 'the', 'an']:
                print('discard:', key, ' | ', text)
                return None

    text = text.lstrip('\n')
    text = text.lstrip()
    text = text.strip()
    return text

def process_gpt3_response(idx_2_videoid_path, response_jsonl_path, output_path):
    idx_2_videoid = json.load(open(idx_2_videoid_path))
    responses = load_jsonl(response_jsonl_path)

    videoid_2_response = {}
    for idx in range(len(responses)):
        video_id = idx_2_videoid[str(idx)]
        assert video_id not in videoid_2_response

        captions = []
        for item in responses[idx]['choices']:
            cleaned_text = clean_text(item['text'])
            if cleaned_text:
                captions.append(cleaned_text)
        if captions:
            videoid_2_response[video_id] = captions

    with open(output_path, 'w') as out:
        json.dump(videoid_2_response, out, indent=4)

    print('output processed file:',output_path)

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--gpt3_response_dir', default="", help="dir containing raw gpt3 response jsonl")
    parser.add_argument('--input_prompts_dir', default="", help="input_prompts dir containing input jsonls and idx2id jsons")
    parser.add_argument('--output_dir', default="", help="ourput dir storing processed gpt3 response json")
    args = parser.parse_args()
    
    
    response_files = glob(os.path.join(args.gpt3_response_dir,"*.jsonl"))
    for response_file in response_files:
        input_name = os.path.basename(response_file)[:-6]
        idx_2_videoid_path = os.path.join(args.input_prompts_dir, f"{input_name}__idx_2_videoid.json")
        output_path = os.path.join(args.output_dir, f"processed_{input_name}.json")
        process_gpt3_response(idx_2_videoid_path, response_file, output_path)
    