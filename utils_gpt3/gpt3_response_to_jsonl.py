import json
import os
import argparse
from glob import glob

def video_level_only(input_path, output_dir, selected_ids = None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f'{os.path.basename(input_path)[:-5]}.jsonl') 

    lines = []
    if selected_ids is None:
        selected_ids = json.load(open(input_path)).keys()
    for key,value in json.load(open(input_path)).items():
        if key not in selected_ids:
            continue
        for i in range(len(value)):
            # caption = value[i]['text'].strip()
            caption = value[i].strip()
            clip_name = key
            sen_id = len(lines)
            line = {'caption':caption,'clip_name':clip_name,'sen_id':sen_id}
            lines.append(line)
    print(len(lines))
    with open(output_path, 'w') as out:
        for line in lines:
            out.write(json.dumps(line))
            out.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--dataset', default="msrvtt", help="chosen dataset")
    parser.add_argument('--gpt3_processed_dir', default="", help="processed gpt3 response json")
    parser.add_argument('--output_dir', default="pseudo_label_ann_example", help="output pseudo labeled dataset ann jsonl")
    args = parser.parse_args()

    dataset = args.dataset
    output_dir = args.output_dir

    input_paths = glob(os.path.join(args.gpt3_processed_dir,"*.json"))
    
    for input_path in input_paths:
        if dataset == 'msrvtt':
            training_ann = json.load(open('shared_datasets/MSRVTT_ret/ann/video_2_text_original_train.json'))
        elif dataset == 'vatex':
            training_ann = json.load(open('shared_datasets/Vatex/value_ann/vatex_en_c/videoid_2_text_vatex_en_c_train.json'))
        
        video_level_only(input_path, output_dir, selected_ids = training_ann.keys())
