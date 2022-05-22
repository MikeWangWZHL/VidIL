from glob import glob
import json
import os
from collections import defaultdict

import argparse
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
from visual_token_generation.prompts import Prompt

def get_prompt_prefix(
        train_visual_tokens, 
        train_frame_captions_filtered, 
        train_frame_captions_unfiltered, 
        training_video_ann, 
        instruction_line, 
        config,
        video_2_question_answer_pairs,
        video_2_asr,
        shot,
        seed
    ):

    dummy_prompt = Prompt("",seed=seed)

    random.seed(seed)

    print(seed, shot)
    # prompt is a instance of class Prompt
    training_example_ids = sorted(list(training_video_ann.keys()))
    chosen_example_ids = []
    while len(chosen_example_ids) != shot:
        cand_example_id = random.choice(training_example_ids)
        cand_video_id = training_video_ann[cand_example_id]['video_id']
        if cand_video_id in train_visual_tokens:
            chosen_example_ids.append(cand_example_id)

    chosen_few_shot_examples = {}
    
    example_strs = []
    for chosen_example_id in chosen_example_ids:
        video_name = training_video_ann[chosen_example_id]['video_id']
        visual_tokens_object = train_visual_tokens[video_name]
        # load frame captions
        if video_name not in train_frame_captions_filtered:
            if config['caption_all_video']:
                if video_name not in train_frame_captions_unfiltered:
                    print('skip loading failed video:',video_name)
                    continue
                frame_captions = train_frame_captions_unfiltered
                print(f'fallback to unfiltered: {video_name}')
            else:
                continue
        else:
            frame_captions = train_frame_captions_filtered
        
        # load asr
        if video_2_asr is not None and video_name in video_2_asr:
            subs = video_2_asr[video_name]
            if subs == []:
                asr = 'no subtitle.'
            else:
                if config['prompt_task'] == 'vlep':
                    new_subs = []
                    total_length = 0
                    for sub in subs:
                        sub = sub.strip()
                        if not sub.endswith(('.', ',', '?', ';', '!',':',"\'","\"")):
                            sub += '.'
                        new_subs.append(sub)
                        total_length += len(sub)
                        if total_length >= 1024:
                            break
                    asr = ' '.join(new_subs)
                else:
                    asr = ' '.join(subs) # list of str
        else:
            asr = None

        prompt_str = dummy_prompt.construct_prompt(video_name, visual_tokens_object, frame_captions, config, question=None, answer=None, asr=asr, vlep_example = training_video_ann[chosen_example_id])
        chosen_few_shot_examples[chosen_example_id] = training_video_ann[chosen_example_id]
    
        example_strs.append(prompt_str)

    final_prompt_prefix_str = "\n\n".join([instruction_line] + example_strs) + "\n\n"


    print(f'### {chosen_example_ids} ###')
    print(final_prompt_prefix_str)
    print()

    # output chosen video ifs
    output_name = os.path.basename(config['output_path'])[:-6]
    output_dirname = os.path.dirname(config['output_path'])
    with open(os.path.join(output_dirname, output_name + '__chosen_samples.json'), 'w') as out:
        json.dump(chosen_few_shot_examples, out, indent=4)

    return final_prompt_prefix_str

def save_prompt_lines(
    target_video_ann,
    visual_tokens, 
    frame_captions_filtered, 
    frame_captions_unfiltered, 
    prompt, 
    config, 
    video_2_question_answer_pairs,
    video_2_asr
    ):
    # prompt is a instance of class Prompt
    print('number of videos:', len(visual_tokens))
    output_lines = []
    line_num_2_example_id = {}
    for example_id, ann in target_video_ann.items():
        video_name = ann['video_id']
        visual_tokens_object = visual_tokens[video_name]
        # load frame captions
        if video_name not in frame_captions_filtered:
            if config['caption_all_video']:
                if video_name not in frame_captions_unfiltered:
                    print('skip loading failed video:',video_name)
                    continue
                frame_captions = frame_captions_unfiltered
                print(f'fallback to unfiltered: {video_name}')
            else:
                continue
        else:
            frame_captions = frame_captions_filtered
        
        # load asr
        if video_2_asr is not None and video_name in video_2_asr:
            subs = video_2_asr[video_name]
            if subs == []:
                asr = 'no subtitle.'
            else:
                if config['prompt_task'] == 'vlep':
                    new_subs = []
                    total_length = 0
                    for sub in subs:
                        sub = sub.strip()
                        if not sub.endswith(('.', ',', '?', ';', '!',':',"\'","\"")):
                            sub += '.'
                        new_subs.append(sub)
                        total_length += len(sub)
                        if total_length >= 1024:
                            break
                    asr = ' '.join(new_subs)
                else:
                    asr = ' '.join(subs) # list of str
        else:
            asr = None


        prompt_str = prompt.construct_prompt(video_name, visual_tokens_object, frame_captions, config, question=None, answer=None, asr=asr, vlep_example = ann)
        request_body = config['request_body']
        request_body["prompt"] = prompt_str
        output_lines.append(json.dumps(request_body))
        line_num_2_example_id[len(output_lines)-1] = example_id
        
    
    # # output prompt
    with open(config['output_path'], 'w') as out:
        for line in output_lines:
            out.write(line)
            out.write('\n')

    # output line idx to videoid
    output_name = os.path.basename(config['output_path'])[:-6]
    output_dirname = os.path.dirname(config['output_path'])
    with open( os.path.join(output_dirname, output_name + '__idx_2_example_id.json'), 'w') as out:
        json.dump(line_num_2_example_id, out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_task', default='caption')
    parser.add_argument('--visual_tokens_dir')
    parser.add_argument('--frame_captions_dir')
    parser.add_argument('--question_answer_path',default='',help='path to a json file: key is videoid, value are question, answer pairs')
    parser.add_argument('--asr_path',default='',help='path to a json file: key is videoid, value is the ASR text')
    parser.add_argument('--prompt_temporal_template', default='temporal_natural', help="choose from ['temporal_natural','temporal_index','static']")
    parser.add_argument('--output_dir')
    parser.add_argument('--output_name',default='gpt3_queries.jsonl')
    parser.add_argument('--caption_all_video', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_objects', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_events', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_attributes', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_scenes', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_original_caption', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_frame_captions', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_ASR', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--add_answer', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--gpt3_temperature', default=0.0, type=float)
    parser.add_argument('--gpt3_max_tokens', default=64, type=int)
    parser.add_argument('--gpt3_top_p', default=1, type=int)
    ### args for randomly select in-context examples to get prompt prefix ###
    parser.add_argument('--trainset_json_ann', help='in order to restrict the selection in training set videoids')
    parser.add_argument('--target_json_ann', help='vlep ann json such as: VLEP/ann/example_id_2_events_answer_dev.json')
    parser.add_argument('--train_dataset_visual_tokens_dir', help='contains trainset visual tokens')
    parser.add_argument('--train_dataset_frame_captions_dir', help='contains trainset captions')
    parser.add_argument('--instruction_line', help='instruction line in prompt prefix')
    parser.add_argument('--shot', default=5, help='how many in-context examples to use')
    parser.add_argument('--seed', default=42, help='random seed')
    
    args = parser.parse_args()

    visual_tokens_json_path = os.path.join(args.visual_tokens_dir, 'visual_tokens.json')
    frame_caption_filtered_json_path = os.path.join(args.frame_captions_dir, 'video_text_CapFilt.json')
    frame_caption_unfiltered_json_path = os.path.join(args.frame_captions_dir, 'video_text_Cap.json')

    """ load frame captions and visual tokens """
    visual_tokens = json.load(open(visual_tokens_json_path))
    frame_captions_filtered = json.load(open(frame_caption_filtered_json_path))
    frame_captions_unfiltered = json.load(open(frame_caption_unfiltered_json_path))
    
    
    """ load question answer dict"""
    if args.prompt_task == 'qa':
        print('prompt for qa task ...')
        assert args.question_answer_path != ''
        video_2_question_answer_pairs = json.load(open(args.question_answer_path))
    else:
        print('prompt for caption/ vlep task ...')
        video_2_question_answer_pairs = None

    if args.asr_path != '' and args.add_ASR:
        print(f'using ASR:{args.add_ASR}')
        video_2_asr = json.load(open(args.asr_path))
    else:
        video_2_asr = None

    
    """ output path """
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir,f"temp_{args.gpt3_temperature}_" + args.output_name)
    
    print('using temperature:',args.gpt3_temperature)
    print('using max tokens:',args.gpt3_max_tokens)
    print('using top p:',args.gpt3_top_p)
    request_body = {
        "engine": "text-davinci-002",
        "prompt": "",
        "temperature": args.gpt3_temperature,
        "max_tokens": args.gpt3_max_tokens,
        "top_p": args.gpt3_top_p,
        "frequency_penalty": 0,
        "presence_penalty": 0
    } 
    config = {
        "prompt_task":args.prompt_task,
        "add_objects":args.add_objects,
        "add_events":args.add_events,
        "add_attributes":args.add_attributes,
        "add_scenes":args.add_scenes,
        "add_original_caption":args.add_original_caption,
        "add_frame_captions":args.add_frame_captions,
        "add_ASR":args.add_ASR,
        "add_answer":args.add_answer,
        "prompt_temporal_template":args.prompt_temporal_template,
        "prompt_version":'v2',
        "visual_token_aggregation_version":'v2',
        "topk":4,
        "output_path":output_path,
        "request_body":request_body,
        "caption_all_video":args.caption_all_video
    }

    '''output prompt'''
    if args.prompt_task == 'caption' and args.caption_all_video:
        print('using caption_all_video: it is gaurenteed to have a query for each video')


    ### generate random in-context examples

    """ load full frame captions and visual tokens """
    train_visual_tokens_json_path = os.path.join(args.train_dataset_visual_tokens_dir, 'visual_tokens.json')
    train_frame_caption_filtered_json_path = os.path.join(args.train_dataset_frame_captions_dir, 'video_text_CapFilt.json')
    train_frame_caption_unfiltered_json_path = os.path.join(args.train_dataset_frame_captions_dir, 'video_text_Cap.json')

    train_visual_tokens = json.load(open(train_visual_tokens_json_path))
    train_frame_captions_filtered = json.load(open(train_frame_caption_filtered_json_path))
    train_frame_captions_unfiltered = json.load(open(train_frame_caption_unfiltered_json_path))

    training_video_ann = json.load(open(args.trainset_json_ann))
    
    # filled with gt annotation:
    config['add_original_caption'] = True
    config['add_answer'] = True
    
    prompt_prefix_str = get_prompt_prefix(
        train_visual_tokens, 
        train_frame_captions_filtered, 
        train_frame_captions_unfiltered, 
        training_video_ann, 
        args.instruction_line, 
        config,
        video_2_question_answer_pairs,
        video_2_asr,
        int(args.shot),
        int(args.seed)
    )
    


    ### output final prompts
    target_video_ann = json.load(open(args.target_json_ann))
    
    prompt = Prompt(prompt_prefix_str, seed=int(args.seed))
    
    # change back to user specified
    config['add_original_caption'] = args.add_original_caption
    config['add_answer'] = args.add_answer
    save_prompt_lines(
        target_video_ann,
        visual_tokens, 
        frame_captions_filtered, 
        frame_captions_unfiltered, 
        prompt, 
        config, 
        video_2_question_answer_pairs = video_2_question_answer_pairs,
        video_2_asr = video_2_asr
    )

