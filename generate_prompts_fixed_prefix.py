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


def save_prompt_lines(
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
    line_num_2_video_id = {}
    for video_name, visual_tokens_object in visual_tokens.items():
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
                asr = ' '.join(subs) # list of str
                if asr == '':
                    asr = 'no subtitle.'
        else:
            asr = None

        if config['prompt_task'] == 'qa':
            if video_name not in video_2_question_answer_pairs:
                print(f'skip video without qa annotation: {video_name}')
                continue
            for qidx in range(len(video_2_question_answer_pairs[video_name])):
                item = video_2_question_answer_pairs[video_name][qidx]
                question = item['question']
                answer = item['answer']
                prompt_str = prompt.construct_prompt(video_name, visual_tokens_object, frame_captions, config, question, answer, asr)
                # print(f'### {video_name} ###')
                # print(prompt_str)
                # print()
                request_body = config['request_body']
                request_body["prompt"] = prompt_str
                output_lines.append(json.dumps(request_body))
                line_num_2_video_id[len(output_lines)-1] = (video_name,qidx)
        
        else:
            prompt_str = prompt.construct_prompt(video_name, visual_tokens_object, frame_captions, config, question=None, answer=None, asr=asr)
            request_body = config['request_body']
            request_body["prompt"] = prompt_str
            output_lines.append(json.dumps(request_body))
            line_num_2_video_id[len(output_lines)-1] = video_name
        
    
    # # output prompt
    with open(config['output_path'], 'w') as out:
        for line in output_lines:
            out.write(line)
            out.write('\n')

    # output line idx to videoid
    output_name = os.path.basename(config['output_path'])[:-6]
    output_dirname = os.path.dirname(config['output_path'])
    with open( os.path.join(output_dirname, output_name + '__idx_2_videoid.json'), 'w') as out:
        json.dump(line_num_2_video_id, out, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_task', default='caption')
    parser.add_argument('--visual_tokens_dir')
    parser.add_argument('--frame_captions_dir')
    parser.add_argument('--question_answer_path',default='',help='path to a json file: key is videoid, value are question, answer pairs')
    parser.add_argument('--asr_path',default='',help='path to a json file: key is videoid, value is the ASR text')
    parser.add_argument('--prompt_prefix', help='path to a txt file')
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
    args = parser.parse_args()

    visual_tokens_json_path = os.path.join(args.visual_tokens_dir, 'visual_tokens.json')
    frame_caption_filtered_json_path = os.path.join(args.frame_captions_dir, 'video_text_CapFilt.json')
    frame_caption_unfiltered_json_path = os.path.join(args.frame_captions_dir, 'video_text_Cap.json')

    """ load frame captions and visual tokens """
    visual_tokens = json.load(open(visual_tokens_json_path))
    frame_captions_filtered = json.load(open(frame_caption_filtered_json_path))
    frame_captions_unfiltered = json.load(open(frame_caption_unfiltered_json_path))
    
    """ set up prompt template """
    prompt = Prompt(args.prompt_prefix)
    
    """ load question answer dict"""
    if args.prompt_task == 'qa':
        print('prompt for qa task ...')
        assert args.question_answer_path != ''
        video_2_question_answer_pairs = json.load(open(args.question_answer_path))
    elif args.prompt_task == 'caption':
        print('prompt for caption task ...')
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

    save_prompt_lines(
        visual_tokens, 
        frame_captions_filtered, 
        frame_captions_unfiltered, 
        prompt, 
        config, 
        video_2_question_answer_pairs = video_2_question_answer_pairs,
        video_2_asr = video_2_asr
    )

