import json
import os
import spacy
from collections import defaultdict

def load_jsonl(jsonl_path):
    lines = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

### processing gpt3 response ###
def clean_text(text, filter_short_response = False, take_first_sentence = True):
    text = text.strip()
    if '\n\nObjects:' in text:
        text = text.split('\n\nObjects:')[0]
    elif '\n\nFrame' in text:
        text = text.split('\n\nFrame')[0]
    else:
        text = text
    if take_first_sentence:
        text = text.split('.')[0].strip()
    # print(text)

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
###

from datasets import list_metrics, load_metric

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def video_caption_eval(gt_jsonl, results_file, output_path = None):
    print('preparing gt and result...')
    if isinstance(gt_jsonl, str):
        gt = load_jsonl(gt_jsonl)
    else:
        gt = gt_jsonl
    
    if isinstance(results_file, str):
        results = json.load(open(results_file))
        if not isinstance(results,list):
            results = [{'video_id':key,'caption':captions} for key,captions in results.items()] 
    else:
        results = results_file
    
    # get keys
    res_keys = set()
    gts_keys = set()
    for item in results:
        res_keys.add(item['video_id'])
    for line in gt:
        gts_keys.add(line['clip_name'])

    # get dict for common keys
    res = defaultdict(list)
    for item in results:
        if item['video_id'] not in gts_keys:
            continue
        if isinstance(item['caption'],list):
            res[item['video_id']].append({'image_id':item['video_id'], 'caption':item['caption'][0]}) # take the first one as hypothesis
        elif isinstance(item['caption'],str):
            res[item['video_id']].append({'image_id':item['video_id'], 'caption':item['caption']})
    

    gts = defaultdict(list)
    for line in gt:
        if line['clip_name'] not in res:
            continue
        if isinstance(line['caption'],list):
            gts[line['clip_name']] += [{'image_id':line['clip_name'], 'caption':c} for c in line['caption']]
        elif isinstance(line['caption'],str):
            gts[line['clip_name']].append({'image_id':line['clip_name'], 'caption':line['caption']})
    
    assert res.keys() == gts.keys()
    print(f'evaluate {len(res)} videos...')
        
    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    print('setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    eval_dict = {}
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        eval_dict[scorer.method()] = score
    print(eval_dict)

    if output_path:
        with open(output_path,'w') as out:
            json.dump(eval_dict, out, indent=4)
        print('')
    
    return eval_dict

def eval_captioning_from_gpt3_response(gpt3_response_file, idx_2_videoid_json, gt_jsonl, output_dir):
    """
        gpt3_response_file: path the raw jsonl output from gpt3, e.g., <output_name>.jsonl
        idx_2_videoid_json: path to the json containing line idx of the output jsonl to videoid, i.e., <output_name>__idx_2_videoid.json; 
            this file can be found in the input prompt dir such as "<OUTPUT_ROOT>/msrvtt/input_prompts", where the <OUTPUT_ROOT> is 
            what user defined for running the pipeline
        gt_jsonl: the ground truth jsonl from dataset, such as "<shared_datasets>/MSRVTT_caption/ann/test_caption.jsonl"
        output_dir: where to store the processed gpt3 captions and eval metrics
    """
    processed_gpt3_output_path = os.path.join(output_dir,'processed_'+os.path.basename(gpt3_response_file)[:-1])
    metric_output_path = os.path.join(output_dir,'metric.json')
    
    process_gpt3_response(idx_2_videoid_json, gpt3_response_file, output_path=processed_gpt3_output_path)
    video_caption_eval(gt_jsonl, processed_gpt3_output_path, output_path = metric_output_path)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()     
    parser.add_argument('--output_dir', default='visual_token_generation/output/tmp')
    parser.add_argument('--pred', help="path to the captioning result json file: a list of objects {'video_id':vid, 'caption':[str]} or raw gpt3 response jsonl")
    parser.add_argument('--pred_idx_2_vid', help="only required for gpt3 response: path to the json containing line idx of the output jsonl to videoid, i.e., <output_name>__idx_2_videoid.json")
    parser.add_argument('--gt', help="path to ground truth captions jsonl, e.g., '<shared_datasets>/MSRVTT_caption/ann/test_caption.jsonl'")
    parser.add_argument('--input_format', default='result_file', help="choose from ['result_file', 'gpt3_response']")
    args = parser.parse_args()


    gt_jsonl = args.gt # 'shared_datasets/MSRVTT_caption/ann/test_caption.jsonl'
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.input_format == 'result_file':
        ### eval from result file
        results_file = args.pred # 'output/eval_captioning_msrvtt_from_BLIP_COCO_finetuned_concat_frame_8/result/test_epoch0.json'
        video_caption_eval(gt_jsonl, results_file, output_path=os.path.join(output_dir,'metric.json'))
    elif args.input_format == 'gpt3_response':
        ### eval from gpt3 response
        gpt3_response_file = args.pred
        gpt3_idx_2_video_id = args.pred_idx_2_vid
        eval_captioning_from_gpt3_response(gpt3_response_file, gpt3_idx_2_video_id, gt_jsonl, output_dir)
    else:
        print("specify --input_format from ['result_file', 'gpt3_response']")