import json
import os
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict
import spacy
from tqdm import tqdm

def load_json(path):
    return json.load(open(path))

def load_jsonl_results_alpro(path, ansid_2_answer):
    preds = json.load(open(path))
    lines = []
    for pred in preds:
        lines.append({
            "question_id":pred['question_id'],
            "answer":ansid_2_answer[pred['answer']]
        })
    return lines

def load_jsonl_gt(path):
    lines = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            obj['question_id'] = len(lines)
            lines.append(obj)
    return lines

def load_jsonl_result(path):
    lines = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            lines.append(obj)
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

def process_gpt3_response_jsonl(response_jsonl_path, output_dir):
    responses = load_jsonl_result(response_jsonl_path)
    lines = []
    for idx in range(len(responses)):
        captions = []
        for item in responses[idx]['choices']:
            cleaned_text = clean_text(item['text'])
            captions.append(cleaned_text)

        lines.append({"samples":captions})

    assert len(lines) == len(responses)
    output_path = os.path.join(output_dir, 'tmp.jsonl')
    with open(output_path, 'w') as out:
        for line in lines:
            out.write(json.dumps(line))
            out.write('\n')
    print('output processed file:',output_path)
    return output_path
### 

def evaluate_ranking_result(prediction_json, groudtruth_jsonl):
    preds = load_json(prediction_json)
    qid_2_pred = {item['question_id']:item['answer'] for item in preds}

    gt = load_jsonl_gt(groudtruth_jsonl)
    qid_2_gt = {item['question_id']:item['answer'] for item in gt}

    pred_list = []
    gt_list = []
    for key in qid_2_gt.keys():
        if key in qid_2_pred:
            gt_list.append(qid_2_gt[key])
            pred_list.append(qid_2_pred[key])

    print(len(pred_list))
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    acc = accuracy_score(gt_array, pred_array)
    print(acc)
    
    # macro_score = precision_recall_fscore_support(gt_array, pred_array, average='macro')
    # micro_score = precision_recall_fscore_support(gt_array, pred_array, average='micro')
    # print('macro:', macro_score)
    # print('micro:', micro_score)

def evaluate_ranking_result_Alpro(prediction_json, groudtruth_jsonl):
    preds = load_json(prediction_json)
    qid_2_pred = {item['question_id']:item['answer'] for item in preds}

    gt = load_jsonl_gt(groudtruth_jsonl)
    qid_2_gt = {item['question_id']:item['answer'] for item in gt}

    pred_list = []
    gt_list = []
    for key in qid_2_gt.keys():
        if key in qid_2_pred:
            gt_list.append(qid_2_gt[key])
            pred_list.append(qid_2_pred[key])

    print(len(pred_list))
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    acc = accuracy_score(gt_array, pred_array)
    print(acc)
    
    # macro_score = precision_recall_fscore_support(gt_array, pred_array, average='macro')
    # micro_score = precision_recall_fscore_support(gt_array, pred_array, average='micro')
    # print('macro:', macro_score)
    # print('micro:', micro_score)

def question_aware_post_processing(question_str, answer_str, nlp = None):
    if "not sure" in answer_str or "There is no" in answer_str:
        if "doing?" in question_str:
            ret = "talk"
        elif "who" in question_str:
            ret = "person"
        else:
            ret = answer_str
    else:
        ret = answer_str
    return ret

        
def evaluate_generation_result(prediction_json, groudtruth_jsonl, answer_list_json):
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # sbert model
    model_name = 'all-mpnet-base-v2'
    print(f'loading {model_name}...')
    model = SentenceTransformer(model_name)
    model.eval()
    model.to(device)

    # get answer list embedding
    answer_list = load_json(answer_list_json)
    answer_embeddings = model.encode(answer_list, convert_to_tensor=True)
    
    # get predictions
    if isinstance(prediction_json, str):
        preds = load_json(prediction_json)
    elif isinstance(prediction_json, list):
        preds = prediction_json

    pred_qid_list = []
    pred_answer_list = []
    for item in preds:
        pred_answer_list.append(item['answer'])
        pred_qid_list.append(item['question_id'])
    
    pred_answer_embeddings = model.encode(pred_answer_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(pred_answer_embeddings, answer_embeddings)
    cosine_scores = cosine_scores.cpu().detach().numpy()

    qid_2_pred = {}
    for i in range(len(pred_qid_list)):
        top_answer_idx = np.argmax(cosine_scores[i])
        qid_2_pred[pred_qid_list[i]] = answer_list[top_answer_idx]
        print('map answer:',pred_answer_list[i],'->',answer_list[top_answer_idx])
    
    # load gt
    gt = load_jsonl_gt(groudtruth_jsonl)
    qid_2_gt = {item['question_id']:item['answer'] for item in gt}
    

    # compute acc
    pred_list = []
    gt_list = []
    for key in qid_2_gt.keys():
        if key in qid_2_pred:
            gt_list.append(qid_2_gt[key])
            pred_list.append(qid_2_pred[key])

    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    
    acc = accuracy_score(gt_array, pred_array)
    print(acc)

def evaluate_generation_result_jsonl(prediction_jsonl, groudtruth_jsonl, answer_list_json, idx_2_video_id, post_processing=False):
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # set up spacy
    if post_processing:
        print("loading spacy...")
        nlp = spacy.load("en_core_web_sm")

    # sbert model
    model_name = 'all-mpnet-base-v2'
    print(f'loading {model_name}...')
    model = SentenceTransformer(model_name)
    model.eval()
    model.to(device)

    # get answer list embedding
    answer_list = load_json(answer_list_json)
    answer_embeddings = model.encode(answer_list, convert_to_tensor=True)
    
    # load gt
    gt = load_jsonl_gt(groudtruth_jsonl)
    video_id_to_gts = defaultdict(list)
    video_id_to_questions = defaultdict(list)
    for item in gt:
        video_id_to_gts[item['video_id']].append(item['answer'])
        video_id_to_questions[item['video_id']].append(item['question'])
    
    # get predictions
    preds = load_jsonl_result(prediction_jsonl)
    video_id_to_answers = defaultdict(list)
    for i in range(len(preds)):
        item = preds[i]
        video_id = idx_2_video_id[str(i)][0]
        if video_id in video_id_to_gts:
            video_id_to_answers[video_id].append(item['samples'][0])
    
    
    # get list
    gt_list = []
    pred_answer_list = []
    print('post-processing prediction...')
    for key in video_id_to_answers.keys():
        assert len(video_id_to_gts[key]) == len(video_id_to_answers[key])
        for i in range(len(video_id_to_gts[key])):
            gt_list.append(video_id_to_gts[key][i])
            
            answer_str = video_id_to_answers[key][i]
            
            if post_processing:
                answer_str = question_aware_post_processing(video_id_to_questions[key][i], answer_str, nlp)

            pred_answer_list.append(answer_str)
    
    pred_answer_embeddings = model.encode(pred_answer_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(pred_answer_embeddings, answer_embeddings)
    cosine_scores = cosine_scores.cpu().detach().numpy()

    pred_list = []
    for i in range(len(pred_answer_list)):
        top_answer_idx = np.argmax(cosine_scores[i])
        pred_list.append(answer_list[top_answer_idx])
   
    # compute scores
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    
    acc = accuracy_score(gt_array, pred_array)
    print(acc)

def evaluate_generation_result_jsonl_majority_vote(prediction_jsonl, groudtruth_jsonl, answer_list_json, idx_2_video_id, post_processing = False):
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # sbert model
    model_name = 'all-mpnet-base-v2'
    print(f'loading {model_name}...')
    model = SentenceTransformer(model_name)
    model.eval()
    model.to(device)

    # get answer list embedding
    answer_list = load_json(answer_list_json)
    answer_embeddings = model.encode(answer_list, convert_to_tensor=True)
    
    # load gt
    gt = load_jsonl_gt(groudtruth_jsonl)
    video_id_to_gts = defaultdict(list)
    video_id_to_questions = defaultdict(list)
    for item in gt:
        video_id_to_gts[item['video_id']].append(item['answer'])
        video_id_to_questions[item['video_id']].append(item['question'])
    
    # get predictions
    preds = load_jsonl_result(prediction_jsonl)
    video_id_to_answers = defaultdict(list)
    for i in range(len(preds)):
        item = preds[i]
        video_id = idx_2_video_id[str(i)][0]
        if video_id in video_id_to_gts:    
            video_id_to_answers[video_id].append(item['samples'])
    
    # get list
    gt_list = []
    pred_answer_list = []
    for key in video_id_to_answers.keys():
        assert len(video_id_to_gts[key]) == len(video_id_to_answers[key])
        for i in range(len(video_id_to_gts[key])):
            gt_list.append(video_id_to_gts[key][i])
 
            answer_strs = video_id_to_answers[key][i]
            if post_processing:
                answer_strs = [question_aware_post_processing(video_id_to_questions[key][i], answer_str, nlp) for answer_str in answer_strs]
            pred_answer_list.append(answer_strs)
    
    print('mapping prediction ...')
    pred_list = []
    for answer_with_sampling in tqdm(pred_answer_list):
        pred_answer_embeddings = model.encode(answer_with_sampling, convert_to_tensor=True)
        cosine_scores = util.cos_sim(pred_answer_embeddings, answer_embeddings)
        cosine_scores = cosine_scores.cpu().detach().numpy()
        cand_dict = defaultdict(int)
        for i in range(len(cosine_scores)):
            top_answer_idx = np.argmax(cosine_scores[i])
            cand_dict[answer_list[top_answer_idx]] += 1
        cand_list = [(key,value) for key,value in cand_dict.items()]
        cand_list = sorted(cand_list,key=lambda x:x[1], reverse=True)
        pred_list.append(cand_list[0][0])
    
    # compute scores
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)
    
    
    acc = accuracy_score(gt_array, pred_array)
    print(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--pred', default="", help="prediction json")
    parser.add_argument('--pred_jsonl', default="", help="prediction jsonl")
    parser.add_argument('--idx_2_video_id', default="", help="idx_2_video_id")
    parser.add_argument('--ans2label', default="", help="ans2label")
    parser.add_argument('--gt', help="ground truth jsonl'")
    parser.add_argument('--answer_list_json', help="required if evaluating generation results'")
    parser.add_argument('--setting', help="'ranking' or 'generation', 'generation_gpt3_raw', 'alpro'")
    args = parser.parse_args()

    assert args.pred!="" or args.pred_jsonl!=""


    if args.setting == 'generation':
        if args.pred_jsonl != "":
            idx_2_video_id = json.load(open(args.idx_2_video_id))
            evaluate_generation_result_jsonl(args.pred_jsonl, args.gt, args.answer_list_json, idx_2_video_id)
        else:
            evaluate_generation_result(args.pred, args.gt, args.answer_list_json)
    elif args.setting == 'ranking':
        evaluate_ranking_result(args.pred, args.gt)

    elif args.setting == 'generation_gpt3_raw':
        output_dir = os.path.dirname(args.pred_jsonl)
        tmp_jsonl_path = process_gpt3_response_jsonl(args.pred_jsonl, output_dir)
        idx_2_video_id = json.load(open(args.idx_2_video_id))
        evaluate_generation_result_jsonl(tmp_jsonl_path, args.gt, args.answer_list_json, idx_2_video_id)

    elif args.setting == 'generation_gpt3_raw_majority_vote':
        output_dir = os.path.dirname(args.pred_jsonl)
        tmp_jsonl_path = process_gpt3_response_jsonl(args.pred_jsonl, output_dir)
        idx_2_video_id = json.load(open(args.idx_2_video_id))
        evaluate_generation_result_jsonl_majority_vote(tmp_jsonl_path, args.gt, args.answer_list_json, idx_2_video_id)

    elif args.setting == 'alpro_generation':
        ansid_2_answer = {value:key for key,value in json.load(open(args.ans2label)).items()}
        processed_json = load_jsonl_results_alpro(args.pred, ansid_2_answer)
        evaluate_generation_result(processed_json, args.gt, args.answer_list_json)