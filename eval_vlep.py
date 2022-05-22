import json
import os

import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict

from tqdm import tqdm

def eval_openended(results, example_id_2_data, model, output_path):
    output_lines = []
    for example_id, ann in tqdm(example_id_2_data.items()):
        video_id = ann['video_id']
        if video_id in results:
            if isinstance(results[video_id],str):
                result = results[video_id]
            elif isinstance(results[video_id],list):
                result = results[video_id][0]
            candidates = ann['events']
            cand_embeddings = model.encode(candidates, convert_to_tensor=True)
            pred_embeddings = model.encode(result, convert_to_tensor=True)
            cosine_scores = util.cos_sim(pred_embeddings, cand_embeddings)
            cosine_scores = cosine_scores.cpu().detach().numpy()
            # print(cosine_scores.shape)
            top_answer_idx = np.argmax(cosine_scores[0])
            print(candidates)
            print(result, ' -> ', candidates[top_answer_idx])
            assert int(top_answer_idx) in [0, 1]
            output_lines.append({"example_id":int(example_id), "pred_ans":int(top_answer_idx)})
    print(len(output_lines))
    with open(output_path, 'w') as out:
        for line in output_lines:
            out.write(json.dumps(line))
            out.write('\n')
  
def eval_multichoice(results, example_id_2_data, model, output_path):
    output_lines = []
    for example_id, ann in tqdm(example_id_2_data.items()):
        if example_id in results:
            if isinstance(results[example_id],str):
                result = results[example_id]
            elif isinstance(results[example_id],list):
                result = results[example_id][0]
            candidates = ann['events']
            cand_embeddings = model.encode(candidates, convert_to_tensor=True)
            pred_embeddings = model.encode(result, convert_to_tensor=True)
            cosine_scores = util.cos_sim(pred_embeddings, cand_embeddings)
            cosine_scores = cosine_scores.cpu().detach().numpy()
            # print(cosine_scores.shape)
            top_answer_idx = np.argmax(cosine_scores[0])
            # print(candidates)
            # print(result, ' -> ', candidates[top_answer_idx])
            assert int(top_answer_idx) in [0, 1]
            output_lines.append({"example_id":int(example_id), "pred_ans":int(top_answer_idx)})
    with open(output_path, 'w') as out:
        for line in output_lines:
            out.write(json.dumps(line))
            out.write('\n')

def gen_dummy_dev_result(output_path='shared_datasets/VLEP/submission/dummy_dev.jsonl'):
    output_lines = []
    for example_id, ann in json.load(open('shared_datasets/VLEP/ann/example_id_2_events_answer_dev.json')).items():
        output_lines.append({"example_id":int(example_id), "pred_ans":0})
    with open(output_path, 'w') as out:
        for line in output_lines:
            out.write(json.dumps(line))
            out.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()     
    parser.add_argument('--gpt3_processed_response', default="", help="processed gpt3 response json")
    parser.add_argument('--output_path', default="", help="vlep output path in official eval format for CodaLab submission")

    args = parser.parse_args()

    ''' set up device '''
    ### use cuda
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    ### sbert model
    model_name = 'all-mpnet-base-v2'
    print(f'loading {model_name}...')
    model = SentenceTransformer(model_name)
    model.eval()
    model.to(device)

    ### load example id to data
    example_id_2_data = json.load(open('shared_datasets/VLEP/ann/example_id_2_events_answer_test.json'))

    ### load result json
    results = json.load(open(args.gpt3_processed_response))
    output_path = args.output_path
    ### generate evaluation file

    if "multichoice" in result_file_name:
        eval_multichoice(results, example_id_2_data, model, output_path)
    elif "openended" in result_file_name:
        eval_openended(results, example_id_2_data, model, output_path)
    