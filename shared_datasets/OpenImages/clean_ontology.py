import json
import re
import os
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def remove_semantically_similar(texts, thresh, output_path = None):
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # sbert model
    model_name = 'all-mpnet-base-v2'
    print(f'loading {model_name}...')
    model = SentenceTransformer(model_name)
    model.eval()
    model.to(device)

    print(f'compute similarity...')
    text_embeddings = model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(text_embeddings, text_embeddings)
    print(cosine_scores.size())

    # top2 = torch.topk(cosine_scores, 2, dim=1)
    # values, indices = top2[0].detach().cpu().numpy(), top2[1].detach().cpu().numpy()

    mapping = {i:i for i in range(len(texts))}
    cosine_scores = cosine_scores.detach().cpu().numpy()
    for i in tqdm(range(len(cosine_scores)-1)):
        for j in range(i+1, len(cosine_scores)):
            if mapping[j] != j:
                continue
            if cosine_scores[i][j] > thresh:
                mapping[j] = i
                from_text = texts[j]
                to_text = texts[i]
                print(f'mapping: {from_text} -> {to_text}')
    new_texts = set()
    for index in mapping.values():
        new_texts.add(texts[index])

    new_texts = sorted(list(new_texts))

    print(f'before:{len(texts)} -> after:{len(new_texts)}')

    if output_path:
        with open(output_path, 'w') as out:
            json.dump(new_texts, out, indent=4)
    return new_texts



def clean_text(texts, output_path = None):
    new_texts = []
    for text in texts:
        if '(fictional character)' not in text:
            new_texts.append(text)
    if output_path:
        with open(output_path, 'w') as out:
            json.dump(new_texts, out, indent=4)

if __name__ == '__main__':
    list_json = './openimage_classes_all.json'
    output_path = './openimage_classes_all_cleaned_fictional_characters.json'
    
    clean_text(json.load(open(list_json)), output_path)


    # output_path = './openimage_classes_all_remove_similar0.9.json'
    # cleaned = remove_semantically_similar(json.load(open(list_json)), thresh = 0.9, output_path = output_path)