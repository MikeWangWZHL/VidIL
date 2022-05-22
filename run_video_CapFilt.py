'''
    code modified from https://github.com/salesforce/BLIP, https://github.com/salesforce/ALPRO
'''

from PIL import Image
import requests

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder
from models.blip_itm import blip_itm

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

from glob import glob
import av
import lmdb
import decord
from decord import VideoReader

import spacy
from tqdm import tqdm
import shutil

decord.bridge.set_bridge('native')

def load_video_from_path_decord(video_path, frm_sampling_strategy, num_frm, height=None, width=None, start_time=None, end_time=None, fps=-1):
    try:
        if not height or not width:
            vr = VideoReader(video_path)
        else:
            vr = VideoReader(video_path, width=width, height=height)

        vlen = len(vr)

        if start_time or end_time:
            assert fps > 0, 'must provide video fps if specifying start and end time.'

            start_idx = min(int(start_time * fps), vlen)
            end_idx = min(int(end_time * fps), vlen)
        else:
            start_idx, end_idx = 0, vlen

        if frm_sampling_strategy == 'uniform':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm, dtype=int)
        elif frm_sampling_strategy == 'nlvl_uniform':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm).astype(int)
        elif frm_sampling_strategy == 'nlvl_rand':
            frame_indices = np.arange(start_idx, end_idx, vlen / num_frm).astype(int)

            # generate some random perturbations
            strides = [frame_indices[i] - frame_indices[i-1] for i in range(1, len(frame_indices))] + [vlen - frame_indices[-1]]
            pertube = np.array([np.random.randint(0, stride) for stride in strides])

            frame_indices = frame_indices + pertube

        elif frm_sampling_strategy == 'rand':
            frame_indices = sorted(random.sample(range(vlen), num_frm))
        elif frm_sampling_strategy == 'headtail':
            frame_indices_head = sorted(random.sample(range(vlen // 2), num_frm // 2))
            frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), num_frm // 2))
            frame_indices = frame_indices_head + frame_indices_tail
        else:
            raise NotImplementedError('Invalid sampling strategy {} '.format(frm_sampling_strategy))

        raw_sample_frms = vr.get_batch(frame_indices).asnumpy() # (num_frm, H, W, C)

    except Exception as e:
        LOGGER.info(e)
        return None
    return raw_sample_frms

@torch.no_grad()
def caption_frames(captioner, images, mode='beam'):
    '''
        mode: "beam", 'sample'
    '''
    with torch.no_grad():
        # beam search
        if mode == 'beam':
            captions = captioner.generate(images, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        else:
            captions = captioner.generate(images, sample=True, top_p=0.9, max_length=20, min_length=5) 
    return captions

@torch.no_grad()
def filter_captions(filterer, images, texts, threshold, mode='max_filter'):
    filtered_captions = []
    for i in range(len(texts)):
        t = texts[i]
        itm_output = filterer(images, [t for i in range(images.size()[0])], match_head='itm')
        # print(itm_output.size())
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1].detach().cpu().numpy()
        # print(itm_score)

        if mode == 'avg_filter':
            prob = np.sum(itm_score)/len(itm_score)
        elif mode == 'max_filter':
            prob = np.max(itm_score)
        
        if prob > threshold:
            filtered_captions.append(t)
        
    # print(f'filtered captions:',filtered_captions)
    return filtered_captions

def process_frame(frame, config, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config["image_size"],config["image_size"]),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    # processed = transform(frame).unsqueeze(0).to(device)
    processed = transform(frame).to(device)
    return processed

@torch.no_grad()
def CapFilt(data, config, device):
    # spacy
    nlp = spacy.load("en_core_web_sm", disable=['ner','tagger','lemmatizer'])

    # load captioner
    captioner = blip_decoder(pretrained=config["caption_model_ckpt"], image_size=config["image_size"], vit=config["vit"])
    captioner.eval()
    captioner = captioner.to(device)
    
    # load filterer
    filterer = blip_itm(pretrained=config["filterer_model_ckpt"], image_size=config["image_size"], vit=config["vit"])
    filterer.eval()
    filterer = filterer.to(device)
    
    # count = 0
    for item in tqdm(data):
        video_path = item['video_path']
        
        # sample frames
        try:
            raw_sample_frms = load_video_from_path_decord(video_path, config["frm_sampling_strategy"], config["num_frm_CapFilt"])
            processed_frms = torch.stack([process_frame(frm, config, device) for frm in raw_sample_frms])
        except:
            print(f'skip video that cannot be loaded: {video_path}')
            continue

        if ('do_sentence_tokenization' not in config) or config['do_sentence_tokenization']:
            original_caption_sentences = []
            for original_cap in item['text']:
                original_caption = original_cap.replace('\n','. ')
                doc = nlp(original_caption)
                for sent in doc.sents:
                    if len(sent.text) > 3:
                        original_caption_sentences.append(sent.text.strip())
        else:
            original_caption_sentences = [cap.replace('\n','. ').strip() for cap in item['text']]

        # captioning
        if not config["caption"]:
            candidate_captions = original_caption_sentences
            item['unfiltered_text'] = candidate_captions
        else:
            generated_captions = caption_frames(captioner, processed_frms, mode=config["generation_mode"])
            
            # filter duplicated frame captions: exact match
            generated_captions_final = []
            for cap in generated_captions:
                if cap not in generated_captions_final:
                    generated_captions_final.append(cap)
            # add original captions
            if config['keep_original_caption']:
                candidate_captions = original_caption_sentences + generated_captions_final
                item['unfiltered_text'] = candidate_captions
            else:
                item['text'] = []
                candidate_captions = generated_captions_final
                item['unfiltered_text'] = candidate_captions
        # filtering
        if config["filter"]:
            if config["filter_generated_only"]:
                item['text'] += filter_captions(filterer, processed_frms, generated_captions_final, config["threshold"], config['filter_mode'])
            else:
                item['text'] = filter_captions(filterer, processed_frms, candidate_captions, config["threshold"], config['filter_mode'])
        else:
            item['text'] = candidate_captions

def main(args, config):
    # load data
    video_format = config["video_formats"]
    video_dir = config["video_roots"] # <video_id>.mp4
    video_2_text_original = json.load(open(config["train_ann_jsons"])) # dict: {<video_id>:[texts],...}
    data = []
    for key, text in video_2_text_original.items():
        video_path = os.path.join(video_dir,f'{key}.{video_format}')
        if os.path.exists(video_path):
            data.append({'video_path':video_path,'text':text,'video_id':key})
        else:
            print(f'skip: {key}')
    print('number of existing videos:', len(data))
    
    # mkdir tmp
    tmp_dir = os.path.join(args.output_dir,'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    # init multi-process
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # do CapFilt
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = len(data)//num_tasks + 1
    start = rank*step
    end = min(len(data),start+step)
    print(f'rank{rank}:device:',device)
    print(f'rank{rank}:start-{start}:end-{end}')
    print(f'rank{rank}:',data[start:start+3])
    CapFilt(data[start:end], config, device)
    print(f'rank{rank}:',data[start:start+3])
    
    # output to tmp dir per process
    video_text_CapFilt = {}
    video_text_Cap_unfiltered = {}
    for item in data[start:end]:
        if 'unfiltered_text' not in item:
            print(f'skip video that cannot be loaded: {video_path}')
            continue
        video_text_Cap_unfiltered[item['video_id']] = item['unfiltered_text']
        if item['text'] != []:
            video_text_CapFilt[item['video_id']] = item['text']
        else:
            print('filter out video:',item['video_id'])

    with open(os.path.join(tmp_dir,f'{rank}_filtered.json'), 'w') as out:
        print(f'rank{rank} filtered output to tmp...')
        json.dump(video_text_CapFilt, out, indent=4)

    with open(os.path.join(tmp_dir,f'{rank}_unfiltered.json'), 'w') as out:
        print(f'rank{rank} unfiltered output to tmp...')
        json.dump(video_text_Cap_unfiltered, out, indent=4)
    
    dist.barrier()

    # aggregate tmp dir
    if utils.is_main_process():
        video_text_CapFilt = {}
        for r in range(num_tasks):
            r_json = json.load(open(os.path.join(tmp_dir, f'{r}_filtered.json')))
            video_text_CapFilt.update(r_json)
        print('filtered video number:', len(video_text_CapFilt))
        
        video_text_Cap_unfiltered = {}
        for r in range(num_tasks):
            r_json = json.load(open(os.path.join(tmp_dir, f'{r}_unfiltered.json')))
            video_text_Cap_unfiltered.update(r_json)
        print('unfiltered video number:', len(video_text_Cap_unfiltered))
        
        with open(os.path.join(args.output_dir,'video_text_CapFilt.json'), 'w') as out:
            json.dump(video_text_CapFilt, out, indent=4)
        
        with open(os.path.join(args.output_dir,'video_text_Cap.json'), 'w') as out:
            json.dump(video_text_Cap_unfiltered, out, indent=4)
        # rm tmp dir
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='none')
    parser.add_argument('--output_dir', default='output/video_CapFilt')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w')) 

    main(args, config)
