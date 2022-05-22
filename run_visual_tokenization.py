'''
    code modified from https://github.com/salesforce/BLIP, https://github.com/salesforce/ALPRO
'''

from PIL import Image
import requests
import json
import os
from transformers import CLIPProcessor, CLIPModel
from glob import glob
from sklearn.cluster import KMeans
from scipy.special import softmax

from torchvision import transforms

from tqdm import tqdm
from data.video_pretrain_dataset import visual_tokenization_dataset
from models.blip_retrieval import blip_retrieval
from torchvision.transforms.functional import InterpolationMode

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path

import utils
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from collections import defaultdict
import shutil

### helper functions ###
def load_json(json_path):
    return json.load(open(json_path))

def save_json(filepath, json_object):
    with open(filepath, 'w') as f:
        json.dump(json_object, f, indent=4)

def save_frames(dir_path, video_name, frames):
    video_dir = os.path.join(dir_path, video_name)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    for frm_idx in range(len(frames)):
        frm = frames[frm_idx]
        frm.save(os.path.join(video_dir,f'frm_{frm_idx}.jpg'))

def get_prefix_prompt_functions(version):
    if version == 'v0':
        object_prompt = lambda x: x
        attribute_prompt = lambda x: x
        scene_prompt = lambda x: x
        verb_prompt = lambda x: x
    elif version == 'v1':
        # object_prompt = lambda x: f'An image of {x}'
        # attribute_prompt = lambda x: f'An image of {x}'
        # scene_prompt = lambda x: f'An image of {x}'
        # verb_prompt = lambda x: f'An image of {x}'
        # object_prompt = lambda x: f'A picture of {x}'
        # attribute_prompt = lambda x: f'A picture of {x}'
        # scene_prompt = lambda x: f'A picture of {x}'
        # verb_prompt = lambda x: f'A picture of {x}'
        object_prompt = lambda x: f'A photo of {x}'
        attribute_prompt = lambda x: f'A photo of {x}'
        scene_prompt = lambda x: f'A photo of {x}'
        verb_prompt = lambda x: f'A photo of {x}'
    return {
        'objects':object_prompt,
        'attributes':attribute_prompt,
        'scenes':scene_prompt,
        'verbs':verb_prompt
    }

### embedding functions ### 
@torch.no_grad()
def get_text_embeddings_clip(model, processor, texts, device):
    num_text = len(texts)
    text_bs = EMBBDING_BATCH_LIMIT_TEXT
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        inputs = processor(text=text, images=DUMMY_IMAGE, return_tensors="pt", padding=True, truncation = True).to(device)
        outputs = model(**inputs)
        txt_emb = outputs.text_embeds
        text_embeds.append(txt_emb)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    return text_embeds, None, None

@torch.no_grad()
def get_text_embeddings_florence(model, processor, texts, device):
    # texts: a list of str
    num_text = len(texts)
    text_bs = EMBBDING_BATCH_LIMIT_TEXT
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        #TODO_florence
        txt_emb = None
        text_embeds.append(txt_emb)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    return text_embeds, None, None

@torch.no_grad()
def get_text_embeddings_blip(model, texts, device):    
    num_text = len(texts)
    text_bs = EMBBDING_BATCH_LIMIT_TEXT
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    return text_embeds, text_ids, text_atts

@torch.no_grad()
def get_image_embeddings_clip(model, processor, images, device):
    # image: a list of PIL images
    inputs = processor(
        text=['hello world'], images=images, return_tensors="pt", padding=True, truncation = True
    ).to(device)
    outputs = model(**inputs)
    im_emb = outputs.image_embeds
    return None, im_emb # (num_frm, 512)

@torch.no_grad()
def get_image_embeddings_florence(model, processor, images, device):
    # image: a list of PIL images
    #TODO_florence
    im_emb = None
    return None, im_emb # (num_frm, 512)

@torch.no_grad()
def get_image_embeddings_blip(model, image, device):
    # image: a batch with size: (num_frm, c, h, w)
    image = image.to(device) 
    image_feat = model.visual_encoder(image)   
    image_embed = model.vision_proj(image_feat[:,0,:])            
    image_embed = F.normalize(image_embed,dim=-1)      
    return image_feat, image_embed

@torch.no_grad()
def predict_video(
        config, 
        video_dataset, 
        model,
        device, 
        visual_token_texts,
        prompt_functions,
        encoder_version='clip',
        processor=None
    ):
    # aggregate based on frequency
    def aggregate_frame_tokens(frame_tokens):
        keys = frame_tokens[0].keys()
        aggregated_tokens = {key:[] for key in keys}
        topk = len(frame_tokens[0]['objects'])
        num_frm = len(frame_tokens)
        for key in keys:
            if frame_tokens[0][key] == []:
                continue
            count_dict = defaultdict(int)
            for j in range(topk):
                for i in range(num_frm):
                    count_dict[frame_tokens[i][key][j]] += 1
            candidates = sorted([(t,c) for t,c in count_dict.items()], key = lambda x:x[1], reverse=True)
            aggregated_tokens[key] = [item[0] for item in candidates[:topk]]
        return aggregated_tokens

    # test
    model.eval()

    print('transform:', video_dataset.transform)

    ### text representations ###
    text_representations = {}
    num_frm = config['num_frm_visual_tokenization']
    print('num of frm:',num_frm)
    for key in visual_token_texts.keys():
        print(f'compute {key} text features...')
        texts = [prompt_functions[key](t) for t in visual_token_texts[key]]
        if args.encoder_version == 'blip':
            text_embeds, text_ids, text_atts = get_text_embeddings_blip(model, texts, device)
        elif args.encoder_version == 'clip':
            # text_ids, text_atts should be None
            text_embeds, text_ids, text_atts = get_text_embeddings_clip(model, processor, texts, device)
        elif args.encoder_version == 'florence':
            # text_ids, text_atts should be None
            text_embeds, text_ids, text_atts = get_text_embeddings_florence(model, processor, texts, device)
         
        text_representations[key] = {
            'text_embeds':text_embeds,
            'text_ids':text_ids,
            'text_atts':text_atts
        }
    
    ### frame representations ###
    print(f'compute image features...') 
    image_feats = []
    image_embeds = []
    video_ids = []
    captions = []
    
    rank = utils.get_rank()

    # for i in range(len(video_dataset)):
    for i in tqdm(range(len(video_dataset))):
        if i == config["early_stop_step"]:
            print(f'early stop at {i}')
            break
        ann = video_dataset.annotation[i]
        video_name = os.path.basename(ann['video'])[:-4]
        # if rank == 1:
        # print(f'rank{rank} | working on:',video_name)
        
        frames, caption = video_dataset[i] # [num_frm, c, h, w], text
        if frames is None:
            print('skip video that cannot be loaded:',video_name)
            continue
        if config['save_frames']:
            save_frames(config['save_frame_dir'],video_name,frames)
        
        if args.encoder_version == 'blip':
            image_feat, image_embed = get_image_embeddings_blip(model, frames, device) 
        elif args.encoder_version == 'clip':
            # image_feat should be None
            image_feat, image_embed = get_image_embeddings_clip(model, processor, frames, device) 
        elif args.encoder_version == 'florence':
            # image_feat should be None
            image_feat, image_embed = get_image_embeddings_florence(model, processor, frames, device) 
        
        if image_feat is not None:
            image_feats.append(image_feat.cpu())
        
        image_embeds.append(image_embed)

        video_ids.append(video_name)
        captions.append(caption)
        # if rank == 1:
        # print(f'rank{rank} | done:',video_name)
    
    if image_feats != []:
        image_feats = torch.cat(image_feats,dim=0)
        print(image_feats.size())
    
    image_embeds = torch.cat(image_embeds,dim=0)
    print(image_embeds.size())

    videoid_2_visual_tokens = { video_ids[i]:{"frame_tokens":[defaultdict(list) for frm in range(num_frm)],"caption":captions[i]} for i in range(len(video_ids)) }
    topk_visualize = config['topk_visualize']
    for key in visual_token_texts.keys():
        print(f'predicting {key}...')
        text_embeds = text_representations[key]['text_embeds']
        text_ids = text_representations[key]['text_ids']
        text_atts = text_representations[key]['text_atts']

        sims_matrix = image_embeds @ text_embeds.t()
        if args.encoder_version == 'blip':
            # BLIP further uses ITM 
            score_matrix_i2t = torch.full((image_embeds.size()[0],text_embeds.size()[0]),-100.0).to(device)

            for i,sims in enumerate(sims_matrix):
                topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
                encoder_output = image_feats[i].repeat(config['k_test'],1,1).to(device)
                encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
                output = model.text_encoder(text_ids[topk_idx], 
                                            attention_mask = text_atts[topk_idx],
                                            encoder_hidden_states = encoder_output,
                                            encoder_attention_mask = encoder_att,                             
                                            return_dict = True,
                                        )
                score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]

                score_matrix_i2t[i,topk_idx] = score + topk_sim
        else:
            # CLIP and Florence directly uses contrastive matrix
            score_matrix_i2t = sims_matrix

        score_matrix_i2t = score_matrix_i2t.view(len(video_ids),num_frm,-1)
        score_matrix_i2t = score_matrix_i2t.cpu().numpy()
        # get frame tokens
        for j in range(len(video_ids)):
            vid = video_ids[j]
            for frm_idx in range(len(score_matrix_i2t[j])):
                frm_score = score_matrix_i2t[j][frm_idx]
                # print(frm_score.shape)
                inds = np.argsort(frm_score)[::-1][:topk_visualize]
                txts = [visual_token_texts[key][ii] for ii in inds]
                videoid_2_visual_tokens[vid]['frame_tokens'][frm_idx][key] = txts
                # print(vid, videoid_2_visual_tokens[vid])

    for vid,obj in videoid_2_visual_tokens.items():
        obj["aggregated_tokens"] = aggregate_frame_tokens(obj['frame_tokens'])

    return videoid_2_visual_tokens


### main ###
def main(args, config):
    ### set up tmp dir ###
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


    #### Model #### 
    print("Creating model")
    if args.encoder_version == 'blip':
        model = blip_retrieval(pretrained=config['blip_model_visual_tokenization'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
        model = model.to(device)
        processor = None
    elif args.encoder_version == 'clip':
        model_name = config['clip_model_visual_tokenization']
        print(f'loading {model_name}...')
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    elif args.encoder_version == 'florence':
        #TODO_florence
        model = None
        model.eval()
        model.to(device)
        processor = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module  

    # set up prompts
    prompt_version = config['prompt_version_visual_tokenization']
    prompt_functions = get_prefix_prompt_functions(prompt_version)
    print('prompts:', prompt_functions)

    ''' load ontology '''
    if config['ontology'] == 'vg':
        print('using openimage objects and vg srl selected events')
        # object_json_path = f'shared_datasets/OpenImages/openimage_classes_600.json'
        object_json_path = f'visual_token_ontology/vg/openimage_classes_all_cleaned_fictional_characters.json'
        attribute_json_path = f'visual_token_ontology/vg/vg_original_attributes_synsets_keys_cleaned_remove_similar0.9.json'
        scene_json_path = f'visual_token_ontology/vg/place365_ontology.json'
        verb_json_path = f'visual_token_ontology/vg/vg_srl_selected_object_synsets_keys_remove_similar0.9.json'
    elif config['ontology'] == 'vg_tencent':
        print('using tencent-ml-image objects and vg srl selected events')
        object_json_path = f'visual_token_ontology/vg_tencent/tencent_ml_images_objects.json'
        attribute_json_path = f'visual_token_ontology/vg_tencent/vg_original_attributes_synsets_keys_cleaned_remove_similar0.9.json'
        scene_json_path = f'visual_token_ontology/vg/place365_ontology.json'
        verb_json_path = f'visual_token_ontology/vg_tencent/vg_srl_selected_object_synsets_keys_remove_similar0.9.json'

    object_texts = load_json(object_json_path)
    attribute_texts = load_json(attribute_json_path)
    scene_texts = load_json(scene_json_path)
    verb_texts = load_json(verb_json_path)
    if isinstance(verb_texts,dict):
        verb_texts = list(verb_texts.keys())
    for key in attribute_texts: 
        if key in object_texts:
            attribute_texts.remove(key)
    for key in OMIT_KEYWORDS:
        if key in object_texts: object_texts.remove(key)
        if key in attribute_texts: attribute_texts.remove(key)
        if key in scene_texts: scene_texts.remove(key)
        if key in verb_texts: verb_texts.remove(key)
    print('num of objects:', len(object_texts))
    print('num of attributes:', len(attribute_texts))
    # print('num of scenes:', len(scene_texts))
    print('num of verbs:', len(verb_texts))
    visual_token_texts = {
        'objects':object_texts,
        'attributes':attribute_texts,
        'scenes':scene_texts,
        'verbs':verb_texts
    }
    print('objects examples:', visual_token_texts['objects'][:5])
    # print('attributes examples:',visual_token_texts['attributes'][:5])

    ''' load query image '''
    if args.encoder_version == 'blip':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))       
        transform_ = transforms.Compose([
            transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])
    elif args.encoder_version == 'clip':
        transform_ = transforms.Compose([])
    elif args.encoder_version == 'florence':
        transform_ = transforms.Compose([]) # can be replaced with desirable transform
    
    video_dataset = visual_tokenization_dataset(transform_, config, max_words=64)
    
    
    # divide work amoung ranks
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = len(video_dataset)//num_tasks + 1
    start = rank*step
    end = min(len(video_dataset),start+step)
    print(f'rank{rank}:device:',device)
    print(f'rank{rank}:start-{start}:end-{end}')
    video_dataset.annotation = video_dataset.annotation[start:end]

    videoid_2_visual_tokens = predict_video(
        config, 
        video_dataset,
        model_without_ddp,
        device, 
        visual_token_texts,
        prompt_functions,
        processor = processor,
        encoder_version = args.encoder_version
    )

    with open(os.path.join(tmp_dir,f'{rank}.json'), 'w') as out:
        print(f'rank{rank} output to tmp...')
        json.dump(videoid_2_visual_tokens, out, indent=4)

    dist.barrier()

    if utils.is_main_process():
        videoid_2_visual_tokens = {}
        for r in range(num_tasks):
            r_json = json.load(open(os.path.join(tmp_dir, f'{r}.json')))
            videoid_2_visual_tokens.update(r_json)
        
        with open(os.path.join(args.output_dir,'visual_tokens.json'), 'w') as out:
            json.dump(videoid_2_visual_tokens, out, indent=4)
        
        # rm tmp dir
        shutil.rmtree(tmp_dir)

        

if __name__ == '__main__':
    ###
    DUMMY_IMAGE = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    EMBBDING_BATCH_LIMIT_TEXT = 512
    OMIT_KEYWORDS = ['media player','video','playing video','audio','sound','taking video', 'water mark', 'water marked', 'watermark', 'watermarks', 'for sale in', 'sold from', 'stock', 'sold on','by viewers',
        'are provided by','are posted on','for more','tag with','stream from','viewed from','showing video of','are on at', 'shuttlecock', 'shutter', 'shutter is white', 'shutters have bones','tape is looped', 'bliss wants you','thumbnail','technique']
    ### 
    
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='visual_token_generation/config/visual_tokenization_main.yaml')
    parser.add_argument('--output_dir', default='visual_token_generation/output/tmp')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--encoder_version', default='clip')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # for matching the argument name for constructing the video dataset
    config['num_frm_train'] = config['num_frm_visual_tokenization']

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w')) 

    main(args, config)