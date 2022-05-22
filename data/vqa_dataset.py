import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

from glob import glob
import av
import decord
from decord import VideoReader
import copy
from torchvision import transforms

decord.bridge.set_bridge("torch")

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        
        if split=='train':
            urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
            self.annotation = []
            for f in train_files:
                download_url(urls[f],ann_root)
                self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        else:
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json',ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,'vqa_test.json'),'r'))    
            
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',ann_root)
            self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))    
                
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            question = pre_question(ann['question'])        
            
            if ann['dataset']=='vqa':               
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.2]  

            return image, question, answers, weights

class msrvtt_qa_dataset(Dataset):
    def __init__(self, transform, config, split):
        self.split = split
        self.config = config
        
        if 'video_fmt' not in config:
            self.video_fmt = '.mp4'
        else:
            self.video_fmt = config['video_fmt']

        self.transform = transform
        self.transform.transforms.insert(0, transforms.ToPILImage())

        if split == 'train':
            if ('use_val' not in config) or config['use_val']:
                print('adding valset as training samples')
                ann_jsonls = [self.config['train_ann_jsonl'], self.config['val_ann_jsonl']]
            else:
                print('not adding valset as training samples')
                ann_jsonls = [self.config['train_ann_jsonl']]

        elif split == 'test':
            ann_jsonls = [self.config['test_ann_jsonl']]
            self.answer_list = json.load(open(config['test_answer_list']))
        
        self.annotation = []
        skip_count = 0
        for ann_jsonl in ann_jsonls:
            with open(ann_jsonl, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    video_path = os.path.join(self.config['video_root'],obj['video_id']+self.video_fmt)  
                    if not os.path.exists(video_path):
                        skip_count += 1
                        continue
                    obj['question_id'] = len(self.annotation)
                    self.annotation.append(obj)

        print('skip non-exist number:',skip_count)
        print('qa pairs number:',len(self.annotation))
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]

        video_id = ann['video_id']
        video_path = os.path.join(self.config['video_root'],f'{video_id}{self.video_fmt}')     
        
        raw_sample_frms = self._load_video_from_path_decord(video_path)
        processed_frms = torch.stack([self.transform(frm) for frm in raw_sample_frms]) # [num_frm, c, h, w]
        
        if 'timesformer' in self.config["vit"]:
            processed_frms = processed_frms.permute(1,0,2,3)
        
        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']            
            return processed_frms, question, question_id

        elif self.split=='train':                       
            question = pre_question(ann['question'])
            answers = [ann['answer']]
            weights = [0.2]
            return processed_frms, question, answers, weights
    
    def _load_video_from_path_decord(self, video_path):
        frm_sampling_strategy=self.config['frm_sampling_strategy']
        num_frm=self.config['num_frm_train']
        height=self.config['height']
        width=self.config['width']
        start_time=self.config['start_time']
        end_time=self.config['end_time']
        fps=self.config['fps']
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
            elif frm_sampling_strategy == 'clip-kmeans':
                frame_indices = self._CLIP_selection(vr, num_frm)
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices).detach().cpu().numpy() # (num_frm, H, W, C)
            # raw_sample_frms = vr.get_batch(frame_indices).asnumpy() # (num_frm, H, W, C)

        except Exception as e:
            print(e)
            return None
        # raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # torch tensor
        # raw_sample_frms = np.transpose(raw_sample_frms, (0, 3, 1, 2)) # numpy
        return raw_sample_frms
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        