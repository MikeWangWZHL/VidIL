import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption, wait_for_file

from collections import defaultdict
import numpy as np
import random
import torch

from glob import glob
import av
import decord
from decord import VideoReader


from torchvision import transforms

# from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
# from sklearn.cluster import KMeans

import copy

decord.bridge.set_bridge("torch")

class msrvtt_ret_train(Dataset):
    def __init__(self, transform, config, max_words=30, prompt = '', device='cpu'):        
        '''
        config:
            video_root (string): Root directory of videos (e.g. msrvtt_ret/videos/)
            train_ann_jsonl (string): train/val/test.jsonl 
                train: each line is a {"caption": "a cartoon animals runs through an ice cave in a video game", "clip_name": "video2960", "sen_id": 0}...
            num_frm (string): number of frames to be sampled
            frm_sampling_strategy (string): ["uniform","nlvl_uniform","nlvl_rand","rand","headtail"]
            height=None, 
            width=None, 
            start_time=None, 
            end_time=None, 
            fps=-1
        '''

        self.config = config
        
        if 'video_fmt' in config:
            self.video_fmt = config['video_fmt']
        else:
            self.video_fmt = '.mp4'

        # video dir
        if 'train_video_root' in config:
            video_roots = config['train_video_root']
        else:
            video_roots = config['video_root']
        if isinstance(config['train_ann_jsonl'],str):
            ann_jsonls = [config['train_ann_jsonl']]
            self.video_roots = [video_roots]
        else:
            ann_jsonls = config['train_ann_jsonl']
            self.video_roots = video_roots

        # load annotation
        self.annotation = []
        self.video_id_2_caption = [defaultdict(list) for i in range(len(ann_jsonls))]
        skip_count = 0
        for i in range(len(ann_jsonls)):
            with open(ann_jsonls[i], 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    video_id = obj['clip_name']
                    caption = obj['caption']
                    obj['video_root_idx'] = i
                    
                    video_path = os.path.join(self.video_roots[i],f'{video_id}{self.video_fmt}')

                    if os.path.exists(video_path):
                        self.video_id_2_caption[i][video_id].append(caption)
                        self.annotation.append(obj)
                    else:
                        skip_count += 1
        print("training video-text pair number:", len(self.annotation))
        print("skipped non-exist video number:", skip_count)
        self.transform = transform
        # add ToPILImage: to let the extracted frames from decord be able to use the training transform 
        self.transform.transforms.insert(0, transforms.ToPILImage())

        self.max_words = max_words      
        self.prompt = prompt
        
        self.video_id_2_index = [{} for i in range(len(ann_jsonls))]  
        n = 0
        for i in range(len(ann_jsonls)):
            for vid_id in self.video_id_2_caption[i].keys():
                self.video_id_2_index[i][vid_id] = n
                n += 1

        if self.config["frm_sampling_strategy"] == 'clip-kmeans':
            self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        clip_name = ann['clip_name']
        video_root_idx = ann['video_root_idx']
        video_path = os.path.join(self.video_roots[video_root_idx],f'{clip_name}.mp4')

        # try loading video
        for _ in range(3):
            raw_sample_frms = self._load_video_from_path_decord(video_path)
            if raw_sample_frms is not None:
                break
        # having trouble loading this video, load another random one instead
        if raw_sample_frms is None:
            idx = random.randint(0, len(self.annotation)-1)
            print(f'ERROR: cannot load video:{video_path}; load random instead')
            return self.__getitem__(idx)
        
        processed_frms = torch.stack([self.transform(frm) for frm in raw_sample_frms]) # [num_frm, c, h, w]
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        
        # selected_frame = self._select_frame(processed_frms, caption) 
        
        if 'timesformer' in self.config["vit"]:
            processed_frms = processed_frms.permute(1,0,2,3)

        return processed_frms, caption, self.video_id_2_index[video_root_idx][ann['clip_name']] 

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
    
    def _CLIP_selection(self, vr, num_frm, downsample_ratio = 2):
        vlen = len(vr)
        downsampled_indices = np.arange(vlen, step=downsample_ratio, dtype=int)
        # calculate vision embedding for each frame
        all_frames = vr.get_batch(downsampled_indices) # a tensor, since bridged
        all_frames = all_frames.numpy()
        all_frames_pil = [Image.fromarray(frm) for frm in all_frames]
        inputs = self.clip_processor(images=all_frames_pil, return_tensors="pt")
        outputs = self.clip_model(**inputs)
        pooled_output = outputs.pooler_output  # pooled CLS states
        frm_embeddings = pooled_output.detach().numpy()
        # cluster into num_frm clusters
        kmeans = KMeans(n_clusters=num_frm, random_state=0).fit(frm_embeddings)
        labels = kmeans.labels_ 
        frame_indices = []
        # random sample one frame from each cluster
        for i in range(num_frm):
            masked = np.where(labels == i)[0]
            idx = np.random.choice(masked)
            frame_indices.append(downsampled_indices[idx])
        frame_indices = sorted(frame_indices)
        return frame_indices
