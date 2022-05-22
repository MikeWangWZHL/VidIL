from glob import glob
import json
import os
from collections import defaultdict
import numpy as np
import random
# key: (int) list length
# value: (lambda function) x is a list of string
TEMPORAL_TEMPLATES_NATURAL = {
    1:lambda x: f"First, {x[0]}.",
    2:lambda x: f"First, {x[0]}. Then, {x[1]}.",
    3:lambda x: f"First, {x[0]}. Then, {x[1]}. Finally, {x[2]}.",
    4:lambda x: f"First, {x[0]}. Then, {x[1]}. After that, {x[2]}. Finally, {x[3]}.",
    5:lambda x: f"First, {x[0]}. Then, {x[1]}. Then, {x[2]}. Then, {x[3]}. Finally, {x[4]}.",
    6:lambda x: f"First, {x[0]}. Then, {x[1]}. Then, {x[2]}. Then, {x[3]}. Then, {x[4]}. Finally, {x[5]}.",
    7:lambda x: f"First, {x[0]}. Then, {x[1]}. Then, {x[2]}. Then, {x[3]}. Then, {x[4]}. Then, {x[5]}. Finally, {x[6]}.",
    8:lambda x: f"First, {x[0]}. Then, {x[1]}. Then, {x[2]}. Then, {x[3]}. Then, {x[4]}. Then, {x[5]}. Then, {x[6]}. Finally, {x[7]}."
}
TEMPORAL_TEMPLATES_INDEX = {
    1:lambda x: f"[1] {x[0]}.",
    2:lambda x: f"[1] {x[0]}. [2] {x[1]}.",
    3:lambda x: f"[1] {x[0]}. [2] {x[1]}. [3] {x[2]}.",
    4:lambda x: f"[1] {x[0]}. [2] {x[1]}. [3] {x[2]}. [4] {x[3]}.",
    5:lambda x: f"[1] {x[0]}. [2] {x[1]}. [3] {x[2]}. [4] {x[3]}. [5] {x[4]}.",
    6:lambda x: f"[1] {x[0]}. [2] {x[1]}. [3] {x[2]}. [4] {x[3]}. [5] {x[4]}. [6] {x[5]}.",
    7:lambda x: f"[1] {x[0]}. [2] {x[1]}. [3] {x[2]}. [4] {x[3]}. [5] {x[4]}. [6] {x[5]}. [7] {x[6]}.",
    8:lambda x: f"[1] {x[0]}. [2] {x[1]}. [3] {x[2]}. [4] {x[3]}. [5] {x[4]}. [6] {x[5]}. [7] {x[6]}. [8] {x[7]}."
}
STATIC_TEMPLATES = {
    1:lambda x: f"{x[0]}.",
    2:lambda x: f"{x[0]}. {x[1]}.",
    3:lambda x: f"{x[0]}. {x[1]}. {x[2]}.",
    4:lambda x: f"{x[0]}. {x[1]}. {x[2]}. {x[3]}.",
    5:lambda x: f"{x[0]}. {x[1]}. {x[2]}. {x[3]}. {x[4]}.",
    6:lambda x: f"{x[0]}. {x[1]}. {x[2]}. {x[3]}. {x[4]}. {x[5]}.",
    7:lambda x: f"{x[0]}. {x[1]}. {x[2]}. {x[3]}. {x[4]}. {x[5]}. {x[6]}.",
    8:lambda x: f"{x[0]}. {x[1]}. {x[2]}. {x[3]}. {x[4]}. {x[5]}. {x[6]}. {x[7]}."
}


class Prompt():
    def __init__(self, template_txt, seed=42):
        random.seed(seed)
        if os.path.exists(template_txt):
            template = open(template_txt,'r').read()
        else:
            if isinstance(template_txt, str):
                template = template_txt
        self.template = template
        # print(template)
  
    def _get_top_visual_tokens_v2(self, video_name, visual_tokens_object, topk):
        frame_tokens = visual_tokens_object['frame_tokens']
        
        count_dict = defaultdict(lambda:defaultdict(int)) # for weighting for selection
        index_sum_dict = defaultdict(lambda:defaultdict(int)) # for final sorting on temporal order

        frm_candidate_k = 2
        for i in range(len(frame_tokens)):
            frame_token = frame_tokens[i]
            for key in ['objects','attributes','scenes', 'verbs']:
                if key in frame_token:
                    for s in frame_token[key][:frm_candidate_k]:
                        count_dict[key][s] += 1
                        index_sum_dict[key][s] += i
        topk_tokens = {}
        for key in ['objects','attributes','scenes', 'verbs']:
            candidate_list = sorted([(text, count, index_sum_dict[key][text]/count) for text, count in count_dict[key].items()], key = lambda x: x[1], reverse = True)
            candidate_list = candidate_list[:topk]
            candidate_list = sorted(candidate_list, key = lambda x: x[2]) # temporal sorting
            topk_tokens[key] = [item[0].rstrip('.') for item in candidate_list]
        
        return topk_tokens
    
    def _get_top_visual_tokens_v3(self, video_name, visual_tokens_object, topk):
        frame_tokens = visual_tokens_object['frame_tokens']
        indices = np.linspace(0, len(frame_tokens), num=topk, dtype=int, endpoint=False)
        topk_tokens = defaultdict(list)
        # divide into blocks
        blocks = []
        for i in range(len(indices)):
            if i == len(indices) - 1:
                blocks.append((indices[i],len(frame_tokens)))
            else:
                blocks.append((indices[i],indices[i+1]))
        # get visual token for each block
        candidate_tokens = defaultdict(list)
        for key in frame_tokens[0].keys():
            # print(f'-- {key} --')
            for b in blocks:
                start_i, end_i = b
                frm_candidate_k = 2
                count_dict = defaultdict(int)
                rank_dict = defaultdict(int)
                for i in range(start_i, end_i):
                    for r in range(frm_candidate_k):
                        text = frame_tokens[i][key][r]
                        count_dict[text]+=1
                        rank_dict[text]+=r
                cand_list = [(key, -count_dict[key], rank_dict[key]) for key in count_dict.keys()]
                cand_list = sorted(cand_list, key=lambda x: (x[1],x[2]))
                # print(cand_list)
                chosen_text = ', '.join([item[0].rstrip('.').strip() for item in cand_list[:frm_candidate_k]])
                # chosen_text = cand_list[0][0].rstrip('.').strip()
                candidate_tokens[key].append(chosen_text)
        # remove duplicate neighbors
        topk_tokens = {}
        for key in ['objects','attributes','scenes', 'verbs']:
            cand_tokens = candidate_tokens[key]
            select_ids = []
            for i in range(len(cand_tokens)-1):
                if i == 0:
                    select_ids.append(i)
                else:
                    if cand_tokens[i] != cand_tokens[select_ids[-1]]:
                        select_ids.append(i)
            topk_tokens[key] = [cand_tokens[ii] for ii in select_ids]
        return topk_tokens

    def construct_prompt(self, video_name, visual_tokens_object, frame_captions, config, question = None, answer = None, asr = None, vlep_example = None):
        ## temporal prompt with natural language template ##
        topk = config['topk']

        if config['visual_token_aggregation_version'] == 'v2':
            topk_tokens = self._get_top_visual_tokens_v2(video_name, visual_tokens_object, topk)
        elif config['visual_token_aggregation_version'] == 'v3':
            topk_tokens = self._get_top_visual_tokens_v3(video_name, visual_tokens_object, topk)

        prompt_temporal_template = config['prompt_temporal_template']

        if prompt_temporal_template == 'temporal_natural':
            # print('using temporal natural language prompt template')
            TEMPLATE_FUNC = TEMPORAL_TEMPLATES_NATURAL
        elif prompt_temporal_template == 'temporal_index':
            # print('using temporal index prompt template')
            TEMPLATE_FUNC = TEMPORAL_TEMPLATES_INDEX
        elif prompt_temporal_template == 'static':
            # print('using static prompt template')
            TEMPLATE_FUNC = STATIC_TEMPLATES
        else:
            raise NotImplementedError

        
        if not config['add_objects']:
            objects_str = None
        else:
            objects_str = TEMPLATE_FUNC[len(topk_tokens['objects'])](topk_tokens['objects'])

        if not config['add_events']:
            events_str = None
        else:
            events_str = TEMPLATE_FUNC[len(topk_tokens['verbs'])](topk_tokens['verbs'])
        
        if not config['add_attributes']:
            attributes_str = None
        else:
            attributes_str = TEMPLATE_FUNC[len(topk_tokens['attributes'])](topk_tokens['attributes'])
        
        if not config['add_scenes']:
            scenes_str = None
        else:
            scenes_str = topk_tokens['scenes'][0]
        
        if not config['add_ASR']:
            ASR_str = None
        else:
            ASR_str = asr
        

        original_caption = visual_tokens_object['caption']

        # frame captions str
        if len(frame_captions[video_name]) > topk:
            caption_list = [cap.rstrip(".").strip() for cap in frame_captions[video_name][:topk]]
            frame_captions_str =TEMPLATE_FUNC[topk](caption_list)
        else:
            caption_list = [cap.rstrip(".").strip() for cap in frame_captions[video_name]]
            frame_captions_str =TEMPLATE_FUNC[len(frame_captions[video_name])](caption_list)

        if vlep_example is not None:
            return self._construct_prompt_base_vlep_multichoice(
                config, 
                objects_str, 
                events_str, 
                attributes_str, 
                scenes_str, 
                frame_captions_str, 
                ASR_str, 
                original_caption,
                question,
                answer,
                vlep_example
                )
        else:
            return self._construct_prompt_base(
                config, 
                objects_str, 
                events_str, 
                attributes_str, 
                scenes_str, 
                frame_captions_str, 
                ASR_str, 
                original_caption,
                question,
                answer
                )


    def _construct_prompt_base(self, 
        config, 
        objects_str, 
        events_str, 
        attributes_str, 
        scenes_str, 
        frame_captions_str, 
        ASR_str, 
        original_caption, 
        question_str, 
        answer_str):
        
        if not config['add_original_caption']:
            original_caption = None
        if not config['add_frame_captions']:
            frame_captions_str = None
        if not config['add_answer']:
            answer_str = None

        # write prompt
        p = self.template
        if scenes_str:
            p += "Scene: " + scenes_str + '\n'
        if objects_str:
            p += "Objects: " + objects_str + '\n'
        if events_str:
            p += "Events: " + events_str + '\n'
        if attributes_str:
            p += "Attributes: " + attributes_str + '\n'
        if frame_captions_str:
            p += "Frame Captions: " + frame_captions_str + '\n'
        if ASR_str:
            if config['prompt_task'] == 'vlep':
                p += "Dialogue: " + ASR_str + '\n'
            else:
                p += "Subtitle: " + ASR_str + '\n'
                
        
        if config['prompt_task'] == 'caption':
            p += "Video Caption:"
            if original_caption:
                if isinstance(original_caption,str):
                    p += " " + original_caption.strip()
                elif isinstance(original_caption,list):
                    random.shuffle(original_caption)
                    p += " " + original_caption[0].strip()

        elif config['prompt_task'] == 'qa':
            assert question_str is not None
            p += "Question: " + question_str + '\n'
            p += "Answer:"
            if answer_str:
                p += " " + answer_str

        elif config['prompt_task'] == 'vlep':
            p += "What is likely to happen next?"
            if original_caption:
                if isinstance(original_caption,str):
                    p += " " + original_caption.strip()
                elif isinstance(original_caption,list):
                    random.shuffle(original_caption)
                    p += " " + original_caption[0].strip()
        return p

    def _construct_prompt_base_vlep_multichoice(self, 
        config, 
        objects_str, 
        events_str, 
        attributes_str, 
        scenes_str, 
        frame_captions_str, 
        ASR_str, 
        original_caption, 
        question_str, 
        answer_str,
        vlep_example):
        
        if not config['add_frame_captions']:
            frame_captions_str = None

        # write prompt
        p = self.template
        if scenes_str:
            p += "Scene: " + scenes_str + '\n'
        if objects_str:
            p += "Objects: " + objects_str + '\n'
        if events_str:
            p += "Events: " + events_str + '\n'
        if attributes_str:
            p += "Attributes: " + attributes_str + '\n'
        if frame_captions_str:
            p += "Frame Captions: " + frame_captions_str + '\n'
        if ASR_str:

            p += "Dialogue: " + ASR_str + '\n'
        
        event_A, event_B = vlep_example['events']

        p += f"Question: What is more likely to happen next? A:{event_A} B:{event_B}\nAnswer:"
        if config['add_original_caption']:
            p += " " + vlep_example['answer'].strip()

        return p


