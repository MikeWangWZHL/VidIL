'''
    code modified from https://github.com/salesforce/BLIP, https://github.com/salesforce/ALPRO
'''

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

from models.blip import blip_decoder, blip_decoder_video
from models.blip_itm import blip_itm
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval, video_caption_eval
import spacy

def select_frame(filterer, images, text):
    itm_output = filterer(images, [text for i in range(images.size()[0])], match_head='itm')
    # print(itm_output.size())
    itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1].detach().cpu().numpy()
    # pick max score frame:
    idx = np.argmax(itm_score)
    return images[idx]

def train(model, data_loader, optimizer, epoch, device, config, filterer):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image_b, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if config['video_representation'] == 'single_frame':
            assert filterer is not None
            image_b = image_b.to(device,non_blocking=True) # (B, num_frm, C, H, W)
            # select one frame using a pretrained BLIP filterer:
            picked_frms = []
            for j in range(image_b.size()[0]):
                picked_frms.append(select_frame(filterer,image_b[j],caption[j]))
            image = torch.stack(picked_frms).to(device,non_blocking=True) # (B, C, H, W)
        elif config['video_representation'] == 'concat_frame':
            image = image_b.to(device,non_blocking=True) # (B, num_frm, C, H, W)

        loss = model(image, caption)      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image_b, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        if config['video_representation'] == 'single_frame':
            # pick middle frame
            num_frm = image_b.size()[1]
            image = image_b[:,int(num_frm/2),:,:,:].to(device,non_blocking=True)
        elif config['video_representation'] == 'concat_frame':
            image = image_b.to(device,non_blocking=True) # (B, num_frm, C, H, W)
 
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"video_id": img_id, "caption": caption})

    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_msrvtt', config)  
    print('train dataset size:',len(train_dataset))
    print('val dataset size:',len(val_dataset))
    print('test dataset size:',len(test_dataset))
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['train_batch_size'],config['test_batch_size'],config['test_batch_size']],
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         
    #### set up spacy ####
    # nlp = spacy.load("en_core_web_sm", disable=['ner','tagger','lemmatizer'])
    
    #### set up filterer ####
    if config['video_representation'] == 'single_frame':
        filterer = blip_itm(pretrained=config["filterer_model_ckpt"], image_size=config["image_size"], vit=config["vit"])
        filterer.eval()
        filterer.to(device)
    else:
        filterer = None


    #### Model #### 
    print("Creating model")
    if config['video_representation'] == 'single_frame':
        print('represent video as single frame by selecting the best matched frame to caption')
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            prompt=config['prompt'])
    elif config['video_representation'] == 'concat_frame':
        print('represent video as concat frames')
        model = blip_decoder_video(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                            vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                            prompt=config['prompt'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            # train
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])    
            train_stats = train(model, train_loader, optimizer, epoch, device, config, filterer)
        # evaluate:
        if epoch == config['max_epoch'] -1 or args.evaluate:
            if ('skip_val' not in config) or (not config['skip_val']):
                val_result = evaluate(model_without_ddp, val_loader, device, config)  
                val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='video_id')        
    
            test_result = evaluate(model_without_ddp, test_loader, device, config)  
            test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='video_id')  

            if utils.is_main_process():
                if ('skip_val' not in config) or (not config['skip_val']):            
                    video_val = video_caption_eval(config['val_ann_jsonl'], val_result_file)
                
                video_test = video_caption_eval(config['test_ann_jsonl'], test_result_file)

                if args.evaluate:            
                    if ('skip_val' not in config) or (not config['skip_val']):
                        log_stats = {**{f'val_{k}': v for k, v in video_val.items()},
                                    **{f'test_{k}': v for k, v in video_test.items()},                       
                                    }
                    else:
                        log_stats = {**{f'test_{k}': v for k, v in video_test.items()}}

                    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                   
                else:             
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }

                    # using Bleu 4
                    if ('skip_val' not in config) or (not config['skip_val']):
                        if video_val['CIDEr'] + video_val['Bleu'][3] > best:
                            best = video_val['CIDEr'] + video_val['Bleu'][3]
                            best_epoch = epoch                
                            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                            
                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                    **{f'val_{k}': v for k, v in video_val.items()},
                                    **{f'test_{k}': v for k, v in video_test.items()},                       
                                    'epoch': epoch,
                                    'best_epoch': best_epoch,
                                    }
                    else:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                    **{f'test_{k}': v for k, v in video_test.items()},                       
                                    'epoch': epoch
                                    }

                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    print(config)
    main(args, config)