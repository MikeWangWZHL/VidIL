'''
    code modified from https://github.com/salesforce/BLIP, https://github.com/salesforce/ALPRO
'''
import argparse
import os
import ruamel_yaml as yaml
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

from models.blip_retrieval import blip_retrieval, blip_retrieval_video
from models.blip import blip_decoder
from models.blip_itm import blip_itm
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

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
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image_b, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
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

        idx = idx.to(device,non_blocking=True)
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    start_time = time.time()  

    texts = data_loader.dataset.text   
    print('Computing text features for evaluation...')
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = tokenizer.additional_special_tokens_ids[0]
    
    print('Computing video features for evaluation...')
    
    video_feats = []
    video_embeds = []
    for video, video_id in data_loader: 

        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H)
        video = video.to(device,non_blocking=True) 
        video_feat = model.visual_encoder(video)        
        video_embed = model.vision_proj(video_feat[:,0,:])   
        video_embed = video_embed.view(B,N,-1).mean(dim=1)
        video_embed = F.normalize(video_embed,dim=-1)  
       
        video_feat = video_feat.view(B,-1,video_feat.shape[-1])
        video_feats.append(video_feat.cpu())
        video_embeds.append(video_embed)
     
    video_feats = torch.cat(video_feats,dim=0)
    video_embeds = torch.cat(video_embeds,dim=0)
    
    sims_matrix = video_embeds @ text_embeds.t()
    score_matrix_v2t = torch.full((len(texts),len(texts)),-100.0).to(device) 
    
    print('Done computing embedding')

    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        
        encoder_output = video_feats[start+i].repeat(config['k_test'],1,1).to(device,non_blocking=True) 
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True) 
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_v2t[start+i,topk_idx] = score + topk_sim

        
    sims_matrix = sims_matrix.t()
    score_matrix_t2v = torch.full((len(texts),len(texts)),-100.0).to(device) 
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = video_feats[topk_idx].to(device,non_blocking=True) 
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True) 
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2v[start+i,topk_idx] = score + topk_sim

        
    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_v2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2v, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_v2t.cpu().numpy(), score_matrix_t2v.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_v2t, scores_t2v, txt2vmg, vid2txt):
    
    #Video->Text 
    ranks = np.zeros(scores_v2t.shape[0])
    for index,score in enumerate(scores_v2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == vid2txt[index])[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Video 
    ranks = np.zeros(scores_t2v.shape[0])
    
    for index,score in enumerate(scores_t2v):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2vmg[index])[0][0]
    
    mdR = np.median(ranks+1)
        
    # Compute metrics
    vr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    vr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    vr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    vr_mean = (vr1 + vr5 + vr10) / 3
    r_mean = (tr_mean + vr_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'vid_r1': vr1,
                    'vid_r5': vr5,
                    'vid_r10': vr10,
                    'vid_r_mean': vr_mean,
                    'vid_mdR': mdR,
                    'r_mean': r_mean}
    return eval_result


def main(args, config):
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
    if config['video_representation'] == 'single_frame':
        print('represent video as single frame by selecting the best matched frame to caption')
        model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
    elif config['video_representation'] == 'concat_frame':
        print('represent video as concat frames')
        model = blip_retrieval_video(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                 vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                 queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config['dataset'], config)
    print('train dataset size:',len(train_dataset))
    print('val dataset size:',len(val_dataset))
    print('test dataset size:',len(test_dataset))
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                        batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                        num_workers=[4,4,4],
                                                        is_trains=[True, False, False], 
                                                        collate_fns=[None,None,None])

    #### main loop ####
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            if config['video_representation'] == 'single_frame':
                # set up filter model
                filterer = blip_itm(pretrained=config["filterer_model_ckpt"], image_size=config["image_size"], vit=config["vit"])
                filterer.eval()
                filterer.to(device)
            else:
                filterer = None
            # train
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, config, filterer)
            
            # release before evaluation
            if not filterer:
                del filterer
        
        if "eval_only_once" in config and epoch != config['max_epoch']-1:
            if config['eval_only_once']:
                print('skip eval...')
                continue

        if ('skip_val' not in config) or (not config['skip_val']):
            score_val_v2t, score_val_t2v = evaluation(model_without_ddp, val_loader, model_without_ddp.tokenizer, device, config)
        
        score_test_v2t, score_test_t2v = evaluation(model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config)

        if utils.is_main_process():
            if ('skip_val' not in config) or (not config['skip_val']):
                val_result = itm_eval(score_val_v2t, score_val_t2v, val_loader.dataset.txt2video, val_loader.dataset.video2txt)
                print(val_result)
                                
                if val_result['r_mean']>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_result['r_mean']        
                    best_epoch = epoch  
                    
                    test_result = itm_eval(score_test_v2t, score_test_t2v, test_loader.dataset.txt2video, test_loader.dataset.video2txt)
                    print(test_result)
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                
                test_result = itm_eval(score_test_v2t, score_test_t2v, test_loader.dataset.txt2video, test_loader.dataset.video2txt)
                print(test_result)

                best = test_result['r_mean']        
                best_epoch = epoch

                
            if args.evaluate:
                if ('skip_val' not in config) or (not config['skip_val']):        
                    log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},                  
                                }
                else:
                    log_stats = {**{f'test_{k}': v for k, v in test_result.items()}}

                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                if ('skip_val' not in config) or (not config['skip_val']):
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},  
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }
                else:
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},  
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")           
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_msrvtt.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_msrvtt')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # train with single frame or concat frame, default 'single_frame'
    if 'video_representation' not in config:
       config['video_representation'] = 'single_frame'
    print(config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)