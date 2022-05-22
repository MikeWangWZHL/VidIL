mkdir -p ckpt/pretrained
mkdir -p ckpt/finetuned
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth -P ckpt/pretrained
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth -P ckpt/pretrained
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth -P ckpt/finetuned
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth -P ckpt/finetuned
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth -P ckpt/finetuned