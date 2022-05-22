import json
from glob import glob
import os

full_ann = json.load(open('/shared/nas/data/m1/wangz3/Shared_Datasets/VL/Vatex/value_ann/vatex_en_c/videoid_2_text_vatex_en_c_train.json'))
train_video_subset_dir = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/Vatex/video_clips/training_subset'

subset_ann = {}
for video_path in glob(os.path.join(train_video_subset_dir,'*.mp4')):
    video_id = os.path.basename(video_path)[:-4]
    subset_ann[video_id] = full_ann[video_id]
print(subset_ann)

with open('vatex_en_c_train_subset.json','w') as out:
    json.dump(subset_ann,out,indent=4)
