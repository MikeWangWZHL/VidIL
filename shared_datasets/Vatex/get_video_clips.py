import ffmpeg
import os
from glob import glob
import json
import sys
from tqdm import tqdm

def output_selection(video_path, output_path, start, end):  
    duration = end - start
    ffmpeg_input_stream = ffmpeg.input(video_path)
    (
        ffmpeg
        .output(
            ffmpeg_input_stream,
            output_path,
            ss=start, t=duration
        )
        .global_args('-loglevel', 'quiet')
        .run()
    )
    

if __name__ == "__main__":
    input_root = './videos'
    output_root = './video_clips'

    splits = [
        'training',
        'testing_public'
    ]
    anns = [
        './value_ann/vatex_en_c/videoid_2_text_vatex_en_c_train.json',
        './value_ann/vatex_en_c/videoid_2_text_vatex_en_c_test_public.json'
    ]
    
    for split,ann_path in zip(splits,anns):
        
        if not os.path.exists(os.path.join(output_root, split)):
            os.makedirs(os.path.join(output_root, split), exist_ok=True)

        ann = json.load(open(ann_path))
        for key in ann.keys():
            yt_video_id = '_'.join(key.split('_')[:-2])
            start_time = key.split('_')[-2:][0].lstrip('0')
            end_time = key.split('_')[-2:][1].lstrip('0')
            if start_time == '':
                start_time = '0'
            if end_time == '':
                print(f'end time can not be 0: skip {key}')
                continue
            start_time = int(start_time)
            end_time = int(end_time)

            input_path = os.path.join(input_root, split, f'{yt_video_id}.mp4')
            output_path = os.path.join(output_root, split, f'{key}.mp4')
            if os.path.exists(input_path):
                try: 
                    output_selection(input_path, output_path, start_time, end_time)
                    print(f'Successful: {key}')
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    print(f"Failed: {key}") 

