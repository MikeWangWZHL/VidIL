import ffmpeg
import os
from glob import glob
import json

def multi_output_selection(input_root, output_root, video_id, video_ann):
    # trim one video into many clips according to video_ann dict
    
    split = video_ann["subset"]
    
    # find input path
    for extension in ['.mp4', '.mkv', '.webm']:
        input_path = os.path.join(input_root,split,f'{video_id}{extension}')
        if os.path.exists(input_path):
            break
        else:
            input_path = None
    
    if input_path is None:
        print(f'skip video: {split}/{video_id}')
        return False
    
    # load video
    ffmpeg_input_stream = ffmpeg.input(input_path)
    ffmpeg_output_streams = []
    for clip in video_ann['annotations']:
        clip_id = clip['id']
        start, end = clip['segment']
        duration = end - start
        output_path = os.path.join(output_root, split, f'{video_id}_{clip_id}.mp4')

        ffmpeg_output_streams.append(
            ffmpeg.output(
                ffmpeg_input_stream,
                output_path,
                ss=start, t=duration
            )
        )

    output_streams = ffmpeg.merge_outputs(*ffmpeg_output_streams)
    ffmpeg.run(output_streams)
    return True

if __name__ == "__main__":
    input_root = './raw_videos'
    output_root = './video_clips'

    for split in ['training','validation','testing']:
        if not os.path.exists(os.path.join(output_root, split)):
            os.makedirs(os.path.join(output_root, split), exist_ok=True)
    
    # load raw annotation
    train_val_annotation_json = './youcookii_annotations_trainval.json'
    train_val_annotation = json.load(open(train_val_annotation_json))


    total_count = len(train_val_annotation['database'].keys())
    print('total num of videos:', total_count)
    
    success_count = 0
    fail_count = 0
    for video_id, video_ann in train_val_annotation['database'].items():
        success = multi_output_selection(input_root, output_root, video_id, video_ann)
        if success:
            success_count += 1
        else:
            fail_count += 1
        print(f'INFO: success {success_count} | fail {fail_count} | total {total_count}')