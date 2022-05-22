from pytube import YouTube 
import csv
import os
import sys
import json

def load_csv(csv_file):
    links = []
    ids = []
    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            links.append(row[2])
            ids.append(row[1])
    print('num of video links:', len(links))
    assert len(ids) == len(links)
    return links, ids

def get_download_links(json_file):
    annotations = json.load(open(json_file))
    video_ids = []
    video_links = []
    for item in annotations:
        video_id = '_'.join(item['videoID'].split('_')[:-2])
        video_link = 'www.youtube.com/watch?v='+video_id
        video_ids.append(video_id)
        video_links.append(video_link)
    return video_ids, video_links

if __name__ == '__main__':
    ann_jsons = ['./official_ann/vatex_public_test_english_v1.1.json', './official_ann/vatex_training_v1.0.json']
    output_dirs = ['./videos/testing_public','./videos/training']
    for ann_json, output_dir in zip(ann_jsons, output_dirs):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        ids, links = get_download_links(ann_json)

        success_count = 0
        failed_count = 0
        for i in range(len(links)):
            link = links[i]
            id_ = ids[i]
            if os.path.exists(os.path.join(output_dir, f'{id_}.mp4')):
                print(f'already exist: {id_}')
                continue
            try: 
                # object creation using YouTube
                # which was imported in the beginning
                yt = YouTube(link)         
                #filters out all the files with "mp4" extension
                # mp4file = yt.streams.filter(file_extension='mp4').get_by_itag(22) # 22: 720p 
                mp4file = yt.streams.filter(file_extension='mp4').get_by_itag(18) # 18: 360p 

                # downloading the video 
                mp4file.download(output_dir, f'{id_}.mp4')
                print(f'downloaded {i}/{len(links)}: {id_}')
                success_count += 1
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                # print(e)
                print(f"Failed {i}/{len(links)}: {id_}") 
                failed_count += 1
        print(f'Task Completed! success: {success_count} failed: {failed_count}') 