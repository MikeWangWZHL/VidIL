# Download Dataset Annotations
Download dataset annotations zip from [here](https://uofi.box.com/s/wwh4phgetakycvzeyxoih5qupevzj9xe). Then unzip the downloaded datasets under `shared_datasets/`. The resulting shared_dataset folder structure is expected to be:
```
shared_datasets
├── README.md
├── MSRVTT_caption
├── MSRVTT_qa
...
```

# Download & Preprocess Raw Videos
Instruction for downloading the videos and preprocessing the datasets. The annotation files are already included in the repo. The preprocessing scripts require installing the following additional package:
```
pytube
ffmpeg
```

## MSR-VTT
Refer to https://github.com/salesforce/ALPRO#data-preparation to download the videos and put **all** videos (including train, val, test) under `shared_datasets/MSRVTT_ret/videos`

## YouCook2
- (step 1) Refer to http://youcook2.eecs.umich.edu/download to download the raw videos (note that some videos may not be donwloadable anymore), which is expected to have the following folder structure:
```
    shared_datasets/Youcook2/raw_videos
    ├── testing
    ├── training
    └── validation
```
- (step 2) Under `shared_datasets/Youcook2`, run the following script to extract the video clips from raw videos:
```
    python get_video_clips.py
```
- (step 3) Copy or move videos in `shared_datasets/Youcook2/raw_videos/training` and `shared_datasets/Youcook2/raw_videos/validation` into `shared_datasets/Youcook2/video_clips/train_val`


## VaTex
- (step 1) Under `shared_datasets/Vatex`, download **training** and **public_testing** videos using the following scripts (note that some videos may not be donwloadable anymore)
    ```
        python download_vatex_pytube.py
    ```
- (step 2) Preprocess raw videos to get video clips, under `shared_datasets/Vatex`, run:
    ```
        python get_video_clips.py
    ```
- (step 3) Put all  **training and public_testing** video clips under `shared_datasets/Vatex/video_clips/train_test`

## MSVD
Refer to https://github.com/salesforce/ALPRO#data-preparation to download the videos for MSVD and put **all** videos under `shared_datasets/msvd_qa/videos`


## VLEP
Refer to https://github.com/jayleicn/VideoLanguageFuturePred/blob/main/data/README.md to request downloading the videos. Put all **.mp4** videos under `shared_datasets/VLEP/vlep_clips` and all **.srt** files under `shared_datasets/VLEP/vlep_srt` 