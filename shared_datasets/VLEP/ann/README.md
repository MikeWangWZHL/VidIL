## VLEP DATASET

This directory contains the data for our Video-and-Language Event Prediction (VLEP) dataset. 
Given a video with associated dialogue as premise, and two possible future events, the VLEP 
task requires systems to predict which one is more likely to happen.


#### Annotations

We split VLEP into 3 subsets: 

| Split | #examples | Filename | Description|
| --- | --- | --- | --- |
| train | 20,142 | [vlep_train_release.jsonl](vlep_train_release.jsonl) | Model Training |
| dev | 4,392 | [vlep_dev_release.jsonl](vlep_dev_release.jsonl) | Hyperparameter tuning |
| test | 4,192 | [vlep_test_release.jsonl](vlep_test_release.jsonl) | Model Testing, ground-truth answer is removed. |

An annotated example from VLEP `dev` split is shown below.
```
{
  "example_id": 21411,
  "vid_name": "OO9kSxcT9Rg_subs_012_00:13:00_00:14:00_ep",  # video name.
  "ts": [10.54, 13.55],  # start and end time of the premise event (in seconds).
  "events": [  
    "The customer eats the food from the tray.",
    "The customer takes the tray of food."
  ],  # the two possible future events after the premise.
  "answer": 1,  # ground-truth answer, in {0, 1}
  "split": "dev"
}
```

For YouTube videos, `vid_name` is formatted as 
`{youtube_id}_subs_{sequence_idx}_{video_start_time}_{video_end_time}_ep`. 
You can use the information from `vid_name` and `ts` (`[event_st_time, event_ed_time]`) to compile a link to 
this premise event clip on YouTube:
`https://www.youtube.com/embed/{youtube_id}?start={video_start_time + event_st_time}&end={video_start_time + event_ed_time}&version=3`.
For example, the link to the premise event shown above is https://www.youtube.com/embed/OO9kSxcT9Rg?start=790&end=793&version=3. 
In order to make this link work, you need to convert `{video_start_time + event_st_time}` into seconds and round it to a neighboring integer. 
Since rounding is applied here, it does not show exactly the premise event, but a approximated version.

#### Subtitles
The original subtitles in `.srt` format is available at this [link](https://drive.google.com/file/d/1hFddQQfBU0jh0zMZUYmTVQVD7A9KdeK0/view?usp=sharing). 
For your convenience, we attach the preprocessed subtitles in a single `.jsonl` file [data/vlep_subtitles.jsonl](vlep_subtitles.jsonl). 
It is preprocessed minimally using this [script](../utils/text_utils/preprocess_subtitles.py) from [TVRetrieval](https://github.com/jayleicn/TVRetrieval).

#### Video Clips and Frames
To access VLEP clips (YouTube only) and 3 FPS frames (TV show + YouTube), please fill out this [form](https://forms.gle/szvKNp77CxNVevNS8). 
The download link for the VLEP files will be sent to you in around a week if your form is valid. 
Please do not share the video frames with others.

#### Evaluation
Test split answers are reserved, please see [standalone_eval](../standalone_eval/README.md) for instructions on how to evaluate performance on test split.
