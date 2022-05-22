MSR-VTT 10K

All video info and caption sentences are stored using the JSON file format. All data share the basic data structure below:

{
  "info" : info,
  "videos": [video],
  "sentences": [sentence],
}

info{
  "year" : str,
  "version" : str,
  "description": str,
  "contributor": str,
  "data_created": str,
}

video{
  "id": int,
  "video_id": str,
  "category": int,
  "url": str,
  "start time": float,
  "end time": float,
  "split": str,
}

sentence{
  "sen_id": int,
  "video_id": str,
  "caption": str,
}

The time is counted by seconds.
category name can refer to 'category.txt' for the detailed information.

Data split(video_id):
Train: video0 : video6512 (6513)
Val: video6513 : video7009 (497)
Test: video7010 : video9999 (2990) 


