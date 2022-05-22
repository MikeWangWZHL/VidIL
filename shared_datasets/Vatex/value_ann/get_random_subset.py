import json
import random

original_ann = '/shared/nas/data/m1/wangz3/Shared_Datasets/VL/Vatex/value_ann/vatex_en_c/videoid_2_text_vatex_en_c_train.json'

original_ann = json.load(open(original_ann))

keys = list(original_ann.keys())
random.shuffle(keys)

k = 50

subset = {}
for key in keys[:k]:
    subset[key] = original_ann[key]

with open(f'vatex_en_c_train_random_subset_{k}.json', 'w') as out:
    json.dump(subset, out, indent=4)