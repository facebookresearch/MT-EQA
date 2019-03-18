"""
Sometimes we want object_id/room_id --> category_name, e.g. for semantic segmentation.
We construct such mapping for eqa_v1 houses.
house_id.target_id --> cat_name
"""
import argparse
import os
import os.path as osp
import sys
import random
import time
import json
from tqdm import tqdm
from house_parse import HouseParse

this_dir = osp.dirname(__file__)
eqa_v1_file = osp.join(this_dir, '../data/eqa_v1/eqa_v1.json')
hid_to_qns = json.load(open(eqa_v1_file))['questions']

# get house info, as well as id_to_cat
dataDir = osp.join(this_dir, '../data')
HouseApiDir = osp.join(this_dir, '../pyutils/House3D')
Hp = HouseParse(dataDir=osp.join(dataDir, 'SUNCGdata'), objrenderPath=osp.join(HouseApiDir, 'House3D'))

id_to_cat = {}
for hid in tqdm(hid_to_qns.keys()):
  Hp.parse(hid)
  for obj_id, obj in Hp.objects.items():
    id_to_cat[str(hid)+'.'+str(obj_id)] = obj['fine_class']

output_file = osp.join(this_dir, '../cache/question-gen-outputs/object_id_to_cat.json')
with open(output_file, 'w') as f:
  json.dump(id_to_cat, f)
  