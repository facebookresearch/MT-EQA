# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
In case I will change qid for questions, e.g., I re-generated the questions, then I
have to re-compute their shortest-paths again, which is super time-consuming.

Instead of saving the shortest-paths by ``qid'', we save them using 
[house_id.target_id1.target_id2.json] 
[house_id.target_id1.target_id2.target_id3.json]

This is a post-processing step merging the computed paths from qid.json into above.
Hopefully we won't use this script again after updating ``generate-shortest-paths-v2.py''
"""
import json
import os
import os.path as osp
import sys
this_dir = osp.dirname(__file__)

# config
cfg = {}
cfg['pruned_questions_file'] = osp.join(this_dir, '../cache/question-gen-outputs/questions_pruned_v2.json')
cfg['shortest_path_dir'] = osp.join(this_dir, '../cache/shortest-paths-v2/nearby_source_best_view')

# 1) load pruned questions (before shortest-path filtering)
questions = json.load(open(cfg['pruned_questions_file']))
Questions = {qn['id']: qn for qn in questions}
print('There are %s questions loaded from %s.' % (len(questions), cfg['pruned_questions_file']))

# 2) hid.tid.tid --> [qid, qid, ...]
htt_to_qids = {}  # htt: hid.tid.tid
for qn in questions:
  house_id = str(qn['house'])
  box_ids = '.'.join([str(box['id']) for box in qn['bbox']])
  htt = house_id+'.'+box_ids
  if htt not in htt_to_qids: 
    htt_to_qids[htt] = []
  htt_to_qids[htt].append(str(qn['id']))
print('%s hid.tid.tid found.' % len(htt_to_qids))
num_questions = len([qid for qids in htt_to_qids.values() for qid in qids])

# 3) merge them
num_merged = 0
for htt, qids in htt_to_qids.items():
  merged_paths = []
  for qid in qids:
    qid_path_file = osp.join(cfg['shortest_path_dir'], str(qid)+'.json') 
    if osp.exists(qid_path_file):
      # list of [{positions, actions, best_iou, ordered}]
      qid_paths = json.load(open(qid_path_file))  
      merged_paths += qid_paths
  # save
  if len(merged_paths) > 0:
    with open(osp.join(cfg['shortest_path_dir'], htt+'.json'), 'w') as f:
      json.dump(merged_paths, f)
    print('%s paths merged for %s' % (len(merged_paths), htt))
    num_merged += 1

print('Done. There are %s merged paths for %s questions.' % (num_merged, num_questions))

