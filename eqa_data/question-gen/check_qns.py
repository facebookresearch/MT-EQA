# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This code check qtype-to-cnt for EQA V1 data.
"""
import os
import os.path as osp
import json
import sys
from pprint import pprint

#######################################################
# Check V1 data
#######################################################
this_dir = osp.dirname(__file__)
data_dir = osp.join(this_dir, '../data')
data = json.load(open(osp.join(data_dir, 'eqa_v1', 'eqa_v1.json')))

# house_id --> split
hid_to_split = {hid: split for split, hids in data['splits'].items() for hid in hids}

# count questions for each type
qtype_to_qns = {}
for house_id, qns in data['questions'].items():
  for qn in qns:
    if qn['type'] not in qtype_to_qns: qtype_to_qns[qn['type']] = []
    qtype_to_qns[qn['type']] += [qn]
# print
total = 0
for qtype, qns in qtype_to_qns.items():
  total += len(qns)
  print('%s questions for [%s]' % (len(qns), qtype))
print('There are %s questions in all for EQA v1.' % total)


#######################################################
# Check generated data (after pruning)
#######################################################
print('\nv1 questions_pruned:')
cache_dir = osp.join(this_dir, '../cache')
if osp.exists(osp.join(cache_dir, 'question-gen-outputs', 'questions_pruned_v1.json')):
  data = json.load(open(osp.join(cache_dir, 'question-gen-outputs', 'questions_pruned_v1.json')))

  # count questions for each type
  qtype_to_qns = {}
  for qn in data:
    if qn['type'] not in qtype_to_qns: qtype_to_qns[qn['type']] = []
    qtype_to_qns[qn['type']] += [qn]
  # print
  for qtype, qns in qtype_to_qns.items():
    print('%s questions for [%s]' % (len(qns), qtype))
  print('There are %s questions in all for my processed EQA v1.' % len(data))

#######################################################
# Check generated data (after pruning)
#######################################################
print('\nv2 questions_pruned:')
cache_dir = osp.join(this_dir, '../cache')
if osp.exists(osp.join(cache_dir, 'question-gen-outputs', 'questions_pruned_v2.json')):
  data = json.load(open(osp.join(cache_dir, 'question-gen-outputs', 'questions_pruned_v2.json')))

  # count questions for each type
  qtype_to_qns = {}
  for qn in data:
    if qn['type'] not in qtype_to_qns: qtype_to_qns[qn['type']] = []
    qtype_to_qns[qn['type']] += [qn]
  # print
  for qtype, qns in qtype_to_qns.items():
    print('%s questions for [%s]' % (len(qns), qtype))
  print('There are %s questions in all for my processed EQA v2.' % len(data))

  # stats on splits
  split_to_hids = {}
  for qn in data:
    split = hid_to_split[qn['house']]
    if split not in split_to_hids: split_to_hids[split] = []
    split_to_hids[split].append(qn['house'])
  for split in split_to_hids:
    split_to_hids[split] = list(set(split_to_hids[split]))
    print('[%s] has %s house_ids.' % (split, len(split_to_hids[split])))

#######################################################
# Check generated data (after pruning)
#######################################################
print('\nmt questions_pruned:')
cache_dir = osp.join(this_dir, '../cache')
if osp.exists(osp.join(cache_dir, 'question-gen-outputs', 'questions_pruned_mt.json')):
  data = json.load(open(osp.join(cache_dir, 'question-gen-outputs', 'questions_pruned_mt.json')))

  # count questions for each type
  qtype_to_qns = {}
  for qn in data:
    if qn['type'] not in qtype_to_qns: qtype_to_qns[qn['type']] = []
    qtype_to_qns[qn['type']] += [qn]
  # print
  for qtype, qns in qtype_to_qns.items():
    print('%s questions for [%s]' % (len(qns), qtype))
  print('There are %s questions in all for my processed EQA mt.' % len(data))

  # stats on splits
  split_to_hids = {}
  for qn in data:
    split = hid_to_split[qn['house']]
    if split not in split_to_hids: split_to_hids[split] = []
    split_to_hids[split].append(qn['house'])
  for split in split_to_hids:
    split_to_hids[split] = list(set(split_to_hids[split]))
    print('[%s] has %s house_ids.' % (split, len(split_to_hids[split])))