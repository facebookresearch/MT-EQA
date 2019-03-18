"""
Give the final questions, let's add program for each question.
Specifically, we provide the following programs:
1)  'nav_room[xxx]'        --> key frame
2)  'nav_object[xxx]'      --> key frame, object mask
3)  'query_object_color'   --> object color
4)  'query_object_size'    --> object size
5)  'query_room_size'      --> room size
6)  'query_trajectory'     --> trajectory length between source to current, current frame
7)  'equal_object_color'                --> 0/1
8)  'equal_object_size[bigger/smaller]' --> 0/1
9)  'equal_room_size[bigger/smaller]'   --> 0/1
10) 'equal_object_dist[closer/farther]' --> 0/1

We will formulate each program as the following example:
"is the bedroom in the living room bigger than the car in the garage?"
[
  {'function': 'scene', 'inputs': [], 'value_inputs': []},
  {'function': 'nav_room', 'inputs': [0], 'value_inputs': ['living room']},
  {'function': 'nav_object', 'inputs': [1], 'value_inputs': ['bedroom']},
  {'function': 'query_object_color', 'inputs': [2], 'value_inputs': []},
  {'function': 'scene', 'inputs': [], 'value_inputs': []},
  {'function': 'nav_room', 'inputs': [4], 'value_inputs': ['garage']},
  {'function': 'nav_object', 'inputs': [5], 'value_inputs': ['car']},
  {'function': 'query_object_color', 'inputs': [6], 'value_inputs': []},
  {'function': 'equal_object_color', 'inputs': [3, 7], 'value_inputs': []}
]
"""
import os
import os.path as osp
import sys
import json
import argparse
import operator
import collections
from tqdm import tqdm
from pprint import pprint

import _init_paths


# room types
ROOM_TYPES = ['bedroom', 'garage', 'gym', 'living room', 'dining room', 'kitchen', 'bathroom']

def convert_object_name(box_name):
  return ' '.join(box_name.lower().split('_'))

"""
1) object_color_compare_xroom
"""
def program_object_color_compare_xroom(qn):
  obj1_name = convert_object_name(qn['bbox'][0]['name'])
  obj1_color = qn['bbox'][0]['color']
  obj1_cat = qn['bbox'][0]['fine_class']
  obj1_id = qn['bbox'][0]['id']
  room1_id = qn['bbox'][0]['room_id']
  room1_name = qn['bbox'][0]['room_name']
  obj2_name = convert_object_name(qn['bbox'][1]['name'])
  obj2_color = qn['bbox'][1]['color']
  obj2_cat = qn['bbox'][1]['fine_class']
  obj2_id = qn['bbox'][1]['id']
  room2_id = qn['bbox'][1]['room_id']
  room2_name = qn['bbox'][1]['room_name']
  assert room1_id != room2_id and obj1_id != obj2_id
  assert room1_name in qn['question'] and room2_name in qn['question']
  # make program
  program = [
    {'function': 'scene', 'inputs': [], 'value_inputs': []},
    {'function': 'nav_room', 'inputs': [0], 'value_inputs': [room1_name], 'id': [room1_id]},
    {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
    {'function': 'query_object_color', 'inputs': [2], 'value_inputs': [], 'color': [obj1_color]},
    {'function': 'nav_room', 'inputs': [4], 'value_inputs': [room2_name], 'id': [room2_id]},
    {'function': 'nav_object', 'inputs': [5], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
    {'function': 'query_object_color', 'inputs': [6], 'value_inputs': [], 'color': [obj2_color]},
    {'function': 'equal_object_color', 'inputs': [3, 7], 'value_inputs': []},
  ]
  # return
  return program

"""
2) object_size_compare_xroom
"""
def program_object_size_compare_xroom(qn):
  obj1_name = convert_object_name(qn['bbox'][0]['name'])
  obj1_cat = qn['bbox'][0]['fine_class']
  obj1_id = qn['bbox'][0]['id']
  room1_id = qn['bbox'][0]['room_id']
  room1_name = qn['bbox'][0]['room_name']
  obj2_name = convert_object_name(qn['bbox'][1]['name'])
  obj2_cat = qn['bbox'][1]['fine_class']
  obj2_id = qn['bbox'][1]['id']
  room2_id = qn['bbox'][1]['room_id']
  room2_name = qn['bbox'][1]['room_name']
  assert room1_id != room2_id and obj1_id != obj2_id
  assert room1_name in qn['question'] and room2_name in qn['question']
  # operator
  op = 'bigger' if 'bigger' in qn['question'] else 'smaller'
  # make program
  program = [
    {'function': 'scene', 'inputs': [], 'value_inputs': []},
    {'function': 'nav_room', 'inputs': [0], 'value_inputs': [room1_name], 'id': [room1_id]},
    {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
    {'function': 'query_object_size', 'inputs': [2], 'value_inputs': []},
    {'function': 'nav_room', 'inputs': [4], 'value_inputs': [room2_name], 'id': [room2_id]},
    {'function': 'nav_object', 'inputs': [5], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
    {'function': 'query_object_size', 'inputs': [6], 'value_inputs': []},
    {'function': 'equal_object_size', 'inputs': [3, 7], 'value_inputs': [op]},
  ]
  # return
  return program

"""
3) object_color_compare_inroom
"""
def program_object_color_compare_inroom(qn):
  obj1_name = convert_object_name(qn['bbox'][0]['name'])
  obj1_color = qn['bbox'][0]['color']
  obj1_cat = qn['bbox'][0]['fine_class']
  obj1_id = qn['bbox'][0]['id']
  room1_id = qn['bbox'][0]['room_id']
  room1_name = qn['bbox'][0]['room_name']
  obj2_name = convert_object_name(qn['bbox'][1]['name'])
  obj2_color = qn['bbox'][1]['color']
  obj2_cat = qn['bbox'][1]['fine_class']
  obj2_id = qn['bbox'][1]['id']
  room2_id = qn['bbox'][1]['room_id']
  room2_name = qn['bbox'][1]['room_name']
  assert room1_id == room2_id and room1_name == room2_name
  room_id = room1_id
  room_name = room1_name
  # make program
  if qn['bbox'][0]['unique'] and qn['bbox'][1]['unique']:
    assert ' in the ' not in qn['question'], qn['question']
    program = [
      {'function': 'scene', 'inputs': [], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
      {'function': 'query_object_color', 'inputs': [2], 'value_inputs': [], 'color': [obj1_color]},
      {'function': 'nav_object', 'inputs': [4], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
      {'function': 'query_object_color', 'inputs': [5], 'value_inputs': [], 'color': [obj2_color]},
      {'function': 'equal_object_color', 'inputs': [3, 6], 'value_inputs': []},
    ]
  else:
    assert ' in the ' in qn['question'], qn['question']
    program = [
      {'function': 'scene', 'inputs': [], 'value_inputs': []},
      {'function': 'nav_room', 'inputs': [0], 'value_inputs': [room_name], 'id': [room_id]},
      {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
      {'function': 'query_object_color', 'inputs': [2], 'value_inputs': [], 'color': [obj1_color]},
      {'function': 'nav_object', 'inputs': [4], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
      {'function': 'query_object_color', 'inputs': [5], 'value_inputs': [], 'color': [obj2_color]},
      {'function': 'equal_object_color', 'inputs': [3, 6], 'value_inputs': []},
    ]
  # return
  return program

"""
4) object_size_compare_inroom
"""
def program_object_size_compare_inroom(qn):
  obj1_name = convert_object_name(qn['bbox'][0]['name'])
  obj1_cat = qn['bbox'][0]['fine_class']
  obj1_id = qn['bbox'][0]['id']
  room1_id = qn['bbox'][0]['room_id']
  room1_name = qn['bbox'][0]['room_name']
  obj2_name = convert_object_name(qn['bbox'][1]['name'])
  obj2_cat = qn['bbox'][1]['fine_class']
  obj2_id = qn['bbox'][1]['id']
  room2_id = qn['bbox'][1]['room_id']
  room2_name = qn['bbox'][1]['room_name']
  assert room1_id == room2_id and room1_name == room2_name
  room_id = room1_id
  room_name = room1_name
  # operator
  op = 'bigger' if 'bigger' in qn['question'] else 'smaller'
  # make program
  if qn['bbox'][0]['unique'] and qn['bbox'][1]['unique']:
    assert ' in the ' not in qn['question'], qn['question']
    program = [
      {'function': 'scene', 'inputs': [], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
      {'function': 'query_object_size', 'inputs': [2], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [4], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
      {'function': 'query_object_size', 'inputs': [5], 'value_inputs': []},
      {'function': 'equal_object_size', 'inputs': [3, 6], 'value_inputs': [op]},
    ]
  else:
    assert ' in the ' in qn['question'], qn['question']
    program = [
      {'function': 'scene', 'inputs': [], 'value_inputs': []},
      {'function': 'nav_room', 'inputs': [0], 'value_inputs': [room_name], 'id': [room_id]},
      {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
      {'function': 'query_object_size', 'inputs': [2], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [4], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
      {'function': 'query_object_size', 'inputs': [5], 'value_inputs': []},
      {'function': 'equal_object_size', 'inputs': [3, 6], 'value_inputs': [op]},
    ]
  return program

"""
5) room_size_compare
"""
def program_room_size_compare(qn):
  room1_id = qn['bbox'][0]['id']
  room1_name = qn['bbox'][0]['name']
  room2_id = qn['bbox'][1]['id']
  room2_name = qn['bbox'][1]['name']
  assert room1_name in qn['question'], '%s, %s' % (room1_name, qn['question'])
  assert room2_name in qn['question'], '%s, %s' % (room2_name, qn['question'])
  op = 'bigger' if 'bigger' in qn['question'] else 'smaller'
  # make program
  program = [
    {'function': 'scene', 'inputs': [], 'value_inputs': []},
    {'function': 'nav_room', 'inputs': [0], 'value_inputs': [room1_name], 'id': [room1_id]},
    {'function': 'query_room_size', 'inputs': [1], 'value_inputs': []},
    {'function': 'nav_room', 'inputs': [3], 'value_inputs': [room2_name], 'id': [room2_id]},
    {'function': 'query_room_size', 'inputs': [4], 'value_inputs': []},
    {'function': 'equal_room_size', 'inputs': [2, 5], 'value_inputs': [op]},
  ]
  return program

"""
6) object_dist_compare_inroom
"""
def program_object_dist_compare_inroom(qn):
  obj1_name = convert_object_name(qn['bbox'][0]['name'])
  obj1_cat = qn['bbox'][0]['fine_class']
  obj1_id = qn['bbox'][0]['id']
  room1_id = qn['bbox'][0]['room_id']
  room1_name = qn['bbox'][0]['room_name']
  obj2_name = convert_object_name(qn['bbox'][1]['name'])
  obj2_cat = qn['bbox'][1]['fine_class']
  obj2_id = qn['bbox'][1]['id']
  room2_id = qn['bbox'][1]['room_id']
  room2_name = qn['bbox'][1]['room_name']
  obj3_name = convert_object_name(qn['bbox'][2]['name'])
  obj3_cat = qn['bbox'][2]['fine_class']
  obj3_id = qn['bbox'][2]['id']
  room3_id = qn['bbox'][2]['room_id']
  room3_name = qn['bbox'][2]['room_name']
  assert room1_id == room2_id == room3_id
  assert room1_name == room2_name == room3_name
  room_id = room1_id
  room_name = room1_name
  # operator
  op = 'farther' if 'farther' in qn['question'] else 'closer'
  # make program
  if qn['bbox'][0]['unique'] and qn['bbox'][1]['unique'] and qn['bbox'][2]['unique']:
    assert ' in the ' not in qn['question'], qn['question']
    program = [
      {'function': 'scene', 'inputs': [], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
      {'function': 'query_trajectory', 'inputs': [2], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [3], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
      {'function': 'query_trajectory', 'inputs': [5], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [7], 'value_inputs': [obj3_name], 'fine_class': [obj3_cat], 'id': [obj3_id]},
      {'function': 'query_trajectory', 'inputs': [8], 'value_inputs': []},
      {'function': 'query_object_dist', 'inputs': [3, 6, 9], 'value_inputs': [op]},
    ]
  else:
    assert ' in the ' in qn['question'], qn['question']
    program = [
      {'function': 'scene', 'inputs': [], 'value_inputs': []},
      {'function': 'nav_room', 'inputs': [0], 'value_inputs': [room_name], 'id': [room_id]},
      {'function': 'nav_object', 'inputs': [1], 'value_inputs': [obj1_name], 'fine_class': [obj1_cat], 'id': [obj1_id]},
      {'function': 'query_trajectory', 'inputs': [2], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [3], 'value_inputs': [obj2_name], 'fine_class': [obj2_cat], 'id': [obj2_id]},
      {'function': 'query_trajectory', 'inputs': [5], 'value_inputs': []},
      {'function': 'nav_object', 'inputs': [7], 'value_inputs': [obj3_name], 'fine_class': [obj3_cat], 'id': [obj3_id]},
      {'function': 'query_trajectory', 'inputs': [8], 'value_inputs': []},
      {'function': 'query_object_dist', 'inputs': [3, 6, 9], 'value_inputs': [op]},
    ]
  return program


def main(args):

  # hid --> questions
  questions = json.load(open(args.input_questions_json))
  hid_to_qns = collections.defaultdict(list)
  for qn in questions:
    hid_to_qns[qn['house']].append(qn)

  # run
  for hid, qns in tqdm(hid_to_qns.items()):

    # add programs
    for qn in qns:
      qtype = qn['type']
      if qtype == 'object_color_compare_inroom':
        fn = program_object_color_compare_inroom
      elif qtype == 'object_color_compare_xroom':
        fn = program_object_color_compare_xroom
      elif qtype == 'object_size_compare_inroom':
        fn = program_object_size_compare_inroom
      elif qtype == 'object_size_compare_xroom':
        fn = program_object_size_compare_xroom
      elif qtype == 'object_dist_compare_inroom':
        fn = program_object_dist_compare_inroom
      elif qtype == 'room_size_compare':
        fn = program_room_size_compare
      else:
        raise NotImplementedError
      qn['program'] = fn(qn)
      qn['path'] = '.'.join([qn['house']]+[pg['id'][0] for pg in qn['program'] if 'nav' in pg['function']])

  # save
  with open(args.output_questions_json, 'w') as f:
    json.dump(questions, f)
  print('questions with program are saved in %s.' % args.output_questions_json)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_questions_json', type=str, default='cache/question-gen-outputs/questions_pruned_mt_with_conn.json')
  parser.add_argument('--object_id_to_cat_json', default='cache/question-gen-outputs/object_id_to_cat.json')
  parser.add_argument('--output_questions_json', type=str, default='cache/question-gen-outputs/questions_pruned_mt_with_conn_program.json')
  args = parser.parse_args()

  main(args)
  
