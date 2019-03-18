"""
Prepare data.json 
  - questions: list of [{h5_id, house, question, answer, split, bbox, type, path_name, 
                         program, room_labels, room_inroomDists, path_actions, path_positions}]
    where "bbox" is list of [{id, phrase, room_id, room_phrase, type, name, fine_class, box, color}],
    and room_labels: room_id -> #paths of label
    and room_inroomDists: room_id -> #paths of inroomDists
    and path_actions = #paths of action list, {0: 'move forward', 1: 'turn left', 2: 'turn right', 3: 'stop'}
    and path_positions = #paths of positon list - [pos, pos, ...], each pos is (x, y, z, yaw) coord
    We will use path_h5 to get "imgs_h5":
      - num_paths (1, )
      - orderedk (1, )
      - ego_rgbk (L, 224, 224, 3)
      - ego_semk (L, 224, 224, 3)
      - cube_rgbk (L, 4, 224, 224, 3)
      - key_ixsk (k, )
      - positionsk (L, 4)
      - actions (L, )
      and "feats_h5":
      - ego_rgbk (L, 32, 10, 10)
      - cube_rgbk (L, 4, 32, 10, 10)
  - wtoi: question vocab
  - atoi: answer vocab
  - ctoi: color vocab
  - wtov: word2vec
Prepare data.h5
  - encoded_questions  (N, L)
  - encoded_answers (N, )
"""
import argparse
import h5py
import os
import os.path as osp
import sys
import json
import copy
import collections
import numpy as np
from tqdm import tqdm
from pprint import pprint
import random
random.seed(24)

# room-specific objects (specific version)
room_to_objects = {}
room_to_objects['bathroom'] = [ 
  'toilet', 'sink', 'shower', 'bathtub', 'towel_rack',  'mirror', 'hanger', 'partition', 'bookshelf', 
  'curtain', 'toy', 'dryer', 'heater'
  ]
room_to_objects['kitchen'] = [
  'kitchen_cabinet', 'hanging_kitchen_cabinet', 'refrigerator', 
  'microwave', 'kettle', 'coffee_machine', 'knife_rack', 'cutting_board', 'food_processor', 
  'glass', 'pan', 'plates', 'utensil_holder', 'cup', 'water_dispenser', 'range_hood', 
  'beer', 'fruit_bowl', 'dishwasher', 'range_oven'
  ]
room_to_objects['bedroom'] = [
  'double_bed', 'single_bed', 'baby_bed', 'table_lamp',  'dressing_table', 'desk', 
  'laptop', 'books', 'tv_stand', 'mirror', 'computer'
  ]
room_to_objects['living room'] = [
  'sofa', 'coffee_table', 'tv_stand', 'books', 'fireplace', 'stereo_set', 'playstation', 'clock', 
  'fish_tank', 'table_lamp', 'piano', 'xbox'
  ]
room_to_objects['dining room'] = [
  'dining_table', 'cup', 'plates', 'glass', 'candle', 'fruit_bowl', 'bottle', 'coffee_table'
  ]
room_to_objects['gym'] = [
  'gym_equipment', 'game_table', 'mirror', 'curtain', 'toy', 'outdoor_seating', 
  'vacuum_cleaner', 'basketball_hoop', 'piano', 'stereo_set', 'ironing_board', 'partition', 
  ]
room_to_objects['garage'] = [
  'car', 'garage_door', 'motorcycle', 'column', 'outdoor_lamp', 'vacuum_cleaner', 'gym_equipment', 
  'partition', 'shoes_cabinet', 'ironing_board', 
  ]
room_to_objects['balcony'] = [
  'fence', 'chair', 'outdoor_lamp', 'wall_lamp', 'grill', 'rug', 'door', 'trash_can',
  ]

def tokenize(seq, delim=' ', punctToRemove=None, addStartToken=True, addEndToken=True):
  """
  Tokenize a sequence, converting a string seq into a list of (string) tokens by
  splitting on the specified delimiter. Optionally add start and end tokens.
  """
  if punctToRemove is not None:
    for p in punctToRemove:
      seq = str(seq).replace(p, '')
  tokens = str(seq).split(delim)
  if addStartToken:
    tokens.insert(0, '<START>')
  if addEndToken:
    tokens.append('<END>')
  return tokens

def buildVocab(sequences, minTokenCount=1, delim=' ', punctToRemove=None, addSpecialTok=False):
  SPECIAL_TOKENS = {'<NULL>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
  tokenToCount = {}
  for seq in sequences:
    seqToTokens = tokenize(seq, delim=delim, punctToRemove=punctToRemove, addStartToken=False, addEndToken=False)
    for token in seqToTokens:
      tokenToCount[token] = tokenToCount.get(token, 0) + 1
  tokenToIdx = {}
  if addSpecialTok == True:
    for token, idx in SPECIAL_TOKENS.items():
      tokenToIdx[token] = idx
  for token, count in sorted(tokenToCount.items()):
    if count >= minTokenCount:
      tokenToIdx[token] = len(tokenToIdx)
  return tokenToIdx

def encode(seqTokens, tokenToIdx, allowUnk=False):
  """
  Given seqTokens (list of tokens), encode to seqIdx (list of token_ixs)
  """
  seqIdx = []
  for token in seqTokens:
    if token not in tokenToIdx:
      if allowUnk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seqIdx.append(tokenToIdx[token])
  return seqIdx

def decode(seqIdx, idxToToken, delim=None, stopAtEnd=True):
  """
  Given seqIdx (list of token_ixs), decode to a sentence.  
  """
  tokens = []
  for idx in seqIdx:
    tokens.append(idxToToken[idx])
    if stopAtEnd and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)

def convert_to_phrase(target_name):
  """
  we only eliminate _, not /, 
  so "fish tank/bowl" still remains.
  """
  wds = target_name.split(' ')
  wds = [wd for _ in wds for wd in _.split('_')]
  return ' '.join(wds)

def make_wordvec(wtoi, atoi, ctoi, args):
  """
  Convert questions', answers', colors' words to vectors, becoming w2v.
  We keep "tank/bowl", encode "<START>" etc, and normal word.
  """
  with open(args.vectors_file, 'r') as f:
    vectors = {}
    for line in tqdm(f):
      vals = line.rstrip().split(' ')
      vectors[vals[0]] = [float(x) for x in vals[1:]]
  wtov = {}
  words = list(set(list(wtoi.keys()) + list(atoi.keys()) + list(ctoi.keys())))
  for wd in words:
    if '/' in wd: 
      # tank/bowl
      wtov[wd] = np.array([vectors[w] for w in wd.split('/')]).mean(0).tolist()
    elif '<' in wd:
      # <START> <NULL> <END>
      wtov[wd] = np.array(vectors[wd.replace('<', '').replace('>', '').lower()]).tolist()
    elif '_' in wd:
      # living_room, for answers only
      wtov[wd] = np.array([vectors[w] for w in wd.split('_')]).mean(0).tolist()
    else:
      # normal word, I like it
      wtov[wd] = np.array(vectors[wd]).tolist()
  return wtov

def get_positions(path):
  """
  Given one path = {positions, actions, best_iou, ordered}, note positions is a list of segment path ,
  we compute:
  - positions = [pos, pos, pos, ...], each is (x, y, z, yaw)
  - actions = [act, act, act, ...] whose last token is <STOP> (3)
  - key_ixs = [ix, ix], where agent at positions[ix] faces the target objects.
  - ordered = True or False
  """
  positions = [pos for seg_path in path['positions'] for pos in seg_path[1:]]
  positions.insert(0, path['positions'][0][0])  # add the spawned point
  actions = [act for seg_path in path['actions'] for act in seg_path[:-1]]
  actions.append(3) # add <STOP> in the end
  key_ixs, key_pos = [], []
  total_len = 0
  for i, seg_path in enumerate(path['positions'][:-1]):
    # last seg_path is toward end, no need being considered for key_ixs
    total_len += len(seg_path)
    key_ixs.append(total_len-i-1)
    key_pos.append(seg_path[-1])
  # check
  for ix, pos in zip(key_ixs, key_pos):
    assert positions[ix] == pos
  return positions, actions, key_ixs, path['ordered']

def main(args):

  # load questions = [{question, answer, type, bbox, id, house, etc}]
  questions = json.load(open(args.question_json, 'r'))
  print('%s questions are prepared.' % len(questions))

  # prepare raw questions
  print('adding shortest_path info ...')
  for qn in tqdm(questions):
    path_name = qn['path']
    imgs_h5 = osp.join(args.path_imgs_dir, path_name+'.h5')  # ego_rgb, ego_depth, ego_sem, cube_rgb, key_ixs, ...
    room_meta_file = osp.join(args.room_meta_dir, path_name+'.json') # room_id -> #paths of [{inroomDists, obj_to_iou}]
    shortest_path_file = osp.join(args.shortest_path_dir, path_name+'.json')  # positions, actions, best_iou, ordered

    # imgs_info
    imgs_info = h5py.File(imgs_h5, 'r')
    num_paths = int(imgs_info['num_paths'][0])
    key_ixs_set = [imgs_info['key_ixs%s'%i][...].tolist() for i in range(num_paths)]
    for i, box in enumerate(qn['bbox']):
      box['phrase'] = convert_to_phrase(box['name'])

    # add more info to qn
    qn['path_name'] = path_name
    qn['num_paths'] = num_paths
    qn['key_ixs_set'] = key_ixs_set
    qn['path_lengths'] = [imgs_info['positions%d'%i].shape[0] for i in range(num_paths)]
    qn['answer'] = '_'.join(qn['answer'].split(' '))

    # prepare #paths of actions, each actions is of length path_len
    paths_info = json.load(open(shortest_path_file)) #paths of positions, actions
    qn['path_actions'] = []
    qn['path_positions'] = []
    for pix in range(qn['num_paths']):
      path = paths_info[pix]
      positions, actions, _, _ = get_positions(path)
      assert len(actions) == qn['path_lengths'][pix]
      qn['path_actions'].append(actions)
      qn['path_positions'].append(positions)

    # nav_ids
    nav_ids = [pg['id'][0] for pg in qn['program'] if 'nav' in pg['function']]
    assert len(nav_ids) == len(key_ixs_set[0])

    # reference nav_id, used for evaluation
    qn['ref_nav_id'] = random.sample(nav_ids, 1)[0]

    # prepare room_labels for each qn if necessary
    room_ids = [pg['id'][0] for pg in qn['program'] if pg['function'] == 'nav_room']
    room_names = [pg['value_inputs'][0] for pg in qn['program'] if pg['function'] == 'nav_room']
    if len(room_ids) > 0:
      room_meta = json.load(open(room_meta_file))  # room_id --> #paths of [{inroomDists, obj_to_iou}]
      qn['room_labels'] = collections.defaultdict(list)      # room_id --> #paths of label
      qn['room_inroomDists'] = collections.defaultdict(list) # room_id --> #paths of inroomDists

      for room_id, room_name in zip(room_ids, room_names):
        # meta = #paths of [{inroomDists, obj_to_iou}]
        for pix in range(num_paths):
          assert room_id in room_meta
          meta = room_meta[room_id][pix]
          assert qn['path_lengths'][pix] == len(meta['inroomDists'])
          label = []
          for d, obj_to_iou in zip(meta['inroomDists'], meta['obj_to_iou']):
            # room-specific object's iou
            room_specific_objs = room_to_objects[room_name]
            room_specific_ious = [obj_to_iou[obj] for obj in room_specific_objs]
            if max(room_specific_ious) > args.iou_thresh and d > 0:
              label.append(1)  # inside room facing mearningful objects
            elif d > 0:
              label.append(0)  # inside room facing "nothing"
            else:
              label.append(-1) # outside room
          # also consider key_ix
          key_ix = key_ixs_set[pix][nav_ids.index(room_id)]
          label[key_ix] = 1
          # add to room_labels
          qn['room_labels'][room_id].append(label)
          # add to room_inroomDists
          qn['room_inroomDists'][room_id].append(meta['inroomDists'])

  print('meta_info added.')

  # vocab for questions (all words)
  print('encoding ...')
  wtoi = buildVocab([qn['question'] for qn in questions], punctToRemove=['?'], addSpecialTok=True)

  # vocab for answers
  atoi = buildVocab([qn['answer'] for qn in questions])

  # vocab for colors
  colors = []
  for qn in questions:
    if 'color' in qn['type']:
      colors.append(qn['bbox'][0]['color'])
      colors.append(qn['bbox'][1]['color'])
  ctoi = buildVocab(colors) 

  # encode questions and answers
  encoded_questions, encoded_answers = [], []
  for h5_id, qn in enumerate(questions):
    qn['h5_id'] = h5_id
    questionTokens = tokenize(qn['question'], punctToRemove=['?'], addStartToken=False, addEndToken=True)  # xxx<END>
    encoded_questions.append(encode(questionTokens, wtoi))
    encoded_answers.append(atoi[qn['answer']]) 
  maxQLength = max(len(x) for x in encoded_questions)
  for qe in encoded_questions:
    while len(qe) < maxQLength:
      qe.append(wtoi['<NULL>'])
  print('questions and answers encoded.')

  # make word vectors
  wtov = make_wordvec(wtoi, atoi, ctoi, args)
  print('%d wtov prepared.' % len(wtov))

  # hid_tid --> best_iou
  hid_tid_to_best_iou = {}
  for qn in questions:
    for pg in qn['program']:
      if 'nav_object' in pg['function']:
        tid = pg['id'][0]
        hid_tid = qn['house']+'_'+tid
        best_iou = json.load(open(osp.join(args.target_obj_bestview_pos_dir, hid_tid+'.json'), 'r'))['ious'][0]
        hid_tid_to_best_iou[hid_tid] = best_iou

  # save
  data = {}
  data['questions'] = questions
  data['wtoi'] = wtoi
  data['atoi'] = atoi  # yes: 1, no: 0
  data['ctoi'] = ctoi
  data['wtov'] = wtov
  data['hid_tid_to_best_iou'] = hid_tid_to_best_iou
  with open(osp.join(args.output_dir, 'data.json'), 'w') as f:
    json.dump(data, f)
  print('data.json written to %s.' % osp.join(args.output_dir, 'data.json'))

  # save h5
  f = h5py.File(osp.join(args.output_dir, 'data.h5'), 'w')
  f.create_dataset('encoded_questions', data=np.asarray(encoded_questions), dtype=np.int32)  
  f.create_dataset('encoded_answers', data=np.asarray(encoded_answers), dtype=np.int32)
  f.close()
  print('data.h5 written to ', osp.join(args.output_dir, 'data.h5'))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--question_json', default='data/question-gen-outputs/questions_mt_paths_nearby_source_best_view_program.json', help='bigger question set')
  parser.add_argument('--shortest_path_dir', default='data/shortest-paths-mt/nearby_source_best_view', type=str, help='directory saving sampled paths: qid.json')
  parser.add_argument('--target_obj_bestview_pos_dir', type=str, default='data/target-obj-bestview-pos')
  parser.add_argument('--path_feats_dir', default='cache/path_feats', help='run extract_feats.py to get it.')
  parser.add_argument('--path_imgs_dir', default='cache/path_images', help='run generate_path_imgs.py to get it.')
  parser.add_argument('--room_meta_dir', default='cache/path_to_room_meta', help='run compute_meta_info.py to get it.')
  parser.add_argument('--vectors_file', default='data/glove/glove.6B.300d.txt')
  # options
  parser.add_argument('--iou_thresh', default=0.1, type=float, help='mininum iou of meaningful objects.')
  # output
  parser.add_argument('--output_dir', default='cache/prepro/reinforce')
  args = parser.parse_args()

  # run
  if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

  main(args)

