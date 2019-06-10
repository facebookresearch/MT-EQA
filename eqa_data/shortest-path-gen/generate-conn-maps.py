# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This code generates:
1) movMap, obsMap, levelConnMap for each house.
2) connMap, connectedCoors, inroomDist, maxConnDist for each object and room, mentioned by the question.
"""
import argparse
import os, sys, json
import os.path as osp
from tqdm import tqdm
from pprint import pprint

from queue import Queue
from threading import Thread, Lock

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../pyutils/House3D'))
from House3D import objrender, Environment, load_config
from House3D.core import local_create_house

sys.path.insert(0, osp.join(this_dir, '../pyutils'))
from house3d import House3DUtils


def main(house_to_qns, args):

  # config
  cfg = {}
  cfg['colorFile'] = osp.join(args.house3d_metadata_dir, 'colormap_fine.csv')
  cfg['roomTargetFile'] = osp.join(args.house3d_metadata_dir, 'room_target_object_map.csv')
  cfg['modelCategoryFile'] = osp.join(args.house3d_metadata_dir, 'ModelCategoryMapping.csv')
  cfg['prefix'] = osp.join(args.suncg_data_dir, 'house')
  for d in cfg.values():
    assert osp.exists(d), d

  invalid = []

  # data pool
  lock = Lock()
  q = Queue()
  for i, house_id in enumerate(list(house_to_qns.keys())):
    qns = house_to_qns[house_id]
    q.put((i, house_id, qns))

  # worker
  def worker():
    api_thread = objrender.RenderAPIThread(w=224, h=224)
    while True:
      i, house_id, qns = q.get()
      print('Processing house[%s] %s/%s' % (house_id, i+1, len(house_to_qns)))
      # api_thread = objrender.RenderAPIThread(w=224, h=224)
      env = Environment(api_thread, house_id, cfg, ColideRes=args.colide_resolution)
      build_graph = True
      if osp.exists(osp.join(args.graph_dir, env.house.house['id']+'.pkl')):
        build_graph = False
      h3d = House3DUtils(env, build_graph=build_graph, graph_dir=args.graph_dir, 
              target_obj_conn_map_dir=args.target_obj_conn_map_dir)

      # make connMap for each qn
      for qn in qns:
        try:
          if 'object' in qn['type']:
            # object_color_compare_xroom(inroom), object_size_compare_xroom(inroom), object_dist_compare_inroom
            for bbox in qn['bbox']:
              assert bbox['type'] == 'object'
              obj = h3d.objects[bbox['id']]
              h3d.set_target_object(obj)  # connMap computed and saved
              # we also store its room connMap
              room = h3d.rooms[bbox['room_id']]
              h3d.set_target_room(room)   # connMap computed and saved
          
          elif qn['type'] in ['room_size_compare']:
            for bbox in qn['bbox']:
              assert 'room' in qn['type']
              room = h3d.rooms[bbox['id']]
              h3d.set_target_room(room)   # connMap computed and saved

        except:
          print('Error found for qn[%s]' % qn['id'])
          invalid.append(qn['id'])

      q.task_done()

  for i in range(args.num_workers):
    t = Thread(target=worker)
    t.daemon = True
    t.start()
  q.join()

  return invalid


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--question_json', default='cache/question-gen-outputs/questions_pruned_mt.json')
  parser.add_argument('--house3d_metadata_dir', default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--suncg_data_dir', default='data/SUNCGdata')
  parser.add_argument('--graph_dir', default='cache/3d-graphs', help='directory for saving graphs')
  parser.add_argument('--target_obj_conn_map_dir', default='cache/target-obj-conn-maps')
  parser.add_argument('--colide_resolution', default=500, type=int, help='house grid resolution')
  parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
  parser.add_argument('--invalid_json', default='cache/invalid_conn_maps.json', help='storing question ids that cannot produce connMap.')
  parser.add_argument('--output_question_json', default='cache/question-gen-outputs/questions_pruned_mt_with_conn.json')
  args = parser.parse_args()

  # load questions = [{bbox, id, house, question, answer, type}]
  questions = json.load(open(args.question_json, 'r'))
  house_to_qns = {}
  for qn in questions:
    house_id = qn['house']
    if house_id not in house_to_qns: house_to_qns[house_id] = []
    house_to_qns[house_id].append(qn)

  # cache conn_maps
  if not osp.isdir(args.target_obj_conn_map_dir):
    os.makedirs(args.target_obj_conn_map_dir)
  if not osp.isdir(args.graph_dir):
    os.makedirs(args.graph_dir)

  # run 
  invalid = main(house_to_qns, args)

  # invalid
  with open(args.invalid_json, 'w') as f:
    json.dump(invalid, f)
  print('%s invalid questions found, saved in %s' % (len(invalid), args.invalid_json))
  print('Done.')

  # update the pruned questions
  qns = [qn for qn in questions if qn['id'] not in invalid]
  with open(args.output_question_json, 'w') as f:
    json.dump(qns, f)
  print('%s (out of %s) questions with conn_map are saved in %s.' % (len(qns), len(questions), args.output_question_json))
