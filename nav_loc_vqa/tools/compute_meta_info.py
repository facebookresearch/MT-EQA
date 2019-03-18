"""
This code prepares meta-info for computing label for each frame along shortest-path.
We consider two conditions:
1) if current position is inside the target room.
2) if current frame contains meaningful objects.
1+2    -> label = 1
1 only -> label = unknown
not 1  -> label = 0

Thus in this code we compute 
1) [inroomDist for each position] along the path
2) [cls_to_iou for each frame] along the path

The computed meta_info is saved in cache/path_to_room_meta
Each hid.tid.tid contains {room_id: #paths of [{inroomDists, obj_to_iou}] }
"""
import h5py
import os
import os.path as osp
import sys
import json
import argparse
import numpy as np
import csv
import collections
from tqdm import tqdm
from pprint import pprint

import _init_paths
from house3d_thin import House3DThin


# room_to_objects
room_to_objects = {}
room_to_objects['bathroom'] = [ 
  'toilet', 'sink', 'shower', 'shelving', 'bathtub', 'rug', 'towel_rack', 
  'mirror', 'hanger', 'wall_lamp', 'switch', 'partition', 'wardrobe_cabinet', 'bookshelf', 'curtain',
  'washer', 'plant', 'trash_can', 'toy', 'vase', 'dresser', 'air_conditioner', 'dryer', 'heater', 
  'picture_frame', 'iron'
]
room_to_objects['kitchen'] = [
  'kitchen_cabinet', 'chair', 'hanging_kitchen_cabinet', 'refrigerator', 
  'shelving', 'microwave', 'kettle', 'coffee_machine', 'knife_rack', 'cutting_board', 'food_processor', 
  'glass', 'pan', 'plates', 'utensil_holder', 'trash_can', 'cup', 'water_dispenser', 'plant', 'range_hood', 
  'beer', 'fruit_bowl', 'dishwasher', 'range_oven'
  ]
room_to_objects['bedroom'] = [
  'wardrobe_cabinet', 'double_bed', 'single_bed', 'baby_bed', 'table_lamp', 
  'rug', 'television', 'picture_frame', 'shelving', 'dressing_table', 'dresser', 'desk', 'plant', 
  'switch', 'laptop','books', 'tv_stand', 'floor_lamp', 'air_conditioner', 'ottoman', 'armchair', 
  'mirror', 'office_chair', 'wall_lamp', 'computer'
  ]
room_to_objects['living room'] = [
  'sofa', 'coffee_table', 'plant', 'television', 'chandelier', 'rug', 
  'tv_stand', 'picture_frame', 'loudspeaker', 'armchair', 'vase', 'floor_lamp', 'shelving', 'ottoman', 
  'wall_lamp', 'books', 'fireplace', 'stereo_set', 'air_conditioner', 'playstation', 'clock', 
  'fish_tank', 'table_lamp', 'piano', 'xbox'
  ]
room_to_objects['dining room'] = [
  'chair', 'dining_table', 'plant', 'cup', 'plates', 'armchair', 'glass',
  'rug', 'candle', 'picture_frame', 'vase', 'wall_lamp', 'floor_lamp', 'air_conditioner', 'fruit_bowl', 
  'bottle', 'wardrobe_cabinet', 'dresser', 'television', 'ceiling_fan', 'ottoman', 'coffee_table'
  ]
room_to_objects['gym'] = [
  'gym_equipment', 'game_table', 'mirror', 'plant', 'loudspeaker', 'curtain', 'toy', 'outdoor_seating', 
  'air_conditioner', 'television', 'floor_lamp', 'picture_frame', 'ceiling_fan', 'washer', 
  'vacuum_cleaner', 'iron', 'basketball_hoop', 'piano', 'stereo_set', 'ironing_board', 'partition', 
  'wall_lamp'
  ]
room_to_objects['garage'] = [
  'shelving', 'car', 'garage_door', 'motorcycle', 'column', 'chandelier', 'wall_lamp', 'plant', 
  'washer', 'outdoor_lamp', 'wardrobe_cabinet', 'ceiling_fan', 'chair', 'vacuum_cleaner', 'gym_equipment', 
  'partition', 'shoes_cabinet', 'iron', 'ironing_board', 'ottoman', 'floor_lamp', 'television', 
  'loudspeaker'
  ]
room_to_objects['balcony'] = [
  'fence', 'chair', 'outdoor_lamp', 'wall_lamp', 'coffee_table', 'grill', 'rug', 'door', 'trash_can',
  'window', 'curtain', 'ironing_board', 'ottoman'
  ]

# global ref_mask (won't be changed)
REF_MASK = np.zeros((224, 224), np.int8)
REF_MASK[int(0.25*224):int(0.85*224), int(0.25*224):int(0.75*224)] = 1

# fine_class -> ix, rgb
def get_semantic_classes(semantic_color_file):
    """
    cls_to_ix, cls_to_rgb 
    """
    cls_to_ix, cls_to_rgb = {}, {}
    with open(semantic_color_file) as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            fine_cat = row['name']
            cls_to_rgb[fine_cat] = (row['r'], row['g'], row['b'])
            cls_to_ix[fine_cat] = i
    return cls_to_ix, cls_to_rgb 

# iou computation
def compute_iou(cand_mask, ref_mask=None):
  """
  Given (h, w) cand_mask, we wanna our ref_mask to be in the center of image,
  with [0.25h:0.75h, 0.25w:0.75w] occupied.
  """
  if ref_mask is None:
    h, w = cand_mask.shape[0], cand_mask.shape[1]
    ref_mask = np.zeros((h,w), np.int8)
    ref_mask[int(0.25*h):int(0.85*h), int(0.25*w):int(0.75*w)] = 1
  inter = (cand_mask > 0) & (ref_mask > 0)
  union = (cand_mask > 0) | (ref_mask > 0)
  iou = inter.sum() / (union.sum() + 1e-5)
  return iou

# meta_info given one position
def compute_meta_info(h3d_thin, room_to_objects, cls_to_rgb, room_id, pos, ego_sem):
  """
  Inputs:
  - h3d_thin        : house3d_thin instance
  - room_to_objects : top-K objects for each room type
  - cls_to_rgb      : fine_class to rgb
  - room_id         : target room_id
  - pos             : (x, y, z, yaw)
  - ego_sem         : (h, w, 3) egocentric semantics map at [pos]
  Outputs:
  - dist            : inroomDist
  - obj_to_iou      : ious of each object (specific to this room type)
  """
  room = h3d_thin.rooms[room_id]
  assert room_id == room['id']
  h3d_thin.set_target_room(room)
  connMap, _, inroomDists, _ = h3d_thin.connMapDict[room_id]
  # inroomDist at pos
  x, y = h3d_thin.to_grid(pos[0], pos[2])
  dist = float(inroomDists[x, y])
  # iou of frequent objects for this room type
  room_type = h3d_thin.rooms[room_id]['type']
  room_name = h3d_thin.convert_room_name(room_type)
  top_objects = room_to_objects[room_name]
  obj_to_iou = {}
  for obj_cls in top_objects:
    c = np.array(cls_to_rgb[obj_cls]).astype(np.uint8)  # (3, )
    obj_seg = np.all(ego_sem == c, axis=2).astype(np.uint8)  # (224, 224)
    obj_to_iou[obj_cls] = float(compute_iou(obj_seg, ref_mask=REF_MASK))
  # return 
  return dist, obj_to_iou

def main(args):
  """
  For each question, we have program [nav_room] where we  could compute cls_to_iou and inroomDist for it.
  "inroomDist" and "cls_to_iou" are computed from each path and room in hid.tid_tid
  i.e., hid.tid_tid --> room_id --> #paths of {dists, list of cls_to_iou}
  """
  # set up h3d_thin instance
  h3d_thin = House3DThin(args.suncg_data_dir, args.house3d_metadata_dir, args.target_obj_conn_maps)

  # set up model_id --> ix/rgb
  color_file = osp.join(args.house3d_metadata_dir, 'colormap_fine.csv')
  cls_to_ix, cls_to_rgb = get_semantic_classes(color_file)

  # load questions = [{question, answer, type, bbox, id, house, programs, etc}]
  questions = json.load(open(args.input_json))
  hids = sorted(list(set(qn['house'] for qn in questions)))  # maintain hids order for no reason
  hid_to_qns = collections.defaultdict(list)
  for qn in questions:
    hid_to_qns[qn['house']].append(qn)
 
  # run
  hindex = 0
  for hid in tqdm(hids):
    qns = hid_to_qns[hid]

    # parse this house
    h3d_thin.parse(hid)

    # run
    for qindex, qn in enumerate(qns):
      # path_info = {num_paths, ego_rgbk, cube_rgbk, key_ixsk, positionsk, actionsk, etc}
      path_name = qn['path']

      # output_file saving {room_id: [{dist, obj_to_iou}]}
      output_file = osp.join(args.output_dir, path_name+'.json')
      if osp.exists(output_file):
        continue

      path_info = h5py.File(osp.join(args.path_images_dir, path_name+'.h5'), 'r')
      num_paths = path_info['num_paths'][0]
      room_ids = [pg['id'][0] for pg in qn['program'] if pg['function'] == 'nav_room']  # room_ids
      if len(room_ids) == 0: 
        continue

      room_meta = {}
      rindex = 0
      for room_id in room_ids:
        room_meta[room_id] = []

        # room_name        
        room_name = h3d_thin.convert_room_name(h3d_thin.rooms[room_id]['type'])
        assert room_name in qn['question']
        # meta_info along each path for this room
        for pix in range(num_paths):
          meta_info = collections.defaultdict(list)
          path_positions = path_info['positions%s'%pix] # (path_len, 4), [x, y, z, yaw]
          path_ego_sem = path_info['ego_sem%s'%pix]     # (path_len, 224, 224, 3)
          path_len = path_ego_sem.shape[0]
          for step in range(path_len):
            dist, obj_to_iou = compute_meta_info(h3d_thin, room_to_objects, cls_to_rgb, room_id, 
                path_positions[step], path_ego_sem[step])
            meta_info['inroomDists'].append(dist)
            meta_info['obj_to_iou'].append(obj_to_iou)

          room_meta[room_id].append(meta_info)

        print('hid(%s/%s) qid[%s](%s/%s) room_id[%s](%s/%s) done.' % \
          (hindex+1, len(hids), qn['id'], qindex+1, len(qns), room_id, rindex+1, len(room_ids)))
        rindex += 1

      # save room_meta for this path_file 
      with open(output_file, 'w') as f:
        json.dump(room_meta, f)
      print('saved to %s' % output_file)

    hindex += 1


if __name__ == '__main__':

  parser = argparse.ArgumentParser()  
  # input and output
  parser.add_argument('--input_json', default='data/question-gen-outputs/questions_mt_paths_nearby_source_best_view_program.json', help='filtered(sampled) questions json')
  parser.add_argument('--path_images_dir', default='cache/path_images', type=str, help='navigation images directory')
  parser.add_argument('--output_dir', default='cache/path_to_room_meta', type=str)
  # house3d_thin 
  parser.add_argument('--suncg_data_dir', default='data/SUNCGdata')
  parser.add_argument('--house3d_metadata_dir', type=str, default='pyutils/House3D/House3D/metadata')
  parser.add_argument('--target_obj_conn_maps', type=str, default='data/target-obj-conn-maps')
  args = parser.parse_args()

  if not osp.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  main(args)
