# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a thin version of customized house3dutils.
We won't use renderer here.
"""
import csv
import copy
import sys
import time
import pickle
import os
import os.path as osp
import json
import numpy as np

class House3DThin():

  def __init__(self, dataDir, house3d_metadata_dir, target_obj_conn_map_dir):
    # house data directory
    self.dataDir = dataDir
    # model_id --> fine_class
    csvFile = csv.reader(open(os.path.join(house3d_metadata_dir, 'ModelCategoryMapping.csv'), 'r'))
    headers = next(csvFile)
    self.modelCategoryMapping = {}
    for row in csvFile:
      self.modelCategoryMapping[row[headers.index('model_id')]] = {
        headers[x]: row[x]
        for x in range(2, len(headers))  # 0 is index, 1 is model_id
      }
    # each pickle = [connMap, connectedCoors, inroomDist, maxConnDist] 
    self.target_obj_conn_map_dir = target_obj_conn_map_dir


  def parse(self, house_id, levelsToExplore=[0], ColideRes=500):
    # load house info
    json_path = osp.join(self.dataDir, 'house', house_id, 'house.json')
    data = json.load(open(json_path, 'r'))

    # construct rooms and objects, both are dicts
    rooms, objects = {}, {}
    for i in levelsToExplore:
      for j in range(len(data['levels'][i]['nodes'])):
        assert data['levels'][i]['nodes'][j]['type'] != 'Box'
        # Rooms
        if data['levels'][i]['nodes'][j]['type'] == 'Room':
          if 'roomTypes' not in data['levels'][i]['nodes'][j]:
            continue
          # Can rooms have more than one type?
          # Yes, they can; just found ['Living_Room', 'Dining_Room', 'Kitchen']
          roomType = [
            # ' '.join(x.lower().split('_'))
            x.lower() for x in data['levels'][i]['nodes'][j]['roomTypes']
          ]
          nodes = data['levels'][i]['nodes'][j]['nodeIndices'] \
            if 'nodeIndices' in data['levels'][i]['nodes'][j] else []
          rooms[data['levels'][i]['nodes'][j]['id']] = {
            'id': data['levels'][i]['nodes'][j]['id'], 
            'type': roomType,
            'bbox': data['levels'][i]['nodes'][j]['bbox'],
            'nodes': nodes,
            'model_id': data['levels'][i]['nodes'][j]['modelId']
          }
        # Objects
        elif data['levels'][i]['nodes'][j]['type'] == 'Object':
          if 'materials' not in data['levels'][i]['nodes'][j]:
            material = []
          else:
            material = data['levels'][i]['nodes'][j]['materials']
          objects[data['levels'][i]['nodes'][j]['id']] = {
            'id': data['levels'][i]['nodes'][j]['id'],
            'model_id': data['levels'][i]['nodes'][j]['modelId'],
            'fine_class': self.modelCategoryMapping[data['levels'][i]['nodes'][j]['modelId']]['fine_grained_class'],
            'coarse_class': self.modelCategoryMapping[data['levels'][i]['nodes'][j]['modelId']]['coarse_grained_class'],
            'bbox': data['levels'][i]['nodes'][j]['bbox'],
            'mat': material
          }
    room_to_object_ids, object_to_room_id = {}, {}
    for room in rooms.values():
      room_id = room['id']
      for node_idx in room['nodes']:
        node = data['levels'][0]['nodes'][node_idx]
        assert node['type'] == 'Object'
        object_id = node['id']
        room_to_object_ids[room_id] = room_to_object_ids.get(room_id, []) + [object_id]
        object_to_room_id[object_id] = room_id

    # register
    self.house_id = house_id
    self.rooms = rooms
    self.objects = objects 
    self.room_to_object_ids = room_to_object_ids 
    self.object_to_room_id = object_to_room_id
    self.connMapDict = {}

    # more info
    level = data['levels'][0]
    self.level = level
    self.L_min_coor = _L_lo = np.array(level['bbox']['min'])
    self.L_lo = min(_L_lo[0], _L_lo[2])
    self.L_max_coor = _L_hi = np.array(level['bbox']['max'])
    self.L_hi = max(_L_hi[0], _L_hi[2])
    self.L_det = self.L_hi - self.L_lo       # longer edge of the house
    self.n_row = ColideRes                   # 2d map resolution for collision check
    self.grid_det = self.L_det / self.n_row  # length per grid

  def set_target_room(self, room):
    """
    Set current target as referred room.
    """
    room_id = room['id']
    # load cached file
    if room_id not in self.connMapDict:
      f = osp.join(self.target_obj_conn_map_dir, self.house_id+'_'+room_id+'.pkl')
      assert osp.exists(f), '%s does not exist.' % f
      connMap, connectedCoors, inroomDist, maxConnDist = pickle.load(open(f, 'rb'))
      self.connMapDict[room_id] = (connMap, connectedCoors, inroomDist, maxConnDist)


  def convert_room_name(self, room_types):
    """
    Revised (copy & paste) from engine_v2's roomEntity,
    converting room_types to room_name
    """
    def my_convert(name):
      return ' '.join(name.lower().split('_'))
    translations = {
      'toilet': 'bathroom',
      'guest room': 'bedroom',
      'child room': 'bedroom',
    }
    names = []
    for x in room_types:
      x = my_convert(x)
      if x in translations:
        names += [translations[x]]
      else:
        names += [x]
    names.sort(key=str.lower)
    return names[0]

  ###### more utility function ######
  def to_grid(self, x, y, n_row=None):
    """
    Convert the true-scale coordinate in SUNCG dataset to grid location
    """
    if n_row is None: n_row = self.n_row
    tiny = 1e-9
    tx = np.floor((x - self.L_lo) / self.L_det * n_row + tiny)
    ty = np.floor((y - self.L_lo) / self.L_det * n_row + tiny)
    return int(tx), int(ty)

  def rescale(self,x1,y1,x2,y2,n_row=None):
    if n_row is None: n_row = self.n_row
    tiny = 1e-9
    tx1 = np.floor((x1 - self.L_lo) / self.L_det * n_row+tiny)
    ty1 = np.floor((y1 - self.L_lo) / self.L_det * n_row+tiny)
    tx2 = np.floor((x2 - self.L_lo) / self.L_det * n_row+tiny)
    ty2 = np.floor((y2 - self.L_lo) / self.L_det * n_row+tiny)
    return int(tx1),int(ty1),int(tx2),int(ty2)

  def to_coor(self, x, y, shft=False):
    """
    Convert grid location to SUNCG dataset continuous coordinate (the grid center will be returned when shft is True)
    """
    tx, ty = x * self.grid_det + self.L_lo, y * self.grid_det + self.L_lo
    if shft:
        tx += 0.5 * self.grid_det
        ty += 0.5 * self.grid_det
    return tx, ty