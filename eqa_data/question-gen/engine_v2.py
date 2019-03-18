import csv
import random
import argparse
import operator
import numpy as np
import os, sys, json
import os.path as osp
from tqdm import tqdm
from scipy import spatial
from numpy.random import choice
from random import shuffle

from house_parse import HouseParse
from question_string_builder import QuestionStringBuilder

from nltk.stem import WordNetLemmatizer

# for reproducibility
random.seed(0)
np.random.seed(0)

# hard thresholding
OBJECT_RATIO_THRESH = 1.5
ROOM_RATIO_TRHESH = 1.5
DIST_LOW_THRESH, DIST_HIGH_THRESH = 0.2, 1.2

blacklist_objects = {
  'color': [
    'container', 'containers', 'stationary_container', 'candle',
    'coffee_table', 'column', 'door', 'floor_lamp', 'mirror', 'person',
    'rug', 'sofa', 'stairs', 'outdoor_seating', 'kitchen_cabinet',
    'kitchen_set', 'switch', 'storage_bench', 'table_lamp', 'vase',
    'candle', 'roof', 'stand', 'beer', 'chair', 'chandelier',
    'coffee_table', 'column', 'trinket', 'grill', 'book', 'books',
    'curtain', 'desk', 'door', 'floor_lamp', 'hanger', 'workplace',
    'glass', 'headstone', 'kitchen_set', 'mirror', 'plant', 'shelving',
    'place_setting', 'ceiling_fan', 'stairs', 'storage_bench',
    'switch', 'table_lamp', 'vase', 'decoration', 'coffin',
    'wardrobe_cabinet', 'window', 'pet', 'cup', 'arch',
    'household_appliance'
  ],
  'dist_compare': [
    'column', 'door', 'switch', 'person', 'household_appliance',
    'decoration', 'trinket', 'place_setting', 'coffin', 'book'
    'cup', 'chandelier', 'arch', 'pet', 'container', 'containers',
    'stationary_container', 'shelving', 'stand', 'kitchen_set',
    'books', 'ceiling_fan', 'workplace', 'glass', 'grill', 'roof',
    'outdoor_seating', 'kitchen_cabinet', 'headstone', 'beer'
  ],
  'object_compare': [
    'container', 'containers', 'stationary_container', 'candle',
    'coffee_table', 'column', 'door', 'floor_lamp', 'mirror', 'person',
    'rug', 'sofa', 'stairs', 'outdoor_seating', 'kitchen_cabinet',
    'kitchen_set', 'switch', 'storage_bench', 'table_lamp', 'vase',
    'candle', 'roof', 'stand', 'beer', 'chair', 'chandelier',
    'coffee_table', 'column', 'trinket', 'grill', 'book', 'books',
    'curtain', 'desk', 'door', 'floor_lamp', 'hanger', 'workplace',
    'glass', 'headstone', 'kitchen_set', 'mirror', 'plant', 'shelving',
    'place_setting', 'ceiling_fan', 'stairs', 'storage_bench',
    'switch', 'table_lamp', 'vase', 'decoration', 'coffin',
    'wardrobe_cabinet', 'window', 'pet', 'cup', 'arch',
    'household_appliance', 'garage_door'
  ]
}

blacklist_rooms = [
  'loggia', 'storage', 'guest room', 'hallway', 'wardrobe', 'hall',
  'boiler room', 'terrace', 'room', 'entryway', 'aeration', 'lobby',
  'office', 'freight elevator', 'passenger elevator'
]

class roomEntity():
  translations = {
    'toilet': 'bathroom',
    'guest room': 'bedroom',
    'child room': 'bedroom',
  }

  def __init__(self, name, bbox, meta):
    self.name = list(
      set([
        self.translations[str(x)]
        if str(x) in self.translations else str(x) for x in name
      ]))
    self.id = meta['id']
    self.bbox = bbox
    self.meta = meta  # {id, type, valid, modelId, nodeIndices, roomTypes, bbox}
    self.type = 'room'
    self.name.sort(key=str.lower)
    self.entities = self.objects = []

  def addObject(self, object_ent):
    self.objects.append(object_ent)

  def isValid(self):
    return len(self.objects) != 0


class objectEntity():
  translations = {
    'bread': 'food',
    'hanging_kitchen_cabinet': 'kitchen_cabinet',
    'teapot': 'kettle',
    'coffee_kettle': 'kettle',
    'range_hood_with_cabinet': 'range_hood',
    'dining_table': 'table',
    'coffee_table': 'table',
    'game_table': 'table',
    'office_chair': 'chair',
    'bench_chair': 'chair',
    'chair_set': 'chair',
    'armchair': 'chair',
    'fishbowl': 'fish_tank/bowl',
    'fish_tank': 'fish_tank/bowl',
    'single_bed': 'bed',
    'double_bed': 'bed',
    'baby_bed': 'bed'
  }
  def __init__(self, name, bbox, meta, obj_id, room_id):
    if name in self.translations: self.name = self.translations[name]
    else: self.name = name
    self.bbox = bbox
    self.meta = meta
    self.type = 'object'
    self.id = obj_id
    self.room_id = room_id
    assert self.id == meta['id']  # check if named_id equals to provided id
    self.entities = self.rooms = []

  def addRoom(self, room_ent):
    self.rooms.append(room_ent)

  def isValid(self):
    return len(self.rooms) != 0


class Engine():
  '''
    Templates and functional forms.
  '''
  template_defs = {
    'object_dist_compare': [
        'filter.rooms', 'unique.rooms', 'filter.objects', 'unique.objects', 'blacklist.dist_compare', 
        'object_dist_pair', 'query.object_dist_compare'
    ],
    'object_color_compare': [
        'filter.rooms', 'unique.rooms', 'filter.objects', 'unique.objects', 'blacklist.object_compare', 
        'object_color_pair', 'query.object_color_compare'
    ],
    'object_size_compare' : [
        'filter.rooms', 'unique.rooms', 'filter.objects', 'unique.objects', 'blacklist.object_compare', 
        'object_size_pair', 'query.object_size_compare'
    ],
    'room_size_compare': [
        'filter.rooms', 'unique.rooms', 'room_size_pair', 'query.room_size_compare'
    ],
    'room_dist_compare': [
        'filter.rooms', 'unique.rooms', 'room_dist_pair', 'query.room_dist_compare'
    ],
  }
  templates = {
    # object distance comparisons
    'object_closer_inroom': 'is the <OBJ> closer to the <OBJ> than to the <OBJ> in the <ROOM>?',
    'object_farther_inroom': 'is the <OBJ> farther from the <OBJ> than from the <OBJ> in the <ROOM>?',
    # object color comparison
    'object_color_compare_inroom': 'does the <OBJ> have same color as the <OBJ> in the <ROOM>?',
    'object_color_compare_xroom': 'does the <OBJ1> in the <ROOM1> have same color as the <OBJ2> in the <ROOM2>?',
    # object size comparison
    'object_bigger_inroom': 'is the <OBJ> bigger than <OBJ> in the <ROOM>?',
    'object_smaller_inroom': 'is the <OBJ> smaller than <OBJ> in the <ROOM>?',
    'object_bigger_xroom': 'is the <OBJ1> in the <ROOM1> bigger than <OBJ2> in the <ROOM2>?',
    'object_smaller_xroom': 'is the <OBJ1> in the <ROOM1> smaller than <OBJ2> in the <ROOM2>?',
    # room size comparison
    'room_bigger': 'is the <ROOM1> bigger than the <ROOM2>?',
    'room_smaller': 'is the <ROOM1> smaller than the <ROOM2>?',
    # room distance comparison
    'room_closer': 'is the <ROOM> closer to <ROOM> than to the <ROOM>?',
    'room_farther': 'is the <ROOM> farther from <ROOM> than from the <ROOM>?',
  }

  def __init__(self, object_counts_by_room_file, env_obj_colors_file, debug=False):
    self.template_fns = {
      'filter': self.filter,
      'unique': self.unique,
      'query': self.query,
      'blacklist': self.blacklist,
      'thresholdSize': self.thresholdSize,
      'object_dist_pair': self.objectDistPair,
      'object_color_pair': self.objectColorPair,
      'object_size_pair': self.objectSizePair,
      'room_size_pair': self.roomSizePair,
      'room_dist_pair': self.roomDistPair,
    }
    self.query_fns = {
      'query_object_dist_compare': self.queryObjectDistCompare,
      'query_object_color_compare': self.queryObjectColorCompare,
      'query_object_size_compare': self.queryObjectSizeCompare,
      'query_room_size_compare': self.queryRoomSizeCompare,
      'query_room_dist_compare': self.queryRoomDistCompare,
    }
    self.blacklist_objects = blacklist_objects
    self.blacklist_rooms = blacklist_rooms
    self.use_threshold_size = True
    self.use_blacklist = True
    self.debug = debug
    self.ent_queue = None
    self.q_str_builder = QuestionStringBuilder()
    self.q_obj_builder = self.questionObjectBuilder

    # update
    if os.path.isfile(object_counts_by_room_file) == True:
      self.global_obj_by_room = json.load(open(object_counts_by_room_file, 'r'))
      self.negative_exists = {}
    else:
      print('Not loading env_lists/800env_object_counts_by_room.json')

    # load colors
    assert osp.isfile(env_obj_colors_file)
    self.env_obj_color_map = json.load(open(env_obj_colors_file, 'r'))

  def cacheHouse(self, Hp):
    """
    Get objects and rooms info for current parsed house.
    """
    self.house = Hp
    self.entities = {'rooms': [], 'objects': []}
    for i in self.house.rooms:
      room = roomEntity(i['type'], i['bbox'], i)
      for j in room.meta['nodes']:
        obj = objectEntity(
          self.house.objects['0_' + str(j)]['fine_class'],
          self.house.objects['0_' + str(j)]['bbox'],
          self.house.objects['0_' + str(j)],
          obj_id='0_' + str(j),
          room_id=room.id)
        room.addObject(obj)
        obj.addRoom(room)
        self.entities['objects'].append(obj)
      self.entities['rooms'].append(room)
    self.isValid()

  def isValid(self):
    # print('checking validity...')
    for i in self.entities['rooms']:
      if i.isValid() == False and self.debug == True:
        print('ERROR', i.meta)
        continue
    for i in self.entities['objects']:
      if i.isValid() == False and self.debug == True:
        print('ERROR', i.meta)
        continue

  def clearQueue(self):
    self.ent_queue = None

  def remain_single_name_rooms(self):
    """filter those elements with no/multiple room names."""
    ent = self.ent_queue['elements']
    if self.ent_queue['type'] == 'objects':
      self.ent_queue['elements'] = [x for x in ent if len(x.rooms) == 1 and len(x.rooms[0].name) == 1]
    if self.ent_queue['type'] == 'rooms':
      self.ent_queue['elements'] = [x for x in ent if len(x.name) == 1]
  
  def executeFn(self, template):
    for i in template:
      if '.' in i:
        _ = i.split('.')
        fn = _[0]
        param = _[1]
      else:
        fn = i
        param = None
      res = self.template_fns[fn](param)

    if isinstance(res, dict):
      return res
    else:
      # return unique questions only
      return list({x['question']: x for x in res}.values())

  def getVolume(self, bbox):
    # return volume of bbox
    return (bbox['max'][0]-bbox['min'][0]) * (bbox['max'][1]-bbox['min'][1]) * (bbox['max'][2]-bbox['min'][2])
  
  def getArea(self, bbox):
    # return 2D bird-view area
    return (bbox['max'][0]-bbox['min'][0]) * (bbox['max'][2]-bbox['min'][2])
  
  def thresholdSize(self, *args):
    assert self.ent_queue != None
    assert self.ent_queue['type'] == 'objects'
    ent = self.ent_queue
    sizes = [self.getVolume(x.bbox) for x in ent['elements']]
    idx = [i for i, v in enumerate(sizes) if v < 0.0005]
    for i in idx[::-1]:
      del ent['elements'][i]
    self.ent_queue = ent
    return self.ent_queue
  
  def blacklist(self, *args):
    assert self.ent_queue != None
    ent = self.ent_queue
    if ent['type'] == 'objects':
      template = args[0]
      names = [x.name for x in ent['elements']]
      idx = [i for i, v in enumerate(names) if v in self.blacklist_objects[template]]
      for i in idx[::-1]:
        del ent['elements'][i]
    elif ent['type'] == 'rooms':
      names = [x.name for x in ent['elements']]
      idx = [
        i for i, v in enumerate([
          any([k for k in x if k in self.blacklist_rooms])
          for x in names
        ]) if v == True
      ]
      for i in idx[::-1]:
        del ent['elements'][i]
    self.ent_queue = ent
    return self.ent_queue
  
  def filter(self, *args):
    """select object/rooms according to args[0] or self.ent_queue['type']"""
    # if ent_queue is empty, execute on parent env entitites
    if self.ent_queue == None:
      self.ent_queue = {'type': args[0], 'elements': self.entities[args[0]]}
    else:
      ent = self.ent_queue
      assert args[0] != ent['type']
      ent = {
        'type': args[0],
        'elements': [z for y in [x.entities for x in ent['elements']] for z in y]
      }
      self.ent_queue = ent

    # remove blacklisted rooms
    if self.ent_queue['type'] == 'rooms' and self.use_blacklist == True:
      self.ent_queue = self.blacklist()

    if self.ent_queue['type'] == 'objects' and self.use_threshold_size == True:
      self.ent_queue = self.thresholdSize()

    return self.ent_queue
  
  def unique(self, *args):
    """select those objects/rooms that occurs only once in this house"""
    assert self.ent_queue != None
    ent = self.ent_queue

    # unique based on either rooms or objects (only)
    if ent['type'] == 'objects':
      names = [x.name for x in ent['elements']]
      idx = [i for i, v in enumerate([names.count(x) for x in names]) if v != 1]
    elif ent['type'] == 'rooms':
      # for room = ['dining room', 'kitchen'], we count all appeared room names
      names = [name for x in ent['elements'] for name in x.name]
      idx = []
      for i, x in enumerate(ent['elements']):
        for name in x.name:
          if names.count(name) != 1:
            idx.append(i)
            break
    else:
      raise NotImplementedError

    for i in idx[::-1]:
      del ent['elements'][i]

    names = [x.name for x in ent['elements']]
    self.ent_queue = ent
    return self.ent_queue
  
  def query(self, *args):
    assert self.ent_queue != None
    ent = self.ent_queue
    return self.query_fns['query_' + args[0]](ent)

  """
  Returned ent_queue is list of (obj1, obj2, obj3, 'closer') and (obj3, obj2, obj1, 'farther')
  where d(obj1, obj2) < d(obj2, obj3)
  """
  # only works with objectEntities for now
  def objectDistPair(self, *args):
    self.remain_single_name_rooms()  # remove 0/multiple-name rooms
    ent = self.ent_queue
    assert ent['type'] == 'objects'
    h_low_threshold, h_high_threshold = DIST_LOW_THRESH, DIST_HIGH_THRESH
    pairwise_distances = self.house.getAllPairwiseDistances(ent['elements']) # list of [(obj1, obj2, distance)]
    updated_ent_queue = {'type': ent['type'], 'elements': []}
    for i in ent['elements']:
      sub_list = [
        x for x in pairwise_distances if x[0].meta['id'] == i.meta['id'] or x[1].meta['id'] == i.meta['id']
      ]
      sub_list = [
        x for x in sub_list if x[0].rooms[0].name == x[1].rooms[0].name
      ]
      far = [x for x in sub_list if x[2] >= h_high_threshold]
      close = [x for x in sub_list if x[2] <= h_low_threshold]
      if len(far) == 0 or len(close) == 0:
        continue
      for j in far:
        far_ent = 1 if j[0].name == i.name else 0
        for k in close:
          close_ent = 1 if k[0].name == i.name else 0
          updated_ent_queue['elements'].append([k[close_ent], i, j[far_ent], 'closer'])
          updated_ent_queue['elements'].append([j[far_ent], i, k[close_ent], 'farther'])
    self.ent_queue = updated_ent_queue
    return self.ent_queue

  def queryObjectDistCompare(self, ent):
    qns = []
    for i in ent['elements']:
      template = 'object_%s_inroom' % i[3]
      qns.append(self.q_obj_builder(template, i[:3], 'yes', 'object_dist_compare_inroom'))
      qns.append(self.q_obj_builder(template, i[:3][::-1], 'no', 'object_dist_compare_inroom'))
    return qns

  """
  Returned ent_queue is list of [(room1, room2, room3, farther/closer)]
  """
  def roomDistPair(self, *args):
    self.remain_single_name_rooms()  # remove 0/multiple-name rooms
    ent = self.ent_queue
    assert ent['type'] == 'rooms'
    h_low_threshold, h_high_threshold = 2, 8
    # TODO: replace geodesic distance with shortest path
    pairwise_distances = self.house.getAllPairwiseRoomDistances(ent['elements'])  # list of [(room1, room2, distance)]
    updated_ent_queue = {'type': ent['type'], 'elements': []}
    for i in ent['elements']:
      sub_list = [
        x for x in pairwise_distances if x[0].meta['id'] == i.meta['id'] or x[1].meta['id'] == i.meta['id']
      ]
      far = [x for x in sub_list if x[2] >= h_high_threshold]
      close = [x for x in sub_list if x[2] <= h_low_threshold]
      if len(far) == 0 or len(close) == 0:
        continue
      for j in far:
        far_ent = 1 if j[0].name == i.name else 0
        for k in close:
          close_ent = 1 if k[0].name == i.name else 0
          updated_ent_queue['elements'].append([k[close_ent], i, j[far_ent], 'closer'])
          updated_ent_queue['elements'].append([j[far_ent], i, k[close_ent], 'farther'])
    self.ent_queue = updated_ent_queue
    return self.ent_queue
  
  def queryRoomDistCompare(self, ent):
    qns = []
    for i in ent['elements']:
      template = 'room_%s' % i[3]
      qns.append(self.q_obj_builder(template, i[:3], 'yes', 'room_dist_compare'))
      qns.append(self.q_obj_builder(template, i[:3][::-1], 'no', 'room_dist_compare'))
    return qns

  """
  Returned ent_queue is list of (obj1, color1, obj2, color2)
  """
  def objectColorPair(self, *args):
    self.remain_single_name_rooms()  # remove 0/multiple-name rooms
    ent = self.ent_queue
    assert ent['type'] == 'objects'
    updated_ent_queue = {'type': ent['type'], 'elements': []}
    num_objects = len(ent['elements'])
    for i in range(num_objects):
      for j in range(num_objects):
        object_i, object_j = ent['elements'][i], ent['elements'][j]
        if object_i.id == object_j.id:
          continue
        if (self.house.id + '.' + object_i.id not in self.env_obj_color_map) or \
            (self.house.id + '.' + object_j.id not in self.env_obj_color_map):
          continue
        # get colors
        color_i = self.env_obj_color_map[self.house.id + '.' + object_i.id]
        color_j = self.env_obj_color_map[self.house.id + '.' + object_j.id]
        updated_ent_queue['elements'].append([object_i, color_i, object_j, color_j])
    self.ent_queue = updated_ent_queue
    return self.ent_queue

  def queryObjectColorCompare(self, ent):
    # ent = {type, elements: [(object1, color1, object2, color2)]}
    qns = []
    for obj1, color1, obj2, color2 in ent['elements']:
      rm = 'inroom' if obj1.rooms[0].name == obj2.rooms[0].name else 'xroom'
      template = 'object_color_compare_%s' % rm
      ans = 'yes' if color1 == color2 else 'no'
      q_type = 'object_color_compare_%s' % rm
      qns.append(self.q_obj_builder(template , [obj1, obj2], ans, q_type))
    return qns
  """
  Returned ent_queue is list of [(obj1, obj2, size_cmp)]
  """
  def objectSizePair(self, *args):
    self.remain_single_name_rooms()  # remove 0/multiple-name rooms
    RATIO_TRHESH = OBJECT_RATIO_THRESH
    ent = self.ent_queue
    assert ent['type'] == 'objects'
    updated_ent_queue = {'type': 'objects', 'elements': []}
    num_objects = len(ent['elements'])
    for i in range(num_objects):
      for j in range(num_objects):
        object_i, object_j = ent['elements'][i], ent['elements'][j]
        if object_i.id == object_j.id:
          continue
        # get 3D volume
        size_i = self.getVolume(object_i.bbox)
        size_j = self.getVolume(object_j.bbox)
        if max(size_i, size_j) > min(size_i, size_j) * RATIO_TRHESH:
          size_cmp = 'bigger' if size_i > size_j else 'smaller'
          updated_ent_queue['elements'].append([object_i, object_j, size_cmp])
    self.ent_queue = updated_ent_queue
    return self.ent_queue

  def queryObjectSizeCompare(self, ent):
    # ent = {type, elements: [(object1, object2, bigger/smaller)]}
    qns = []
    for obj1, obj2, size_cmp in ent['elements']:
      rm = 'inroom' if obj1.rooms[0].name==obj2.rooms[0].name else 'xroom'
      template = 'object_%s_%s' % (size_cmp, rm)
      q_type = 'object_size_compare_%s' % rm
      qns.append(self.q_obj_builder(template, [obj1, obj2], 'yes', q_type))
      qns.append(self.q_obj_builder(template, [obj2, obj1], 'no', q_type))
    return qns
  """
  Returned ent_queue is list of [(room1, room2, size_cmp)]
  """
  def roomSizePair(self, ent):
    self.remain_single_name_rooms()  # remove 0/multiple-name rooms
    RATIO_TRHESH = ROOM_RATIO_TRHESH
    ent = self.ent_queue
    assert ent['type'] == 'rooms'
    updated_ent_queue = {'type': 'rooms', 'elements': []}
    num_rooms = len(ent['elements'])
    for i in range(num_rooms):
      for j in range(num_rooms):
        room_i, room_j = ent['elements'][i], ent['elements'][j]
        if room_i.id == room_j.id: continue
        # get 2D bird-view area
        size_i = self.getArea(room_i.bbox)
        size_j = self.getArea(room_j.bbox)
        if max(size_i, size_j) > min(size_i, size_j) * RATIO_TRHESH:
          size_cmp = 'bigger' if size_i > size_j else 'smaller'
          updated_ent_queue['elements'].append([room_i, room_j, size_cmp])
    self.ent_queue = updated_ent_queue
    return self.ent_queue
  
  def queryRoomSizeCompare(self, ent):
    # ent = {type: rooms, elements: [(room1, room2, bigger/smaller)]}
    qns = []
    for room1, room2, size_cmp in ent['elements']:
      template = 'room_%s' % size_cmp
      q_type = 'room_size_compare' 
      qns.append(self.q_obj_builder(template, [room1, room2], 'yes', q_type))
      qns.append(self.q_obj_builder(template, [room2, room1], 'no', q_type))
    return qns

  """
  Question Builder
  """
  def questionObjectBuilder(self, template, q_ent, a_str, q_type=None):
    if q_type == None:
      q_type = template

    q_str = self.templates[template]
    bbox = []

    # object_dist_compare (we don't need xroom here)
    if q_type in ['object_dist_compare_inroom']:
      for ent in q_ent:
        q_str = self.q_str_builder.prepareString(q_str, ent.name, ent.rooms[0].name[0])
        bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name, 'room_id': ent.room_id, 'target': True})
      mat = {}

    # room_dist_compare
    if q_type == 'room_dist_compare':
      for ent in q_ent:
        q_str = self.q_str_builder.prepareString(q_str, '', ent.name[0])
        bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name[0], 'target': True})
      mat = {}

    # object_color_compare
    if q_type in ['object_color_compare_inroom', 'object_color_compare_xroom']:
      if 'inroom' in template:
        for ent in q_ent:
          q_str = self.q_str_builder.prepareString(q_str, ent.name, ent.rooms[0].name[0])
          color = self.env_obj_color_map[self.house.id + '.' + ent.id]
          bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name, 'color': color, 'room_id': ent.room_id, 'target': True})
      else:
        q_str = self.q_str_builder.prepareStringForTwo(q_str, q_ent[0].name, q_ent[1].name,
                  q_ent[0].rooms[0].name[0], q_ent[1].rooms[0].name[0])
        for ent in q_ent:
          color = self.env_obj_color_map[self.house.id + '.' + ent.id]
          bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name, 'color': color, 'room_id': ent.room_id, 'target': True})
      mat = {}

    # object_size_compare 
    if q_type in ['object_size_compare_inroom', 'object_size_compare_xroom']:
      if 'inroom' in template:
        for ent in q_ent:
          q_str = self.q_str_builder.prepareString(q_str, ent.name, ent.rooms[0].name[0])
          size = self.getVolume(ent.bbox)
          bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name, 'size': size, 'room_id': ent.room_id, 'target': True})
      else:
        q_str = self.q_str_builder.prepareStringForTwo(q_str, q_ent[0].name, q_ent[1].name,
                  q_ent[0].rooms[0].name[0], q_ent[1].rooms[0].name[0])
        for ent in q_ent:
          size = self.getVolume(ent.bbox)
          bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name, 'size': size, 'room_id': ent.room_id, 'target': True})
      mat = {}
    
    # room_size_compare
    if q_type == 'room_size_compare':
      q_str = self.q_str_builder.prepareStringForTwo(q_str, '', '', q_ent[0].name[0], q_ent[1].name[0])
      for ent in q_ent:
        size = self.getArea(ent.bbox)
        bbox.append({'id': ent.id, 'type': ent.type, 'box': ent.bbox, 'name': ent.name[0], 'size': size, 'target': True})
      mat = {}
        
    return {
      'question': q_str,
      'answer': a_str,
      'type': q_type,
      'meta': mat,
      'bbox': bbox
    }



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataDir', default='../data', help='Data directory')
  parser.add_argument('--dataJson', default='eqa_v1.json', help='questions and splits')
  parser.add_argument('--HouseApiDir', default='../pyutils/House3D', help='house3d api dir')
  parser.add_argument('--cacheDir', default='../cache/question-gen-outputs', help='directory for saving generated questions')
  parser.add_argument('--outputJson', default='questions_from_engine_v2.json', help='output json file')
  parser.add_argument('--object_counts_by_room_file', default='env_lists/800env_object_counts_by_room.json', help='roomTp to objT to cnt')
  parser.add_argument('--env_obj_colors_file', default='env_lists/env_obj_colors_v2.json', help='obj to color mapping')
  args = parser.parse_args()

  # load splits
  splits = json.load(open(osp.join(args.dataDir, 'eqa_v1', args.dataJson), 'r'))['splits']
  for split, hids in splits.items():
    print('There are %s %s house_ids.' % (len(hids), split))
  house_ids = [hid for split, hids in splits.items() for hid in hids]
  print('There are in all %s house_ids.' % len(house_ids))

  # HouseParse and QA-engine
  Hp = HouseParse(dataDir=osp.join(args.dataDir, 'SUNCGdata'), 
                  objrenderPath=osp.join(args.HouseApiDir, 'House3D'))
  E = Engine(args.object_counts_by_room_file, args.env_obj_colors_file)

  # # try one house
  # hid = splits['train'][2]
  # Hp.parse(hid); E.cacheHouse(Hp)
  # qns = E.executeFn(E.template_defs['room_size_compare'])
  # pprint(qns) 

  # SAVE QUESTIONS TO A JSON FILE
  T = ['object_dist_compare', 'object_color_compare', 'object_size_compare', 'room_size_compare']
  # T = E.template_defs.keys()
  num_envs = len(house_ids)
  idx, all_qns = 0, []
  empty_envs = []
  for i in tqdm(range(num_envs)):
    Hp.parse(house_ids[i])
    num_qns_for_house = 0
    for t in T:
      E.cacheHouse(Hp)
      qns = E.executeFn(E.template_defs[t])
      num_qns_for_house += len(qns)
      E.clearQueue()

      for k in qns:
        k['id'] = idx
        k['house'] = house_ids[i]
        idx += 1
        all_qns.append(k)

    if num_qns_for_house == 0:
      empty_envs.append(house_ids[i])

  print('Houses with no questions generated (if any) : %d' % len(empty_envs))
  print('%s qns generated for %s.' % (len(all_qns), T))

  # simple stats for each type
  qtype_to_qns = {}
  for qn in all_qns:
    if qn['type'] not in qtype_to_qns: qtype_to_qns[qn['type']] = []
    qtype_to_qns[qn['type']] += [qn]
  for qtype in qtype_to_qns.keys(): 
    print('%s questions for [%s]' % (len(qtype_to_qns[qtype]), qtype))

  # save
  if not osp.isdir(args.cacheDir):
    os.makedirs(args.cacheDir)
  output_file = osp.join(args.cacheDir, args.outputJson)
  json.dump(all_qns, open(output_file, 'w'))
  print('Written to %s.' % output_file)
