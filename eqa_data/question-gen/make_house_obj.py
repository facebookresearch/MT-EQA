from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path as osp
import sys

this_dir = osp.dirname(__file__)
splits = json.load(open(osp.join(this_dir, '../data/eqa_v1/eqa_v1.json')))['splits']
print('There are %s train, %s val, %s test houses.' % (len(splits['train']), len(splits['val']), len(splits['test'])))

# get each house.json and make mtl and json
house_dir = osp.join(this_dir, '../data/SUNCGdata/house')
gaps = '../../../SUNCGtoolbox/gaps/bin/x86_64/scn2scn'
for split in ['train', 'val', 'test']:
    hids = splits[split]
    for i, hid in enumerate(hids):
        hfolder = osp.join(house_dir, str(hid))
        if osp.exists(hfolder+'/house.obj'):
            continue
        os.system('cd %s; %s house.json house.obj' % (hfolder, gaps))
        print('[%s]%s/%s [%s] done.' % (split, i+1, len(hids), hid))
print('Done.')

