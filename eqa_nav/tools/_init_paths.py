import os
import os.path as osp
import sys

this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../pyutils/House3D'))
sys.path.insert(0, osp.join(this_dir, '../pyutils'))

# localized vqa
sys.path.insert(0, osp.join(this_dir, '..'))

# modular_vqa
sys.path.insert(0, osp.join(this_dir, '../nav'))
