# eqa_nav
Room and Object Navigation for EQA

## Prepare House Data (conn-maps, graphs, shortest-paths, images, features, etc)
1. Copy or Symlink `../eqa_data/cache/question-gen-outputs`, `../eqa_data/cache/shortest-paths-mt`, `../eqa_data/target-obj-bestview-pos` and `../eqa_data/target-obj-conn-maps` to `./data`
2. Copy or Symlink `../nav_loc_eqa/cache/path_feats`, `../nav_loc_eqa/cache/path_images` and `../nav_loc_eqa/cache/path_to_room_meta` to `./cache`.

## Prepare Navigation Data
1. Run `./tools/prepro_im.py` for imitation learning data.
2. Run `./tools/prepro_rl.py` for reinforcement finetuning data.

## Train IL-based room-navigator and object-navigator
1. Run `./tools/train_im_object.py` and `./tools/train_im_room.py` for imitation learning.
2. Run `./tools/eval_im_object.py` and `./tools/eval_im_room.py` for imitation learning evaluation.

## Finetune using RL
1. Run `./tools/train_rl_object.py` and `./tools/train_rl_room.py` for reinforcement finetuning.
2. Run `./tools/eval_nav_object.py` and `./tools/eval_nav_room.py` for navigation evaluation.
