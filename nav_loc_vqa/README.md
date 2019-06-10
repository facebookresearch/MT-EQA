# Full model implementation
Navigator + Controller + cVQA

## Prepare House Data (conn-maps, graphs, shortest-paths, images, features, etc)
1. Copy or Symlink `../eqa_data/cache/question-gen-outputs`, `../eqa_data/cache/shortest-paths-mt`, `../eqa_data/target-obj-bestview-pos` and `../eqa_data/target-obj-conn-maps` to `./data`
2. Run `./tools/generate_path_imgs.py` and `./tools/generate_path_cube_imgs.py` to extract 1st-person images along paths.
3. Run `./tools/extract_feats.py` and `./tools/extract_cube_feats.py` to extract 1st-person features along paths.
4. Run `./tools/compute_meta_info.py` to compute room meta info.

## Train and Eval IL
1. Run `./tools/prepro_imitation_data.py` and `./tools/prepro_reinforce_data.py`
2. Run `./tools/eval_gd_path.py` to get results on ground-truth paths.

## Evaluate RL-finetuned Model (after checking eqa_nav)
1. Run `./tools/evaleqa_nr.py`


