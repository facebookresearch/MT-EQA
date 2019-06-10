# eqa_data
Embodied QA Data Generation


## How to generate question-answer pairs
1. Run `./scripts/run_question_generation.sh`, this would save qas to `cache/question-gen-outputs/questions_pruned_v2.json`.
2. Run `python question-gen/make_object_id_to_cat.py` to make mapping from hid.oid to fine_class.

## Generate graphs, connMaps, and shortest-paths
1. Run `python shortest-path-gen/generate-conn-maps.py` to build graph and connMaps. This would:
    - save graphs to `cache/3d-graphs`.
    - save connMaps to `cache/target-obj-conn-maps`.
    - update qas to `cache/question-gen-outputs/questions_pruned_mt_with_conn.json`.
2. Run `python shortest-path-gen/get_best_view.py` saving best-view points and ious. Note the points information [x, 1.0, z, yaw] is in coordinate system (not grid system). 
3. Run `python shortest-path-gen/add_program_mt.py` to add parsed programs to questions, update qas to `cache/question-gen-outputs/questions_pruned_mt_with_conn_program.json`.
4. Run `python shortest-path-gen/generate-bestview-paths-mt.py` to compute shortest paths connecting start, best-view, end points. Note we intentionally make a faked end point for testing if agent learns to stop at right position (before the faked end point).
5. Run `python shortest-path-gen/filter_questions_mt.py` to filter out those questions without shortest paths or of low entropy, saving filtered questions to `questions_mt_paths_nearby_source_best_view_program.json`.

## For installing House3D
Follow the instructions[https://github.com/facebookresearch/House3D].

