
VERSION=$1
NUM_WORKERS=$2

python shortest-path-gen/generate-conn-maps.py \
    --question_json cache/question-gen-outputs/questions_pruned_${VERSION}.json \
    --house3d_metadata_dir pyutils/House3D/House3D/metadata \
    --suncg_data_dir data/SUNCGdata \
    --graph_dir cache/3d-graphs \
    --target_obj_conn_map_dir cache/target-obj-conn-maps \
    --colide_resolution 500 \
    --invalid_json cache/invalid_conn_maps.json \
    --output_question_json cache/question-gen-outputs/questions_pruned_${VERSION}t_with_conn.json \
    --num_workers ${NUM_WORKERS}

