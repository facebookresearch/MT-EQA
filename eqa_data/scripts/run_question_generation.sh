# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

VERSION=$1
CACHE_DIR=cache/question-gen-outputs
GEN_QUESTION_SET=questions_from_engine_${VERSION}.json
PRUNED_QUESTION_SET=questions_pruned_${VERSION}.json
ENV_WISE_STATS_JSON=env_wise_stats_${VERSION}.json

python question-gen/engine_${VERSION}.py \
    --dataDir data \
    --dataJson eqa_v1.json \
    --cacheDir ${CACHE_DIR} \
    --HouseApiDir pyutils/House3D \
    --object_counts_by_room_file question-gen/env_lists/800env_object_counts_by_room.json \
    --env_obj_colors_file question-gen/env_lists/env_obj_colors_v2.json \
    --outputJson ${GEN_QUESTION_SET}

python question-gen/entropy.py \
    --cacheDir ${CACHE_DIR} \
    --inputJson ${GEN_QUESTION_SET} \
    --qnStatsSubDir entropy_stats 

python question-gen/entropy_based_filtering.py \
    --cacheDir ${CACHE_DIR} \
    --inputJson ${GEN_QUESTION_SET} \
    --qnStatsSubDir entropy_stats \
    --prunedOutputJson ${PRUNED_QUESTION_SET} \
    --envWiseStatsJson ${ENV_WISE_STATS_JSON}

python question-gen/check_qns.py
