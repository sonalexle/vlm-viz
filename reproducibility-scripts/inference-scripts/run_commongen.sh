#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

python ${SCRIPTS_DIR}/eval_commongen.py
python ${SCRIPTS_DIR}/eval_commongen.py --image-path "data/commongen-sd_images.pkl"

python ${SCRIPTS_DIR}/eval_commongen.py --fshot
python ${SCRIPTS_DIR}/eval_commongen.py --fshot --image-path "data/commongen-sd_images.pkl"

python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template
python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template --image-path "data/commongen-sd_images.pkl"

python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template --fshot
python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template --fshot --image-path "data/commongen-sd_images.pkl"
