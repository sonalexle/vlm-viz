#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

python ${SCRIPTS_DIR}/eval_dust.py
python ${SCRIPTS_DIR}/eval_dust.py --image-path data/dust-sd_images.pkl

python ${SCRIPTS_DIR}/eval_dust.py --use-chat-template
python ${SCRIPTS_DIR}/eval_dust.py --use-chat-template --image-path data/dust-sd_images.pkl
