#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

for split in dev test
do
    python ${SCRIPTS_DIR}/eval_ambient.py --split $split
    python ${SCRIPTS_DIR}/eval_ambient.py --split $split \
        --image-path data/ambient_${split}-pixart_images.pkl
    
    python ${SCRIPTS_DIR}/eval_ambient.py --split $split --use-chat-template
    python ${SCRIPTS_DIR}/eval_ambient.py --split $split --use-chat-template \
        --image-path data/ambient_${split}-pixart_images.pkl
done