#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

for dataset in piqa arc_easy arc_challenge
do
    python ${SCRIPTS_DIR}/eval_harness_loglik.py --dataset $dataset
    python ${SCRIPTS_DIR}/eval_harness_loglik.py --dataset $dataset \
        --image-path data/${dataset}-sd_images.pkl

    python ${SCRIPTS_DIR}/eval_harness_loglik.py --dataset $dataset --use-chat-template
    python ${SCRIPTS_DIR}/eval_harness_loglik.py --dataset $dataset  --use-chat-template \
        --image-path data/${dataset}-sd_images.pkl
done