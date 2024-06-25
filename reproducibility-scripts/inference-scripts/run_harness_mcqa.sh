#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

for dataset in arc_challenge piqa arc_easy
do
    IMAGE_PATH=data/hf_${dataset}_validation-sd_images.pkl

    python ${SCRIPTS_DIR}/eval_harness_mcqa.py --dataset $dataset

    python ${SCRIPTS_DIR}/eval_harness_mcqa.py --dataset $dataset \
        --image-path ${IMAGE_PATH}

    python ${SCRIPTS_DIR}/eval_harness_mcqa.py --dataset $dataset --use-chat-template

    python ${SCRIPTS_DIR}/eval_harness_mcqa.py --dataset $dataset  --use-chat-template \
        --image-path ${IMAGE_PATH}
done