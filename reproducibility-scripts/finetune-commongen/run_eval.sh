#!/bin/bash

SCRIPTS_DIR="reproducibility-scripts/finetune-commongen"
BASE_DATA_DIR="../../data/iNLG"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz


python ${SCRIPTS_DIR}/run_eval.py \
    --base-data-dir ${BASE_DATA_DIR} \
    --checkpoint outputs/checkpoints/idefics2-8b/completed \
    --tokenizer-checkpoint HuggingFaceM4/idefics2-8b \
    --half-precision