#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
# IMAGE_PATH=data/commongen_test-sd_images.pkl
IMAGE_PATH=data/commongen_validation-sd_images.pkl
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

python ${SCRIPTS_DIR}/eval_commongen.py
# python ${SCRIPTS_DIR}/eval_commongen.py --image-path ${IMAGE_PATH}

# python ${SCRIPTS_DIR}/eval_commongen.py --nshot fixed
# python ${SCRIPTS_DIR}/eval_commongen.py --nshot fixed --image-path ${IMAGE_PATH}

# python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template
# python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template --image-path ${IMAGE_PATH}

# python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template --nshot fixed
# python ${SCRIPTS_DIR}/eval_commongen.py --use-chat-template --nshot fixed --image-path ${IMAGE_PATH}
