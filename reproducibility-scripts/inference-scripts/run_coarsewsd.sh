#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

python ${SCRIPTS_DIR}/eval_coarsewsd_yn.py --disable-tqdm
python ${SCRIPTS_DIR}/eval_coarsewsd_yn.py --image-path data/wsd-senses-images --run-clip --disable-tqdm

python ${SCRIPTS_DIR}/eval_coarsewsd.py --disable-tqdm
python ${SCRIPTS_DIR}/eval_coarsewsd.py --image-path data/coarsewsd-sd_images.pkl --disable-tqdm


python ${SCRIPTS_DIR}/eval_coarsewsd_yn.py --use-chat-template --disable-tqdm
python ${SCRIPTS_DIR}/eval_coarsewsd_yn.py --use-chat-template --image-path data/wsd-senses-images --run-clip --disable-tqdm

python ${SCRIPTS_DIR}/eval_coarsewsd.py --use-chat-template --disable-tqdm
python ${SCRIPTS_DIR}/eval_coarsewsd.py --use-chat-template --image-path data/coarsewsd-sd_images.pkl --disable-tqdm