#!/bin/bash

set -e

if [ ! -L ~/.cache ]; then
    ln -s /nlp-rcp-scratch/home/le/.cache ~/.cache
fi
source /nlp-rcp-scratch/home/le/envs/general-env/bin/activate

SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

# for dataset in coarsewsd, coarsewsd_senses, ambient_dev, ambient_test, dust, commongen_train, commongen_validation, commongen_test
# do
#     echo "Running image generation on ${dataset}"
#     python ${SCRIPTS_DIR}/eval_imagen.py --dataset ${dataset}
# done

# for dataset in piqa arc_easy arc_challenge
# do
#     echo "Running image generation on ${dataset} with promptgen"
#     python ${SCRIPTS_DIR}/eval_imagen.py --dataset ${dataset} \
#         --harness-context-type context \
#         --harness-use-promptgen
# done

# for dataset in hf_arc_challenge hf_arc_easy hf_piqa
# do
#     echo "Running image generation on ${dataset} with promptgen"
#     ARTIFACTS_DIR=data/pixart python ${SCRIPTS_DIR}/eval_imagen.py --dataset ${dataset} \
#         --use-promptgen --split validation --model pixart-sigma
# done

ARTIFACTS_DIR=data/pixart python ${SCRIPTS_DIR}/eval_imagen.py --dataset commongen_validation \
    --use-promptgen --split train --model pixart-sigma
