#!/bin/bash

set -e
SCRIPTS_DIR="reproducibility-scripts/inference-scripts"
export PYTHONPATH=$PYTHONPATH:$(pwd)/vlm_viz

for dataset in coarsewsd, coarsewsd_senses, ambient_dev, ambient_test, dust, commongen_train, commongen_validation, commongen_test
do
    echo "Running image generation on ${dataset}"
    python run_image.py --dataset ${dataset}
done

for dataset in piqa, arc_easy, arc_challenge
do
    echo "Running image generation on ${dataset} with promptgen"
    python run_imagen.py --dataset ${dataset} --harness-use-promptgen
done
