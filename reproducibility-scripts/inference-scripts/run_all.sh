#!/bin/bash

set -e

bash reproducibility-scripts/inference-scripts/run_coarsewsd.sh
bash reproducibility-scripts/inference-scripts/run_dust.sh
bash reproducibility-scripts/inference-scripts/run_ambient.sh
# bash reproducibility-scripts/inference-scripts/run_commongen.sh
# bash reproducibility-scripts/inference-scripts/run_harness_acc_loglik.sh
