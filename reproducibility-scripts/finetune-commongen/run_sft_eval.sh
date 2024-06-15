#!/bin/bash

LOG_DIR=$1

ln -s /nlp-rcp-scratch/home/le/.cache ~/.cache
source /nlp-rcp-scratch/home/le/envs/general-env/bin/activate


# python run_eval.py \
#     --checkpoint ./checkpoints/llava-1.5-7b-hf_lm/completed \
#     --tokenizer-checkpoint llava-hf/llava-1.5-7b-hf \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/vicuna-7b-v1.5/completed \
#     --tokenizer-checkpoint lmsys/vicuna-7b-v1.5 \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/gemma-2b/completed \
#     --tokenizer-checkpoint google/gemma-2b \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/paligemma-3b-pt-224_lm/completed \
#     --tokenizer-checkpoint google/paligemma-3b-pt-224 \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/Llama-2-7b-hf/completed \
#     --tokenizer-checkpoint meta-llama/Llama-2-7b-hf \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/gemma-1.1-2b-it/completed \
#     --tokenizer-checkpoint google/gemma-1.1-2b-it \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/Meta-Llama-3-8B/completed \
#     --tokenizer-checkpoint meta-llama/Meta-Llama-3-8B \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/llava-1.5-7b-hf/completed \
#     --tokenizer-checkpoint llava-hf/llava-1.5-7b-hf \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/Meta-Llama-3-8B-Instruct/completed \
#     --tokenizer-checkpoint meta-llama/Meta-Llama-3-8B-Instruct \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/idefics2-8b/completed \
#     --tokenizer-checkpoint HuggingFaceM4/idefics2-8b \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/bakLlava-v1-hf/completed \
#     --tokenizer-checkpoint llava-hf/bakLlava-v1-hf \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/paligemma-3b-pt-224/completed \
#     --tokenizer-checkpoint google/paligemma-3b-pt-224 \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1


# python run_eval.py \
#     --checkpoint ./checkpoints/paligemma-3b-pt-448/completed \
#     --tokenizer-checkpoint google/paligemma-3b-pt-448 \
#     --half-precision \
#     >> ${LOG_DIR} 2>&1