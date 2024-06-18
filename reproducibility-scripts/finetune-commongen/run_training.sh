#!/bin/bash

ln -s /nlp-rcp-scratch/home/le/.cache ~/.cache
source /nlp-rcp-scratch/home/le/envs/general-env/bin/activate

export CUTLASS_PATH=/nlp-rcp-scratch/home/le/cutlass

ZERO2_CONFIG=reproducibility-scripts/finetune-commongen/zero2.json
ZERO3_CONFIG=reproducibility-scripts/finetune-commongen/zero3.json
CHECKPOINT_DIR=outputs/commongen-checkpoints
LOG_DIR=outputs/commongen-logs/$1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --output_dir ${CHECKPOINT_DIR}/llava-1.5-7b-hf_lm \
    --run_name llava1.5lm_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --output_dir ${CHECKPOINT_DIR}/vicuna-7b-v1.5 \
    --run_name vicuna1.5_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path google/gemma-2b \
    --output_dir ${CHECKPOINT_DIR}/gemma-2b \
    --run_name gemma2b_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path google/paligemma-3b-pt-224 \
    --output_dir ${CHECKPOINT_DIR}/paligemma-3b-pt-224_lm \
    --run_name pali224lm_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir ${CHECKPOINT_DIR}/Llama-2-7b-hf \
    --run_name llama2_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path google/gemma-1.1-2b-it \
    --output_dir ${CHECKPOINT_DIR}/gemma-1.1-2b-it \
    --run_name gemma2bit_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --output_dir ${CHECKPOINT_DIR}/Meta-Llama-3-8B \
    --run_name llama3_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ${CHECKPOINT_DIR}/Meta-Llama-3-8B-Instruct \
    --run_name llama3it_lr1e5_warmup01_1epoch_decay1e3 \
    --dataset_text_field text --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft_mllm.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --run_name llava1.5_1epoch_lr1e5_warmup0.1_decay1e3 \
    --output_dir ${CHECKPOINT_DIR}/llava-1.5-7b-hf \
    --remove_unused_columns False --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft_mllm.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path google/paligemma-3b-pt-224 \
    --run_name pali224_1epoch_lr1e5_warmup0.1_decay1e3 \
    --output_dir ${CHECKPOINT_DIR}/paligemma-3b-pt-224 \
    --remove_unused_columns False --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft_mllm.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path google/paligemma-3b-pt-448 \
    --run_name pali448_1epoch_lr1e5_warmup0.1_decay1e3 \
    --output_dir ${CHECKPOINT_DIR}/paligemma-3b-pt-448 \
    --remove_unused_columns False --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft_mllm.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path HuggingFaceM4/idefics2-8b \
    --run_name idefics2_1epoch_lr1e5_warmup0.1_decay1e3 \
    --output_dir ${CHECKPOINT_DIR}/idefics2-8b \
    --remove_unused_columns False --report_to wandb \
    >> ${LOG_DIR} 2>&1


deepspeed sft_mllm.py \
    --deepspeed ${ZERO3_CONFIG} \
    --model_name_or_path llava-hf/bakLlava-v1-hf \
    --run_name bakLlava_1epoch_lr1e5_warmup0.1_decay1e3 \
    --output_dir ${CHECKPOINT_DIR}/bakLlava-v1-hf \
    --remove_unused_columns False --report_to wandb \
    >> ${LOG_DIR} 2>&1
