#!/bin/bash


# python sft.py \
#     --model_name_or_path facebook/opt-2.7b \
#     --output_dir ./checkpoints_peft/opt-2.7b_double \
#     --use_lora True \
#     --lora_r 256 \
#     --use_flash_attn False \
#     --double_data True \
#     --overwrite_output_dir True \
#     --run_name double_opt2.7b_lora_2e4_r256 \
#     --num_train_epochs 1 \
#     --optim adamw_torch_fused \
#     --learning_rate 2e-4 \
#     --lr_scheduler_type cosine \
#     --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --do_eval False \
#     --logging_steps 20 \
#     --remove_unused_columns True \
#     --save_steps 100 --save_total_limit 1 \
#     --seed 0 \
#     --on_completions_only False \
#     --max_seq_length 2048 --dataset_text_field text --packing True \
#     --report_to wandb


# python sft.py \
#     --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
#     --output_dir ./checkpoints_peft/Phi-3-mini-4k-instruct \
#     --run_name diverseprompt_phi3mini_lora_5e5_r256_warmup0.5 \
#     --use_lora True \
#     --lora_r 256 \
#     --overwrite_output_dir True \
#     --num_train_epochs 1 \
#     --optim adamw_torch_fused \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.5 \
#     --lr_scheduler_type cosine \
#     --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --do_eval False \
#     --logging_steps 20 \
#     --remove_unused_columns True \
#     --eval_strategy steps --eval_steps 50 \
#     --save_steps 100 --save_total_limit 1 \
#     --seed 0 \
#     --on_completions_only False \
#     --max_seq_length 2048 --dataset_text_field text --packing True \
#     --report_to wandb


# python sft.py \
#     --model_name_or_path google/gemma-1.1-2b-it \
#     --output_dir ./checkpoints_peft/gemma-1.1-2b-it \
#     --run_name gemma2bit_lora_5e5_r64_warmup0.5 \
#     --use_lora True \
#     --lora_r 64 \
#     --overwrite_output_dir True \
#     --num_train_epochs 1 \
#     --optim adamw_torch_fused \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.5 \
#     --lr_scheduler_type cosine \
#     --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --do_eval False \
#     --logging_steps 20 \
#     --remove_unused_columns True \
#     --eval_strategy steps --eval_steps 50 \
#     --save_steps 100 --save_total_limit 1 \
#     --seed 0 \
#     --on_completions_only False \
#     --max_seq_length 2048 --dataset_text_field text --packing True \
#     --report_to wandb


# python sft_paligemma.py \
#     --model_name_or_path google/paligemma-3b-pt-224 \
#     --output_dir ./checkpoints_peft/paligemma-3b-pt-224 \
#     --run_name pali224_lora_2e4_r16_warmup0.5 \
#     --use_lora False \
#     --lora_r 16 \
#     --overwrite_output_dir True \
#     --num_train_epochs 1 \
#     --optim adamw_torch_fused \
#     --learning_rate 2e-4 \
#     --warmup_ratio 0.5 \
#     --lr_scheduler_type cosine \
#     --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --do_eval False \
#     --logging_steps 20 \
#     --remove_unused_columns False \
#     --save_steps 100 --save_total_limit 1 \
#     --seed 0 \
#     --dataset_text_field target \
#     --report_to none

python sft_paligemma.py \
    --model_name_or_path google/paligemma-3b-pt-224 \
    --output_dir ./checkpoints/paligemma-3b-pt-224 \
    --run_name pali224_2e5_warmup0.1_fp32 \
    --overwrite_output_dir True \
    --num_train_epochs 1 \
    --optim adamw_torch_fused \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --bf16 False \
    --logging_steps 20 \
    --remove_unused_columns False \
    --save_steps 500 --save_total_limit 1 \
    --seed 0
