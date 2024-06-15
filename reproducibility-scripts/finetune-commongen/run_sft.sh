#!/bin/bash

# python sft.py \
#     --model_name_or_path facebook/opt-350m \
#     --output_dir ./checkpoints/opt-350m \
#     --run_name opt350m_lr3e5_decay1e3_warmup01_3epoch \
#     --num_train_epochs 5 \
#     --learning_rate 3e-5 --weight_decay 1e-3 \
#     --warmup_ratio 0.1 \
#     --logging_steps 50 \
#     --per_device_train_batch_size 256 --gradient_accumulation_steps 1 \
#     --per_device_eval_batch_size 256 \
#     --eval_strategy steps --eval_steps 100 \
#     --save_strategy steps --save_steps 500 \
#     --load_best_model_at_end True \
#     --save_total_limit 3 \
#     --on_completions_only True \
#     --report_to wandb


# python sft.py \
#     --model_name_or_path facebook/opt-iml-max-1.3b \
#     --output_dir ./checkpoints/opt-iml-max-1.3b_instruct \
#     --run_name newprompt_opt1.3bimlmax_lr1e6_3epoch_decay0001 \
#     --num_train_epochs 3 \
#     --logging_steps 20 \
#     --learning_rate 1e-6 --weight_decay 1e-3 \
#     --warmup_steps 50 \
#     --per_device_train_batch_size 64 --gradient_accumulation_steps 1 \
#     --per_device_eval_batch_size 32 \
#     --eval_strategy steps --eval_steps 500 \
#     --save_strategy steps --save_steps 1000 \
#     --load_best_model_at_end True \
#     --save_total_limit 3 \
#     --on_completions_only True \
#     --report_to wandb


# python sft.py \
#     --model_name_or_path facebook/opt-2.7b \
#     --output_dir ./checkpoints/opt-2.7b \
#     --run_name opt2.7b_lr1e6_warmup01_3epoch_decay1e3 \
#     --num_train_epochs 3 \
#     --learning_rate 1e-6 --weight_decay 1e-3 \
#     --warmup_ratio 0.1 \
#     --logging_steps 20 \
#     --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 32 \
#     --eval_strategy steps --eval_steps 500 \
#     --save_strategy steps --save_steps 1000 \
#     --load_best_model_at_end True \
#     --save_total_limit 3 \
#     --on_completions_only True \
#     --report_to none


# python sft.py \
#     --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
#     --output_dir ./checkpoints/Phi-3-mini-4k-instruct \
#     --run_name phi3mini_lr5e6_3epoch_decay1e3_warmup01 \
#     --num_train_epochs 3 \
#     --logging_steps 20 \
#     --learning_rate 5e-6 --weight_decay 1e-3 \
#     --per_device_train_batch_size 64 --gradient_accumulation_steps 1 \
#     --warmup_ratio 0.1 \
#     --per_device_eval_batch_size 32 \
#     --eval_strategy steps --eval_steps 200 \
#     --save_strategy steps --save_steps 1000 \
#     --load_best_model_at_end True \
#     --save_total_limit 3 \
#     --on_completions_only True \
#     --gradient_checkpointing \
#     --bf16 \
#     --report_to wandb


# python sft.py \
#     --model_name_or_path llava-hf/llava-1.5-7b-hf \
#     --output_dir ./checkpoints/llava-1.5-7b-hf_lm \
#     --run_name llava1.5lm_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


# python sft.py \
#     --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --output_dir ./checkpoints/vicuna-7b-v1.5 \
#     --run_name vicuna1.5_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


# python sft.py \
#     --model_name_or_path google/gemma-2b \
#     --output_dir ./checkpoints/gemma-2b \
#     --run_name gemma2b_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


# python sft.py \
#     --model_name_or_path google/paligemma-3b-pt-224 \
#     --output_dir ./checkpoints/paligemma-3b-pt-224_lm \
#     --run_name pali224lm_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


# python sft.py \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --output_dir ./checkpoints/Llama-2-7b-hf \
#     --run_name llama2_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


# python sft.py \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --output_dir ./checkpoints/Meta-Llama-3-8B-Instruct \
#     --run_name llama3it_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


# python sft.py \
#     --model_name_or_path google/gemma-1.1-2b-it \
#     --output_dir ./checkpoints/gemma-1.1-2b-it \
#     --run_name gemma2bit_lr1e5_warmup01_1epoch_decay1e3 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --dataset_text_field text --report_to wandb


python sft.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --output_dir ./checkpoints/Meta-Llama-3-8B \
    --run_name llama3_lr1e5_warmup01_1epoch_decay1e3 \
    --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
    --num_train_epochs 1 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
    --gradient_checkpointing True \
    --bf16 True \
    --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
    --dataset_text_field text --report_to wandb


# python sft_mllm.py \
#     --model_name_or_path HuggingFaceM4/idefics2-8b \
#     --run_name idefics2_1epoch_lr1e5_warmup0.1_decay1e3 \
#     --output_dir ./checkpoints/idefics2-8b \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --remove_unused_columns False --report_to none


# python sft_mllm.py \
#     --model_name_or_path google/paligemma-3b-pt-448 \
#     --run_name pali448_1epoch_lr1e5_warmup0.1_decay1e3 \
#     --output_dir ./checkpoints/paligemma-3b-pt-448 \
#     --learning_rate 1e-5 --warmup_ratio 0.1 --weight_decay 1e-3 \
#     --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
#     --per_device_eval_batch_size 8 --eval_strategy steps --eval_steps 100 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --logging_steps 20 --save_strategy steps --save_steps 500 --save_total_limit 2 \
#     --remove_unused_columns False --report_to none