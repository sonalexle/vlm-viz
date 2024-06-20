from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
from vlm_viz.utils.model_utils import (
    load_model_and_processor,
)
from params import parse_args

instruction_bank = [
    'Write a natural, human-like sentence containing these keywords "{}"',
    'Create a natural, conversational sentence using these concepts: {}',
    'Craft a sentence that sounds natural and includes these words "{}"',
    'Compose a human-like sentence that incorporates these terms: {}',
    'Formulate a sentence that naturally integrates the following keywords "{}"',
    'Construct a sentence that feels natural and contains these concepts: {}',
    'Develop a sentence that uses these terms in a human-like way "{}"',
    'Make a sentence that includes these words and sounds like normal speech: {}',
    'Write a sentence that flows naturally and features these keywords "{}"',
    'Generate a sentence that naturally uses these concepts in a human-like manner: {}',
    '# Instruction\n\nGiven several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.\nThe sentence should describe a common scene in daily life, and the concepts should be used in a natural way.\n\n# Your Task\n\n- Concepts: "{}"\n- Sentence:'
]


def seed_all(seed=0):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raw_dataset(
    base_data_dir: str,
    task='concept2text',
    dataset='commongen_inhouse',
    phases=['train', 'validation', 'test']
):
    # if base_data_dir is a relative path, make it absolute
    if not os.path.isabs(base_data_dir):
        base_data_dir = os.path.abspath(base_data_dir)
    base_dir = os.path.join(base_data_dir, task, dataset)
    kwargs = {}
    if isinstance(phases, list):
        data_files = {
            phase: os.path.join(base_dir, f'{phase}.jsonl') 
            for phase in phases
        }
        kwargs["data_files"] = data_files
    elif isinstance(phases, str):
        data_files = os.path.join(base_dir, f'{phases}.jsonl')
        kwargs["data_files"] = data_files
    raw_dataset = datasets.load_dataset('json', **kwargs)
    return raw_dataset


def get_chat_template(checkpoint):
    instruction = """# Instruction

Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

# Your Task

- Concepts: "{}"
- Sentence:"""
    
    if "llava" in checkpoint or "vicuna" in checkpoint:
        instruct_template = 'USER: {}\nASSISTANT:'
    elif checkpoint == "google/gemma-1.1-2b-it":
        instruct_template = '<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n'
    else:
        instruct_template = '{}'

    if "llava" in checkpoint or "vicuna" in checkpoint:
        response_template = '\nASSISTANT:'
    elif "Phi-3" in checkpoint:
        response_template = "<|end|>\n<|assistant|>\n"
    elif checkpoint == "meta-llama/Meta-Llama-3-8B-Instruct":
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif checkpoint == "google/gemma-1.1-2b-it":
        response_template = "<start_of_turn>model\n"
    else:
        response_template = ''

    return instruct_template.format(instruction), response_template


def apply_chat_template(
    example,
    tokenizer,
    instruct_template,
    randomized_instruction=False
):
    if randomized_instruction:
        instruct_template = random.choice(instruction_bank)
    else:
        instruct_template = instruction_bank[-1]
    concepts = ", ".join(example["concepts"])
    messages = [
        {"role": "user", "content": instruct_template.format(concepts)},
        {"role": "assistant", "content": example["target"]}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return text


def should_use_chat_template(checkpoint):
    return checkpoint == "google/gemma-1.1-2b-it" \
        or checkpoint == "meta-llama/Meta-Llama-3-8B-Instruct" \
        or "Phi-3" in checkpoint


def format_input_text(
    example,
    checkpoint,
    tokenizer,
    template,
    eos_token=None,
):
    if should_use_chat_template(checkpoint):
        text = apply_chat_template(example, tokenizer, template)
        if "gemma-1.1-2b-it" in checkpoint:
            text = text.strip() # remove the last '\n' character
        if "gemma-1.1-2b-it" in checkpoint and eos_token is not None:
            text += eos_token # otherwise the model does not stop
    else:
        concepts = ", ".join(example["concepts"])
        text = template.format(concepts)
        if not (template.endswith(" ") or template.endswith("\n")):
            text += " " if "pali" not in checkpoint else "\n"
        text += example['target']
        if eos_token is not None:
            text += eos_token
    example["text"] = text
    return example


def formatting_prompts_func(
    examples,
    checkpoint,
    tokenizer,
    template,
    eos_token=None,
):
    # examples: dict of lists with keys 'concepts' and 'target'
    # call format_input_text for each example
    formatted_examples = []
    for i in range(len(examples['target'])):
        example = {
            'concepts': examples['concepts'][i],
            'target': examples['target'][i],
        }
        example = format_input_text(
            example, checkpoint, tokenizer, template, eos_token)
        formatted_examples.append(example["text"])
    return formatted_examples


def get_peft_config(model_args):
    peft_config = {
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "modules_to_save": None,
    }
    peft_config["r"] = model_args.lora_r
    peft_config["lora_dropout"] = model_args.lora_dropout
    peft_config["target_modules"] = model_args.lora_target_modules
    peft_config["lora_alpha"] = int(peft_config["r"]*2)
    peft_conf = LoraConfig(**peft_config)
    return peft_conf


def prepare_model_tokenizer(model_args, training_args):
    old_model = None
    kwargs = {
        "half_precision": training_args.bf16,
        "padding_side": "right",
        "use_flash_attn": model_args.use_flash_attn or "Phi-3" in model_args.model_name_or_path,
        "device_map": "cuda:0" if training_args.deepspeed is None else None
    }
    model, tokenizer = load_model_and_processor(
        model_args.model_name_or_path,
        use_flash_attn=model_args.use_flash_attn,
        **kwargs
    )
    if "paligemma" in model_args.model_name_or_path or "llava" in model_args.model_name_or_path:
        old_model = model
        model = model.language_model
        tokenizer = tokenizer.tokenizer
    assert tokenizer.padding_side == "right"

    if training_args.gradient_checkpointing:
        model.config.use_cache = False # still does not disable the error message

    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    if "pali" in model_args.model_name_or_path:
        tokenizer.add_bos_token = True # when loading the paligemma processor, add_bos_token was set to False
    if "Phi-3" in model_args.model_name_or_path:
        eos_token = "<|end|>" # end of turn token
    else:
        eos_token = tokenizer.eos_token
    return model, tokenizer, old_model, eos_token


def main():
    model_args, data_args, training_args, _ = parse_args()
    assert training_args.packing + data_args.on_completions_only <= 1, "Packing is not supported with ignoring prompt loss"
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    seed_all(training_args.seed)

    model, tokenizer, old_model, eos_token = prepare_model_tokenizer(model_args, training_args)

    raw_dataset = load_raw_dataset(
        data_args.base_data_dir,
        task=data_args.task,
        dataset=data_args.dataset,
        phases=["train", "validation"]
    )
    splits = {
        "train": raw_dataset["train"],
        "val": raw_dataset["validation"],
    }
    column_names = list(splits["train"].features)

    # double the training data
    if data_args.double_data:
        splits["train"] = datasets.concatenate_datasets([splits["train"], splits["train"]])

    instruct_template, response_template = get_chat_template(model_args.model_name_or_path)
    fn_kwargs = {
        "checkpoint": model_args.model_name_or_path,
        "tokenizer": tokenizer,
        "template": instruct_template,
        "eos_token": eos_token
    }
    for phase in ["train", "val"]:
        splits[phase] = splits[phase].map(
            format_input_text,
            fn_kwargs=fn_kwargs,
            num_proc=10,
            remove_columns=column_names,
        )
    # shuffle the training data
    splits["train"] = splits["train"].shuffle(seed=training_args.seed)
    print(splits["train"]["text"][:5])

    trainer_kwargs = {}
    if model_args.use_lora:
        trainer_kwargs["peft_config"] = get_peft_config(model_args)
        print("NOTE: Using LoRA with the following params:")
        print(trainer_kwargs["peft_config"])
    if data_args.on_completions_only:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        trainer_kwargs["data_collator"] = collator
    else:
        collator = None

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=splits['train'],
        eval_dataset=splits['val'],
        args=training_args,
        **trainer_kwargs
    )
    if training_args.resume_from_checkpoint == "True":
        training_args.resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if "paligemma" in model_args.model_name_or_path or "llava" in model_args.model_name_or_path:
        trainer.model = old_model
    trainer.save_model(os.path.join(training_args.output_dir, "completed"))


if __name__ == "__main__":
    main()