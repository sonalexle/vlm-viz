from dataclasses import dataclass, field
from typing import Optional, Union, List
from trl import SFTConfig
from transformers import HfArgumentParser
from pathlib import Path
import os
import argparse


IMAGE_FOLDER = os.environ.get("IMAGE_FOLDER", "data")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-350m",
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA or not when finetuning."}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Use flash attention from either optimum or flash-attn package."}
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "freeze the vision encoder in a VLM"}
    )
    freeze_projector: bool = field(
        default=True,
        metadata={"help": "freeze the projection layer in a VLM"}
    )
    lora_r: int = field(
        default=16, metadata={"help": "The rank in LoRA."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout in LoRA."}
    )
    lora_target_modules: Union[str, List[str]] = field(
        default="all-linear", metadata={"help": "The target modules in LoRA."}
    )


@dataclass
class DataArguments:
    base_data_dir: str = field(
        metadata={"help": "Path to the data folder."}
    )
    task: str = field(
        default="concept2text",
        metadata={"help": "The task to train on."}
    )
    dataset: str = field(
        default="commongen",
        metadata={"help": "The dataset to train on."}
    )
    double_data: bool = field(
        default=False,
        metadata={"help": "Double the training data by concatenating with itself."}
    )
    on_completions_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Only compute loss on assistant tokens."}
    )
    debug_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Debug mode."}
    )


@dataclass
class MyTrainingArguments(SFTConfig):
    """This class is a copy of trl.SFTConfig (which itself is a copy of transformers.TrainingArguments)
    with some additional arguments added. If something is not documented here,
    it is the same as in transformers.TrainingArguments, with the same default values.
    """

    report_to: Union[None, str, List[str]] = field(
        default="none", metadata={"help": "The list of integrations to report the results and logs to."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using a datasets.Dataset."}
    )
    output_dir: str = field(default='outputs/commongen-checkpoints', metadata={"help": 'The output dir for logs and checkpoints'})

    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used'})
    learning_rate: float = field(default=1e-5, metadata={"help": 'The initial learning rate'})
    warmup_ratio: float = field(default=0.1, metadata={"help": 'The ratio of the total steps to perform linear learning rate warmup for'})
    weight_decay: float = field(default=1e-3, metadata={"help": 'The weight decay to apply (if any)'})

    num_train_epochs: int = field(default=1, metadata={"help": 'The number of epochs to train'})
    per_device_train_batch_size: int = field(default=4, metadata={"help": 'The training batch size per GPU'})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": 'How many gradient steps to accumulate before to perform an optimizer step'})

    eval_strategy: str = field(default='steps', metadata={"help": 'The evaluation strategy to adopt during training'})
    eval_steps: int = field(default=100, metadata={"help": 'The number of steps between evaluations'})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": 'The evaluation batch size per GPU'})

    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    bf16: bool = field(default=True, metadata={"help": 'Use bfloat16 precision. You want to use this.'})

    save_strategy: str = field(default='steps', metadata={"help": 'The checkpoint save strategy to adopt during training'})
    save_steps: int = field(default=500, metadata={"help": 'The number of steps between checkpoints'})
    save_total_limit: int = field(default=2, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    logging_steps: int = field(default=20, metadata={"help": 'The number of steps between logging'})

    max_seq_length: int = field(default=2048, metadata={"help": 'Maximum sequence length for the model, for use in ConstantLengthDataset packing.'})


def parse_args():
    hfparser = HfArgumentParser((
        ModelArguments, DataArguments, MyTrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print("output_dir:", args.output_dir)
    return model_args, data_args, training_args, extra_args
