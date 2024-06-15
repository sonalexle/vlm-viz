import os
from typing import List, Union, Dict, Any
from PIL import Image

from transformers import Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer

from vlm_viz.utils.model_utils import load_model_and_processor
from sft import load_raw_dataset, get_peft_config
from params import parse_args, IMAGE_FOLDER


COMMONGEN_IMAGE_PATH = os.path.join(IMAGE_FOLDER, "commongen_{}-sd_images.pkl")


def get_instruction_template():
    return """# Instruction

Given the image and several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

# Your Task

- Concepts: "{}"
- Sentence:"""


def load_pickled_images(pth):
    import pickle
    with open(pth, "rb") as f:
        images = pickle.load(f)
    return images


def prepare_dataset_for_inference(
    base_data_dir: str,
    task: str = 'concept2text',
    dataset: str = 'commongen_inhouse',
    split: str = 'test',
    with_image=True,
    prompt_template='concepts "{}" sentence\n'
):
    dataset = load_raw_dataset(
        base_data_dir,
        task=task,
        dataset=dataset,
        phases=[split]
    )[split]
    if with_image:
        pth = COMMONGEN_IMAGE_PATH.format(split)
        concept2image = load_pickled_images(pth)
    else:
        concept2image = None
    def preprocess_fn(example):
        concepts = ", ".join(example["concepts"])
        text = prompt_template.format(concepts)
        return {"text": text}
    dataset = dataset.map(preprocess_fn)
    return dataset, concept2image


class MLLMDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        processor,
        is_paligemma=False,
        chat_template="",
        ignore_index=-100,
        eos_token=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.is_paligemma = is_paligemma
        self.chat_template = chat_template
        self.ignore_index = ignore_index
        if eos_token is None:
            self.eos_token = processor.tokenizer.eos_token
        else:
            self.eos_token = eos_token

    def example2text(self, example):
        concepts = ", ".join(example["concepts"])
        text = get_instruction_template().format(concepts)
        text = self.chat_template.format(text)
        if not self.is_paligemma: # note paligemma handles the target differently and it adds the eos token automatically
            assert text.endswith(":"), "only support chat templates that end with a colon"
            text += f" {example['target']}"
            text += self.eos_token # need to add eos_token for non-chat models (those without an end of turn token)
        return text

    @staticmethod
    def example2image(example, concept2image=None):
        concept = ', '.join(sorted(example['concepts']))
        if DEBUG:
            image = Image.new('RGB', (512, 512), (0, 0, 0)) # debug: return black image
        else:
            image = concept2image[concept].convert('RGB')
        return image

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # TODO: somehow make use of/adapt the super class' torch_call method (DataCollatorForCompletionOnlyLM.torch_call)
        # TODO: should we ignore_index the instruction text?
        texts = [self.example2text(example) for example in examples]
        images = [self.example2image(example, concept2image) for example in examples]
        if "idefics2" in type(self.processor).__name__.lower():
            images = [[im] for im in images]
        kwargs = {"return_tensors": "pt", "padding": "longest"}
        if self.is_paligemma:
            suffixes = [example["target"] for example in examples]
            kwargs["suffix"] = suffixes
        batch = self.processor(text=texts, images=images, **kwargs)
        if not self.is_paligemma:
            labels = batch["input_ids"].clone()
            # get image token id
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            labels[labels == image_token_id] = self.ignore_index
            batch["labels"] = labels
        return batch


def freeze_modules(model, model_args):
    if "idefics" in model_args.model_name_or_path:
        vision_tower = model.model.vision_model
        projector = model.model.connector
    else:
        vision_tower = model.vision_tower
        projector = model.multi_modal_projector

    for param in vision_tower.parameters():
        param.requires_grad = not model_args.freeze_vision_tower

    for param in projector.parameters():
        param.requires_grad = not model_args.freeze_projector


def get_chat_template(model_name_or_path):
    if "paligemma" in model_name_or_path:
        template = "{}"
    elif "llava" in model_name_or_path or "bakLlava" in model_name_or_path:
        template = "USER: <image>\n{}\nASSISTANT:"
    elif "HuggingFaceM4/idefics2-8b" == model_name_or_path:
        template = 'User:<image>{}<end_of_utterance>\nAssistant:'
    else:
        raise NotImplementedError
    return template


def main():
    model_args, data_args, training_args, _ = parse_args()
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    global DEBUG
    global concept2image
    DEBUG = data_args.debug_mode
    if not DEBUG:
        concept2image_train = load_pickled_images(COMMONGEN_IMAGE_PATH.format("train"))
        concept2image_val = load_pickled_images(COMMONGEN_IMAGE_PATH.format("validation"))
        # merge the two dictionaries
        concept2image = {**concept2image_train, **concept2image_val}
    else:
        concept2image = None

    dataset = load_raw_dataset(
        data_args.base_data_dir,
        task=data_args.task,
        dataset=data_args.dataset,
        phases=["train", "validation"]
    )

    # load the model
    model, processor = load_model_and_processor(
        model_args.model_name_or_path,
        half_precision=training_args.bf16,
        padding_side="right",
        device_map="cuda:0" if training_args.deepspeed is None else None
    )
    if "llava" in model_args.model_name_or_path or "idefics2" in model_args.model_name_or_path:
        model.config.hidden_size = 4096 # for deepspeed auto mode
    if training_args.gradient_checkpointing:
        model.config.use_cache = False # THIS DOES NOT SUPPRESS THE WARNING AT ALL
    assert processor.tokenizer.pad_token_id != processor.tokenizer.eos_token_id
    assert processor.tokenizer.padding_side == "right"

    # IMPORTANT: freeze vision modules
    if not model_args.use_lora:
        freeze_modules(model, model_args)

    kwargs = {
        "is_paligemma": "paligemma" in model_args.model_name_or_path,
        "chat_template": get_chat_template(model_args.model_name_or_path),
    }
    if "HuggingFaceM4/idefics2-8b" == model_args.model_name_or_path:
        kwargs["eos_token"] = "<end_of_utterance>"


    collator = MLLMDataCollator(
        processor,
        tokenizer=processor.tokenizer,
        mlm=False,
        **kwargs
    )

    # shuffle the training data
    dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)

    trainer_kwargs = {}
    if model_args.use_lora:
        trainer_kwargs["peft_config"] = get_peft_config(model_args)
        print("NOTE: Using LoRA with the following params:")
        print(trainer_kwargs["peft_config"])
        training_args.dataset_text_field = "target"

    trainer_cls = SFTTrainer if model_args.use_lora else Trainer
    trainer = trainer_cls(
        model,
        tokenizer=processor.tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        args=training_args,
        data_collator=collator,
        **trainer_kwargs
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "completed"))


if __name__ == "__main__":
    main()
        

