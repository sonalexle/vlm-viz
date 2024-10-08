import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from vlm_viz.utils.model_utils import (
    load_model_and_processor, get_sd_images, apply_chat_template,
    get_pad_token_id, postprocess_output_ids, seed_all
)

torch.backends.cuda.matmul.allow_tf32 = True
BASE_DATA_DIR = Path("../..") / "data" / "coarse-wsd-20" # assuming the data is in the parent directory of the project root
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def load_coarsewsd(data_folder=None, split="test"):
    if data_folder is None:
        data_folder = BASE_DATA_DIR
    with open(os.path.join(data_folder, 'word-senses-short.json'), 'r') as f:
        wordsenses = json.load(f)

    with open(os.path.join(data_folder, f'coarse-wsd-{split}.jsonl'), 'r') as f:
        coarsewsd = [json.loads(line) for line in f]

    return wordsenses, coarsewsd


def get_mcqa_prompts(wordsenses, test_instances, with_image=False, use_chat_template=False, checkpoint=None):
    prompts = []
    for test_instance in test_instances:
        word = test_instance["word"]
        test_instance = test_instance["sentence"]
        senses = wordsenses[word]
        instruction = '# Instruction\n\nThe word "{}" may mean one of the following: {}. '.format(word, ", ".join(senses.values()))
        if with_image:
            instruction += 'This image illustrates the word "{}" in the sentence below. '.format(word)
        instruction += 'Given the sentence below as context, '
        if with_image:
            instruction += 'observe this image carefully, then '
        instruction += 'select the answer option that best describes the meaning of the word "{}" in the sentence.\n\n'.format(word)
        instruction += "## Your Task\n\n"
        prompt = 'Sentence: {}\nQuestion: What does the word "{}" in the above sentence mean?\n'.format(test_instance, word)
        prompt = instruction + prompt
        for idx in range(len(senses)):
            choice = chr(ord('A') + idx)
            prompt += "({}) {}\n".format(choice, senses[str(idx)]) # data format: {0: sense1, 1: sense2, ...}
        if use_chat_template:
            prompt = prompt.rstrip()
            prompt = apply_chat_template(prompt, checkpoint, with_image=with_image)
            prompt += " " if prompt[-1] != "\n" else ""
        elif with_image:
            prompt = "<image>\n" + prompt
        prompt += "Answer: ("
        prompts.append(prompt)
    return prompts


def get_debug_data(data_list, images=None, limit=8):
    indices = np.random.choice(len(data_list), limit, replace=False)
    data_list = [d for idx, d in enumerate(data_list) if idx in indices]
    if images is not None:
        images = [i for idx, i in enumerate(images) if idx in indices]
    return data_list, images


def run_model_inference(model, processor, prompts, sd_images=None, batch_size=16, disable_tqdm=False):
    seed_all()
    predictions = []

    dataloader = DataLoader(prompts, batch_size=batch_size)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if sd_images is None and "paligemma" in type(processor).__name__.lower():
        processor = processor.tokenizer # paligemma processor must have images as input

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    generation_kwargs = dict(
        max_new_tokens=1,
        pad_token_id=get_pad_token_id(processor),
    )
    if "Llama-3" in tokenizer.name_or_path:
        generation_kwargs["eos_token_id"] = terminators

    i = 0
    for batch in tqdm(dataloader, disable=disable_tqdm):
        bsize = len(batch)
        kwargs = {"padding": batch_size>1, "return_tensors": "pt"}
        if sd_images is not None and "processor" in type(processor).__name__.lower():
            images = sd_images[i:i+bsize]
            if "idefics" in type(processor).__name__.lower():
                images = [[i] for i in images if type(i) != list]
            kwargs["images"] = images
        inputs = processor(batch, **kwargs)
        if sd_images is None and "pixel_values" in inputs: del inputs["pixel_values"]
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        outputs = postprocess_output_ids(inputs["input_ids"], output_ids, tokenizer)
        predictions.extend(outputs)
        i += bsize
    
    return predictions


def get_metrics(word2preds, word2labels):
    f1 = {}
    accuracy = {}
    for word in word2preds:
        labels = word2labels[word]
        predictions = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(l) for l in word2preds[word]]
        predictions = [p if p > -1 else 0 for p in predictions]
        f1[word] = round(f1_score(labels, predictions, average='macro') * 100, 1)
        accuracy[word] = round(accuracy_score(labels, predictions) * 100, 1)
    f1["average"] = round(sum(f1.values()) / len(f1), 1)
    accuracy["average"] = round(sum(accuracy.values()) / len(accuracy), 1)
    return f1, accuracy


def main(
    data_folder=None,
    results_path=None,
    image_path=None,
    black_images=False,
    use_chat_template=False,
    debug=False,
    disable_tqdm=False,
):
    checkpoints = [
        "TIGER-Lab/Mantis-8B-Idefics2",
        "HuggingFaceM4/idefics2-8b",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/bakLlava-v1-hf",
    ]
    if image_path is None and not black_images:
        checkpoints += [
            "lmsys/vicuna-7b-v1.5",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ]
    if not use_chat_template:
        checkpoints += ["HuggingFaceM4/idefics2-8b-base"]
    if not use_chat_template and image_path is None:
        checkpoints += ["mistralai/Mistral-7B-v0.1"]

    # sd_images = get_sd_images(image_path) if image_path is not None else None
    wordsenses, coarsewsd = load_coarsewsd(data_folder)
    # if debug:
    #     coarsewsd, sd_images = get_debug_data(coarsewsd, sd_images)

    if black_images:
        sd_images = get_sd_images(black_images=len(coarsewsd))
    elif image_path is not None:
        sd_images = get_sd_images(image_path=image_path)
    else:
        sd_images = None

    words = sorted(list(set([sample["word"] for sample in coarsewsd])))
    wordsenses = {word: wordsenses[word] for word in words}

    results = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )
        with_image = sd_images is not None and "paligemma" not in checkpoint
        prompts = get_mcqa_prompts(
            wordsenses, coarsewsd, with_image=with_image,
            use_chat_template=use_chat_template, checkpoint=checkpoint
        )
        print(prompts[:3])
        word2preds = {}
        word2labels = {}
        predictions = run_model_inference(
            model, processor, prompts,
            sd_images=sd_images, batch_size=16,
            disable_tqdm=disable_tqdm
        )
        for word in words:
            word2preds[word] = [pred for pred, sample in zip(predictions, coarsewsd) if sample["word"] == word]
            word2labels[word] = [sample["label"] for sample in coarsewsd if sample["word"] == word]
        del model, processor
        f1, accuracy = get_metrics(word2preds, word2labels)
        results[checkpoint] = {"f1": f1, "accuracy": accuracy}
        print(checkpoint, results[checkpoint])

    if not debug:
        if results_path is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            if black_images:
                wimage = '_black'
            elif image_path is not None:
                wimage = '_wimage'
            else:
                wimage = ''
            wchat = '_chat' if use_chat_template else ''
            results_path = f"coarsewsd{wimage}{wchat}-results.json"
            results_path = RESULTS_DIR / results_path
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print(results)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data-folder", type=str, default=None)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--black-images", action="store_true", default=False)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args()
    main(
        data_folder=args.data_folder,
        results_path=args.results_path,
        image_path=args.image_path,
        black_images=args.black_images,
        use_chat_template=args.use_chat_template,
        debug=args.debug,
        disable_tqdm=args.disable_tqdm
    )
