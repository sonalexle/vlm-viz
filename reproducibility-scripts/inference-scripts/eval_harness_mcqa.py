import json, pickle, os
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import datasets
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from vlm_viz.utils.model_utils import (
    load_model_and_processor, get_sd_images, apply_chat_template
)
from eval_coarsewsd import run_model_inference, get_debug_data

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def get_harness_hf_data(
    dataset_name
):
    assert dataset_name in ["piqa", "arc_easy", "arc_challenge"]

    split = "validation"
    if dataset_name == "arc_challenge":
        df = datasets.load_dataset("ai2_arc", data_dir="ARC-Challenge", split=split, trust_remote_code=True)
    elif dataset_name == "arc_easy":
        df = datasets.load_dataset("ai2_arc", data_dir="ARC-Easy", split=split, trust_remote_code=True)
    elif dataset_name == "piqa":
        df = datasets.load_dataset("piqa", split=split, trust_remote_code=True)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    
    if dataset_name == "piqa":
        df = df.map(lambda doc: {'answerKey': ['A', 'B'][doc['label']], 'question': doc['goal'], 'choices': [doc['sol1']] + [doc['sol2']]})
    else:
        for doc in df:
            choices = doc['choices']['label']
            assert choices == sorted(choices), f"Choices are not sorted: {doc}"
        df = df.map(lambda doc: {'choices': doc['choices']['text']})
    
    return df


def get_mcqa_prompts(df, with_image=False, use_chat_template=False, checkpoint=None):
    prompts = []
    for doc in df:
        prompt = "# Instruction\n\n"
        prompt += "Given " + ("the image and " if with_image else "") +  "the following question, select the option that best answers the question."
        if with_image:
            prompt += " The image above provides additional context for the question, which may help you answer it."
        prompt += "\n\n"
        prompt += "# Your Task\n\n"
        prompt += 'Question: {}\n'.format(doc['question'])
        choices = doc['choices']
        for idx in range(len(choices)):
            choice = chr(ord('A') + idx)
            prompt += "({}) {}\n".format(choice, choices[idx])
        if use_chat_template:
            prompt = apply_chat_template(prompt, checkpoint, with_image=with_image)
            prompt += " " if prompt[-1] != "\n" else ""
        else:
            if with_image:
                prompt = "<image>\n" + prompt
            prompt += "\n"
        prompt += "Answer: ("
        prompts.append(prompt)
    return prompts


def get_metrics(predictions, labels, dataset_name):
    # process the predictions so that they must be in valid labels, else Z, a dummy label (does not exist in labels)
    valid_labels = ['A', 'B'] if dataset_name == "piqa" else ['A', 'B', 'C', 'D']
    predictions = [p if p in valid_labels else "Z" for p in predictions]
    f1 = round(f1_score(labels, predictions, average='macro', labels=valid_labels) * 100, 1)
    accuracy = round(accuracy_score(labels, predictions) * 100, 1)
    return f1, accuracy


def main(dataset_name, results_path=None, image_path=None, use_chat_template=False, debug=False, disable_tqdm=False):
    checkpoints = [
        "TIGER-Lab/Mantis-8B-Idefics2",
        "HuggingFaceM4/idefics2-8b",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/bakLlava-v1-hf",
    ]
    if image_path is None:
        checkpoints += [
            "lmsys/vicuna-7b-v1.5",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ]

    sd_images = get_sd_images(image_path) if image_path is not None else None
    df = get_harness_hf_data(dataset_name)
    if debug:
        df = get_debug_data(df, sd_images)

    results = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )
        with_image = sd_images is not None and "paligemma" not in checkpoint
        prompts = get_mcqa_prompts(
            df, with_image=with_image,
            use_chat_template=use_chat_template, checkpoint=checkpoint
        )
        print(prompts[:3])
        predictions = run_model_inference(model, processor, prompts, sd_images=sd_images, batch_size=16, disable_tqdm=disable_tqdm)
        print(predictions[:3])
        del model, processor
        f1, accuracy = get_metrics(predictions, df['answerKey'], dataset_name)
        results[checkpoint] = {"f1": f1, "accuracy": accuracy}
        print(checkpoint, results[checkpoint])

    if not debug:
        if results_path is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            wimage = '_wimage' if image_path else ''
            wchat = '_chat' if use_chat_template else ''
            results_path = f"{dataset_name}{wimage}{wchat}-mcqa_results.json"
            results_path = RESULTS_DIR / results_path
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print(results)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args()
    main(
        args.dataset,
        results_path=args.results_path, image_path=args.image_path,
        use_chat_template=args.use_chat_template, debug=args.debug,
        disable_tqdm=args.disable_tqdm
    )
