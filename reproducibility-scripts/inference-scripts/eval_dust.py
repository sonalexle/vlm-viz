import json
import click
import os
from pathlib import Path

import pandas as pd
import numpy as np

from vlm_viz.utils.model_utils import (
    load_model_and_processor,
    get_sd_images, get_perplexity,
    apply_chat_template
)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def load_exp1_data(sd_images=None):
    data_path = "https://raw.githubusercontent.com/frank-wildenburg/DUST/main/experiment_1_data.csv"
    exp1_data = pd.read_csv(data_path)
    exp1_data = exp1_data[exp1_data["type"] != 4].reset_index(drop=True)
    test_instances = []
    images = []
    for i, row in exp1_data.iterrows():
        sentence = row['underspecified sentence']
        control_sentence = row['control sentence']
        uuoo_prompt = f"This is an underspecified sentence: '{sentence}'. This is its more specified counterpart: '{control_sentence}'."
        uoou_prompt = f"This is an underspecified sentence: '{control_sentence}'. This is its more specified counterpart: '{sentence}'."
        oouu_prompt = f"This is a more specified sentence: '{control_sentence}'. This is its underspecified counterpart: '{sentence}'."
        ouuo_prompt = f"This is a more specified sentence: '{sentence}'. This is its underspecified counterpart: '{control_sentence}'."
        test_instances.extend(
            [
                {'row_idx': i, 'type': 'over_under', 'order': 'uuoo', 'prompt': uuoo_prompt},
                {'row_idx': i, 'type': 'over_under', 'order': 'uoou', 'prompt': uoou_prompt},
                {'row_idx': i, 'type': 'over_under', 'order': 'oouu', 'prompt': oouu_prompt},
                {'row_idx': i, 'type': 'over_under', 'order': 'ouuo', 'prompt': ouuo_prompt}
            ]
        )
        if sd_images is not None:
            images.extend([sd_images[i]] * 4)
    if len(images) == 0: images = None
    prompts = [i['prompt'] for i in test_instances]
    return exp1_data, test_instances, prompts, images


def compute_accuracy(test_instances, perplexities):
    for row, perp in zip(test_instances, perplexities):
        row['perplexity'] = perp
    accuracies = []
    for j in range(0, len(test_instances), 4):
        instance = test_instances[j:j+4]
        perp_correct = np.sum([i["perplexity"] for i in instance if i["order"] in ["uuoo", "oouu"]])
        perp_incorrect = np.sum([i["perplexity"] for i in instance if i["order"] in ["uoou", "ouuo"]])
        acc = perp_correct < perp_incorrect
        accuracies.append(acc)
    return np.mean(accuracies)


@click.command()
@click.option('--image-path', type=str, default=None)
@click.option('--use-chat-template', is_flag=True, default=False, help='Apply chat template')
@click.option('--disable-tqdm', is_flag=True, default=False, help='Disable tqdm progress bar')
@click.option('--debug', is_flag=True)
def main(image_path=None, use_chat_template=False, disable_tqdm=False, debug=False):
    checkpoints = [
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
    _, test_instances, orig_prompts, sd_images = load_exp1_data(sd_images=sd_images)
    metrics = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True, patch_perplexity=True
        )

        if sd_images is not None and "paligemma" not in checkpoint:
            format_fn = lambda p: apply_chat_template(p, checkpoint, with_image=True) if use_chat_template else f"<image>\n{p}"
            prompts = [format_fn(p) for p in orig_prompts]
        else:
            prompts = orig_prompts
        
        perplexities = get_perplexity(model, processor, prompts, sd_images=sd_images, batch_size=16, disable_tqdm=disable_tqdm)
        del model, processor

        metrics[checkpoint] = {
            "accuracy": compute_accuracy(test_instances, perplexities),
            "avg_perplexity": np.mean(perplexities),
        }
        print(f"{checkpoint} metrics: {metrics[checkpoint]}")

    if not debug:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        wimage = '_wimage' if image_path else ''
        wchat = '_chat' if use_chat_template else ''
        results_path = RESULTS_DIR / f"dust_exp1{wimage}{wchat}-results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    main()