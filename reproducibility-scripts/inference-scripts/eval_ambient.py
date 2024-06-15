
import torch
import random
import click
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlm_viz.utils.model_utils import (
    get_model_class, load_model_and_processor,
    get_sd_images, apply_chat_template
)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


TF_token_dict = {
    'gpt3': {True: ' True', False: ' False'},
    'chat': {True: 'True', False: 'False'},
    'flan': {True: 10998, False: 10747}, # 10747: token ID for 'Fal'
    "llama3": {True: 2575, False: 4139},
    'llama': {True: 5852, False: 7700},
    "gemma": {True: 5036, False: 8393},
    "mistral": {True: 6110, False: 8250},
}


TF_templates = {
    'This may mean:': True,
    'This does not necessarily mean:': True,
    'This cannot mean:': False,
    'This can only mean:': False,
}


def logits_to_TF(logits, model_class):
    """
    logits: batch_size x vocab_size
    """
    TF_tokens = TF_token_dict[model_class]
    TF = (logits[:, TF_tokens[True]] > logits[:, TF_tokens[False]]).tolist()

    probs = torch.softmax(logits, dim=-1)
    prob_mass = (probs[:, TF_tokens[True]] + probs[:, TF_tokens[False]]).tolist()

    return TF, prob_mass


def TF_query(checkpoint, prompts: list, sd_images=None, model=None, processor=None, batch_size=None):
    model_class = get_model_class(checkpoint)
    if sd_images is None and "paligemma" in type(processor).__name__.lower():
        processor = processor.tokenizer

    bsize = batch_size if batch_size else 1
    outputs, prob_mass = [], []
    for i in tqdm(range(0, len(prompts), bsize), desc=f'Running {checkpoint} on {len(prompts)} prompts'):
        batch_prompts = prompts[i:i+bsize]
        kwargs = {"padding": batch_size>1, "return_tensors": "pt"}
        if sd_images is not None and "processor" in type(processor).__name__.lower():
            images = sd_images[i:i+bsize]
            if "idefics" in type(processor).__name__.lower():
                images = [[i] for i in images if type(i) != list]
            kwargs["images"] = images
        inputs = processor(batch_prompts, **kwargs)
        if sd_images is None and "pixel_values" in inputs: del inputs["pixel_values"]
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        for j in range(len(batch_prompts)):
            o, p = logits_to_TF(logits[j:j+1, -1, :], model_class)
            outputs += o
            prob_mass += p

    return outputs, prob_mass


def create_TF_test_instances(test_df):
    test_instances = []
    for i, row in test_df.iterrows():
        for sentence_key in ['premise', 'hypothesis']:
            if row[f'{sentence_key}_ambiguous']:
                test_instances.append({
                    'id': row['id'],
                    'ambiguous_sentence_key': sentence_key,
                    'ambiguous_sentence': row[sentence_key],
                    'disambiguations': list(set([l[sentence_key] for l in row['disambiguations']])),
                })
    return test_instances


def load_ambient_TF(split, sd_images=None):
    data_path = f"https://raw.githubusercontent.com/alisawuffles/ambient/main/AmbiEnt/{split}.jsonl"
    try:
        test_df = pd.read_json(data_path, lines=True)
    except Exception as e:
        print(f"Failed to read {data_path}: {e}")
        raise e
    # only consider rows with at least one ambiguous sentence (premise or hypothesis or both)
    test_df = test_df[test_df['premise_ambiguous'] | test_df['hypothesis_ambiguous']]
    test_instances = create_TF_test_instances(test_df)
    print(f'Number of test instances: {len(test_instances)}')

    test_examples = []
    images = []
    with_image = sd_images is not None
    for i, row in enumerate(test_instances):
        for disambiguation in row['disambiguations']:
            for template_id, (template, answer) in enumerate(TF_templates.items()):
                if with_image: images.append(sd_images[i])
                prompt = f"Q: {row['ambiguous_sentence']} {template} {disambiguation} True or False?\nA:"
                test_examples.append({
                    'example_id': row['id'],
                    'ambiguous_sentence_key': row['ambiguous_sentence_key'],
                    'disambiguation': disambiguation,
                    'prompt': prompt,
                    'template_id': template_id,
                    'answer': answer,
                })
    if len(images) == 0: images = None

    prompts = [e['prompt'] for e in test_examples]
    if images is not None:
        assert len(prompts) == len(images)
    print(random.sample(prompts, 1)[0])

    return test_df, test_instances, test_examples, prompts, images


@click.command()
@click.option('--split', type=str, default='dev')
@click.option('--image-path', type=str, default=None)
@click.option('--use-chat-template', is_flag=True, default=False, help='Apply chat template')
@click.option('--debug', is_flag=True, default=False, help='Debug mode')
def main(split, image_path=None, use_chat_template=False, debug=False):
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
    test_df, test_instances, test_examples, orig_prompts, sd_images = load_ambient_TF(split, sd_images=sd_images)
    metrics = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )

        if sd_images is not None and "paligemma" not in checkpoint:
            format_fn = lambda p: apply_chat_template(p, checkpoint, with_image=True) if use_chat_template else f"<image>\n{p}"
            prompts = [format_fn(p) for p in orig_prompts]
        else:
            prompts = orig_prompts

        outputs, prob_mass = TF_query(checkpoint, prompts, sd_images=sd_images, model=model, processor=processor, batch_size=16)
        del model, processor

        results = []
        for ex, output, mass in zip(test_examples, outputs, prob_mass):
            ex['prediction'] = output
            ex['TF_prob_mass'] = mass
            results.append(ex)

        results_df = pd.DataFrame(results)
        acc = (results_df.prediction == results_df.answer).sum()/len(results_df.index)
        print(f'{checkpoint} Accuracy: {acc}')
        print(f'{checkpoint} Average probability mass of True, False tokens: {results_df.TF_prob_mass.mean()}')
        metrics[checkpoint] = {"accuracy": acc, "avg_prob_mass": results_df.TF_prob_mass.mean()}

    if not debug:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        wimage = '_wimage' if image_path else ''
        wchat = '_chat' if use_chat_template else ''
        results_path = RESULTS_DIR / f"ambient_TF_{split}{wimage}{wchat}-results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        print(metrics)


if __name__ == '__main__':
    main()