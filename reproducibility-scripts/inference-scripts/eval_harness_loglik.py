import json, pickle, os
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

import lm_eval.api
from lm_eval.tasks import Task, TaskManager, get_task_dict
from lm_eval.api.instance import Instance
from lm_eval.evaluator_utils import get_task_list, get_sample_size

from vlm_viz.utils.model_utils import (
    load_model_and_processor, seed_all, apply_chat_template
)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def get_harness_data(dataset_name="piqa", task_dict_orig=None, num_fewshot=None):
    assert dataset_name in ["piqa", "arc_easy", "arc_challenge"]
    seed_all()
    if task_dict_orig is None:
        task_manager = TaskManager("INFO")
        task_dict_orig = get_task_dict([dataset_name], task_manager)
    
    task_dict = deepcopy(task_dict_orig)
    task_obj = task_dict[dataset_name]
    if num_fewshot is not None:
        task_obj.set_config(key="num_fewshot", value=num_fewshot)
    elif (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
        task_obj.set_config(key="num_fewshot", value=0)

    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    # get lists of group hierarchy and each type of request
    task_hierarchy, eval_tasks = get_task_list(task_dict)

    limit = None
    for task_output in eval_tasks: # only one task for piqa
        task: Task = task_output.task
        limit = get_sample_size(task, limit) # returns None
        task.build_all_requests(rank=0, world_size=1, limit=limit)

        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type # always loglikelihood for piqa
            requests[reqtype].append(instance) # flattened list, #docs * #choices per doc
    
    return eval_tasks, requests, limit, task_dict_orig


def get_requests_submit(requests: List[Instance], use_chat_template=False, checkpoint=None, with_image=False):
    if not isinstance(requests[0], Tuple):
        requests = [req.args for req in requests]
    new_reqs = []
    for context, continuation in requests:
        if use_chat_template:
            context = apply_chat_template(context, checkpoint, with_image)
        elif with_image:
            context = "<image>\n" + context
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        new_reqs.append((context, continuation))
    return new_reqs


def run_loglikelihood(
    requests: List[Instance], model, processor, sd_images=None,
    use_chat_template=False, checkpoint=None, disable_tqdm=False
):
    reqs = requests.items()
    assert len(reqs) == 1 # only 1 request type for piqa, arc
    _, reqs = list(reqs)[0]

    requests: List[Tuple[str, str]] = get_requests_submit(reqs, use_chat_template, checkpoint, with_image=sd_images is not None)

    seed_all()
    resps = []
    perplexities = []

    for i, (context, continuation) in tqdm(enumerate(requests), total=len(requests), disable=disable_tqdm):
        image = None if sd_images is None else sd_images[i]
        loglik, perplexity = loglikelihood_tokens(model, processor, context, continuation, image)
        resps.append((loglik, False))
        perplexities.append(perplexity)

    for x, req in zip(resps, reqs):
        req.resps.append(x)
    perplexity = np.mean(perplexities)

    return perplexity


def loglikelihood_tokens(
    model,
    processor,
    context: str,
    continuation: str,
    image=None
):
    kwargs = {}
    if image is not None and "<image>" in context:
        kwargs = {"images": image}
        assert hasattr(processor, "image_processor"), "You are passing images into a model that does not support images."
    encodings = processor(context+continuation, **kwargs, return_tensors="pt")
    if image is None and "pixel_values" in encodings: del encodings["pixel_values"]
    ctx_encodings = processor(context, return_tensors="pt")["input_ids"]
    ctx_len = ctx_encodings.shape[1] # NOTE THIS LINE
    cont_len = encodings["input_ids"].shape[1]-ctx_len  
    encodings = encodings.to(model.device)
    labels = encodings["input_ids"].clone() # [1, seq]
    labels[:, :ctx_len] = -100
    encodings.update({"labels": labels})
    with torch.inference_mode():
        loss = model(**encodings).loss * cont_len
        cont_loglik = -loss.item()
    perplexity = torch.exp(loss / cont_len).item()
    return cont_loglik, perplexity


def get_sd_images(dataset_name, image_path=None):
    if image_path is None:
        image_path = f"../artifacts/{dataset_name}-sd_images.pkl"
    with open(image_path, "rb") as f:
        images_by_doc = pickle.load(f)
    sd_images = []
    for doc_id, image_list in enumerate(images_by_doc):
        sd_images.extend(image_list)
    return sd_images


def process_doc_results(task, doc, results):
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = task.doc_to_choice(doc)
    completion_len = np.array([float(len(i)) for i in choices])

    pred = np.argmax(lls)
    pred_norm = np.argmax(lls / completion_len)
    gold = task.doc_to_target(doc)

    if isinstance(gold, int):
        gold = gold if gold < len(choices) else -100
    elif isinstance(gold, str):
        gold = choices.index(gold) if gold in choices else -100
    else:
        raise ValueError(f"Invalid gold type, expected int or str but got: {gold}")

    gold_index_error = gold == -100

    if gold_index_error:
        raise ValueError(
            f"Label index was not in within range of available choices,"
            f"Sample:\n\n{doc}\n\n"
        )

    acc = 1.0 if pred == gold else 0.0
    acc_norm = 1.0 if pred_norm == gold else 0.0
    # TODO: this gets score of 0 on arc_challenge for pythia-70m. need to test that this works properly
    exact_match = int(is_greedy[gold]) if gold != -100 else 0

    result_dict = {
        "acc": acc,
        "acc_norm": acc_norm,
        # "exact_match": exact_match,
    }

    return result_dict


def get_metrics(eval_tasks, limit=None, no_metric_average=False):
    RANK = 0
    WORLD_SIZE = 1
    ### Postprocess outputs ###
    metrics_dict = {"acc": [], "acc_norm": []}

    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters() # for piqa, does not do anything (TakeFirstFilter, stateless apply)

        ### Collect values of metrics on all datapoints ###
        # unpack results and sort back in order and return control to Task
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used - only 1 filter for piqa: 'none'
        doc_iterator = task.doc_iterator(
            rank=RANK, limit=limit, world_size=WORLD_SIZE
        )
        for doc_id, doc in doc_iterator:
            requests = instances_by_doc_id[doc_id]
            metrics = process_doc_results(
                task, doc, [req.filtered_resps['none'] for req in requests]
            )
            for metric, value in metrics.items():
                metrics_dict[metric].append(value)

    if not no_metric_average: # double negative makes a positive
        for metric in metrics_dict:
            metrics_dict[metric] = np.mean(metrics_dict[metric])

    return metrics_dict


def main(
    dataset_name, results_path=None, image_path=None, use_chat_template=False,
    no_metric_average=False, disable_tqdm=False
):
    assert dataset_name in ["piqa", "arc_easy", "arc_challenge"]
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

    sd_images = get_sd_images(dataset_name, image_path) if image_path is not None else None

    results = {}
    task_dict_orig = None
    for checkpoint in checkpoints:
        eval_tasks, requests, limit, task_dict_orig = get_harness_data(
            dataset_name=dataset_name, task_dict_orig=task_dict_orig
        )
        model, processor = load_model_and_processor(checkpoint, use_flash_attn=True)
        perplexity = run_loglikelihood(
            requests, model, processor, sd_images, use_chat_template,
            checkpoint, disable_tqdm=disable_tqdm
        )
        metrics = get_metrics(eval_tasks, limit, no_metric_average)
        metrics["perplexity"] = perplexity
        results[checkpoint] = metrics
        print(checkpoint, results)

    if results_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        wimage = '_wimage' if image_path else ''
        wchat = '_chat' if use_chat_template else ''
        results_path = RESULTS_DIR / f"{dataset_name}{wimage}{wchat}_loglik_ranking-results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--no-metric-average", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args()

    main(
        args.dataset, args.results_path, args.image_path,
        args.use_chat_template, args.no_metric_average, args.disable_tqdm
    )
