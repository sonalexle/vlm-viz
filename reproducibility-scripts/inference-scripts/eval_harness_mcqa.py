import json, os
from pathlib import Path

import numpy as np
import datasets
from sklearn.metrics import f1_score, accuracy_score

from vlm_viz.utils.model_utils import (
    load_model_and_processor, get_sd_images, apply_chat_template,
    seed_all
)
from eval_coarsewsd import run_model_inference, get_debug_data

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def get_instruct_template(with_image=False, with_instruct=True):
    if with_instruct:
        prompt = "# Instruction\n\n"
        prompt += "Given " + ("the image and " if with_image else "") +  "the following question, select the option that best answers the question."
        if with_image:
            prompt += " The image above provides additional context for the question, which may help you answer it."
        prompt += "\n\n"
    else:
        prompt = ""
    prompt += "{}" # slot for few-shot examples, if any
    if with_instruct:
        prompt += "# Your Task\n\n"
    prompt += "{}" # slot for the question and choices
    return prompt


def get_harness_hf_data(
    dataset_name
):
    assert dataset_name in ["piqa", "arc_easy", "arc_challenge"]

    if dataset_name == "arc_challenge":
        df = datasets.load_dataset("ai2_arc", data_dir="ARC-Challenge", trust_remote_code=True)
    elif dataset_name == "arc_easy":
        df = datasets.load_dataset("ai2_arc", data_dir="ARC-Easy", trust_remote_code=True)
    elif dataset_name == "piqa":
        df = datasets.load_dataset("piqa", trust_remote_code=True)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    
    if dataset_name == "piqa":
        df = df.map(lambda doc: {'answerKey': ['A', 'B'][doc['label']], 'question': doc['goal'], 'choices': [doc['sol1']] + [doc['sol2']]})
    else:
        df = df.map(check_and_preprocess_doc)
    
    return df


def check_and_preprocess_doc(doc):
    choices = doc['choices']['label']
    assert choices == sorted(choices), f"Choices are not sorted: {doc}"
    try: # check if doc['answerKey'] can be converted to int
        # there are samples where the keys are 1 2 3 4 instead of A B C D
        int(doc['answerKey'])
        answerKey = int(doc['answerKey'])
        if choices[0] == "1":
            answerKey -= 1
        answerKey = chr(ord('A') + int(answerKey))
    except ValueError:
        answerKey = doc['answerKey']
    return {'answerKey': answerKey, 'choices': doc['choices']['text']}


def construct_few_shot_examples(df, nshot=8, with_instruct=True):
    random_example_indices = np.random.choice(len(df), nshot, replace=False)
    few_shot_examples = []
    for idx in random_example_indices:
        few_shot_examples.append(construct_mcqa(df[int(idx)], include_answer=True))
    if with_instruct:
        prompt = "## Examples\n\n"
    else:
        prompt = ""
    for idx, example in enumerate(few_shot_examples):
        if with_instruct:
            prompt += f"### Example {idx+1}\n"
        prompt += example
    return prompt


def construct_mcqa(doc, include_answer=False):
    prompt = "Question: {}\n".format(doc['question'])
    choices = doc['choices']
    for idx in range(len(choices)):
        choice = chr(ord('A') + idx)
        prompt += "({}) {}\n".format(choice, choices[idx])
    if include_answer:
        assert ord(doc['answerKey']) - ord('A') >= 0, f"something wrong with the doc: {doc}"
        prompt += "Answer: ({}) {}".format(doc['answerKey'], choices[ord(doc['answerKey']) - ord('A')])
        prompt += "\n"
    prompt += "\n"
    return prompt


def get_mcqa_prompts(df, with_image=False, use_chat_template=False, nshot=None, checkpoint=None, with_instruct=True):
    if use_chat_template:
        assert checkpoint is not None
    prompts = []
    template = get_instruct_template(with_image=with_image, with_instruct=with_instruct)
    seed_all()
    for doc in df["validation"]:
        if nshot is not None and nshot > 0:
            icl = construct_few_shot_examples(df["train"], nshot, with_instruct=with_instruct)
        else:
            icl = ""
        task_prompt = construct_mcqa(doc)
        prompt = template.format(icl, task_prompt)
        if use_chat_template:
            prompt = prompt.rstrip()
            prompt = apply_chat_template(prompt, checkpoint, with_image=with_image)
            prompt += " " if prompt[-1] != "\n" else ""
        else:
            if with_image:
                prompt = "<image>\n" + prompt
            prompt = prompt[:-1] # remove the last newline
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


def main(
    dataset_name,
    results_path=None,
    image_path=None,
    black_images=False,
    use_chat_template=False,
    nshot=None,
    debug=False,
    disable_tqdm=False
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
    if not use_chat_template and image_path is None and not black_images:
        checkpoints += ["mistralai/Mistral-7B-v0.1"]

    df = get_harness_hf_data(dataset_name)
    # if debug:
    #     df = get_debug_data(df, sd_images)
    if black_images:
        sd_images = get_sd_images(black_images=len(df["validation"]))
    elif image_path is not None:
        sd_images = get_sd_images(image_path=image_path)
    else:
        sd_images = None

    results = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )
        with_image = sd_images is not None and "paligemma" not in checkpoint
        prompts = get_mcqa_prompts(
            df,
            with_image=with_image,
            use_chat_template=use_chat_template,
            nshot=nshot,
            checkpoint=checkpoint
        )
        # print(prompts[:3])
        predictions = run_model_inference(model, processor, prompts, sd_images=sd_images, batch_size=16, disable_tqdm=disable_tqdm)
        # print(predictions[:3])
        del model, processor
        f1, accuracy = get_metrics(predictions, df["validation"]['answerKey'], dataset_name)
        metrics = {"f1": f1, "accuracy": accuracy}
        results[checkpoint] = metrics
        print(checkpoint, results[checkpoint])

        if not debug:
            if results_path is None:
                results_path = get_results_path(
                    black_images=black_images,
                    image_path=image_path,
                    use_chat_template=use_chat_template,
                    dataset_name=dataset_name
                )
            save_model_results(
                ids=df["validation"]["id"] if dataset_name != "piqa" else list(range(len(df["validation"]))),
                predictions=predictions,
                references=df["validation"]['answerKey'],
                metrics=metrics,
                checkpoint=checkpoint,
                results_path=results_path,
                dataset_name=dataset_name
            )

    if debug:
        print(results.to_string())


def get_results_path(*, black_images, image_path, use_chat_template, dataset_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if black_images:
        wimage = 'black'
    elif image_path is not None:
        wimage = 'wimage'
    else:
        wimage = 'woimage'
    wchat = '_chat' if use_chat_template else ''
    results_path = f"{wimage}{wchat}"
    results_path = RESULTS_DIR / "pixart" /dataset_name / results_path
    os.makedirs(results_path, exist_ok=True)
    return results_path


def save_model_results(*, ids, predictions, references, metrics, checkpoint, results_path, dataset_name):
    results = {
        "dataset": dataset_name,
        "n_examples": len(predictions),
        "model": checkpoint,
        "summary": metrics,
        "examples": {i: {"prediction": p, "gt_answer": r} for i, p, r in zip(ids, predictions, references)}
    }
    checkpoint = checkpoint.replace("/", "__")
    results_path = results_path / f"{checkpoint}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="piqa")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--black-images", action="store_true", default=False)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--nshot", type=int, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args()
    main(
        args.dataset,
        results_path=args.results_path,
        image_path=args.image_path,
        black_images=args.black_images,
        use_chat_template=args.use_chat_template,
        nshot=args.nshot,
        debug=args.debug,
        disable_tqdm=args.disable_tqdm
    )
