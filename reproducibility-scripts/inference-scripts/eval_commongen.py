import json
import os
from functools import partial
from pathlib import Path
from PIL import Image

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
torch.backends.cuda.matmul.allow_tf32 = True

from vlm_viz.utils.model_utils import (
    load_model_and_processor, get_sd_images, apply_chat_template,
    get_pad_token_id, postprocess_output_ids, seed_all
)
from vlm_viz.utils.eval_utils import evaluator, compute_bertscore

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))


def get_instruct_template(with_image=False, with_instruct=True):
    if with_instruct:
        template = '# Instruction\n\nGiven {}several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.\n'.format("the image and " if with_image else "")
        template += 'The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.\n\n'
    else:
        template = ''
    template += "{}" # slot for few-shot examples, if any
    if with_instruct:
        template += '# Your Task\n\n'
        prefix = '- '
    else:
        prefix = ''
    template += prefix + 'Concepts: "{}"\n'
    template += prefix + 'Sentence:'
    if with_image:
        template = "<image>\n" + template
    return template


def get_fixed_few_shot_examples(with_instruct=True):
    template = ''
    prefix = '- ' if with_instruct else ''
    if with_instruct:
        template += '# Examples\n\n## Example 1\n'
    template += prefix + 'Concepts: "dog, frisbee, catch, throw"\n'
    template += prefix + 'Sentence: The dog catches the frisbee when the boy throws it into the air.\n\n'
    if with_instruct:
        template += '## Example 2\n'
    template += prefix + 'Concepts: "apple, place, tree, pick"\n'
    template += prefix + 'Sentence: A girl picks some apples from a tree and places them into her basket.\n\n'
    if with_instruct:
        template += '## Example 3\n'
    template += prefix + 'Concepts: "canoe, lake, paddle"\n'
    template += prefix + 'Sentence: A man paddles his canoe on the lake.\n\n'
    return template


def construct_one_example(doc, with_instruct=True, include_answer=False):
    prefix = '- ' if with_instruct else ''
    concepts = doc["concepts"]
    sentence = f' {doc["target"]}' if include_answer else ''
    return f'{prefix}Concepts: "{concepts}"\n{prefix}Sentence:{sentence}'


def construct_few_shot_examples(df, nshot=8, with_instruct=True):
    random_example_indices = np.random.choice(len(df), nshot, replace=False)
    few_shot_examples = []
    for idx in random_example_indices:
        few_shot_examples.append(construct_one_example(df[int(idx)], with_instruct=with_instruct, include_answer=True))
    if with_instruct:
        prompt = "## Examples\n\n"
    else:
        prompt = ""
    for idx, example in enumerate(few_shot_examples):
        if with_instruct:
            prompt += f"### Example {idx+1}\n"
        prompt += example
        prompt += "\n\n"
    return prompt


def doc_to_chat(doc, df_train=None, checkpoint=None, nshot=None, use_chat_template=False, with_image=False, with_instruct=True):
    template = get_instruct_template(with_image=with_image, with_instruct=with_instruct)

    if nshot is not None:
        if nshot == "fixed":
            icl = get_fixed_few_shot_examples(with_instruct=with_instruct)
        elif nshot > 0:
            assert df_train is not None, "Training data must be provided when using random few-shot prompting"
            icl = construct_few_shot_examples(df_train, nshot=nshot, with_instruct=with_instruct)
        else:
            raise ValueError("Invalid value for nshot: {}".format(nshot))
    else:
        icl = ''
    
    prompt = template.format(icl, doc["concepts"])
    if use_chat_template:
        assert checkpoint is not None, "Checkpoint must be provided when using chat template"
        prompt = apply_chat_template(prompt, checkpoint, with_image=False)

    return {"prompt": prompt}


def load_commongen(dataset_name="commongen_hf"):
    assert dataset_name in ["commongen_hf", "commongen_imgverb"]
    if dataset_name == "commongen_imgverb":
        raise NotImplementedError("TODO: commongen_imgverb is not supported yet")
    df = load_dataset("GEM/common_gen", trust_remote_code=True)
    df = df.map(lambda x: {"concepts": ', '.join(sorted(x['concepts']))})
    return df


def run_model_inference(model, processor, prompts, concept2image=None, batch_size=16, disable_tqdm=False):
    seed_all()
    predictions = []

    dataloader = DataLoader(prompts, batch_size=batch_size)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if concept2image is None and "paligemma" in type(processor).__name__.lower():
        processor = processor.tokenizer # paligemma processor must have images as input

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    generation_kwargs = dict(
        max_new_tokens=100,
        pad_token_id=get_pad_token_id(processor),
    )
    if "Llama-3" in tokenizer.name_or_path:
        generation_kwargs["eos_token_id"] = terminators

    i = 0
    for batch in tqdm(dataloader, disable=disable_tqdm):
        bsize = len(batch)
        kwargs = {"padding": batch_size>1, "return_tensors": "pt"}
        if concept2image is not None and "processor" in type(processor).__name__.lower():
            images = [concept2image[c] for c in batch["concepts"]]
            if "idefics" in type(processor).__name__.lower():
                images = [[i] for i in images if type(i) != list]
            kwargs["images"] = images
        inputs = processor(batch["prompt"], **kwargs)
        if concept2image is None and "pixel_values" in inputs: del inputs["pixel_values"]
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        outputs = postprocess_output_ids(inputs["input_ids"], output_ids, tokenizer)
        predictions.extend(outputs)
        i += bsize
    
    return predictions


def postprocess_predictions(predictions):
    # Get rid of unnecessary generations after the prediction of the sentence
    # if we don't do this, the evaluation function crashes with an error
    # even when using few-shot prompting
    def remove_quotations(sent):
        # check if the sentence is wrapped in "" or '' and ends with full stop: ".", ""., '.', ''.
        if (sent.startswith('"') and (sent.endswith('".') or sent.endswith('."'))) or \
            (sent.startswith("'") and (sent.endswith("'.") or sent.endswith(".'"))):
            sent = sent[1:-2]
        if not sent.endswith('.'):
            sent += '.'
        return sent

    cleaned_predictions = []
    for i, p in enumerate(predictions):
        if "\n" in p:
            p = p.split("\n")[0]
        p = remove_quotations(p)
        cleaned_predictions.append(p)
    return cleaned_predictions


def prepare_for_eval(predictions, references):
    assert len(predictions) == len(references)
    gts = {i: v for i, v in enumerate(references)}
    res = {i: [v] for i, v in enumerate(predictions)}
    gts_bert = [v for v in references]
    res_bert = [v for v in predictions]
    return gts, res, gts_bert, res_bert


def get_debug_data(df, indices=None):
    if indices is None:
        indices = np.random.choice(len(df), 8, replace=False)
    df = df.select(indices)
    return df


def main(
    dataset_name="commongen_hf",
    results_path=None,
    image_path=None,
    black_images=False,
    nshot=None,
    use_chat_template=False,
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
    
    special_checkpoints = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-v0.1"
    ]

    df = load_commongen(dataset_name)
    references = df["validation"]["references"]

    if black_images:
        black_img = Image.new("RGB", (512, 512), (0, 0, 0))
        concept2image = {c: black_img for c in df["validation"]["concepts"]}
    elif image_path is not None:
        concept2image = get_sd_images(image_path)
    else:
        concept2image = None

    results = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )
        with_image = concept2image is not None and "paligemma" not in checkpoint
        map_fn = partial(
            doc_to_chat,
            df_train=df["train"],
            checkpoint=checkpoint,
            nshot=nshot,
            use_chat_template=use_chat_template,
            with_image=with_image,
            with_instruct=checkpoint not in special_checkpoints
        )
        test_df = df["validation"].map(map_fn)
        test_df = test_df.remove_columns([c for c in test_df.column_names if c not in ["prompt", "concepts"]])
        # print(test_df["prompt"][:3])
        predictions = run_model_inference(
            model, processor, test_df, concept2image=concept2image, batch_size=32, disable_tqdm=disable_tqdm
        )
        # print(predictions[:3])
        del model, processor
        concepts = test_df["concepts"]
        predictions = postprocess_predictions(predictions)
        gts, res, gts_bert, res_bert = prepare_for_eval(predictions, references)
        metrics = evaluator(gts, res)
        metrics['BERTScore'] = compute_bertscore(cand_list=res_bert, refer_list=gts_bert)
        metrics = {k: str(round(v*100, 2)) if k != "CIDEr" else str(round(v*10, 2)) for k, v in metrics.items()}
        results[checkpoint] = metrics
        print(checkpoint, metrics)

        if not debug:
            if results_path is None:
                results_path = get_results_path(
                    nshot=nshot,
                    black_images=black_images,
                    image_path=image_path,
                    use_chat_template=use_chat_template
                )
            save_model_results(
                concepts=concepts,
                predictions=predictions,
                references=references,
                metrics=metrics,
                checkpoint=checkpoint,
                results_path=results_path,
                dataset_name="commongen"
            )

    if debug:
        print(results.to_string())


def get_results_path(*, nshot, black_images, image_path, use_chat_template, dataset_name="commongen"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if nshot is None:
        fshot = 'zshot'
    elif nshot == "fixed":
        fshot = 'fshot'
    else:
        fshot = f'{nshot}shot'
    if black_images:
        wimage = '_black'
    elif image_path is not None:
        wimage = '_wimage'
    else:
        wimage = ''
    wchat = '_chat' if use_chat_template else ''
    results_path = f"{fshot}{wimage}{wchat}"
    results_path = RESULTS_DIR / "pixart" / dataset_name / results_path
    os.makedirs(results_path, exist_ok=True)
    return results_path


def save_model_results(*, concepts, predictions, references, metrics, checkpoint, results_path, dataset_name):
    results = {
        "dataset": dataset_name,
        "n_examples": len(predictions),
        "model": checkpoint,
        "summary": metrics,
        "examples": {c: {"prediction": p, "references": r} for c, p, r in zip(concepts, predictions, references)}
    }
    checkpoint = checkpoint.replace("/", "__")
    results_path = results_path / f"{checkpoint}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def nshot_type(value):
    from argparse import ArgumentTypeError
    if value == "fixed":
        return value
    try:
        ivalue = int(value)
        if ivalue > 0:
            return ivalue
        else:
            raise ArgumentTypeError(f"Integer value must be > 0: {value}")
    except ValueError:
        raise ArgumentTypeError(f"Invalid value: {value}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="commongen_hf")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--black-images", action="store_true", default=False)
    parser.add_argument("--nshot", type=nshot_type, default=None, help='"fixed" or int > 0 if specified, otherwise None')
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        results_path=args.results_path,
        image_path=args.image_path,
        black_images=args.black_images,
        nshot=args.nshot,
        use_chat_template=args.use_chat_template,
        debug=args.debug
    )