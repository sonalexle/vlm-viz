import json
import os
from pathlib import Path

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


def get_fshot_template():
    fshot_instruct_template = """Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

Concepts: dog, frisbee, catch, throw
Sentence: The dog catches the frisbee when the boy throws it into the air.

Concepts: apple, place, tree, pick
Sentence: A girl picks some apples from a tree and places them into her basket.

Concepts: canoe, lake, paddle
Sentence: A man paddles his canoe on the lake.

{}Concepts: {}"""
    return fshot_instruct_template


def get_zshot_template():
    zeroshot_instruct_template = """Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words.
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

{}Concepts: {}"""
    return zeroshot_instruct_template


def doc_to_chat(doc, checkpoint, zshot=True, use_chat_template=False, with_image=False):
    if zshot:
        template = get_zshot_template()
    else:
        template = get_fshot_template()
    if with_image:
        template = template.replace("Given several concepts", "Given the image and several concepts")

    image_token = "<image>" if with_image else ""
    prompt = template.format(image_token, ", ".join(doc["concepts"]))
    if use_chat_template:
        if "<image>" in prompt:
            prompt = prompt.replace("<image>", "")
        chat = apply_chat_template(prompt, checkpoint, with_image=with_image) + " Sentence:"
    else:
        if "<image>" in prompt:
            prompt = prompt.replace("<image>", "<image>\n")
        chat = f"{prompt}\nSentence:"
    return {"chat": chat}


def load_commongen(dataset_name="commongen_hf", split="validation"):
    assert dataset_name in ["commongen_hf", "commongen_imgverb"]
    if dataset_name == "commongen_imgverb":
        raise NotImplementedError("TODO: commongen_imgverb is not supported yet")
    df = load_dataset("GEM/common_gen", split=split, trust_remote_code=True)
    return df


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
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6, top_p=0.9,
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


def postprocess_predictions(predictions):
    def remove_quotations(sent):
        # check if the sentence is wrapped in "" or '' and ends with full stop: ".", ""., '.', ''.
        if (sent.startswith('"') and (sent.endswith('".') or sent.endswith('."'))) or (sent.startswith("'") and (sent.endswith("'.") or sent.endswith(".'"))):
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


def prepare_for_eval(predictions, df):
    references = [df[i]["references"] for i in range(len(df))]
    assert len(predictions) == len(references)
    gts = {i: v for i, v in enumerate(references)}
    res = {i: [v] for i, v in enumerate(predictions)}
    gts_bert = [v for v in references]
    res_bert = [v for v in predictions]
    return gts, res, gts_bert, res_bert


def get_debug_data(df, images=None, indices=None):
    if indices is None:
        indices = np.random.choice(len(df), 8, replace=False)
    df = df.select(indices)
    if images is not None:
        images = [i for idx, i in enumerate(images) if idx in indices]
    return df, images


def main(
    dataset_name="commongen_hf", results_path=None, image_path=None,
    fshot=False, use_chat_template=False, debug=False,
    disable_tqdm=False
):
    checkpoints = [
        # "HuggingFaceM4/idefics2-8b",
        # "llava-hf/llava-v1.6-mistral-7b-hf",
        # "llava-hf/llava-1.5-7b-hf",
        # "llava-hf/bakLlava-v1-hf",
    ]
    if image_path is None:
        checkpoints += [
            # "lmsys/vicuna-7b-v1.5",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ]
    sd_images = get_sd_images(image_path) if image_path is not None else None
    df = load_commongen(dataset_name)
    if debug:
        df, sd_images = get_debug_data(df, sd_images)
    results = {}
    for checkpoint in checkpoints:
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )
        with_image = sd_images is not None and "paligemma" not in checkpoint
        test_df = df.map(lambda x: doc_to_chat(x, checkpoint, not fshot, use_chat_template, with_image))
        print(test_df["chat"][:3])
        predictions = run_model_inference(
            model, processor, test_df["chat"], sd_images=sd_images, batch_size=32, disable_tqdm=disable_tqdm
        )
        del model, processor
        predictions = postprocess_predictions(predictions)
        print(predictions[:3])
        gts, res, gts_bert, res_bert = prepare_for_eval(predictions, test_df)
        metrics = evaluator(gts, res)
        metrics['BERTScore'] = compute_bertscore(cand_list=res_bert, refer_list=gts_bert)
        metrics = {k: str(round(v*100, 2)) if k != "CIDEr" else str(round(v*10, 2)) for k, v in metrics.items()}
        results[checkpoint] = metrics
        print(checkpoint, metrics)
    raise NotImplementedError

    if not debug:
        if results_path is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            fshot = '_fshot' if fshot else '_zshot'
            wimage = '_wimage' if image_path else ''
            wchat = '_chat' if use_chat_template else ''
            results_path = f"commongen{fshot}{wimage}{wchat}-results.json"
            results_path = RESULTS_DIR / results_path
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print(results.to_string())


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="commongen_hf")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--fshot", action="store_true", default=False)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    main(args.dataset_name, args.results_path, args.image_path, args.fshot, args.use_chat_template, args.debug)