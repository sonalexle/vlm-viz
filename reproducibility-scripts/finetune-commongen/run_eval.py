import os
import json
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from bert_score import score as bert_score

from vlm_viz.utils.model_utils import (
    load_model_and_processor, apply_chat_template,
    get_pad_token_id, postprocess_output_ids
)
from vlm_viz.utils.eval_utils import evaluator as commongen_evaluator
from sft_mllm import prepare_dataset_for_inference, get_instruct_template


CHECKPOINT_FOLDER = Path(os.environ.get("CHECKPOINT_FOLDER", "outputs/commongen-checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
MODEL_TOKENIZER_MAPPING = {
    'Llama-2-7b-hf': "meta-llama/Llama-2-7b-hf",
    'Meta-Llama-3-8B': "meta-llama/Meta-Llama-3-8B",
    'Meta-Llama-3-8B-Instruct': "meta-llama/Meta-Llama-3-8B-Instruct",
    'vicuna-7b-v1.5': "lmsys/vicuna-7b-v1.5",
    'llava-1.5-7b-hf': "llava-hf/llava-1.5-7b-hf",
    'llava-1.5-7b-hf_lm': "llava-hf/llava-1.5-7b-hf",
    'bakLlava-v1-hf': "llava-hf/bakLlava-v1-hf",
    'idefics2-8b': "HuggingFaceM4/idefics2-8b",
    'gemma-1.1-2b-it': "google/gemma-1.1-2b-it",
    'gemma-2b': "google/gemma-2b",
    'paligemma-3b-pt-224': "google/paligemma-3b-pt-224",
    'paligemma-3b-pt-224_lm': "google/paligemma-3b-pt-224",
    'paligemma-3b-pt-448': "google/paligemma-3b-pt-448"
}


def prepare_bertscore_input(rst_pred, rst_ref):
    """
    Inputs are in the format for pycocoeval.
    Outputs are prepared for BERTScore computing.
    """
    cand_list, refer_list = [], []
    for k in rst_pred:
        cand_list.append(rst_pred[k][0])
        refer_list.append(rst_ref[k])
    return cand_list, refer_list


def compute_bertscore(cand_list, refer_list):
    P_mul, R_mul, F_mul = bert_score(cand_list, refer_list, lang="en", rescale_with_baseline=True)
    return F_mul.mean().item()


def get_checkpoint(model_name="Llama-2-7b-hf"):
    checkpoint = str(CHECKPOINT_FOLDER / model_name / "completed")
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist."
    tokenizer_checkpoint = MODEL_TOKENIZER_MAPPING[model_name]
    return checkpoint, tokenizer_checkpoint


def run_model_inference(model, processor, dataset, concept2image=None, batch_size=16, disable_tqdm=False):
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

    example2image = lambda concepts: concept2image[', '.join(sorted(concepts))]
    
    predictions = []
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Running inference", disable=disable_tqdm):
        batch = dataset[idx:idx+batch_size]
        kwargs = {"padding": batch_size>1, "return_tensors": "pt"}
        if concept2image is not None:
            images = [example2image(c) for c in batch["concepts"]]
            if "idefics" in type(processor).__name__.lower():
                images = [[im] for im in images]
            kwargs["images"] = images
        inputs = processor(batch["text"], **kwargs)
        if "pixel_values" in inputs and inputs["pixel_values"] is None: del inputs["pixel_values"]
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        outputs = postprocess_output_ids(inputs["input_ids"], output_ids, tokenizer)
        predictions.extend(outputs)

    return predictions


def postprocess_predictions(predictions):
    # Get rid of unnecessary generations after the prediction of the sentence
    # if we don't do this, the evaluation function crashes with an error
    # even when using few-shot prompting
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


def process_prediction_output(concept_list, pred_list, ref_list, remove_repeat=True, with_bertscore=False):
    assert len(concept_list) == len(pred_list) == len(ref_list)
    print(f'#raw_pred = {len(pred_list)}\t#raw_gt = {len(ref_list)}')
    print(f'Will remove repeat predictions?\t{remove_repeat}')
    # for pycocoeval
    rst_pred = {}
    rst_ref = defaultdict(list)
    for idx, (concept, pred, ref) in enumerate(list(zip(concept_list, pred_list, ref_list))):
        key = ' '.join(sorted(concept)) if remove_repeat else idx
        rst_pred[key] = [pred]  # will remove repeating predictions for concept2text
        rst_ref[key].append(ref)
    print(f'#raw pred = {len(pred_list)}\t#processed pred = {len(rst_pred)}')
    if with_bertscore:
        # for bert-score
        cand_list, refer_list = prepare_bertscore_input(rst_pred, rst_ref)
        retval = rst_pred, rst_ref, cand_list, refer_list
    else:
        retval = rst_pred, rst_ref
    return retval


def main(base_data_dir, model_name=None, checkpoint=None, tokenizer_checkpoint=None, use_chat_template=False, half_precision=False):
    if model_name is not None:
        checkpoint, tokenizer_checkpoint = get_checkpoint(model_name)
    else:
        assert checkpoint is not None
        assert tokenizer_checkpoint is not None
    print("Evaluating checkpoint", model_name or checkpoint)

    kwargs = {"half_precision": half_precision, "padding_side": "left"}
    if "Phi-3" in tokenizer_checkpoint:
        kwargs.update({"half_precision": True, "use_flash_attn": True})
    model, processor = load_model_and_processor(
        checkpoint,
        tokenizer_checkpoint,
        **kwargs
    )
    if "lm" not in checkpoint and hasattr(processor, "tokenizer"):
        with_image = True
    else:
        with_image = False
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor
        if "pali" in checkpoint:
            tokenizer.add_bos_token = True

    prompt_template = get_instruct_template(fshot=False, with_image=with_image)
    if "pali" not in checkpoint and use_chat_template:
            prompt_template = apply_chat_template(prompt_template, tokenizer_checkpoint, with_image=False)
        # elif with_image:
        #     prompt_template = "<image>\n" + prompt_template
    if "pali" in checkpoint and not with_image:
        prompt_template += "\n" # paligemma was trained with the newline, but unlike the processor, the tokenizer does not append '\n' automatically
    dataset, concept2image = prepare_dataset_for_inference(
        base_data_dir,
        split='test',
        with_image=with_image,
        prompt_template=prompt_template
    )
    print(dataset["text"][:5])

    preds = run_model_inference(model, processor, dataset, concept2image=concept2image, batch_size=32)

    # preds = postprocess_predictions(preds)

    print("NOTE: Inference loop completed.")
    print(preds[:10])

    rst_pred, rst_ref, cand_list, refer_list = process_prediction_output(
        dataset["concepts"], preds, dataset["target"], remove_repeat=True, with_bertscore=True
    )
    metrics = commongen_evaluator(rst_ref, rst_pred, skip_spice=False)
    metrics['BERTScore'] = compute_bertscore(cand_list=cand_list, refer_list=refer_list)
    metrics = {k: f'{v*100:.2f}' if k != "CIDEr" else f'{v*10:.2f}' for k, v in metrics.items()}
    print("Metrics of model", checkpoint)
    print(metrics)
    return metrics


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--base-data-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer-checkpoint", type=str, default=None)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--half-precision", action="store_true", default=False)
    args = parser.parse_args()

    if args.model_name is None and args.checkpoint is None and args.tokenizer_checkpoint is None:
        print(f"WARN: since no model name or checkpoint is provided, all models in {CHECKPOINT_FOLDER} will be evaluated.")
        model_names = os.listdir(CHECKPOINT_FOLDER)
        print("Discovered model checkpoints to evaluate:", model_names)
        all_metrics = {}
        for model_name in model_names:
            checkpoint, tokenizer_checkpoint = get_checkpoint(model_name)
            metrics = main(
                base_data_dir=args.base_data_dir, model_name=model_name,
                use_chat_template=args.use_chat_template,
                half_precision=args.half_precision
            )
            all_metrics[model_name] = metrics
        results_path = RESULTS_DIR / "commongen_finetune-results.json"
        with open(results_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
    else:
        main(
            args.base_data_dir,
            args.model_name,
            args.checkpoint,
            args.tokenizer_checkpoint,
            args.use_chat_template,
            args.half_precision
        )