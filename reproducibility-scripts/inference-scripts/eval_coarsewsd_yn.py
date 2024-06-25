import os, json, pickle
import pprint
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
from tqdm import tqdm

from eval_coarsewsd import load_coarsewsd
from vlm_viz.utils.model_utils import (
    load_model_and_processor, apply_chat_template, get_model_class
)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "data"))


def build_instruction(with_image=False):
    instruction = """# Instruction\n\nThe word "{}" may mean one of the following: {}. """
    if with_image:
        instruction += """This image illustrates the "{}" meaning of the word "{}" in the sentence below. """
    instruction += 'Given the sentence below as context, '
    if with_image:
        instruction += 'observe this image carefully, then '
    instruction += 'read the following sentence and determine the meaning of the word "{}".\n\n'
    instruction += "## Your Task\n\n"
    instruction += """Sentence: {}\nQuestion: Does the word "{}" in the given sentence mean "{}"? Please answer Yes or No.\nAnswer:"""
    return instruction


class CLIPScore:
    def __init__(self, checkpoint, half_precision=False, device_map="cuda:0"):
        kwargs = {}
        if half_precision:
            kwargs["torch_dtype"] = torch.bfloat16
        if device_map:
            kwargs["device_map"] = device_map
        self.model = AutoModel.from_pretrained(checkpoint, **kwargs)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    @torch.inference_mode()
    def imgimg_sim(self, image1, image2):
        inputs = self.processor(images=[image1, image2], return_tensors="pt").to(self.model.device).to(self.model.dtype)
        image_embeds = self.model.get_image_features(**inputs).unsqueeze(1)
        sim = F.cosine_similarity(image_embeds[0], image_embeds[1])
        return sim.item()

    @torch.inference_mode()
    def imgtext_sim(self, texts, images):
        inputs = self.processor(text=texts, images=images, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        if hasattr(self.model, "logit_bias"):
            logits_per_image -= self.model.logit_bias
        return logits_per_image.diag()


def get_sense2image(wordsenses, image_folder=None):
    sense2image = {}
    for word, senses in wordsenses.items():
        sense2image[word] = {}
        for sense in senses.values():
            if image_folder is None:
                img = None
            else:
                img_path = Path(image_folder) / f"{word}_{sense}.png"
                img = Image.open(img_path)
            sense2image[word][sense] = img
    return sense2image


YN_token_dict = {
    'gpt3': {True: ' Yes', False: ' No'},
    'chat': {True: 'Yes', False: 'No'},
    'flan': {True: 19739, False: 4168},
    "llama3": {True: 9642, False: 2822},
    'llama': {True: 8241, False: 3782},
    "gemma": {True: 3553, False: 1294},
    "mistral": {True: 5613, False: 2501},
}


def logits_to_True_prob(logits, model_class):
    """
    logits: batch_size x vocab_size
    """
    # TF_tokens = TF_token_dict[model_class]
    TF_tokens = YN_token_dict[model_class]

    # probs = torch.softmax(logits, dim=-1)
    probs = logits
    prob_mass = probs[:, TF_tokens[True]]

    return prob_mass


def TF_query(checkpoint, instances: list, model=None, processor=None, batch_size=None, disable_tqdm=False):
    model_class = get_model_class(checkpoint)
    bsize = batch_size if batch_size is not None else 1
    prob_mass = []
    for i in tqdm(
        range(0, len(instances), bsize),
        desc=f'Running {checkpoint} on {len(instances)} instances',
        disable=disable_tqdm
    ):
        batch_instances = instances[i:i+bsize]
        kwargs = {"return_tensors": "pt", "padding": True}
        with_image = "image" in batch_instances[0]
        if with_image:
            images = [s["image"] for s in batch_instances]
            if "idefics" in type(processor).__name__.lower():
                images = [[s] for s in images] # idefics2 requires list of lists of images
            kwargs["images"] = images
        inputs = processor([s["prompt"] for s in batch_instances], **kwargs)
        if not with_image and "pixel_values" in inputs: del inputs["pixel_values"]
        with torch.inference_mode():
            logits = model(**inputs.to(model.device)).logits
        for j in range(len(batch_instances)):
            p = logits_to_True_prob(logits[j:j+1, -1, :], model_class)
            prob_mass.append(p.item())
    return prob_mass


def prepare_instances(checkpoint, coarsewsd, wordsenses, sense2image, given_word=None, use_chat_template=False):
    instances = []
    for example in coarsewsd:
        if given_word is not None and example["word"] != given_word: continue
        word = example["word"]
        sentence = example["sentence"]
        all_senses = ", ".join(wordsenses[word].values())
        for sense, image in sense2image[word].items():
            question_prompt_template = build_instruction(with_image=image is not None)
            if image is None:
                prompt = question_prompt_template.format(word, all_senses, word, sentence, word, sense)
            else:
                prompt = question_prompt_template.format(word, all_senses, sense, word, word, sentence, word, sense)
            if use_chat_template:
                prompt = apply_chat_template(prompt, checkpoint, with_image=image is not None)
            elif image is not None:
                prompt = "<image>\n" + prompt
            instance_set = []
            instance = {
                "uid": example["uid"], "prompt": prompt,
                "image_sense": sense,
                "label_text": example["label_text"]
            }
            if image is not None:
                instance["image"] = image
            instance_set.append(instance)
            instances.extend(instance_set)
    return instances


def prepare_instances_clip(coarsewsd, sense2image, given_word=None):
    instances = []
    for example in coarsewsd:
        if given_word is not None and example["word"] != given_word: continue
        word = example["word"]
        sentence = example["sentence"]
        for sense, image in sense2image[word].items():
            assert image is not None, "CLIPScore requires images"
            instance = {
                "uid": example["uid"], "prompt": sentence,
                "image": image, "image_sense": sense,
                "label_text": example["label_text"]
            }
            instances.append(instance)
    return instances


def run_clipscore(clip_score, instances: list, batch_size=None, disable_tqdm=False):
    bsize = batch_size if batch_size is not None else 1
    scores = []
    for i in tqdm(
        range(0, len(instances), bsize),
        desc=f'Running CLIPScore on {len(instances)} instances',
        disable=disable_tqdm
    ):
        batch_instances = instances[i:i+bsize]
        texts = [s["prompt"] for s in batch_instances]
        images = [s["image"] for s in batch_instances]
        scores.extend([s.item() for s in clip_score.imgtext_sim(texts, images)])
    return scores


def get_metrics(instances, prob_mass):
    curr_uid = -1
    curr_class = None
    curr_probs = []
    curr_senses = []
    preds = []
    labels = []
    for it, prob in zip(instances, prob_mass):
        if it["uid"] != curr_uid:
            if len(curr_senses) > 0:
                pred_class = curr_senses[np.argmax(curr_probs)]
                preds.append(pred_class)
                labels.append(curr_class)
            curr_uid = it["uid"]
            curr_class = it["label_text"]
            curr_senses = [it["image_sense"]]
            curr_probs = [prob]
        else:
            curr_senses.append(it["image_sense"])
            curr_probs.append(prob)
    else:
        pred_class = curr_senses[np.argmax(curr_probs)]
        preds.append(pred_class)
        labels.append(curr_class)
    f1 = round(f1_score(labels, preds, average='macro') * 100, 1)
    accuracy = round(accuracy_score(labels, preds) * 100, 1)
    return {"f1": f1, "accuracy": accuracy}


def compute_average_metrics(metrics):
    return {
        "f1": np.mean([v["f1"] for v in metrics.values()]),
        "accuracy": np.mean([v["accuracy"] for v in metrics.values()])
    }


def main(
    results_path, image_path=None,
    run_clip=True, clip_weight=0.3,
    use_chat_template=False,
    disable_tqdm=False, debug=False, no_clip_cache=False
):
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
    if not use_chat_template:
        checkpoints += ["HuggingFaceM4/idefics2-8b-base"]
    if not use_chat_template and image_path is None:
        checkpoints += ["mistralai/Mistral-7B-v0.1"]

    wordsenses, coarsewsd = load_coarsewsd(split="test")
    sense2image = get_sense2image(wordsenses, image_folder=image_path)
    words = wordsenses.keys()

    # run clip model
    if run_clip:
        cache_dir = ARTIFACTS_DIR / "wsd-clip-cache.pkl"
        clip_score_metrics = {}
        word2clipscores = None
        if not debug and not no_clip_cache and os.path.exists(cache_dir):
            print("Loading pre-computed CLIPScore results from cache...")
            with open(cache_dir, "rb") as f:
                word2clipscores = pickle.load(f)
            for word in words:
                instances_clip = prepare_instances_clip(coarsewsd, sense2image, word)
                clip_score_metrics[word] = get_metrics(instances_clip, word2clipscores[word].tolist())
            clip_score_metrics["average"] = compute_average_metrics(clip_score_metrics)
        else:
            clip_score = CLIPScore("google/siglip-so400m-patch14-384")
            word2clipscores = {}
            for word in words:
                print("Running CLIPScore on word", word)
                instances_clip = prepare_instances_clip(coarsewsd, sense2image, word)
                if debug:
                    instances_clip = instances_clip[:10]
                cscores = run_clipscore(clip_score, instances_clip, 128, disable_tqdm=disable_tqdm)
                word2clipscores[word] = np.array(cscores)
                clip_score_metrics[word] = get_metrics(instances_clip, cscores)
                print("clip", word, clip_score_metrics[word])
            clip_score_metrics["average"] = compute_average_metrics(clip_score_metrics)
            if not debug:
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                with open(cache_dir, "wb") as f:
                    pickle.dump(word2clipscores, f)
                    pickle.dump(clip_score_metrics, f)
                print("CLIPScore results saved to cache at", cache_dir)
            del clip_score
        pprint.pp(clip_score_metrics)

    # run TF model
    mllm_metrics = {}
    if run_clip:
        combined_metrics = {}
    for checkpoint in checkpoints:
        # run MLLM
        mllm_metrics[checkpoint] = {}
        if run_clip:
            combined_metrics[checkpoint] = {}
        model, processor = load_model_and_processor(
            checkpoint, padding_side="left", use_flash_attn=True,
        )
        for word in words:
            print("Running", checkpoint, "on word", word)
            instances = prepare_instances(checkpoint, coarsewsd, wordsenses, sense2image, word, use_chat_template)
            if debug:
                instances = instances[:10]
            mllm_logits = TF_query(checkpoint, instances, model, processor, batch_size=16, disable_tqdm=disable_tqdm)
            mllm_metrics[checkpoint][word] = get_metrics(instances, mllm_logits)
            print(word, mllm_metrics[checkpoint][word])
            if not run_clip: continue
            cscores = word2clipscores[word]
            logits = (1-clip_weight) * np.array(mllm_logits) + clip_weight * cscores
            combined_metrics[checkpoint][word] = get_metrics(instances, logits.tolist())
            print("combined", word, combined_metrics[checkpoint][word])
        mllm_metrics[checkpoint]["average"] = compute_average_metrics(mllm_metrics[checkpoint])
        print("average", mllm_metrics[checkpoint]["average"])
        if run_clip:
            combined_metrics[checkpoint]["average"] = compute_average_metrics(combined_metrics[checkpoint])
            print("combined average", combined_metrics[checkpoint]["average"])
        del model, processor


    results_dict = {"mllm": mllm_metrics}
    if run_clip:
        results_dict["clip"] = clip_score_metrics
        results_dict["combined"] = combined_metrics
    print(results_dict)

    if not debug:
        if results_path is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            wimage = '_wimage' if image_path else ''
            wchat = '_chat' if use_chat_template else ''
            results_path = RESULTS_DIR / f"coarsewsd_tf{wimage}{wchat}-results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--run-clip", action="store_true", default=False)
    parser.add_argument("--clip-weight", type=float, default=0.3)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--no-clip-cache", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(
        args.results_path, image_path=args.image_path,
        run_clip=args.run_clip, clip_weight=args.clip_weight,
        use_chat_template=args.use_chat_template,
        disable_tqdm=args.disable_tqdm, debug=args.debug, no_clip_cache=args.no_clip_cache
    )