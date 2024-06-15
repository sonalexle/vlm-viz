import os
import pickle
from pathlib import Path
from typing import List

import torch
import datasets
from diffusers import AutoPipelineForText2Image
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True

from vlm_viz.utils.model_utils import (
    seed_all, load_model_and_processor, get_pad_token_id,
    apply_chat_template, postprocess_output_ids
)

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "data"))
HF_TOKEN = os.environ.get("HF_TOKEN") or Path('.hf_token').read_text().strip()


def load_pixart_sigma(compiled=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to(device)
    if compiled:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sdxl_turbo():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd3(compiled=False):
    from diffusers import StableDiffusion3Pipeline
    if compiled:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision("high")
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16, token=HF_TOKEN,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if compiled:
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

    return pipe


def generate_images(pipe, batch: List[str]):
    seed_all()
    pipe_cls = type(pipe).__name__.lower()
    if "pixartsigma" in pipe_cls:
        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            images = pipe(prompt=batch, num_images_per_prompt=1).images
    elif "stablediffusionxl" in pipe_cls: # same class as base SDXL
        images = pipe(prompt=batch, num_inference_steps=1, guidance_scale=0.0, num_images_per_prompt=1).images
    elif "stablediffusion3" in pipe_cls:
        images = pipe(prompt=batch, num_inference_steps=28, guidance_scale=7.0).images
    else:
        raise NotImplementedError(f"Unsupported model: {pipe_cls}")
    return images


def run_model_inference(prompts, pipe, batch_size=16, resize_512=True, disable_tqdm=False):
    diffusion_images = []
    dataloader = DataLoader(prompts, batch_size=batch_size)

    pipe_cls = type(pipe).__name__.lower()
    should_break_final_batch = "pixartsigma" in pipe_cls
    for batch in tqdm(dataloader, disable=disable_tqdm):
        if should_break_final_batch and len(batch) != batch_size:
            final_batch = batch
            break
        images = generate_images(pipe, batch)
        if resize_512:
            images = [i.resize((512, 512)) for i in images]
        diffusion_images.extend(images)
    else:
        final_batch = None

    return diffusion_images, final_batch


def run_final_batch(pipe, batch, resize_512=True):
    pipe_cls = type(pipe).__name__.lower()
    if "pixartsigma" in pipe_cls:
        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
            images = pipe(prompt=batch, num_images_per_prompt=1).images
    else:
        raise NotImplementedError("Only compiled models need to run final batch separately")
    if resize_512:
        images = [i.resize((512, 512)) for i in images]
    return images


def get_promptgen_instruction():
    instruct_template = """This is the formula of a Stable Diffusion/Midjourney prompt: A photo of [Main Subject] [Detailed Imagery] [Environment Description], [Mood/Atmosphere Description], detailed, shutterstock.
Your task is to generate such a prompt for a given context. You should only use key words from the context so that the prompt is concise. Always end the prompt with the word shutterstock.

Context: Question: Soil erosion can be best prevented by? Answer: building terraces into the sides of a slope.
Prompt: A photo of terraces on a mountain slope with fresh soil, countryside scene, detailed, shutterstock.

Context: Question: Which equipment will best separate a mixture of iron filings and black pepper? Answer: a magnet.
Prompt: A photo of a magnet separating iron filings from a small pile of black pepper, laboratory scene, detailed, shutterstock.

Context: Question: When ice cream is left out of a freezer, the ice cream changes from a? Answer: solid to a liquid.
Prompt: A photo of a cup of melting ice cream, kitchen scene, detailed, shutterstock.

Context: {}
Prompt:"""
    return instruct_template


def run_llama3_prompt_generation(instructions, checkpoint, batch_size=16, postprocess=True, disable_tqdm=False):
    assert checkpoint == "meta-llama/Meta-Llama-3-8B-Instruct", "Only Meta-Llama-3-8B-Instruct is supported for prompt generation."
    model, tokenizer = load_model_and_processor(checkpoint, padding_side="left", use_flash_attn=True)
    diffusion_prompts = []

    dataloader = DataLoader(instructions, batch_size=batch_size)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    generation_kwargs = dict(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6, top_p=0.9,
        pad_token_id=get_pad_token_id(tokenizer),
    )
    if "Llama-3" in tokenizer.name_or_path:
        generation_kwargs["eos_token_id"] = terminators

    seed_all()
    for batch in tqdm(dataloader, disable=disable_tqdm):
        inputs = tokenizer(batch, padding=batch_size>1, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        outputs = postprocess_output_ids(inputs["input_ids"], output_ids, tokenizer)
        diffusion_prompts.extend(outputs)

    def postprocess_prompt(p):
        if "shutterstock." in p:
            p = p.split("shutterstock.")[0] + "shutterstock."
        if not p.startswith("A photo of "):
            p = "A photo of " + p
        return p

    if postprocess:
        diffusion_prompts = [postprocess_prompt(p) for p in diffusion_prompts]

    return diffusion_prompts


def save_harness_sd_artifacts(prompt_or_image_list, df, task, dataset_name, kind="prompts", save_path=None):
    assert kind in ["prompts", "images"]
    assert len(prompt_or_image_list) == len(df)
    sd_arts_by_doc = []
    prev_id = -1
    for j, d in enumerate(df):
        p = prompt_or_image_list[j]
        if d["doc_id"] == prev_id:
            sd_arts_by_doc[-1].append(p)
        else:
            prev_id = d["doc_id"]
            sd_arts_by_doc.append([p])
    
    # task = eval_tasks[0].task
    doc_iterator = task.doc_iterator(
        rank=0, limit=None, world_size=1
    )
    for doc_id, doc in doc_iterator:
        assert len(sd_arts_by_doc[doc_id]) == (2 if dataset_name == "piqa" else len(doc["choices"]["label"]))

    if save_path is None:
        save_path = f'{dataset_name}-sd_{kind}_llama3.pkl'
    with open(save_path, "wb") as f:
        pickle.dump(sd_arts_by_doc, f)


def get_evalharness_data(dataset_name, use_promptgen=False, use_prompt_cache=False, debug=False, disable_tqdm=False):
    from eval_harness_acc_loglik import get_harness_data
    eval_tasks, requests, limit, task_dict_orig = get_harness_data(dataset_name=dataset_name)

    df_dict = {"doc_id": [], "idx": [], "context": [], "continuation": []}
    for i in range(len(requests["loglikelihood"])):
        doc = requests["loglikelihood"][i]
        df_dict["doc_id"].append(doc.doc_id)
        df_dict["idx"].append(doc.idx)
        df_dict["context"].append(doc.args[0])
        df_dict["continuation"].append(doc.args[1])
        if debug and i == 50:
            break
    df = datasets.Dataset.from_dict(df_dict)
    task = eval_tasks[0].task
    prompt_cache_dir = ARTIFACTS_DIR / f'{dataset_name}-sd_prompts_llama3.pkl'

    if not use_promptgen:
        diffusion_prompts = ["A photo of" + c.lower() for c in df["continuation"]]
    elif use_prompt_cache:
        with open(prompt_cache_dir, "rb") as f:
            prompts_by_doc = pickle.load(f)
        diffusion_prompts = []
        for i, prompt_list in enumerate(prompts_by_doc): # list of list of prompts, each outer list is a doc
            diffusion_prompts.extend(prompt_list)
            if debug and i == 50:
                break
    else:
        checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
        llama3_instruction = get_promptgen_instruction()
        format_fn = lambda doc: {"llama3_prompt": apply_chat_template(llama3_instruction.format(doc["context"] + doc["continuation"]), checkpoint) + "A photo of"}
        diffusion_prompts = run_llama3_prompt_generation(df.map(format_fn, num_proc=10)["llama3_prompt"], checkpoint, disable_tqdm=disable_tqdm)
        if not debug:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            save_harness_sd_artifacts(diffusion_prompts, df, task, dataset_name, kind="prompts", save_path=prompt_cache_dir)

    return diffusion_prompts, df, task


def get_coarsewsd_data(split="test", debug=False):
    from eval_coarsewsd import load_coarsewsd, get_debug_data
    _, prompts = load_coarsewsd(split=split)
    assert [u['uid'] for u in prompts] == sorted([u['uid'] for u in prompts])
    if debug:
        prompts, _ = get_debug_data(prompts, limit=100)
    prompts = [p["sentence"] for p in prompts]
    return prompts


def get_ambient_data(split="dev", debug=False):
    from eval_ambient import load_ambient_TF
    _, test_instances, _, _, _ = load_ambient_TF(split)
    prompts = [t["ambiguous_sentence"] for t in test_instances]
    if debug:
        prompts = prompts[:100]
    return prompts


def get_commongen_data(split="validation"):
    from eval_commongen import load_commongen
    df = load_commongen(split=split)
    concepts = df["concepts"] # list of lists of concepts, may be repeated
    unique_concepts = set() # set of unique concepts
    for concept in concepts:
        c = ", ".join(sorted(concept))
        unique_concepts |= {c}
    unique_concepts = sorted(list(unique_concepts))
    prompts = []
    for c in unique_concepts:
        prompts.append(f"A photo of {c}")
    return prompts


def save_commongen_images(images, prompts):
    assert len(images) == len(prompts)
    concepts = [p.replace("A photo of ", "") for p in prompts]
    images = {c: i for c, i in zip(concepts, images)}
    return images


def get_dust_data(debug=False):
    from eval_dust import load_exp1_data
    exp1_data, _, _, _ = load_exp1_data()
    prompts = exp1_data["underspecified sentence"].tolist()
    if debug:
        prompts = prompts[:100]
    return prompts


def validate_dataset_choice(dataset):
    assert dataset in [
        "coarsewsd", "coarsewsd_senses",
        "ambient_dev", "ambient_test",
        "dust",
        "commongen_train", "commongen_validation", "commongen_test",
        "piqa", "arc_easy", "arc_challenge"
    ], f"Unsupported dataset: {dataset}"


def load_pipe(model, compiled=False):
    if "sdxl-turbo" in model:
        pipe = load_sdxl_turbo()
    elif "pixart-sigma" in model:
        pipe = load_pixart_sigma(compiled=compiled)
    elif "sd3" in model:
        pipe = load_sd3()
    else:
        raise NotImplementedError(f"Unsupported model: {model}")
    return pipe


def run_coarsewsd_senses(pipe, results_path=None):
    sense2prompt = {
        "apple_company": "A photo of Apple Inc. headquarters",
        "apple_fruit": "A photo of an apple in the kitchen",
        "arm_architecture": "A photo of the ARM architecture chip",
        "arm_limb": "A photo of an arm",
        "bank_institution": "A photo of a bank",
        "bank_geography": "A photo of a river bank",
        "bass_guitar": "A photo of an electric bass guitar",
        "bass_voice type": "A photo of a bass singer",
        "bass_double": "A photo of a double bass",
        "bow_ship": "A photo of a ship's bow",
        "bow_weapon": "A photo of a bow and arrow",
        "bow_music": "A photo of a violin bow",
        "chair_chairman": "A photo of a chairman",
        "chair_furniture": "A photo of a chair",
        "club_organization": "A photo of a student club",
        "club_nightclub": "A photo of a nightclub",
        "club_weapon": "A photo of a club weapon",
        "crane_machine": "A photo of a crane in a construction site",
        "crane_bird": "A photo of a crane in a zoo",
        "deck_ship": "A photo of a ship deck",
        "deck_building": "A photo of a deck building",
        "digit_numerical": "A photo of a number digit",
        "digit_anatomy": "A photo of digit fingers and toes",
        "hood_comics": "A photo of the hood comic character",
        "hood_vehicle": "A photo of a car hood",
        "hood_headgear": "A photo of a hood headgear",
        "java_island": "A photo of the java island",
        "java_program": "A photo of a java program",
        "mole_animal": "A photo of a mole",
        "mole_espionage": "A photo of a spy",
        "mole_unit": "A photo of a molar mass",
        "mole_sauce": "A photo of mexican food sauce mole",
        "mole_architecture": "A photo of mole architecture",
        "pitcher_baseball": "A photo of pitcher baseball",
        "pitcher_container": "A photo of pitcher container",
        "pound_mass": "A photo of a scale with an object weighing one pound",
        "pound_currency": "A photo of the pound",
        "seal_pinniped": "A photo of a seal",
        "seal_musician": "A photo of the musician artist Seal",
        "seal_emblem": "A photo of a seal emblem",
        "seal_mechanical": "A photo of a seal to connect two pipes",
        "spring_hydrology": "A photo of a spring in the mountains",
        "spring_season": "A photo of the spring season",
        "spring_device": "A photo of a spring in a physics class",
        "square_shape": "A photo of a square toy",
        "square_company": "A photo of the Square video game company",
        "square_town": "A photo of a town center square",
        "square_number": "A photo of a math equation paper with a square root",
        "trunk_botany": "A photo of a trunk in botany",
        "trunk_automobile": "A photo of a trunk in a car",
        "trunk_anatomy": "A photo of a trunk in the anatomy",
        "yard_unit": "A photo of a yard of ruler",
        "yard_sail": "A photo of a sail yard"
    }
    for sense, prompt in tqdm(sense2prompt.items()):
        image = generate_images(pipe, [prompt])[0]
        if results_path is None:
            results_file = f"../artifacts/wsd-senses-gen/{sense}.png"
        else:
            results_file = Path(results_path) / f"{sense}.png"
        image.save(results_file)


def main(
    dataset: str, model="sdxl-turbo", results_path=None,
    harness_use_promptgen=False, harness_use_prompt_cache=False,
    debug=False, disable_tqdm=False
):
    assert model in ["sdxl-turbo", "pixart-sigma", "sd3"], f"Unsupported model: {model}"
    validate_dataset_choice(dataset)

    pipe = load_pipe(model, compiled=True)

    if dataset == "coarsewsd_senses":
        run_coarsewsd_senses(pipe, results_path)

    use_harness = dataset in ["piqa", "arc_easy", "arc_challenge"]
    if dataset == "coarsewsd":
        prompts = get_coarsewsd_data(debug)
    elif "ambient" in dataset:
        assert dataset in ["ambient_dev", "ambient_test"]
        split = dataset.split("_")[1]
        prompts = get_ambient_data(split, debug)
    elif dataset == "dust":
        prompts = get_dust_data(debug)
    elif "commongen" in dataset:
        assert dataset in ["commongen_train", "commongen_validation", "commongen_test"]
        split = dataset.split("_")[1]
        prompts = get_commongen_data(split)
    elif use_harness:
        prompts, harness_df, harness_task = get_evalharness_data(
            dataset, use_promptgen=harness_use_promptgen,
            use_prompt_cache=harness_use_prompt_cache,
            debug=debug, disable_tqdm=disable_tqdm
        )
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    images, final_batch = run_model_inference(prompts, pipe, batch_size=16)

    should_run_final_batch = final_batch is not None
    if should_run_final_batch:
        print("Running final batch with uncompiled pipe...")
        del pipe # compiled models cannot run the last batch as it has a different length to the rest
        # so we need to run the last batch separately with a new uncompiled pipe
        pipe = load_pipe(model, compiled=False)
        images.extend(run_final_batch(pipe, final_batch))
    assert len(images) == len(prompts)

    if debug: return

    if not results_path:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        results_path = ARTIFACTS_DIR / f"{dataset}-{'sd' if model == 'sdxl-turbo' else 'pixart'}_images.pkl"
    if not use_harness:
        if "commongen" in dataset:
            images = save_commongen_images(images, prompts)
        with open(results_path, "wb") as f:
            pickle.dump(images, f)
    else:
        save_harness_sd_artifacts(images, harness_df, harness_task, dataset, kind="images", save_path=results_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="sdxl-turbo")
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--harness-use-promptgen", action="store_true", default=False)
    parser.add_argument("--harness-use-prompt-cache", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args()

    main(
        args.dataset, args.model, args.results_path,
        args.harness_use_promptgen, args.harness_use_prompt_cache,
        args.debug, args.disable_tqdm
    )
