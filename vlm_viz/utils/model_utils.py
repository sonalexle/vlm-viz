import random
import pickle
import h5py
import os
from pathlib import Path
from PIL import Image
from typing import List, Union, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
HF_TOKEN = os.environ.get("HF_TOKEN") or Path('.hf_token').read_text().strip()


def seed_all():
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)


def maybe_init_pad_token(model, processor):
    """Initialize the pad token for the model if it is not already set.
    In this case, the pad token is initialized to a new special token <<|[PAD]|>>."""

    if "processor" in type(processor).__name__.lower():
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<<|[PAD]|>>"})
        model.resize_token_embeddings(len(tokenizer))
        model.pad_token_initialized = True
        print(f"pad_token initialized for {type(tokenizer).__name__}")
    else:
        model.pad_token_initialized = False
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        print("WARN: pad_token_id is the same as eos_token_id, replacing pad_token with unk_token...")
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


def get_pad_token_id(processor):
    if "processor" in type(processor).__name__.lower():
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    assert hasattr(tokenizer, "pad_token") and tokenizer.pad_token is not None, f"pad_token is not set for {type(tokenizer).__name__}"
    return tokenizer.pad_token_id


def postprocess_output_ids(input_ids, output_ids, tokenizer):
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [o.strip() for o in outputs]
    return outputs


def load_model_and_processor(
    checkpoint, tokenizer_checkpoint=None, patch_perplexity=False, padding_side=None,
    half_precision=True, use_flash_attn=False, device_map="cuda:0", idefics_do_image_splitting=False
):
    if "gemma" in checkpoint:
        return load_gemma(
            checkpoint, tokenizer_checkpoint, patch_perplexity,
            padding_side, half_precision, use_flash_attn, device_map
        )

    if tokenizer_checkpoint is None:
        tokenizer_checkpoint = checkpoint
    processor_kwargs = {}
    processor_cls = AutoProcessor
    if "llava-hf" in tokenizer_checkpoint or "idefics" in tokenizer_checkpoint:
        model_cls = AutoModelForVision2Seq
    else:
        processor_cls = AutoTokenizer
        model_cls = AutoModelForCausalLM
    if "idefics" in tokenizer_checkpoint and "base" not in tokenizer_checkpoint:
        processor_kwargs = {"do_image_splitting": idefics_do_image_splitting}

    # INIT PROCESSOR OR TOKENIZER
    processor = processor_cls.from_pretrained(tokenizer_checkpoint, token=HF_TOKEN, **processor_kwargs)
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    # explicitly disable automatic addition of eos_token as we will manually do this during training
    # TODO: this might change, we might need to enable this
    if hasattr(tokenizer, "add_eos_token") and tokenizer.add_eos_token:
        print("WARN: detected that tokenizer.add_eos_token is True, setting it to False...")
        tokenizer.add_eos_token = False

    # INIT MODEL
    kwargs = {}
    if use_flash_attn:
        kwargs = {"attn_implementation": "flash_attention_2"}
    if half_precision:
        kwargs["torch_dtype"] = torch.bfloat16
    if device_map is not None:
        kwargs["device_map"] = device_map
    if "vicuna" in checkpoint:
        kwargs["do_sample"] = True
    if "Phi-3" in checkpoint:
        kwargs["trust_remote_code"] = True
    model = model_cls.from_pretrained(
        checkpoint,
        token=HF_TOKEN,
        **kwargs
    )
    if "vicuna" in checkpoint:
        model.generation_config.do_sample = True

    maybe_patch_model(model, processor, patch_perplexity)

    return model, processor


def load_gemma(
    checkpoint, tokenizer_checkpoint=None, patch_perplexity=False, padding_side=None,
    half_precision=True, use_flash_attn=False, device_map="cuda:0",
):
    if tokenizer_checkpoint is None:
        tokenizer_checkpoint = checkpoint
    load_kwargs = {}
    processor_kwargs = {}
    if "paligemma" in checkpoint:
        from transformers import PaliGemmaForConditionalGeneration
        processor_cls = AutoProcessor
        model_cls = PaliGemmaForConditionalGeneration
        if half_precision:
            load_kwargs = {"revision": "bfloat16"}
    else:
        processor_cls = AutoTokenizer
        model_cls = AutoModelForCausalLM

    # INIT PROCESSOR OR TOKENIZER
    processor = processor_cls.from_pretrained(tokenizer_checkpoint, token=HF_TOKEN, **processor_kwargs)
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    # explicitly disable automatic addition of eos_token as we will manually do this during training
    # TODO: this might change, we might need to enable this
    if hasattr(tokenizer, "add_eos_token") and tokenizer.add_eos_token:
        print("WARN: detected that tokenizer.add_eos_token is True, setting it to False...")
        tokenizer.add_eos_token = False # paligemma already set this to False anyways

    # INIT MODEL
    kwargs = {}
    if use_flash_attn:
        kwargs = {"attn_implementation": "sdpa"}
    if half_precision:
        kwargs["torch_dtype"] = torch.bfloat16
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = model_cls.from_pretrained(
        checkpoint,
        token=HF_TOKEN,
        **kwargs,
        **load_kwargs
    )

    maybe_patch_model(model, processor, patch_perplexity)

    return model, processor


def maybe_patch_model(model, processor, patch_perplexity=False):
    maybe_init_pad_token(model, processor)
    if patch_perplexity:
        from vlm_viz.utils.patch_models import patch_model_perplexity
        patch_model_perplexity(model)
    else:
        model.patched_perplexity = False


def get_model_class(model_checkpoint):
    class_to_ckpt = {
        "llama": ["lmsys/vicuna-7b-v1.5", "llava-hf/llava-1.5-7b-hf"],
        "llama3": ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B"],
        "mistral": [
            "mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceM4/idefics2-8b", "HuggingFaceM4/idefics2-8b-base",
            "llava-hf/llava-v1.6-mistral-7b-hf", "llava-hf/bakLlava-v1-hf"
        ],
        "gemma": ["google/gemma-1.1-2b-it", "google/paligemma-3b-pt-448"],
    }
    for class_name, ckpts in class_to_ckpt.items():
        if any([ckpt == model_checkpoint for ckpt in ckpts]):
            return class_name


def get_sd_images(image_path):
    with open(image_path, "rb") as f:
        sd_images = pickle.load(f)
    return sd_images


def save_images_to_hdf5(image_dict, hdf5_filepath, compression="gzip", compression_opts=9, disable_tqdm=False):
    """
    Save a dictionary of PIL images to an HDF5 file.

    Parameters:
    image_dict (dict): Dictionary where keys are strings and values are PIL images.
    hdf5_filepath (str): The path of the HDF5 file to save the images to.
    compression (str): The compression algorithm to use. Default is "gzip".
    compression_opts (int): The level of compression to use. Default is 9.
    """
    kwargs = {}
    if compression is not None:
        kwargs["compression"] = compression
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts

    with h5py.File(hdf5_filepath, "w") as f:
        for key, image in tqdm(image_dict.items(), desc="Saving images", disable=disable_tqdm):
            image = image.convert("RGB")
            image = np.array(image)
            f.create_dataset(key, data=image, **kwargs)


def load_images_from_hdf5(hdf5_filepath, disable_tqdm=False):
    """
    Load a dictionary of PIL images from an HDF5 file.

    Parameters:
    hdf5_filepath (str): The path of the HDF5 file to load the images from.

    Returns:
    dict: Dictionary where keys are strings and values are PIL images.
    """
    image_dict = {}
    with h5py.File(hdf5_filepath, "r") as f:
        keys = list(f.keys())
        for key in tqdm(keys, desc='Loading images', disable=disable_tqdm):
            image = Image.fromarray(f[key][...])
            image_dict[key] = image
    return image_dict


def get_perplexity(model, processor, input_texts: Union[str, List[str]], sd_images=None, batch_size: Optional[int]=None, disable_tqdm=False):
    # adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py
    if sd_images is None and "paligemma" in type(processor).__name__.lower():
        processor = processor.tokenizer

    # batch indexing preparation
    single_input = type(input_texts) == str
    input_texts = [input_texts] if single_input else input_texts
    batch_size = len(input_texts) if batch_size is None else batch_size
    batch_ids = list(range(0, len(input_texts), batch_size)) + [len(input_texts)]
    batch_ids = list(zip(batch_ids[:-1], batch_ids[1:]))

    # perplexity computation
    ignore_index = model.config.ignore_index if hasattr(model.config, "ignore_index") else PAD_TOKEN_LABEL_ID
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    forward_func = model.get_perplexity if model.patched_perplexity else model.forward

    loss_list = []
    for s, e in tqdm(batch_ids, disable=disable_tqdm):
        # prepare model inputs, with or without images
        kwargs = {}
        if sd_images is not None and "processor" in type(processor).__name__.lower():
            images = sd_images[s:e]
            if "idefics" in type(processor).__name__.lower():
                images = [[i] for i in images if type(i) != list]
            assert len(images) == e - s
            kwargs = {"images": images}

        # tokenize and prepare model inputs
        model_inputs = processor(input_texts[s:e], **kwargs, padding=True, return_tensors='pt')
        if sd_images is None and "pixel_values" in model_inputs: del model_inputs["pixel_values"]
        model_inputs = model_inputs.to(model.device)
        labels = model_inputs.input_ids.clone()

        if model.patched_perplexity:
            model_inputs.update({'labels': labels})

        with torch.inference_mode():
            outputs = forward_func(**model_inputs)

        if model.patched_perplexity:
            loss: torch.FloatTensor = outputs.loss
        else: # compute the loss ourselves
            logits = outputs.logits
            if model.pad_token_initialized:
                logits = logits[:, :, :-1]
            shift_attention_mask = model_inputs.attention_mask[..., 1:].to(logits.device)
            valid_lengths = shift_attention_mask.sum(dim=-1)
            shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[:, 1:].contiguous()
            shift_labels[shift_attention_mask == 0] = ignore_index
            loss = loss_fct(shift_logits.contiguous().view(-1, shift_logits.size(-1)), shift_labels.contiguous().view(-1))
            loss = loss.view(len(logits), -1)
            loss = torch.sum(loss, -1) / valid_lengths

        assert len(loss) == e - s
        loss_list.extend(loss.cpu().tolist())

    ppl = [np.exp(i) for i in loss_list]
    return ppl[0] if single_input else ppl


def apply_chat_template(prompt, checkpoint, with_image=False):
    image_token = "<image>" if with_image else ""
    if "idefics2" in checkpoint:
        template = 'User:{}{}<end_of_utterance>\nAssistant:'
    elif "mistral" in checkpoint.lower():
        template = "[INST] {}{} [/INST]"
        if "Mistral" in checkpoint:
            image_token = ""
    elif "llava-hf" in checkpoint or "vicuna" in checkpoint:
        template = "USER: {}{}\nASSISTANT:"
        if "llava-hf" in checkpoint:
            image_token += "\n" if with_image else ""
        else:
            image_token = ""
    elif "Meta-Llama-3-8B-Instruct" in checkpoint:
        template = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        image_token = ""
    elif "gemma-1.1-2b-it" in checkpoint:
        template = '<bos><start_of_turn>user\n{}{}<end_of_turn>\n<start_of_turn>model\n'
        image_token = ""
    else:
        raise NotImplementedError(f"Chat template not implemented for checkpoint {checkpoint}")
    return template.format(image_token, prompt)