import torch
from torch import nn
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, image_size_to_num_patches
from transformers.models.llava.modeling_llava import  LlavaCausalLMOutputWithPast
from transformers.models.paligemma.modeling_paligemma import PaliGemmaCausalLMOutputWithPast, Cache
# from transformers.models.idefics2.modeling_idefics2 import Idefics2CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List


def get_perplexity_llava(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
            # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                )

            image_features = self.multi_modal_projector(selected_image_feature)
            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        ignore_index = loss_fct.ignore_index
        # exclude the logits of the pad token id (the last token because we manually added it)
        if self.pad_token_initialized:
            logits = logits[:, :, :-1]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # note here, the original code uses attention_mask to index, but here we just set the ignore_index to block the loss
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            valid_lengths = shift_attention_mask.sum(dim=-1)
            shift_labels[shift_attention_mask == 0] = ignore_index
        else:
            valid_lengths = shift_labels.ne(ignore_index).sum(dim=-1)
        # compute loss, flatten the tokens
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )
        loss = loss.view(len(logits), -1)
        loss = torch.sum(loss, -1) / valid_lengths

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def get_perplexity_llavanext(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    image_sizes: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_length=30)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
        inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
            # ! infer image_num_patches from image_sizes
            image_num_patches = [
                image_size_to_num_patches(
                    image_size=imsize,
                    grid_pinpoints=self.config.image_grid_pinpoints,
                    patch_size=self.config.vision_config.image_size,
                )
                for imsize in image_sizes
            ]
            # figure out if pixel_values is concatenated or stacked
            if pixel_values.dim() == 5:
                # stacking when input is (batch_size, num_patches, num_channels, height, width)
                _pixel_values_list = [
                    pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                ]
                pixel_values = torch.cat(_pixel_values_list, dim=0)
            elif pixel_values.dim() != 4:
                # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

            image_features = self.vision_tower(pixel_values, output_hidden_states=True)
            selected_image_feature = image_features.hidden_states[vision_feature_layer]

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature

            image_features = self.multi_modal_projector(selected_image_feature)

            image_features = torch.split(image_features, image_num_patches, dim=0)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                image_newline=self.image_newline,
            )

            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds, attention_mask, position_ids, labels = self._merge_input_ids_with_image_features(
                image_features,
                feature_lens,
                inputs_embeds,
                input_ids,
                attention_mask,
                position_ids,
                labels=labels,
            )

        # pixel_values is not None but is empty ---> text only cases
        elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
            # there are no images
            pass

        # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
        # generation with cache
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            # Retrieve the first layer to inspect the logits and mask out the hidden states
            # that are set to 0
            first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    logits = outputs[0]

    loss = None
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        ignore_index = loss_fct.ignore_index
        # exclude the logits of the pad token id (the last token because we manually added it)
        if self.pad_token_initialized:
            logits = logits[:, :, :-1]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # note here, the original code uses attention_mask to index, but here we just set the ignore_index to block the loss
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            valid_lengths = shift_attention_mask.sum(dim=-1)
            shift_labels[shift_attention_mask == 0] = ignore_index
        else:
            valid_lengths = shift_labels.ne(ignore_index).sum(dim=-1)
        # compute loss, flatten the tokens
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )
        loss = loss.view(len(logits), -1)
        loss = torch.sum(loss, -1) / valid_lengths

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaNextCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_gemma(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, PaliGemmaCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/PaliGemma-test-224px-hf")
    >>> processor = AutoProcessor.from_pretrained("google/PaliGemma-test-224px-hf")

    >>> prompt = "answer en Where is the cow standing?"
    >>> url = "https://huggingface.co/gv-hf/PaliGemma-test-224px-hf/resolve/main/cow_beach_1.png"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_length=30)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "answer en Where is the cow standing?\nbeach"
    ```"""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # the attention mask is turned 4d after, we keep track of the original one
    input_attention_mask = attention_mask

    if inputs_embeds is None:
        # 1. Extra the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        if pixel_values is not None and input_ids.shape[1] != 1:
            image_outputs = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.multi_modal_projector(selected_image_feature)

            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, labels
            )

        else:
            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                # TODO @molbap this will only work for dynamic cache.
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_seqlen = cache_position[-1] + 1

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses PaliGemma+ Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
    attention_mask = attention_mask.to(inputs_embeds.dtype)
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    logits = outputs.logits
    logits = logits.float()
    # loss = None
    # if labels is not None:
    #     loss_fct = nn.CrossEntropyLoss(reduction="none")
    #     ignore_index = loss_fct.ignore_index
    #     # exclude the logits of the pad token id (the last token because we manually added it)
    #     if self.pad_token_initialized:
    #         logits = logits[:, :, :-1]
    #     # Shift so that tokens < n predict n
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     # note here, the original code uses attention_mask to index, but here we just set the ignore_index to block the loss
    #     if input_attention_mask is not None: # attention_mask is now 4d, use the original 2d one
    #         shift_attention_mask = input_attention_mask[..., 1:]
    #         valid_lengths = shift_attention_mask.sum(dim=-1)
    #         shift_labels[shift_attention_mask == 0] = ignore_index
    #     else:
    #         valid_lengths = shift_labels.ne(ignore_index).sum(dim=-1)
    #     # compute loss, flatten the tokens
    #     loss = loss_fct(
    #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
    #     )
    #     loss = loss.view(len(logits), -1)
    #     loss = torch.sum(loss, -1) / valid_lengths
    if labels is not None:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        if input_attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            shift_attention_mask = input_attention_mask[..., 1:]
            shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(logits.device) != 0].contiguous()
        else:
            shift_logits = shift_logits.contiguous()
            shift_labels = shift_labels.contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()

        flat_logits = shift_logits.view(-1, self.config.vocab_size)
        flat_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(flat_logits, flat_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return PaliGemmaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_model_perplexity(model, model_type=None):
    model.patched_perplexity = True
    if model_type is None:
        model_type = type(model).__name__.lower()
    if "llava" in model_type and "next" not in model_type:
        model.get_perplexity = get_perplexity_llava.__get__(model)
    elif "llavanext" in model_type:
        model.get_perplexity = get_perplexity_llavanext.__get__(model)
    # elif "paligemma" in model_type:
    #     model.get_perplexity = get_perplexity_paligemma.__get__(model)
    # elif "idefics2" in model_type:
    #     model.get_perplexity = get_perplexity_idefics2.__get__(model)
    else:
        model.patched_perplexity = False
        print("Warn: patching requested but model of type {} was not patched".format(model_type))
