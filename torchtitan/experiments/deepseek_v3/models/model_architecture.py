# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on model definition of `deepseek-ai/DeepSeek-V3-Base` on
# Hugging Face Model Hub. Url:
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/resolve/main/configuration_deepseek.py
#
# It has been modified from its original forms to accommodate naming convention
# and usage patterns of the TorchTitan project.

from typing import Optional, Tuple

import torch
from torch import nn

from .attention import Attention
from .mlp import MLP
from .model_config import ModelArgs
from .moe import MoE
from .normalization import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config=config, layer_idx=layer_idx)

        self.mlp = (
            MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else MLP(config)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepseekModel(torch.nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: ModelArgs
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Creating model parts related to my stage
        assert (
            config.stage_idx < config.num_stages
        ), f"Stage {config.stage_idx} is not in the model"
        print(f"Creating model stage {config.stage_idx} of {config.num_stages}")

        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            if config.stage_idx == 0
            else None
        )

        self.layers = torch.nn.ModuleDict()
        division = config.num_hidden_layers // config.num_stages
        residual = config.num_hidden_layers % config.num_stages
        # Some earlier stages may have 1 more layer than latter stages because
        # the division may have residual; this is more even than giving the
        # entire residual to the last stage.
        layers_per_stage = [
            division + 1 if stage < residual else division
            for stage in range(config.num_stages)
        ]
        assert sum(layers_per_stage) == config.num_hidden_layers
        layer_id_start = sum(layers_per_stage[: config.stage_idx])
        layer_id_end = layer_id_start + layers_per_stage[config.stage_idx]
        for layer_id in range(layer_id_start, layer_id_end):
            self.layers[str(layer_id)] = DecoderLayer(config, layer_id)

        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.stage_idx == config.num_stages - 1
            else None
        )

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Embedding
        hidden_states = (
            self.embed_tokens(tokens) if self.embed_tokens is not None else tokens
        )

        # decoder layers
        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        hidden_states = (
            self.norm(hidden_states) if self.norm is not None else hidden_states
        )
        return hidden_states


class DeepseekForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = DeepseekModel(config)
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if config.stage_idx == config.num_stages - 1
            else None
        )

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekForCausalLM

        >>> model = DeepseekForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        hidden_states = self.model(
            tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        logits = (
            self.lm_head(hidden_states) if self.lm_head is not None else hidden_states
        )
        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # Assuming isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    # Setup Symmetric Memory for MoE token shuffle.
    # Supports inference currently.
    def setup_symm_mem(self, dtype: torch.dtype, device: torch.device):
        for layer in self.model.layers.values():
            if not isinstance(layer.mlp, MoE):
                continue
            layer.mlp.setup_symm_mem(dtype, device)
