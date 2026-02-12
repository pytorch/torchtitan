# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common import FeedForward, GQAttention, RoPE, trunc_normal_
from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    create_varlen_metadata_for_document,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import logger


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        config (Qwen3Model.Config): Model configuration.
    """

    def __init__(self, layer_id: int, config: "Qwen3Model.Config"):
        super().__init__()

        self.attention = config.attn_config.build(dim=config.dim)

        self.moe_enabled = config.moe_enabled
        if self.moe_enabled:
            self.moe = config.moe_config.build(dim=config.dim)
        else:
            self.feed_forward = config.ff_config.build(dim=config.dim)

        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * config.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        x = x + self.attention(
            self.attention_norm(x), rope_cache, attention_masks, positions
        )

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, *, buffer_device: torch.device, **kwargs):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Qwen3Model(BaseModel):
    """
    Qwen3Model Module

    Args:
        config (Qwen3Model.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 1024
        n_layers: int = 28
        vocab_size: int = 151936
        norm_eps: float = 1e-6
        depth_init: bool = True

        enable_weight_tying: bool = False

        # MoE params
        moe_enabled: bool = False
        moe_config: MoE.Config = field(default_factory=MoE.Config)

        # Sub-component configs
        ff_config: FeedForward.Config = field(
            default_factory=lambda: FeedForward.Config(hidden_dim=3072)
        )
        rope_config: RoPE.Config = field(
            default_factory=lambda: RoPE.Config(
                dim=128,
                max_seq_len=4096,
                theta=1000000.0,
                backend="cos_sin",
            )
        )
        attn_config: GQAttention.Config = field(
            default_factory=lambda: GQAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_eps=1e-6,
                attn_backend="sdpa",
                rope_backend="cos_sin",
            )
        )

        def update_from_config(
            self,
            *,
            job_config,
            **kwargs,
        ) -> None:
            training = job_config.training
            parallelism = job_config.parallelism
            debug = job_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope_config.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope_config.max_seq_len}."
                )
            # Sync rope_config max_seq_len
            import dataclasses as _dc

            self.rope_config = _dc.replace(self.rope_config, max_seq_len=seq_len)

            self.moe_config._debug_force_load_balance = debug.moe_force_load_balance

            if (
                parallelism.context_parallel_degree > 1
                and self.attn_config.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.attn_config.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert self.attn_config.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.attn_config.n_heads,
                2 * self.attn_config.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.enable_weight_tying = config.enable_weight_tying

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.rope = config.rope_config.build()
        self.register_buffer("rope_cache", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, config)
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        if self.enable_weight_tying:
            self.output.weight = self.tok_embeddings.weight

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
        **kwargs,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Qwen3Model`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.rope_cache.device
        if self.rope is not None:
            self.rope.init_weights(buffer_device=buffer_device)
            self.rope_cache = self.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get rope_cache
            rope = self.config.rope_config.build()
            rope.init_weights(buffer_device=buffer_device)
            self.rope_cache = rope.cache
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                cast(TransformerBlock, layer).init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3

        if self.enable_weight_tying:
            # since when the model is initialized on meta device,
            # the tying in the __init__ may not have worked correctly
            # we ensure the weights are tied here
            assert self.tok_embeddings is not None and self.output is not None
            self.output.weight = self.tok_embeddings.weight

        # The token embedding initialization produces weights with too large
        # standard deviation for the output layer. Reinitialize the output weights
        # with a smaller, truncated normal distribution to improve training stability.
        if self.output is not None:
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]

        match self.config.attn_config.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.config.attn_config.attn_mask_type}"
                )

        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.config.attn_config.attn_backend:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.config.attn_config.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal \
                        attention mask type, got {self.config.attn_config.attn_mask_type}"
                    )
                assert tokenizer.eos_id is not None
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Qwen3Model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output logits after applying the Qwen3Model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
