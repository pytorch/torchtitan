# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_fixed_block_mask_mod,
)
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward,
    GQAttention,
    RoPE,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.utils import get_moe_model_nparams_and_flops, trunc_normal_
from torchtitan.protocols.model import AttentionMasksType, BaseModel
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


def compute_moe_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 256,
    ffn_dim_multiplier: float | None = None,
    auto_scale_hidden_dim: bool = True,
    top_k: int = 1,
    num_shared_experts: int = 1,
) -> int:
    """Compute the MoE expert hidden dimension for Llama4-style models.

    This replicates the original Llama4 computation order:
    1. int(2 * 4 * dim / 3)
    2. Apply ffn_dim_multiplier
    3. Auto-scale (divide by top_k + num_shared_experts)
    4. Round up to multiple_of

    Note: This differs from compute_ffn_hidden_dim which applies multiple_of
    rounding BEFORE any auto-scaling.
    """
    hidden_dim = 4 * dim
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)

    if auto_scale_hidden_dim:
        hidden_dim_denom = top_k + num_shared_experts
        hidden_dim = int(hidden_dim / hidden_dim_denom)

    hidden_dim += -hidden_dim % multiple_of
    return hidden_dim


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        config (Transformer.Config): Model configuration.
    """

    def __init__(self, layer_id: int, config: "Transformer.Config"):
        super().__init__()

        # iRoPE: determine per-layer use_rope and fixed_attn_block_size
        attn_use_rope = True
        if config.every_n_layers_nope is not None:
            if config.every_n_layers_nope <= 1:
                raise ValueError("every_n_layers_nope must be greater than 1")
            if layer_id % config.every_n_layers_nope == 0:
                attn_use_rope = False

        # Create per-layer attention config with potentially overridden use_rope
        if not attn_use_rope:
            layer_attn_config = dataclasses.replace(config.attn_config, use_rope=False)
        else:
            layer_attn_config = config.attn_config

        self.attention = layer_attn_config.build(dim=config.dim)

        # use MoE layer for every interleave_moe_layer_step FFN layers
        self.moe_enabled = (layer_id + 1) % config.interleave_moe_layer_step == 0
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
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            out = h + self.moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Transformer(BaseModel):
    """
    Transformer Module

    Args:
        config (Transformer.Config): Model configuration.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 202048
        norm_eps: float = 1e-5
        max_seq_len: int = 1048576
        # If `True`, then each transformer block init uses its layer ID, and if
        # `False`, each uses the total number of transformer blocks
        depth_init: bool = True
        attn_mask_type: str = "causal"

        # iRoPE settings
        # When ``every_n_layers_nope`` is specified, NoPE (no positional embedding) is
        # used every n layers. Other layers uses RoPE (rotary positional embedding) and
        # the inner attention of those layer will use the fixed block size specified by
        # ``fixed_attn_block_size``. ``fixed_attn_block_size`` means that the query will
        # only attend to the tokens within the same block regardless how long is the
        # sequence.
        every_n_layers_nope: int | None = None
        fixed_attn_block_size: int = 8192

        # MoE
        moe_config: MoE.Config = field(default_factory=MoE.Config)
        # frequency of using MoE layer instead of feedforward layer in a transformer block
        interleave_moe_layer_step: int = 2

        # Sub-component configs
        ff_config: FeedForward.Config = field(
            default_factory=lambda: FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(4096)
            )
        )
        rope_config: RoPE.Config = field(
            default_factory=lambda: RoPE.Config(
                dim=4096 // 32,
                max_seq_len=1048576,
                theta=10000.0,
                format="complex",
                scaling="llama",
                scaling_factor=16.0,
                high_freq_factor=1.0,
            )
        )
        attn_config: GQAttention.Config = field(
            default_factory=lambda: GQAttention.Config(
                n_heads=32,
                attn_type="sdpa",
                rope_format="complex",
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
            if seq_len > self.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
                )
            self.max_seq_len = seq_len
            # Sync rope_config max_seq_len
            self.rope_config = dataclasses.replace(
                self.rope_config, max_seq_len=seq_len
            )

            if self.moe_config.use_grouped_mm and not has_cuda_capability(9, 0):
                logger.warning(
                    "Failed to use grouped mm, which is only supported on SM90 or later",
                )
                self.moe_config.use_grouped_mm = False

            if (
                parallelism.context_parallel_degree > 1
                and self.attn_config.attn_type != "sdpa"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA attention. "
                    f"Got attn_type='{self.attn_config.attn_type}'. "
                    f"FlexAttention and varlen attention are not supported with CP."
                )

            self.moe_config._debug_force_load_balance = debug.moe_force_load_balance

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.attn_config.n_heads,
                2 * (self.dim // self.attn_config.n_heads),
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.rope = config.rope_config.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, config)
        self.norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def init_weights(self, *, buffer_device: torch.device | None = None, **kwargs):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        self.rope.init_weights(buffer_device=buffer_device)
        self.freqs_cis = self.rope.cache
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                cast(TransformerBlock, layer).init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.config.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
                B = input_batch.shape[0]
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.config.attn_mask_type}"
                )

        rope_mask_mod = and_masks(
            *mask_mods,
            get_fixed_block_mask_mod(self.config.fixed_attn_block_size),
        )
        nope_mask_mod = and_masks(*mask_mods)

        seqlen = input_batch.shape[1]
        return {
            "rope": create_attention_mask(rope_mask_mod, B, None, seqlen, seqlen),
            "nope": create_attention_mask(nope_mask_mod, B, None, seqlen, seqlen),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
