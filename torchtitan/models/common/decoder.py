# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import and_masks

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    VarlenAttention,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import Embedding, Linear, RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

__all__ = ["Decoder", "TransformerBlock"]


# TODO: we can unify the TransformerBlock impl across all models when
# there is no special logic for each model, including
# ffn vs. moe naming and creation, rope vs. nope, etc.
class TransformerBlock(Module):
    """Base class for all language model transformer blocks.

    All language model TransformerBlocks share:
    - Attention module (from ``attention.build()``)
    - FFN or MoE (from ``feed_forward.build()`` / ``moe.build()``)
    - Two RMSNorms (``attention_norm``, ``ffn_norm``)
    - Forward: ``x + attn(norm(x), ...); x + ffn(norm(x))``

    Children implement ``__init__`` and ``forward``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: BaseAttention.Config  # required, no default
        feed_forward: FeedForward.Config | None = None
        moe: MoE.Config | None = None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config


class Decoder(BaseModel):
    """Base class for autoregressive decoder-only language models.

    Provides shared ``__init__``, ``forward``, ``init_states``, and
    ``get_attention_masks`` (flex/varlen dispatch) used by most models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int
        vocab_size: int
        lm_head: Linear.Config
        tok_embeddings: Embedding.Config
        norm: RMSNorm.Config
        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3033849265
        # and fix the typing here
        layers: list  # list[TransformerBlock.Config] or subclass configs

        @property
        def max_seq_len(self) -> int:
            # Llama4/iRoPE can have NoPE layers with ``rope=None``; use the
            # first layer that carries RoPE to expose the model context length.
            for layer_cfg in self.layers:
                attention_cfg = getattr(layer_cfg, "attention", None)
                rope_cfg = getattr(attention_cfg, "rope", None)
                if rope_cfg is not None:
                    return rope_cfg.max_seq_len
            raise ValueError("Decoder config does not define RoPE max_seq_len.")

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            """Apply runtime config to model config.

            When *config* is a ``Trainer.Config``, validates
            ``training.seq_len`` against each attention layer's intrinsic
            RoPE max sequence length, resizes RoPE caches, and propagates
            debug flags. Non-trainer callers may pass any config-like
            object with a ``ParallelismConfig`` in its ``parallelism``
            field; in that case the training/debug setup is skipped.
            """
            from torchtitan.config import ParallelismConfig
            from torchtitan.trainer import Trainer

            assert hasattr(config, "parallelism"), (
                "config passed to update_from_config must provide "
                "a parallelism field."
            )
            parallelism = config.parallelism
            assert isinstance(parallelism, ParallelismConfig), (
                "config.parallelism must be a ParallelismConfig, got "
                f"{type(parallelism).__name__}."
            )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                attention = self.layers[0].attention
                n_heads = attention.n_heads
                n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads
                if n_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_heads ({n_heads})."
                    )
                if n_kv_heads % tp != 0:
                    raise ValueError(
                        f"tensor_parallel_degree ({tp}) must divide "
                        f"n_kv_heads ({n_kv_heads})."
                    )

            for layer_cfg in self.layers:
                if hasattr(layer_cfg, "moe") and layer_cfg.moe is not None:
                    from torchtitan.models.common.token_dispatcher import (
                        DeepEPTokenDispatcher,
                        HybridEPTokenDispatcher,
                    )

                    token_dispatcher_cfg = layer_cfg.moe.experts.token_dispatcher
                    if (
                        isinstance(
                            token_dispatcher_cfg,
                            (
                                DeepEPTokenDispatcher.Config,
                                HybridEPTokenDispatcher.Config,
                            ),
                        )
                        and parallelism.expert_parallel_degree == 1
                    ):
                        raise ValueError(
                            f"{type(token_dispatcher_cfg).__qualname__} "
                            "requires expert parallelism "
                            "(expert_parallel_degree > 1)."
                        )

            # NOTE: Inference-only callers such as the RL generator skip
            # training.seq_len sync. Generated sequence length is not known
            # ahead of time, so keep the RoPE cache at the model's max_seq_len.
            if isinstance(config, Trainer.Config):
                debug = config.debug
                seq_len = config.training.seq_len
                max_seq_len = self.max_seq_len
                if seq_len > max_seq_len:
                    raise ValueError(
                        f"Training sequence length {seq_len} exceeds "
                        f"attention RoPE maximum supported sequence "
                        f"length {max_seq_len}."
                    )

                for layer_cfg in self.layers:
                    attention_cfg = getattr(layer_cfg, "attention", None)
                    if (
                        attention_cfg is not None
                        and getattr(attention_cfg, "rope", None) is not None
                    ):
                        rope_cfg = attention_cfg.rope
                        attention_cfg.rope = dataclasses.replace(
                            rope_cfg, max_seq_len=seq_len
                        )
                    if hasattr(layer_cfg, "moe") and layer_cfg.moe is not None:
                        layer_cfg.moe.router._debug_force_load_balance = (
                            debug.moe_force_load_balance
                        )

    # Set by the trainer when ChunkedCELoss is used, so lm_head is applied
    # per-chunk inside the loss function instead of in forward().
    # TODO(#ISSUE): Remove after fixing PP backward to skip non-tensor
    # inputs (bool kwargs cause 'has no attribute requires_grad' errors).
    _skip_lm_head: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tok_embeddings = config.tok_embeddings.build()

        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = layer_config.build()

        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h

        # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
        # calls .requires_grad on all stage inputs, which fails on bool kwargs.
        # TODO: fix PP backward upstream to skip non-tensor inputs
        if self._skip_lm_head:
            return h
        output = self.lm_head(h) if self.lm_head is not None else h
        return output

    def _get_flex_attention_masks(
        self,
        positions: torch.Tensor,
        attn_config: BaseAttention.Config,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]

        match attn_config.mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = positions.shape[0]
                mask_mods.append(
                    get_efficient_causal_mask_mod_for_packed_document(positions)
                )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {attn_config.mask_type}"
                )

        seq_len = positions.shape[1]
        assert isinstance(attn_config.inner_attention, FlexAttention.Config)
        return create_attention_mask(
            and_masks(*mask_mods),
            B,
            None,
            seq_len,
            seq_len,
            device=positions.device,
            BLOCK_SIZE=attn_config.inner_attention.block_size,
            # when separate_full_blocks = True, kernel iterates through
            # full blocks first (blocks where all elements are unmasked)
            # but which blocks are "full" vs "partial" changes depending
            # on the particular batch
            # for batch invariance, we disable this optimization
            separate_full_blocks=not is_in_batch_invariant_mode(),
        )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType:
        attn_config = self.config.layers[0].attention
        inner_attn = attn_config.inner_attention
        if isinstance(inner_attn, FlexAttention.Config):
            return self._get_flex_attention_masks(positions, attn_config)
        elif isinstance(inner_attn, VarlenAttention.Config):
            if attn_config.mask_type != "block_causal":
                raise ValueError(
                    f"varlen attention is only supported with block_causal "
                    f"attention mask type, got {attn_config.mask_type}"
                )
            return create_varlen_metadata_for_document(positions)
        else:
            raise TypeError(
                f"Only VarlenAttention and FlexAttention support attention masks, "
                f"got {type(inner_attn).__name__}"
            )
