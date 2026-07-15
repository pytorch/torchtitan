# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks, BlockMask

from torchtitan.distributed.minimal_async_ep.api import (
    maybe_update_minimal_async_ep_config,
)

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
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict


__all__ = ["Decoder", "MTPDecoder", "TransformerBlock"]


# TODO: we can unify the TransformerBlock impl across all models when
# there is no special logic for each model, including
# ffn vs. moe naming and creation, etc.
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
        # Tie ``tok_embeddings`` and ``lm_head`` to share one weight. Models
        # that support it set this True in their config factories; the tying
        # itself is handled by ``Decoder.__init__`` / ``Decoder.init_states``.
        enable_weight_tying: bool = False

        @property
        def first_attention(self) -> BaseAttention.Config | None:
            """Attention config of the first layer that has one, else None.

            Hybrid models (linear + full attention) don't carry an attention
            config on every layer, so callers needing attention metadata (TP
            validation, FLOPs, mask type) look up the first full-attention
            layer rather than assuming ``layers[0]``.
            """
            return next(
                (
                    layer.attention
                    for layer in self.layers
                    if layer.attention is not None
                ),
                None,
            )

        @property
        def max_seq_len(self) -> int:
            # The first full-attention layer's RoPE defines the context length.
            rope_cfg = getattr(self.first_attention, "rope", None)
            if rope_cfg is None:
                raise ValueError("Decoder config does not define RoPE max_seq_len.")
            return rope_cfg.max_seq_len

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

            if self.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "Weight tying is not supported with Pipeline Parallel."
                )

            if parallelism.pipeline_parallel_degree > 1 and any(
                layer.attention is not None
                and isinstance(layer.attention.inner_attention, VarlenAttention.Config)
                for layer in self.layers
            ):
                raise ValueError(
                    "Pipeline Parallel is not compatible with VarlenAttention. "
                    "Use a FlexAttention backend (attn_backend='flex' or "
                    "'flex_flash') for pipelined models."
                )

            tp = parallelism.tensor_parallel_degree
            attention = self.first_attention
            if tp > 1 and attention is not None:
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
                if layer_cfg.moe is not None:
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

            maybe_update_minimal_async_ep_config(self, config)

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
                    if attention_cfg is not None:
                        attention_cfg.rope = dataclasses.replace(
                            attention_cfg.rope, max_seq_len=seq_len
                        )
                    if hasattr(layer_cfg, "moe") and layer_cfg.moe is not None:
                        layer_cfg.moe.router._debug_force_load_balance = (
                            debug.moe_force_load_balance
                        )

    # Set by the trainer when ChunkedLossWrapper is used, so lm_head is applied
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

        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        if self.enable_weight_tying:
            # Re-tie before init: on meta device the ``__init__`` tying may not
            # have taken effect, and ``tok_embeddings.weight`` is skipped by
            # ``skip_param_init``, so re-point it at the initialized lm_head
            # weight.
            assert self.tok_embeddings is not None and self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight
        super().init_states(buffer_device=buffer_device)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        # positions is listed before attention_masks so AutoParallel's input_fn,
        # which returns (tokens, positions) and binds them positionally, maps
        # positions to the right parameter (it would otherwise land in the
        # attention_masks slot and break the maskless SDPA backend).
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
        return self.lm_head(h) if self.lm_head is not None else h

    def _create_flex_attention_mask(
        self,
        positions: torch.Tensor,
        attn_config: BaseAttention.Config,
        mask_mods: Sequence[_mask_mod_signature],
    ) -> BlockMask:
        """Build a flex-attention BlockMask from mask_mods (ANDed together),
        respecting the config's block_size and batch-invariant mode."""
        assert isinstance(attn_config.inner_attention, FlexAttention.Config)
        B = positions.shape[0]
        seq_len = positions.shape[1]
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

    def _create_flex_attention_mask_for_document(
        self,
        positions: torch.Tensor,
        attn_config: BaseAttention.Config,
    ) -> BlockMask:
        """Build the standard causal + packed-document flex-attention mask."""
        return self._create_flex_attention_mask(
            positions,
            attn_config,
            [
                get_causal_mask_mod(),
                get_efficient_causal_mask_mod_for_packed_document(positions),
            ],
        )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType | None:
        attn_config = self.config.first_attention
        if attn_config is None:
            # No full-attention layers (e.g. a pure linear-attention model, or a
            # pipeline stage holding only linear-attention blocks) → no masks.
            return None
        inner_attn = attn_config.inner_attention
        if isinstance(inner_attn, FlexAttention.Config):
            return self._create_flex_attention_mask_for_document(positions, attn_config)
        elif isinstance(inner_attn, VarlenAttention.Config):
            return create_varlen_metadata_for_document(positions)
        else:
            raise TypeError(
                f"Only VarlenAttention and FlexAttention support attention masks, "
                f"got {type(inner_attn).__name__}"
            )


class MTPDecoder(Decoder):
    """Decoder variant that owns prepared-input MTP blocks.

    Dataloaders provide ``[B, S + K]`` tokens for K MTP depths. The main decoder
    consumes the first S tokens, then the MTP block consumes shifted S-token
    views and returns one hidden-state tensor per MTP depth.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Optional so one model class can serve both regular decoder and
        # decoder-with-MTP configs.
        mtp: "MTPBlock.Config | None" = None
        extra_mtp_tokens: int = 0

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            self.extra_mtp_tokens = (
                self.mtp.num_mtp_layers if self.mtp is not None else 0
            )
            MTPDecoder.materialize_mtp_config(self)

    @staticmethod
    def materialize_mtp_config(config: "MTPDecoder.Config") -> None:
        """Materialize the MTP inner block before model-specific sharding."""
        if config.mtp is None or config.mtp.num_mtp_layers <= 0:
            return

        from torchtitan.models.common.mtp import MTPBlock

        inner_cfg = (
            config.mtp.inner_block_config
            if config.mtp.inner_block_config is not None
            else config.layers[-1]
        )
        config.mtp.inner_block_config = MTPBlock.mtp_layer_config(
            inner_cfg,
            dim=config.dim,
        )

    def __init__(self, config: "MTPDecoder.Config"):
        self.materialize_mtp_config(config)
        super().__init__(config)
        self.mtp_block = (
            config.mtp.build()
            if config.mtp is not None and config.mtp.num_mtp_layers > 0
            else None
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        # positions is listed before attention_masks so AutoParallel's input_fn,
        # which returns (tokens, positions) and binds them positionally, maps
        # positions to the right parameter.
        mtp_tokens = tokens
        if self.mtp_block is not None and self.mtp_block.num_depths > 0:
            if self.tok_embeddings is None:
                raise ValueError("MTP decoder forward requires token embeddings.")
            tokens = self.mtp_block.prepare_main_tokens(tokens)
            if positions is not None and positions.shape[1] == mtp_tokens.shape[1]:
                positions = positions[:, : tokens.shape[1]]

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        mtp_prev_embed = h
        h = self.norm(h) if self.norm is not None else h

        if self.mtp_block is not None and self.mtp_block.num_depths > 0:
            mtp_hidden_states = self.mtp_block(
                mtp_tokens,
                mtp_prev_embed,
                self.tok_embeddings,
                attention_masks,
                positions,
            )
            h = [h] + mtp_hidden_states

        if self._skip_lm_head:
            return h
        if isinstance(h, list):
            return [
                self.lm_head(item) if self.lm_head is not None else item
                for item in h
            ]
        return self.lm_head(h) if self.lm_head is not None else h
