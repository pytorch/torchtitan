# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from spmd_types import SpmdType

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    ContextMetadata,
    InnerAttention,
)
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.token_dispatcher import update_ep_token_dispatcher_config
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

__all__ = ["Decoder", "TransformerBlock"]


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
        max_context_length: int
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
        def first_full_attention_backend(self) -> Module.Config | None:
            """Backend config of the first full-attention layer, else None."""
            attention = self.first_attention
            return attention.inner_attention if attention is not None else None

        @property
        def first_feed_forward(self) -> FeedForward.Config | None:
            """First dense feed-forward config, else None."""
            return next(
                (
                    layer.feed_forward
                    for layer in self.layers
                    if layer.feed_forward is not None
                ),
                None,
            )

        @property
        def first_moe(self) -> MoE.Config | None:
            """First mixture-of-experts config, else None."""
            return next(
                (layer.moe for layer in self.layers if layer.moe is not None),
                None,
            )

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            """Apply runtime config to model config.

            Non-trainer callers may pass any config-like
            object with a ``ParallelismConfig`` in its ``parallelism`` field; in
            that case the training/debug setup is skipped.
            """
            from torchtitan.config.parallelism import ParallelismConfig
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

            moe_configs = list(self.traverse(MoE.Config))
            ep = parallelism.expert_parallel_degree
            if moe_configs and ep < tp:
                raise ValueError(
                    f"MoE models require expert_parallel_degree ({ep}) to be "
                    f"greater than or equal to tensor_parallel_degree ({tp})."
                )
            for moe_fqn, moe, _, _ in moe_configs:
                if moe.num_experts % ep != 0:
                    raise ValueError(
                        f"{moe_fqn}.num_experts ({moe.num_experts}) must be "
                        f"divisible by expert_parallel_degree ({ep})."
                    )

            update_ep_token_dispatcher_config(self, config)

            if isinstance(config, Trainer.Config):
                for layer_cfg in self.layers:
                    if hasattr(layer_cfg, "moe") and layer_cfg.moe is not None:
                        layer_cfg.moe.router._debug_force_load_balance = (
                            config.debug.moe_force_load_balance
                        )

    # Set by the trainer when ChunkedLossWrapper is used, so lm_head is applied
    # per-chunk inside the loss function instead of in forward().
    # TODO(#ISSUE): Remove after fixing PP backward to skip non-tensor
    # inputs (bool kwargs cause 'has no attribute requires_grad' errors).
    _skip_lm_head: bool = False

    def _apply_fsdp(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
        apply_fsdp_to_decoder(
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        )

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
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        # positions is listed before attention_masks so AutoParallel's input_fn,
        # which returns (tokens, positions) and binds them positionally, maps
        # positions to the right parameter (it would otherwise land in the
        # attention_masks slot and break the maskless SDPA backend).
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions, padding_mask=padding_mask)

        h = self.norm(h) if self.norm is not None else h

        # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
        # calls .requires_grad on all stage inputs, which fails on bool kwargs.
        # TODO: fix PP backward upstream to skip non-tensor inputs
        if self._skip_lm_head:
            return h
        output = self.lm_head(h) if self.lm_head is not None else h
        return output

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        """Build masks (flex/varlen), CP-shard, SPMD-wrap, and return the batch."""
        del kwargs
        positions = input_dict.get("positions", None)
        padding_mask = input_dict.get("padding_mask", None)
        if positions is not None:
            attention_masks = self.get_attention_masks(
                positions=positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            if attention_masks is not None:
                input_dict["attention_masks"] = attention_masks

        input_shardings = decoder_input_sharding()
        if parallel_dims.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallel_dims=parallel_dims,
                parallelism=parallelism,
            )
        input_dict = annotate_input_spmd_types(
            parallel_dims, input_dict, input_shardings
        )

        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        return inputs, labels, input_dict

    def _cp_shard(
        self,
        input_dict: dict[str, Any],
        input_shardings: dict[str, SpmdType],
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> dict[str, Any]:
        """Prepare attention metadata and shard model inputs for CP."""
        from torchtitan.distributed import context_parallel
        from torchtitan.models.common.cp_attention import CPInnerAttention

        cp_attention_backends: list[type[CPInnerAttention[Any, Any]]] = []
        for _, config, _, _ in self.config.traverse(
            CPInnerAttention.Config, recurse=True
        ):
            backend = config._owner
            assert backend is not None and issubclass(backend, CPInnerAttention)
            if backend not in cp_attention_backends:
                cp_attention_backends.append(backend)
        context_metadata = input_dict.get("attention_masks")
        load_balancer_metadata = context_metadata
        if isinstance(context_metadata, Mapping):
            load_balancer_metadata = next(
                (
                    context_metadata[backend]
                    for backend in cp_attention_backends
                    if backend in context_metadata
                ),
                context_metadata,
            )
        load_balancer_config = parallelism.context_parallel_load_balancer
        load_balancer = (
            load_balancer_config.build(
                seq_len=context_parallel.get_cp_input_seq_len(
                    input_dict, input_shardings=input_shardings
                ),
                attention_metadata=load_balancer_metadata,
            )
            if load_balancer_config is not None
            else None
        )
        permutation = (
            load_balancer.generate_permutation() if load_balancer is not None else None
        )
        if permutation is not None and "attention_masks" in input_dict:
            context_metadata = input_dict["attention_masks"]
            assert isinstance(context_metadata, dict)
            for backend in cp_attention_backends:
                context_metadata[backend] = backend.prepare_cp_metadata(
                    context_metadata[backend],
                    permutation=permutation,
                )
        return context_parallel.shard_tensors(
            input_dict,
            input_shardings=input_shardings,
            permutation=permutation,
        )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> ContextMetadata | None:
        context_metadata: dict[type[InnerAttention], AttentionMasksType] = {}
        for _, config, _, _ in self.config.traverse(
            InnerAttention.Config, recurse=True
        ):
            backend = config._owner
            assert backend is not None and issubclass(backend, InnerAttention)
            if backend in context_metadata:
                continue
            metadata = backend.build_context_metadata(
                positions,
                config=config,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            if metadata is not None:
                context_metadata[backend] = metadata
        return context_metadata or None
