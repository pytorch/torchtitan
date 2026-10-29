# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym's paged linear attention (GDN/KDA) for the native vLLM runner.

One attention backend and metadata builder serve both linear-attention families.
``configure_hybrid_state`` attaches vLLM's model-level recurrent-state interface;
each family supplies its state shapes, dtypes and copy functions.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import auto, Enum
from typing import Any, TypeVar

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class LinearAttentionExecutionPath(Enum):
    SINGLE_TOKEN = auto()
    PACKED = auto()


@dataclass(kw_only=True)
class TorchTitanLinearAttentionMetadata(GDNAttentionMetadata):
    """vLLM's GDN metadata, which vLLM also uses for KDA, plus the execution path."""

    execution_path: LinearAttentionExecutionPath
    # Block-table column of each request's running state, ``[num_reqs, 1]``; the same
    # in every group, so ``update_block_table`` reuses it.
    state_block_column: torch.Tensor


class TorchTitanLinearAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TORCHTITAN_LINEAR_ATTENTION"

    @staticmethod
    # The upstream abstract method has no return annotation.
    # pyrefly: ignore[bad-override]
    def get_builder_cls() -> type["TorchTitanLinearAttentionMetadataBuilder"]:
        return TorchTitanLinearAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


class TorchTitanLinearAttentionMetadataBuilder(
    AttentionMetadataBuilder[TorchTitanLinearAttentionMetadata]
):
    """Stage shared request buffers before eager execution or native graph replay."""

    reorder_batch_threshold = 1
    supports_update_block_table = True

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport:
        parallel = vllm_config.parallel_config
        if (
            parallel.data_parallel_size != 1
            or parallel.use_ubatching
            or parallel.enable_dbo
        ):
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        # The abstract base initializer has a concrete field-initialization body.
        # pyrefly: ignore[missing-attribute]
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        compilation = vllm_config.compilation_config
        if (
            compilation.cudagraph_mode.has_full_cudagraphs()
            and compilation.cudagraph_num_of_warmups < 1
        ):
            raise ValueError(
                "Linear attention (GDN/KDA) FULL capture requires at least one kernel warmup"
            )
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise ValueError(
                "Attention Gym linear attention (GDN/KDA) does not support "
                "intermediate prefix-cache checkpoints; use 'none' or 'align'"
            )
        self.num_reqs_capacity = vllm_config.scheduler_config.max_num_seqs
        # Always allocate one extra interval in query_start_loc (cu_seqlens)
        # so any tokens left in the padded buffer belong to the last, null request.
        self.query_start_loc = torch.zeros(
            self.num_reqs_capacity + 2, device=device, dtype=torch.int32
        )
        self.state_indices = torch.zeros(
            self.num_reqs_capacity + 1, device=device, dtype=torch.int32
        )
        self.has_initial_state = torch.zeros(
            self.num_reqs_capacity + 1, device=device, dtype=torch.bool
        )
        # Decode has one sequence per physical row, including null padding.
        self.decode_query_start_loc = torch.arange(
            self.num_reqs_capacity + 1, device=device, dtype=torch.int32
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TorchTitanLinearAttentionMetadata:
        m = common_attn_metadata
        capacity = min(self.num_reqs_capacity, m.num_actual_tokens)
        if not 0 <= m.num_reqs <= capacity:
            raise ValueError(
                "Linear attention (GDN/KDA) request count exceeds the batch's metadata capacity"
            )
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            max_state_seq_len = (
                m.block_table_tensor.shape[1] * self.kv_cache_spec.block_size
            )
            if m.max_seq_len > max_state_seq_len:
                raise ValueError(
                    "Linear attention (GDN/KDA) sequence length exceeds the state "
                    f"block-table capacity; got {m.max_seq_len} and "
                    f"{max_state_seq_len}. CUDA graph capture sizes (bounded by "
                    "max_num_batched_tokens) must not exceed max_model_len."
                )
        # vLLM's `mamba_get_block_table_tensor`, one column: "align" keeps the
        # running state in the request's last block, "none" in its only one.
        seq_lens = m.seq_lens[: m.num_reqs]
        if self.vllm_config.cache_config.mamba_cache_mode == "align":
            state_block_column = (seq_lens - 1) // self.kv_cache_spec.block_size
            # 0-length requests in CUDA graph capture read column 0.
            state_block_column = state_block_column.clamp_(min=0).long().unsqueeze(1)
        else:
            state_block_column = seq_lens.new_zeros(m.num_reqs, 1, dtype=torch.long)
        self._stage_state_indices(m.block_table_tensor, state_block_column)
        self.query_start_loc[: m.num_reqs + 1].copy_(
            m.query_start_loc[: m.num_reqs + 1]
        )
        # Under FULL, num_actual_tokens is the padded capacity, while the last
        # query offset is the real token count. The appended interval is null.
        self.query_start_loc[m.num_reqs + 1 :].fill_(m.num_actual_tokens)
        self.has_initial_state[: m.num_reqs].copy_(
            m.compute_num_computed_tokens()[: m.num_reqs] > 0
        )
        self.has_initial_state[m.num_reqs :].zero_()
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(m, decode_threshold=1)
            if m.num_reqs
            else (0, 0, 0, 0)
        )
        # If every request is processing one token, use the single-token
        # (decode-style) path.
        single_token = m.max_query_len == 1
        return TorchTitanLinearAttentionMetadata(
            execution_path=(
                LinearAttentionExecutionPath.SINGLE_TOKEN
                if single_token
                else LinearAttentionExecutionPath.PACKED
            ),
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            # The linear attention (GDN/KDA) layer constructors reject speculative decoding.
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=m.num_actual_tokens,
            non_spec_query_start_loc=(
                self.decode_query_start_loc[: capacity + 1]
                if single_token
                else self.query_start_loc[: capacity + 2]
            ),
            non_spec_state_indices_tensor=(
                self.state_indices[:capacity]
                if single_token
                else self.state_indices[: capacity + 1]
            ),
            has_initial_state=(
                self.has_initial_state[:capacity]
                if single_token
                else self.has_initial_state[: capacity + 1]
            ),
            state_block_column=state_block_column,
        )

    def update_block_table(
        self,
        metadata: TorchTitanLinearAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> TorchTitanLinearAttentionMetadata:
        """Reuse another group's metadata for this group's block table.

        vLLM calls this instead of ``build`` for the step's later groups with the same
        spec. Only the state slots differ; the other buffers are copied into this
        builder's own, which its CUDA graphs captured.
        """
        del slot_mapping
        self._stage_state_indices(blk_table, metadata.state_block_column)
        extent = metadata.has_initial_state.shape[0]
        self.has_initial_state[:extent].copy_(metadata.has_initial_state)
        # Single-token metadata runs eager or replays decode graphs, both reading the
        # static decode offsets, so only the packed path copies the query offsets.
        query_start_loc = self.decode_query_start_loc[: extent + 1]
        if metadata.execution_path is LinearAttentionExecutionPath.PACKED:
            query_start_loc = self.query_start_loc[: extent + 1]
            query_start_loc.copy_(metadata.non_spec_query_start_loc)
        return replace(
            metadata,
            non_spec_query_start_loc=query_start_loc,
            non_spec_state_indices_tensor=self.state_indices[:extent],
            has_initial_state=self.has_initial_state[:extent],
        )

    def _stage_state_indices(
        self, block_table: torch.Tensor, state_block_column: torch.Tensor
    ) -> None:
        """Write each request's state slot; zero the rows past the requests (null slot)."""
        num_reqs = state_block_column.shape[0]
        torch.gather(
            block_table[:num_reqs],
            1,
            state_block_column,
            out=self.state_indices[:num_reqs].unsqueeze(1),
        )
        self.state_indices[num_reqs:].zero_()

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TorchTitanLinearAttentionMetadata:
        metadata = self.build(0, common_attn_metadata)
        self.state_indices.zero_()
        self.has_initial_state.zero_()
        return metadata


_StateShapeT = TypeVar("_StateShapeT", bound=tuple)


def unique_state_shape(
    family: str, state_shapes: Iterable[_StateShapeT]
) -> _StateShapeT | None:
    """Return the one state shape shared by every layer of ``family``.

    vLLM exposes one model-level recurrent-state shape, so every field that
    determines it must match across layers. Returns None for no layers.
    """
    unique = set(state_shapes)
    if not unique:
        return None
    if len(unique) != 1:
        raise ValueError(
            f"All {family} layers must use the same state shape, got {unique}"
        )
    (state_shape,) = unique
    return state_shape


def configure_hybrid_state(
    model_cls: type,
    *,
    get_state_shape: Callable[[VllmConfig], tuple],
    get_state_dtype: Callable[[VllmConfig], tuple],
    get_state_copy_func: Callable[[], tuple],
) -> None:
    """Attach vLLM's hybrid recurrent-state interface to ``model_cls``."""

    def get_state_copy_funcs(cls, mamba_types: Iterable[Any]) -> dict[Any, tuple]:
        copy_funcs = cls.get_mamba_state_copy_func()
        return {mamba_type: copy_funcs for mamba_type in mamba_types}

    model_cls.is_hybrid = True
    model_cls.get_mamba_state_shape_from_config = classmethod(
        lambda cls, vllm_config: get_state_shape(vllm_config)
    )
    model_cls.get_mamba_state_dtype_from_config = classmethod(
        lambda cls, vllm_config: get_state_dtype(vllm_config)
    )
    model_cls.get_mamba_state_copy_func = classmethod(lambda cls: get_state_copy_func())
    model_cls.get_mamba_state_copy_funcs = classmethod(get_state_copy_funcs)
