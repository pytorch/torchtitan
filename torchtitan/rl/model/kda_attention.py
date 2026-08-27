# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from attn_gym.linear.kda import (
    bound_gate,
    l2norm,
    paged_chunk_kda,
    recurrent_kda_decode,
)
from attn_gym.linear.short_conv import causal_conv1d_decode, paged_causal_conv1d

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.protocols.module import Module
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# Shape suffixes:
# T = packed tokens, C = projection channels, H = attention heads,
# K = query/key/value head dimension, W = convolution kernel width.

_CHUNK_SIZE = 64


def _batch_invariant_silu(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    return (x_float * torch.sigmoid(x_float)).to(x.dtype)


def _batch_invariant_l2norm(x_1THK: torch.Tensor) -> torch.Tensor:
    x_float = x_1THK.float()
    return (
        x_float * torch.rsqrt((x_float * x_float).sum(dim=-1, keepdim=True) + 1e-6)
    ).to(x_1THK.dtype)


class VLLMInnerKDA(Module, MambaBase):
    """vLLM replacement for ``InnerKDA`` using paged recurrent state.

    vLLM owns allocation, routing, prefix copies, and the ``[slot, H, V, K]`` recurrent
    cache shape. KDA uses ``K == V``, so Attention Gym can advance the same dense slot
    storage directly without changing vLLM's cache-manager contract.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        conv_kernel_size: int = 4
        lower_bound: float = -5.0
        layer_index: int = 0

    def __init__(self, config: Config) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        tp_degree = vllm_config.parallel_config.tensor_parallel_size
        if config.num_heads % tp_degree != 0:
            raise ValueError(
                f"num_heads ({config.num_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )

        self.tp_size = tp_degree
        self.num_heads = config.num_heads
        self.local_num_heads = config.num_heads // tp_degree
        self.head_dim = config.head_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.lower_bound = config.lower_bound
        self.use_chunk_replay = is_in_batch_invariant_mode()

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_spec = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )
        if self.num_spec != 0:
            raise ValueError("Attention Gym KDA does not support speculative decoding.")
        if is_conv_state_dim_first():
            raise ValueError(
                "Attention Gym KDA requires VLLM_SSM_CONV_STATE_LAYOUT=SD so "
                "the paged convolution history has contiguous channels."
            )

        # vLLM replaces these placeholders after allocating its paged state cache.
        self.kv_cache = tuple(
            torch.tensor([]) for _ in range(8 if self.use_chunk_replay else 2)
        )

        # vLLM keys per-layer metadata and the state cache by this name.
        self.prefix = f"model.layers.{config.layer_index}.kda"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return the convolution, recurrent, and optional replay state shapes."""
        shapes = MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_spec=self.num_spec,
        )
        if not self.use_chunk_replay:
            return shapes
        return (
            *shapes,
            (_CHUNK_SIZE, self.local_num_heads, self.head_dim),
            (_CHUNK_SIZE, self.local_num_heads, self.head_dim),
            (_CHUNK_SIZE, self.local_num_heads, self.head_dim),
            (_CHUNK_SIZE, self.local_num_heads, self.head_dim),
            (_CHUNK_SIZE, self.local_num_heads),
            (1,),
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )
        if not self.use_chunk_replay:
            return dtypes
        return (
            *dtypes,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.int32,
        )

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the vLLM cache operation on rank-local tensors.

        Signature matches :class:`~torchtitan.models.common.attention.InnerKDA`,
        the module this replaces. vLLM derives its own offsets from the per-layer
        metadata, so the caller's ``cu_seqlens`` is unused.
        """
        del cu_seqlens

        num_tokens = query_TC.shape[0]
        mixed_qkv_TC = torch.cat((query_TC, key_TC, value_TC), dim=-1)
        raw_gate_1THK = raw_gate_THK.unsqueeze(0)
        raw_beta_1TH = raw_beta_TH.unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        metadata = self._layer_metadata()
        output_THK = mixed_qkv_TC.new_zeros(
            num_tokens,
            self.local_num_heads,
            self.head_dim,
        )
        if metadata is None:
            # Profiling or cudagraph dummy capture: no live metadata yet.
            return output_THK

        live = metadata.num_actual_tokens
        mixed_qkv_TC = mixed_qkv_TC[:live]
        # The KDA kernels take a leading batch axis of 1.
        raw_gate = raw_gate_1THK.reshape(1, num_tokens, self.local_num_heads, -1)[
            :, :live
        ]
        raw_beta = raw_beta_1TH.reshape(1, num_tokens, -1)[:, :live]

        conv_state, recurrent_state, *replay_state = self.kv_cache
        # vLLM's conv kernels take (channels, width); Conv1d stores (C, 1, W).
        conv_weight = conv_weight_C1W.reshape(
            conv_weight_C1W.size(0), conv_weight_C1W.size(-1)
        )

        state_indices = metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens

        if num_decodes > 0:
            decode_args = (
                mixed_qkv_TC[:num_decode_tokens],
                raw_gate[:, :num_decode_tokens],
                raw_beta[:, :num_decode_tokens],
                A_log_H,
                dt_bias_HK,
                conv_state,
                conv_weight,
                recurrent_state,
                state_indices[:num_decodes],
            )
            output_THK[:num_decode_tokens] = (
                self._kda_replay_decode(*decode_args, *replay_state)
                if self.use_chunk_replay
                else self._kda_decode(*decode_args)
            )[0].to(output_THK.dtype)

        if metadata.num_prefills > 0:
            prefill_slots = metadata.prefill_state_indices
            prefill_has_initial_state = metadata.prefill_has_initial_state
            assert prefill_slots is not None
            assert prefill_has_initial_state is not None
            if num_decodes == 0:
                prefill_cu_seqlens = metadata.non_spec_query_start_loc
            else:
                prefill_cu_seqlens = metadata.prefill_query_start_loc
            assert prefill_cu_seqlens is not None

            prefill_args = (
                mixed_qkv_TC[num_decode_tokens:],
                raw_gate[:, num_decode_tokens:],
                raw_beta[:, num_decode_tokens:],
                A_log_H,
                dt_bias_HK,
                conv_state,
                conv_weight,
                recurrent_state,
                prefill_slots,
                prefill_has_initial_state,
                prefill_cu_seqlens,
            )
            output_THK[num_decode_tokens:live] = (
                self._kda_replay_prefill(*prefill_args, *replay_state)
                if self.use_chunk_replay
                else self._kda_prefill(*prefill_args)
            )[0].to(output_THK.dtype)
        return output_THK

    def _layer_metadata(self) -> GDNAttentionMetadata | None:
        raw = get_forward_context().attn_metadata
        if raw is None:
            return None
        metadata = raw[self.prefix] if isinstance(raw, dict) else raw
        if not isinstance(metadata, GDNAttentionMetadata):
            raise TypeError(
                f"expected GDNAttentionMetadata for {self.prefix}, got {type(metadata)}"
            )
        return metadata

    def _kda_replay_prefill(
        self,
        mixed_qkv_TC,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        conv_state,
        conv_weight,
        recurrent_state,
        state_indices,
        has_initial_state,
        query_start_loc,
        replay_q,
        replay_k,
        replay_v,
        replay_gate,
        replay_beta,
        replay_count,
    ) -> torch.Tensor:
        convolved_1TC = paged_causal_conv1d(
            mixed_qkv_TC.unsqueeze(0),
            conv_weight,
            conv_state,
            state_indices,
            cu_seqlens=query_start_loc,
            has_initial_state=has_initial_state,
        )
        convolved_1TC = _batch_invariant_silu(convolved_1TC)
        query_1THK, key_1THK, value_1THK = (
            tensor.reshape(1, -1, self.local_num_heads, self.head_dim)
            for tensor in convolved_1TC.chunk(3, dim=-1)
        )
        return paged_chunk_kda(
            _batch_invariant_l2norm(query_1THK),
            _batch_invariant_l2norm(key_1THK),
            value_1THK,
            bound_gate(
                raw_gate,
                A_log.float(),
                dt_bias.float(),
                lower_bound=self.lower_bound,
            ),
            raw_beta.float().sigmoid(),
            recurrent_state,
            state_indices,
            cu_seqlens=query_start_loc,
            has_initial_state=has_initial_state,
            autotune=False,
            replay_state=(
                replay_q,
                replay_k,
                replay_v,
                replay_gate,
                replay_beta,
                replay_count,
            ),
        )

    def _kda_replay_decode(
        self,
        mixed_qkv_TC,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        conv_state,
        conv_weight,
        recurrent_state,
        state_indices,
        replay_q,
        replay_k,
        replay_v,
        replay_gate,
        replay_beta,
        replay_count,
    ) -> torch.Tensor:
        convolved_TC = causal_conv1d_decode(
            mixed_qkv_TC,
            conv_weight,
            conv_state,
            state_indices=state_indices,
        )
        convolved_TC = _batch_invariant_silu(convolved_TC)
        query_B1HK, key_B1HK, value_B1HK = (
            tensor.reshape(-1, 1, self.local_num_heads, self.head_dim)
            for tensor in convolved_TC.chunk(3, dim=-1)
        )
        result_B1HK = paged_chunk_kda(
            _batch_invariant_l2norm(query_B1HK),
            _batch_invariant_l2norm(key_B1HK),
            value_B1HK,
            bound_gate(
                raw_gate.transpose(0, 1),
                A_log.float(),
                dt_bias.float(),
                lower_bound=self.lower_bound,
            ),
            raw_beta.transpose(0, 1).float().sigmoid(),
            recurrent_state,
            state_indices,
            autotune=False,
            replay_state=(
                replay_q,
                replay_k,
                replay_v,
                replay_gate,
                replay_beta,
                replay_count,
            ),
        )
        return result_B1HK.transpose(0, 1)

    def _kda_prefill(
        self,
        mixed_qkv_TC,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        conv_state,
        conv_weight,
        recurrent_state,
        state_indices,
        has_initial_state,
        query_start_loc,
    ) -> torch.Tensor:
        """Chunked prefill directly over the paged recurrent-state pool."""
        convolved_1TC = paged_causal_conv1d(
            mixed_qkv_TC.unsqueeze(0),
            conv_weight,
            conv_state,
            state_indices,
            activation="silu",
            cu_seqlens=query_start_loc,
            has_initial_state=has_initial_state,
        )
        query, key, value = (
            tensor.reshape(1, -1, self.local_num_heads, self.head_dim)
            for tensor in convolved_1TC.chunk(3, dim=-1)
        )

        return paged_chunk_kda(
            l2norm(query),
            l2norm(key),
            value,
            bound_gate(
                raw_gate,
                A_log.float(),
                dt_bias.float(),
                lower_bound=self.lower_bound,
            ),
            raw_beta.float().sigmoid(),
            recurrent_state,
            state_indices,
            cu_seqlens=query_start_loc,
            has_initial_state=has_initial_state,
        )

    def _kda_decode(
        self,
        mixed_qkv_TC,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        conv_state,
        conv_weight,
        recurrent_state,
        state_indices,
    ) -> torch.Tensor:
        """Single-token decode: one conv update and one fused paged recurrent step."""
        convolved = causal_conv1d_decode(
            mixed_qkv_TC,
            conv_weight,
            conv_state,
            activation="silu",
            state_indices=state_indices,
        )
        return recurrent_kda_decode(
            convolved,
            raw_gate,
            raw_beta,
            A_log.float(),
            dt_bias.float(),
            recurrent_state,
            state_indices,
            lower_bound=self.lower_bound,
        )
