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
    recurrent_kda,
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
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

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
        """Return ``(conv_state_shape, recurrent_state_shape)``.
        Reuses vLLM's KDA calculator
        """
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_spec=self.num_spec,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
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

        conv_state, recurrent_state = self.kv_cache
        # vLLM's conv kernels take (channels, width); Conv1d stores (C, 1, W).
        conv_weight = conv_weight_C1W.reshape(
            conv_weight_C1W.size(0), conv_weight_C1W.size(-1)
        )

        state_indices = metadata.non_spec_state_indices_tensor
        assert state_indices is not None
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens

        if num_decodes > 0:
            decode_result_1THK = self._kda_decode(
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
            output_THK[:num_decode_tokens] = decode_result_1THK[0].to(output_THK.dtype)

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

            prefill_result_1THK = self._kda_prefill(
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
            output_THK[num_decode_tokens:live] = prefill_result_1THK[0].to(
                output_THK.dtype
            )
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
        if is_in_batch_invariant_mode():
            fresh_slots = state_indices[~has_initial_state]
            if fresh_slots.numel() > 0:
                conv_state[fresh_slots] = 0
                recurrent_state[fresh_slots] = 0

            result_1THK = mixed_qkv_TC.new_empty(
                1,
                mixed_qkv_TC.shape[0],
                self.local_num_heads,
                self.head_dim,
            )
            sequence_starts = query_start_loc[:-1]
            sequence_lengths = query_start_loc[1:] - sequence_starts
            max_sequence_length = int(sequence_lengths.max().item())
            for position in range(max_sequence_length):
                active = sequence_lengths > position
                token_indices = sequence_starts[active] + position
                step_result_1THK = self._kda_decode(
                    mixed_qkv_TC[token_indices],
                    raw_gate[:, token_indices],
                    raw_beta[:, token_indices],
                    A_log,
                    dt_bias,
                    conv_state,
                    conv_weight,
                    recurrent_state,
                    state_indices[active],
                )
                result_1THK[0].index_copy_(
                    0,
                    token_indices.to(torch.int64),
                    step_result_1THK[0],
                )
            return result_1THK

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

        gate = bound_gate(
            raw_gate,
            A_log.float(),
            dt_bias.float(),
            lower_bound=self.lower_bound,
        )
        return paged_chunk_kda(
            l2norm(query),
            l2norm(key),
            value,
            gate,
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
        if is_in_batch_invariant_mode() and mixed_qkv_TC.shape[0] > 1:
            per_sequence = [
                self._kda_decode(
                    mixed_qkv_TC[index : index + 1],
                    raw_gate[:, index : index + 1],
                    raw_beta[:, index : index + 1],
                    A_log,
                    dt_bias,
                    conv_state,
                    conv_weight,
                    recurrent_state,
                    state_indices[index : index + 1],
                )
                for index in range(mixed_qkv_TC.shape[0])
            ]
            return torch.cat(per_sequence, dim=1)

        convolved = causal_conv1d_decode(
            mixed_qkv_TC,
            conv_weight,
            conv_state,
            activation="silu",
            state_indices=state_indices,
        )
        if is_in_batch_invariant_mode():
            num_tokens = convolved.shape[0]
            query, key, value = (
                tensor.reshape(1, num_tokens, self.local_num_heads, self.head_dim)
                for tensor in convolved.chunk(3, dim=-1)
            )
            cu_seqlens = torch.arange(
                num_tokens + 1,
                dtype=torch.int32,
                device=convolved.device,
            )
            gate = bound_gate(
                raw_gate,
                A_log.float(),
                dt_bias.float(),
                lower_bound=self.lower_bound,
            )
            result, _ = recurrent_kda(
                l2norm(query, cu_seqlens=cu_seqlens),
                l2norm(key, cu_seqlens=cu_seqlens),
                value,
                gate,
                raw_beta.float().sigmoid(),
                recurrent_state,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                has_initial_state=torch.ones(
                    num_tokens,
                    dtype=torch.bool,
                    device=convolved.device,
                ),
                batch_invariant=True,
            )
            return result
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
