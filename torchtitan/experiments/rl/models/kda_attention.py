# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from attn_gym.linear.kda import (
    bounded_gate_cumsum,
    chunk_kda,
    l2norm,
    recurrent_kda_decode,
)
from attn_gym.linear.kda.short_conv import causal_conv1d_decode
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from torchtitan.protocols.module import Module


class VLLMKDAWrapper(Module, MambaBase):
    """Adapter from the KDA layer to vLLM's paged recurrent state.

    vLLM owns allocation, routing, prefix copies, and the ``[slot, H, V, K]`` recurrent
    cache shape. KDA uses ``K == V``, so Attention Gym can advance the same dense slot
    storage directly without changing vLLM's cache-manager contract.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        conv_kernel_size: int = 4
        gate_lower_bound: float | None = -5.0
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
        self.gate_lower_bound = config.gate_lower_bound

        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_spec = (
            speculative_config.num_speculative_tokens if speculative_config else 0
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
        mixed_qkv_BLC: torch.Tensor,
        raw_gate_BLNK: torch.Tensor,
        raw_beta_BLN: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the flattened vLLM cache operation on rank-local tensors.

        Signature matches :class:`~torchtitan.models.common.attention.KDAAttention`,
        the module this replaces. vLLM derives its own offsets from the per-layer
        metadata, so the caller's ``cu_seqlens`` is unused.
        """
        del cu_seqlens

        metadata = self._layer_metadata()
        B, L, _ = mixed_qkv_BLC.shape
        num_tokens = B * L
        mixed_qkv_TC = mixed_qkv_BLC.reshape(num_tokens, -1)
        output = mixed_qkv_TC.new_zeros(B, L, self.local_num_heads, self.head_dim)
        if metadata is None:
            # Profiling or cudagraph dummy capture: no live metadata yet.
            return output
        output_TNK = output.view(num_tokens, self.local_num_heads, self.head_dim)

        live = metadata.num_actual_tokens
        mixed_qkv_TC = mixed_qkv_TC[:live]
        # The KDA kernels take a leading batch axis of 1.
        raw_gate = raw_gate_BLNK.reshape(1, num_tokens, self.local_num_heads, -1)[
            :, :live
        ]
        raw_beta = raw_beta_BLN.reshape(1, num_tokens, -1)[:, :live]

        conv_state, recurrent_state = self.kv_cache
        # vLLM's conv kernels take (channels, width); Conv1d stores (C, 1, W).
        conv_weight = conv_weight_C1W.reshape(
            conv_weight_C1W.size(0), conv_weight_C1W.size(-1)
        )

        if metadata.num_prefills > 0:
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            result = self._kda_prefill(
                mixed_qkv_TC,
                raw_gate,
                raw_beta,
                A_log_N,
                dt_bias_NK,
                metadata,
                conv_state,
                conv_weight,
                recurrent_state,
            )
        else:
            if is_conv_state_dim_first():
                raise ValueError(
                    "attention-gym decode requires VLLM_SSM_CONV_STATE_LAYOUT=SD"
                )
            result = self._kda_decode(
                mixed_qkv_TC,
                raw_gate,
                raw_beta,
                A_log_N,
                dt_bias_NK,
                metadata,
                conv_state,
                conv_weight,
                recurrent_state,
            )
        output_TNK[:live] = result[0, :live].to(output.dtype)
        return output

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
        metadata,
        conv_state,
        conv_weight,
        recurrent_state,
    ) -> torch.Tensor:
        """Chunked prefill directly over the paged recurrent-state pool."""
        convolved_TC = causal_conv1d_fn(
            mixed_qkv_TC.transpose(0, 1),
            conv_weight,
            None,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=metadata.has_initial_state,
            cache_indices=metadata.non_spec_state_indices_tensor,
            query_start_loc=metadata.non_spec_query_start_loc,
            metadata=metadata,
        ).transpose(0, 1)
        query, key, value = (
            tensor.reshape(1, -1, self.local_num_heads, self.head_dim)
            for tensor in convolved_TC.unflatten(-1, (-1, 3, self.head_dim)).unbind(-2)
        )

        slots = metadata.non_spec_state_indices_tensor
        has_initial_state = metadata.has_initial_state
        query_start_loc = metadata.non_spec_query_start_loc
        assert slots is not None
        assert has_initial_state is not None
        assert query_start_loc is not None
        cumulative_gate = bounded_gate_cumsum(
            raw_gate.to(torch.bfloat16),
            A_log.float(),
            dt_bias.float(),
            chunk_size=64,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=query_start_loc,
        )
        core_out, _ = chunk_kda(
            l2norm(query),
            l2norm(key),
            value,
            cumulative_gate,
            raw_beta.float().sigmoid(),
            recurrent_state,
            cu_seqlens=query_start_loc,
            state_indices=slots,
            has_initial_state=has_initial_state,
        )
        return core_out

    def _kda_decode(
        self,
        mixed_qkv_TC,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        metadata,
        conv_state,
        conv_weight,
        recurrent_state,
    ) -> torch.Tensor:
        """Single-token decode: one conv update and one fused paged recurrent step."""
        assert metadata.non_spec_state_indices_tensor is not None
        slots = metadata.non_spec_state_indices_tensor[: mixed_qkv_TC.size(0)]
        convolved = causal_conv1d_decode(
            mixed_qkv_TC,
            conv_weight,
            conv_state,
            activation="silu",
            state_indices=slots,
        )
        return recurrent_kda_decode(
            convolved,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            recurrent_state,
            slots,
            lower_bound=self.gate_lower_bound,
        )
