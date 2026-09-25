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

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols.module import Module
from torchtitan.rl.model.linear_attention_backend import (
    GDNExecutionPath,
    TorchTitanGDNAttentionBackend,
    TorchTitanGDNAttentionMetadata,
)
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

# Shape suffixes:
# T = packed tokens, C = projection channels, H = attention heads,
# K = query/key/value head dimension, W = convolution kernel width.

_CHUNK_SIZE = 64
_REPLAY_STATE_DTYPES = (
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
    torch.float32,
    torch.int32,
)


def _replay_state_shapes(
    local_num_heads: int, head_dim: int
) -> tuple[tuple[int, ...], ...]:
    """Return Attention Gym's per-slot q, k, v, gate, beta, and count shapes."""
    return (
        (_CHUNK_SIZE, local_num_heads, head_dim),
        (_CHUNK_SIZE, local_num_heads, head_dim),
        (_CHUNK_SIZE, local_num_heads, head_dim),
        (_CHUNK_SIZE, local_num_heads, head_dim),
        (_CHUNK_SIZE, local_num_heads),
        (1,),
    )


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
            torch.tensor([])
            for _ in range(
                2 + len(_REPLAY_STATE_DTYPES) if self.use_chunk_replay else 2
            )
        )

        # vLLM keys per-layer metadata and the state cache by this name.
        self.prefix = f"model.layers.{config.layer_index}.kda"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return TorchTitanGDNAttentionBackend

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
        return (*shapes, *_replay_state_shapes(self.local_num_heads, self.head_dim))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )
        if not self.use_chunk_replay:
            return dtypes
        return (*dtypes, *_REPLAY_STATE_DTYPES)

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

        mixed_qkv_TC = torch.cat((query_TC, key_TC, value_TC), dim=-1)
        # vLLM's conv kernels take (channels, width); Conv1d stores (C, 1, W).
        conv_weight_CW = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        ).squeeze(1)
        output_THK = mixed_qkv_TC.new_zeros(
            mixed_qkv_TC.shape[0],
            self.local_num_heads,
            self.head_dim,
        )
        self._forward(
            mixed_qkv_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_weight_CW,
            A_log_H,
            dt_bias_HK,
            output_THK,
        )
        return output_THK

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_weight_CW: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        output_THK: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            return
        assert isinstance(attn_metadata, dict)
        metadata = attn_metadata[self.prefix]
        assert isinstance(metadata, TorchTitanGDNAttentionMetadata)

        num_actual_tokens = metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return
        state_indices = metadata.non_spec_state_indices_tensor
        cu_seqlens = metadata.non_spec_query_start_loc
        has_initial_state = metadata.has_initial_state
        assert (
            state_indices is not None
            and cu_seqlens is not None
            and has_initial_state is not None
        )
        conv_state, recurrent_state, *replay_state = self.kv_cache
        gate_args = (A_log_H.float(), dt_bias_HK.float())

        if metadata.execution_path is GDNExecutionPath.SINGLE_TOKEN:
            num_decode_rows = state_indices.numel()
            convolved_BC = causal_conv1d_decode(
                mixed_qkv_TC[:num_decode_rows],
                conv_weight_CW,
                conv_state,
                activation="silu",
                state_indices=state_indices,
                has_initial_state=has_initial_state,
            )
            if not self.use_chunk_replay:
                recurrent_kda_decode(
                    convolved_BC,
                    raw_gate_THK[:num_decode_rows].unsqueeze(0),
                    raw_beta_TH[:num_decode_rows].unsqueeze(0),
                    *gate_args,
                    recurrent_state,
                    state_indices,
                    has_initial_state=has_initial_state,
                    lower_bound=self.lower_bound,
                    out=output_THK[:num_decode_rows].unsqueeze(0),
                )
                return
            query_B1HK, key_B1HK, value_B1HK = (
                tensor.reshape(-1, 1, self.local_num_heads, self.head_dim)
                for tensor in convolved_BC.chunk(3, dim=-1)
            )
            output_B1HK = paged_chunk_kda(
                l2norm(query_B1HK),
                l2norm(key_B1HK),
                value_B1HK,
                bound_gate(
                    raw_gate_THK[:num_decode_rows].unsqueeze(1),
                    *gate_args,
                    lower_bound=self.lower_bound,
                ),
                raw_beta_TH[:num_decode_rows].unsqueeze(1).float().sigmoid(),
                recurrent_state,
                state_indices,
                has_initial_state=has_initial_state,
                replay_state=tuple(replay_state),
            )
            output_THK[:num_decode_rows].copy_(output_B1HK[:, 0])
            return

        convolved_1TC = paged_causal_conv1d(
            mixed_qkv_TC[:num_actual_tokens].unsqueeze(0),
            conv_weight_CW,
            conv_state,
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )
        query_1THK, key_1THK, value_1THK = (
            tensor.reshape(1, -1, self.local_num_heads, self.head_dim)
            for tensor in convolved_1TC.chunk(3, dim=-1)
        )
        output_1THK = paged_chunk_kda(
            l2norm(query_1THK),
            l2norm(key_1THK),
            value_1THK,
            bound_gate(
                raw_gate_THK[:num_actual_tokens].unsqueeze(0),
                *gate_args,
                lower_bound=self.lower_bound,
            ),
            raw_beta_TH[:num_actual_tokens].unsqueeze(0).float().sigmoid(),
            recurrent_state,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            replay_state=tuple(replay_state) if self.use_chunk_replay else None,
        )
        output_THK[:num_actual_tokens].copy_(output_1THK[0])


def maybe_configure_kda_hybrid_model(
    model_cls: type, model_config: Decoder.Config
) -> None:
    """Attach vLLM's hybrid-state interface when the model contains KDA layers."""
    kda_configs = [
        layer.delta_attention
        for layer in model_config.layers
        if getattr(layer, "delta_attention", None) is not None
    ]
    if not kda_configs:
        return

    state_shapes = {
        (
            kda_config.num_heads,
            kda_config.head_dim,
            kda_config.conv_kernel_size,
        )
        for kda_config in kda_configs
    }
    if len(state_shapes) != 1:
        raise ValueError(
            f"All KDA layers must use the same state shape, got {state_shapes}"
        )
    (state_shape,) = state_shapes

    num_heads, head_dim, conv_kernel_size = state_shape

    def get_state_shape(cls, vllm_config):
        speculative_config = vllm_config.speculative_config
        num_speculative_tokens = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )
        shapes = MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            num_heads,
            head_dim,
            conv_kernel_size=conv_kernel_size,
            num_spec=num_speculative_tokens,
        )
        if not is_in_batch_invariant_mode():
            return shapes
        local_num_heads = num_heads // vllm_config.parallel_config.tensor_parallel_size
        return (*shapes, *_replay_state_shapes(local_num_heads, head_dim))

    def get_state_dtype(cls, vllm_config):
        dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )
        if not is_in_batch_invariant_mode():
            return dtypes
        return (*dtypes, *_REPLAY_STATE_DTYPES)

    def get_state_copy_func(cls):
        copy_funcs = MambaStateCopyFuncCalculator.kda_state_copy_func()
        if not is_in_batch_invariant_mode():
            return copy_funcs
        return (*copy_funcs, *(copy_funcs[1] for _ in _REPLAY_STATE_DTYPES))

    def get_state_copy_funcs(cls, mamba_types):
        copy_funcs = cls.get_mamba_state_copy_func()
        return {mamba_type: copy_funcs for mamba_type in mamba_types}

    model_cls.is_hybrid = True
    model_cls.get_mamba_state_shape_from_config = classmethod(get_state_shape)
    model_cls.get_mamba_state_dtype_from_config = classmethod(get_state_dtype)
    model_cls.get_mamba_state_copy_func = classmethod(get_state_copy_func)
    model_cls.get_mamba_state_copy_funcs = classmethod(get_state_copy_funcs)
