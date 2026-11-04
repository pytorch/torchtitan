# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan's Kimi Delta Attention inside SGLang (see ``linear_attention.py``).

Legend: T = tokens of the step, C = rank-local q|k|v channels, H = heads, K = head dim
(key and value), W = conv kernel size (all rank-local under TP).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.runtime_context import get_parallel, linear_attn_tp_size

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attn_gym import causal_conv1d_decode, recurrent_kda_decode
from torchtitan.rl.generator.paged_kda import (
    paged_kda,
    REPLAY_STATE_DTYPES,
    replay_state_shapes,
)
from torchtitan.rl.generator.sglang.linear_attention import (
    linear_attention_weights,
    run_linear_attention,
    SGLangRadixLinearAttention,
    TorchTitanLinearAttentionConfig,
    TorchTitanLinearAttnBackend,
)


class TorchTitanKDAConfig(TorchTitanLinearAttentionConfig):
    """``TorchTitanLinearAttentionConfig`` of a model with KDA layers.

    In batch-invariant mode, each slot also holds Attention Gym's replay state (see
    ``paged_kda``). SGLang's pool keeps a list of conv state tensors in one dtype; the
    replay tensors are extra entries of the list, flattened, sized in bytes and viewed
    back as their own dtypes and shapes by ``TorchTitanKDAAttnBackend``. SGLang clears
    and copies every entry with the slot, so its radix cache (and HiCache) snapshots
    the replay state too.

    Args:
        kda_num_heads, kda_head_dim: The KDA layers' dimensions (global head count),
            shared by every KDA layer.
    """

    def __init__(
        self, *, kda_num_heads: int = 0, kda_head_dim: int = 0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.kda_num_heads = kda_num_heads
        self.kda_head_dim = kda_head_dim

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        tp_size = linear_attn_tp_size()
        # The conv history is [W - 1, C] per slot, with contiguous channels, as
        # Attention Gym's convolution keeps it.
        shape = KimiLinearStateShape.create(
            tp_world_size=tp_size,
            num_heads=self.kda_num_heads,
            head_dim=self.kda_head_dim,
            conv_kernel_size=self.conv_kernel_size,
        )
        conv_dtype = getattr(torch, self.conv_state_dtype)
        if is_in_batch_invariant_mode():
            replay_shapes = [
                _flat_shape_in_dtype(replay_shape, replay_dtype, conv_dtype)
                for replay_shape, replay_dtype in zip(
                    replay_state_shapes(
                        self.kda_num_heads // tp_size, self.kda_head_dim
                    ),
                    REPLAY_STATE_DTYPES,
                    strict=True,
                )
            ]
            shape = dataclasses.replace(shape, conv=[*shape.conv, *replay_shapes])
        return KimiLinearCacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            # The kernels advance a float32 recurrent state: rounding it through
            # bfloat16 after every decode token would differ from one prefill call.
            dtype=Mamba2StateDType(conv=conv_dtype, temporal=torch.float32),
        )


# SGLang's HiCache copies each slot's state entries in 16-byte words.
_STATE_ENTRY_ALIGNMENT_BYTES = 16


def _flat_shape_in_dtype(
    shape: tuple[int, ...], dtype: torch.dtype, storage_dtype: torch.dtype
) -> tuple[int]:
    """The 1-D shape of a ``storage_dtype`` tensor with the bytes of a ``dtype`` tensor
    of ``shape``, padded to ``_STATE_ENTRY_ALIGNMENT_BYTES``; ``_unflatten_state_entry``
    restores the tensor."""
    num_bytes = math.prod(shape) * dtype.itemsize
    num_bytes = (
        -(-num_bytes // _STATE_ENTRY_ALIGNMENT_BYTES) * _STATE_ENTRY_ALIGNMENT_BYTES
    )
    return (num_bytes // storage_dtype.itemsize,)


def _unflatten_state_entry(
    storage: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """View a ``_flat_shape_in_dtype`` pool entry, one row per slot, as the ``dtype``
    tensor of ``shape`` per slot, with dense per-slot rows."""
    num_elements = math.prod(shape) * dtype.itemsize // storage.element_size()
    return storage[:, :num_elements].view(dtype).unflatten(-1, shape)


class TorchTitanKDAAttnBackend(TorchTitanLinearAttnBackend):
    """SGLang's linear attention backend for TorchTitan's KDA layers.

    ``RadixLinearAttention`` passes the raw gate as ``a`` (flattened to ``[T, H * K]``)
    and the raw beta as ``b``.
    """

    def _state(
        self, layer: SGLangRadixLinearAttention
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """The layer's conv history, recurrent state and, in batch-invariant mode,
        replay state pools."""
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_state, *replay_storage = layer_cache.conv
        if not is_in_batch_invariant_mode():
            return conv_state, layer_cache.temporal, None
        # The recurrent state pool is [N, H, K, K].
        replay_shapes = replay_state_shapes(
            layer_cache.temporal.shape[1], layer.head_k_dim
        )
        replay_state = tuple(
            _unflatten_state_entry(storage, shape, dtype)
            for storage, shape, dtype in zip(
                replay_storage, replay_shapes, REPLAY_STATE_DTYPES, strict=True
            )
        )
        return conv_state, layer_cache.temporal, replay_state

    def forward_decode(
        self,
        layer: SGLangRadixLinearAttention,
        forward_batch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """One token per request; returns ``(1, T, H, K)``."""
        conv_weight_CW, A_log_H, dt_bias_HK = linear_attention_weights(layer)
        conv_state, recurrent_state, replay_state = self._state(layer)
        state_indices = self.forward_metadata.mamba_cache_indices.to(torch.int32)
        raw_gate_THK = a.unflatten(-1, (-1, layer.head_k_dim))
        convolved_TC = causal_conv1d_decode(
            mixed_qkv,
            conv_weight_CW,
            conv_state,
            activation="silu",
            state_indices=state_indices,
        )
        if replay_state is None:
            return recurrent_kda_decode(
                convolved_TC,
                raw_gate_THK.unsqueeze(0),
                b.unsqueeze(0),
                A_log_H,
                dt_bias_HK,
                recurrent_state,
                state_indices,
                lower_bound=layer.lower_bound,
            )
        # Attention Gym's replay-backed decode, of one token per row.
        return paged_kda(
            convolved_TC.unsqueeze(1),
            raw_gate_THK.unsqueeze(1),
            b.unsqueeze(1),
            A_log_H,
            dt_bias_HK,
            recurrent_state,
            state_indices,
            cu_seqlens=None,
            has_initial_state=None,
            replay_state=replay_state,
            head_dim=layer.head_k_dim,
            lower_bound=layer.lower_bound,
        ).transpose(0, 1)

    def forward_extend(
        self,
        layer: SGLangRadixLinearAttention,
        forward_batch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Each request's new tokens, resuming the state of its cached prefix; returns
        ``(1, T, H, K)``."""
        conv_weight_CW, A_log_H, dt_bias_HK = linear_attention_weights(layer)
        conv_state, recurrent_state, replay_state = self._state(layer)
        (
            convolved_1TC,
            state_indices,
            cu_seqlens,
            has_initial_state,
        ) = self.convolve_extend(forward_batch, mixed_qkv, conv_weight_CW, conv_state)
        return paged_kda(
            convolved_1TC,
            a.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0),
            b.unsqueeze(0),
            A_log_H,
            dt_bias_HK,
            recurrent_state,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            replay_state=replay_state,
            head_dim=layer.head_k_dim,
            lower_bound=layer.lower_bound,
        )


class SGLangInnerKDA(InnerAttention):
    """Adapter from TorchTitan's KDA to SGLang's ``RadixLinearAttention``.

    The enclosing ``KDA`` owns every parameter and passes them in; this module hands
    them with the step's projections to SGLang, whose ``TorchTitanKDAAttnBackend``
    advances the request's state slots. The inputs are head-sharded under TP and may
    end in padding rows (see ``_DecoderBody``), which it skips.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        layer_id: int
        num_heads: int
        head_dim: int
        conv_kernel_size: int
        lower_bound: float

    def __init__(self, config: Config) -> None:
        super().__init__()
        tp_size = get_parallel().tp_size
        if config.num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({config.num_heads}) must be divisible by "
                f"tensor_parallel_degree ({tp_size})"
            )
        self.conv_kernel_size = config.conv_kernel_size
        local_num_heads = config.num_heads // tp_size
        self.linear_attention = SGLangRadixLinearAttention(
            layer_id=config.layer_id,
            num_q_heads=local_num_heads,
            num_k_heads=local_num_heads,
            num_v_heads=local_num_heads,
            head_q_dim=config.head_dim,
            head_k_dim=config.head_dim,
            head_v_dim=config.head_dim,
            lower_bound=config.lower_bound,
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
        attention_metadata: None,
    ) -> torch.Tensor:
        """Run KDA on SGLang's state slots.

        ``A_log_H`` and ``dt_bias_HK`` arrive in the trainer's master dtype
        (``preserve_dtype_params``); the kernels take them in float32.

        Returns:
            ``(T, H, K)``, zeros in the padding rows.
        """
        if attention_metadata is not None:
            raise ValueError(
                "SGLangInnerKDA does not take attention_metadata; SGLang manages the "
                "request boundaries and states."
            )
        assert conv_q_weight_C1W.shape[-1] == self.conv_kernel_size
        return run_linear_attention(
            self.linear_attention,
            torch.cat((query_TC, key_TC, value_TC), dim=-1),
            raw_gate_THK.flatten(1),
            raw_beta_TH,
            conv_weights=(conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            A_log=A_log_H,
            dt_bias=dt_bias_HK,
        )
