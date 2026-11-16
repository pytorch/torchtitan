# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan's Gated DeltaNet inside SGLang (see ``linear_attention.py``).

Legend: T = tokens of the step, C = rank-local q|k|v channels, H = value heads,
V = value head dim, W = conv kernel size (all rank-local under TP).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.runtime_context import get_parallel, linear_attn_tp_size

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attn_gym import causal_conv1d_decode, recurrent_gdn_decode
from torchtitan.rl.generator.attention.gdn import paged_gdn
from torchtitan.rl.generator.sglang.attention.linear_attention import (
    linear_attention_weights,
    run_linear_attention,
    SGLangRadixLinearAttention,
    TorchTitanLinearAttentionConfig,
    TorchTitanLinearAttnBackend,
)


class TorchTitanGDNConfig(TorchTitanLinearAttentionConfig):
    """``TorchTitanLinearAttentionConfig`` of a model with GDN layers.

    Args:
        num_k_heads, num_v_heads, head_k_dim, head_v_dim: The GDN layers' dimensions
            (global head counts), shared by every GDN layer.
    """

    def __init__(
        self,
        *,
        num_k_heads: int = 0,
        num_v_heads: int = 0,
        head_k_dim: int = 0,
        head_v_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        shape = Mamba2StateShape.create(
            tp_world_size=linear_attn_tp_size(),
            intermediate_size=self.num_v_heads * self.head_v_dim,
            n_groups=self.num_k_heads,
            num_heads=self.num_v_heads,
            head_dim=self.head_v_dim,
            state_size=self.head_k_dim,
            conv_kernel=self.conv_kernel_size,
        )
        # Attention Gym's convolution keeps each slot's history as [W - 1, C] with
        # contiguous channels; SGLang's GDN keeps [C, W - 1]. SGLang's deduplicated
        # conv window buffers (speculative decoding) need the window axis last.
        (conv_shape,) = shape.conv
        shape = dataclasses.replace(
            shape, conv=[conv_shape[::-1]], disable_conv_window_dedup=True
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            # The kernels advance a float32 recurrent state: rounding it through
            # bfloat16 after every decode token would differ from one prefill call.
            dtype=Mamba2StateDType(
                conv=getattr(torch, self.conv_state_dtype), temporal=torch.float32
            ),
        )


class TorchTitanGDNAttnBackend(TorchTitanLinearAttnBackend):
    """SGLang's linear attention backend for TorchTitan's GDN layers."""

    def forward_decode(
        self,
        layer: SGLangRadixLinearAttention,
        forward_batch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """One token per request; returns ``(1, T, H, V)``."""
        conv_weight_CW, A_log_H, dt_bias_H = linear_attention_weights(layer)
        layer_cache = self.state_layer_cache(layer.layer_id)
        state_indices = self.forward_metadata.mamba_cache_indices.to(torch.int32)
        conv_output_TC = causal_conv1d_decode(
            mixed_qkv,
            conv_weight_CW,
            layer_cache.conv[0],
            activation="silu",
            state_indices=state_indices,
        )
        if is_in_batch_invariant_mode():
            # The trainer's sequential scan, as a prefill of one token per row. Every
            # row is its own sequence, as SGLang's offsets leave the rows padding a
            # batch to a CUDA graph's size out; their null slot zeroes them.
            num_rows = mixed_qkv.shape[0]
            return paged_gdn(
                conv_output_TC,
                a,
                b,
                A_log_H,
                dt_bias_H,
                layer_cache.temporal,
                cu_seqlens=torch.arange(
                    num_rows + 1, dtype=torch.int32, device=mixed_qkv.device
                ),
                state_indices=state_indices,
                has_initial_state=torch.ones_like(state_indices, dtype=torch.bool),
                num_k_heads=layer.num_k_heads,
                head_k_dim=layer.head_k_dim,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            ).unsqueeze(0)
        return recurrent_gdn_decode(
            conv_output_TC,
            a.unsqueeze(0),
            b.unsqueeze(0),
            A_log_H,
            dt_bias_H,
            layer_cache.temporal,
            state_indices,
            scale=layer.head_k_dim**-0.5,
        )

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
        ``(1, T, H, V)``."""
        conv_weight_CW, A_log_H, dt_bias_H = linear_attention_weights(layer)
        layer_cache = self.state_layer_cache(layer.layer_id)
        (
            conv_output_1TC,
            state_indices,
            cu_seqlens,
            has_initial_state,
        ) = self.convolve_extend(
            forward_batch, mixed_qkv, conv_weight_CW, layer_cache.conv[0]
        )
        return paged_gdn(
            conv_output_1TC[0],
            a,
            b,
            A_log_H,
            dt_bias_H,
            layer_cache.temporal,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            num_k_heads=layer.num_k_heads,
            head_k_dim=layer.head_k_dim,
            num_v_heads=layer.num_v_heads,
            head_v_dim=layer.head_v_dim,
        ).unsqueeze(0)


class SGLangInnerGDN(InnerAttention):
    """Adapter from TorchTitan's GDN to SGLang's ``RadixLinearAttention``.

    The enclosing ``GDN`` owns every parameter and passes them in; this module hands
    them with the step's projections to SGLang, whose ``TorchTitanGDNAttnBackend``
    advances the request's state slots. The inputs are head-sharded under TP and may
    end in padding rows (see ``_DecoderBody``), which it skips.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        layer_id: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        tp_size = get_parallel().tp_size
        for name in ("num_k_heads", "num_v_heads"):
            if getattr(config, name) % tp_size != 0:
                raise ValueError(
                    f"{name} ({getattr(config, name)}) must be divisible by "
                    f"tensor_parallel_degree ({tp_size})"
                )
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.linear_attention = SGLangRadixLinearAttention(
            layer_id=config.layer_id,
            num_q_heads=config.num_k_heads // tp_size,
            num_k_heads=config.num_k_heads // tp_size,
            num_v_heads=config.num_v_heads // tp_size,
            head_q_dim=config.head_k_dim,
            head_k_dim=config.head_k_dim,
            head_v_dim=config.head_v_dim,
        )

    def forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: None,
    ) -> torch.Tensor:
        """Run GDN on SGLang's state slots.

        ``A_log_H`` and ``dt_bias_H`` arrive in the trainer's master dtype
        (``preserve_dtype_params``); the kernels take them in float32.

        Returns:
            ``(T, H, V)``, zeros in the padding rows.
        """
        if attention_metadata is not None:
            raise ValueError(
                "SGLangInnerGDN does not take attention_metadata; SGLang manages the "
                "request boundaries and states."
            )
        assert key_head_dim == self.head_k_dim
        assert value_head_dim == self.head_v_dim
        assert conv_q_weight_C1W.shape[-1] == self.conv_kernel_size
        return run_linear_attention(
            self.linear_attention,
            mixed_qkv_TC,
            a_TH,
            b_TH,
            conv_weights=(conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            A_log=A_log_H,
            dt_bias=dt_bias_H,
        )
