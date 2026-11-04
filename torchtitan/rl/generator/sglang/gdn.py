# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan's Gated DeltaNet inside SGLang: SGLang owns the per-request conv and
recurrent state slots (its mamba state pool, which its radix cache snapshots), and the
trainer's Attention Gym kernels advance them.

Legend: T = tokens of the step, C = rank-local q|k|v channels, H = value heads,
V = value head dim, W = conv kernel size (all rank-local under TP).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.runtime_context import get_parallel, linear_attn_tp_size
from transformers import PretrainedConfig

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attn_gym import (
    causal_conv1d_decode,
    paged_causal_conv1d,
    recurrent_gdn_decode,
)
from torchtitan.rl.generator.paged_gdn import paged_gdn


class TorchTitanGDNConfig(PretrainedConfig):
    """The HF-shaped config of a TorchTitan model with GDN layers: what SGLang reads to
    allocate their state slots and route layers between its attention backends.

    Args:
        linear_layer_ids: The indices of the GDN layers.
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size: The GDN
            layers' dimensions (global head counts), shared by every GDN layer.
        conv_state_dtype: The model's compute dtype, which the conv history keeps.
    """

    def __init__(
        self,
        *,
        linear_layer_ids: list[int] | None = None,
        num_k_heads: int = 0,
        num_v_heads: int = 0,
        head_k_dim: int = 0,
        head_v_dim: int = 0,
        conv_kernel_size: int = 0,
        conv_state_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.linear_layer_ids = linear_layer_ids or []
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_state_dtype = conv_state_dtype

    @property
    def full_attention_layer_ids(self) -> list[int]:
        linear = set(self.linear_layer_ids)
        return [i for i in range(self.num_hidden_layers) if i not in linear]

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


def _gdn_weights(
    layer: _SGLangRadixLinearAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The layer's ``(conv_weight_CW, A_log_H, dt_bias_H)`` as the kernels take them:
    one ``[C, W]`` convolution weight, and the gate parameters in float32."""
    conv_weight_CW = torch.cat(layer.conv_weights, dim=0).squeeze(1)
    return conv_weight_CW, layer.A_log.float(), layer.dt_bias.float()


class TorchTitanGDNAttnBackend(MambaAttnBackendBase):
    """SGLang's linear attention backend for TorchTitan's GDN layers.

    SGLang's base backend builds the step's state slot of each request
    (``mamba_cache_indices``) and its query offsets, CUDA-graph buffers included; this
    backend runs the trainer's Attention Gym convolution and recurrence on those slots,
    in place, as the vLLM engine's GDN adapter does. Slot 0 is SGLang's padding slot,
    which the kernels skip. Speculative decoding is not supported.
    """

    def forward_decode(
        self,
        layer: _SGLangRadixLinearAttention,
        forward_batch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """One token per request; returns ``(1, T, H, V)``."""
        conv_weight_CW, A_log_H, dt_bias_H = _gdn_weights(layer)
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
        layer: _SGLangRadixLinearAttention,
        forward_batch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Each request's new tokens, resuming the state of its cached prefix; returns
        ``(1, T, H, V)``."""
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError(
                "TorchTitan GDN does not support speculative decoding"
            )
        conv_weight_CW, A_log_H, dt_bias_H = _gdn_weights(layer)
        layer_cache = self.state_layer_cache(layer.layer_id)
        state_indices = self.forward_metadata.mamba_cache_indices.to(torch.int32)
        cu_seqlens = self.forward_metadata.query_start_loc
        has_initial_state = forward_batch.extend_prefix_lens > 0
        conv_output_TC = paged_causal_conv1d(
            mixed_qkv.unsqueeze(0),
            conv_weight_CW,
            layer_cache.conv[0],
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        ).squeeze(0)
        return paged_gdn(
            conv_output_TC,
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


class _SGLangRadixLinearAttention(RadixLinearAttention, Module):
    """SGLang's ``RadixLinearAttention`` as a TorchTitan ``Module``. It holds no
    parameters: ``SGLangInnerGDN`` sets the enclosing GDN's conv and gate weights on
    it for each call, as the backend reads them (``_gdn_weights``)."""


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
        self.linear_attention = _SGLangRadixLinearAttention(
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
        # The parameters themselves, whose storage stays put across model state dict
        # loads: in a breakable prefill CUDA graph, SGLang calls the backend eagerly at
        # replay with the attributes the last capture set, so a tensor computed here
        # would be a stale graph buffer. The backend derives what the kernels take.
        # object.__setattr__: nn.Module would register the parameters as the layer's.
        layer = self.linear_attention
        object.__setattr__(
            layer,
            "conv_weights",
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
        )
        object.__setattr__(layer, "A_log", A_log_H)
        object.__setattr__(layer, "dt_bias", dt_bias_H)
        forward_batch = get_forward_batch()
        # The rows SGLang scheduled.
        num_tokens = forward_batch.out_cache_loc.shape[0]
        num_padding = mixed_qkv_TC.shape[0] - num_tokens
        out_THV = layer(
            forward_batch,
            mixed_qkv_TC[:num_tokens],
            a_TH[:num_tokens],
            b_TH[:num_tokens],
        ).squeeze(0)
        if num_padding:
            out_THV = F.pad(out_THV, (0, 0, 0, 0, 0, num_padding))
        return out_THV
