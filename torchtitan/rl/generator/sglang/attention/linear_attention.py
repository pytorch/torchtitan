# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What TorchTitan's linear attention layers (GDN, KDA) share inside SGLang: SGLang
owns the per-request conv and recurrent state slots (its mamba state pool, which its
radix cache snapshots), and the trainer's Attention Gym kernels advance them.

Legend: T = tokens of the step, C = rank-local q|k|v channels, H = heads,
V = value head dim, W = conv kernel size (all rank-local under TP).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from transformers import PretrainedConfig

from torchtitan.models.base.module import Module
from torchtitan.models.common.attn_gym import paged_causal_conv1d


class TorchTitanLinearAttentionConfig(PretrainedConfig):
    """The HF-shaped config of a TorchTitan model with linear attention layers: what
    SGLang reads to allocate their state slots and route layers between its attention
    backends. Each kind of layer subclasses it with its dimensions and state shapes
    (``mamba2_cache_params``).

    Args:
        linear_layer_ids: The indices of the linear attention layers.
        conv_kernel_size: The layers' conv kernel size.
        conv_state_dtype: The model's compute dtype, which the conv history keeps.
    """

    def __init__(
        self,
        *,
        linear_layer_ids: list[int] | None = None,
        conv_kernel_size: int = 0,
        conv_state_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.linear_layer_ids = linear_layer_ids or []
        self.conv_kernel_size = conv_kernel_size
        self.conv_state_dtype = conv_state_dtype

    @property
    def full_attention_layer_ids(self) -> list[int]:
        linear = set(self.linear_layer_ids)
        return [i for i in range(self.num_hidden_layers) if i not in linear]


class SGLangRadixLinearAttention(RadixLinearAttention, Module):
    """SGLang's ``RadixLinearAttention`` as a TorchTitan ``Module``. It holds no
    parameters: ``run_linear_attention`` sets the enclosing layer's conv and gate
    weights on it for each call, as the backends read them (``linear_attention_weights``).
    """


def run_linear_attention(
    layer: SGLangRadixLinearAttention,
    mixed_qkv_TC: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    conv_weights: tuple[torch.Tensor, ...],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Run ``layer`` on the step's rows SGLang scheduled, skipping the padding rows the
    inputs may end in (see ``_DecoderBody``).

    Args:
        a, b: The layer's per-token gate inputs, 2-D: SGLang trims padded extends along
            dim 0 of 2-D gates only.
        conv_weights, A_log, dt_bias: The enclosing layer's parameters themselves, whose
            storage stays put across model state dict loads: in a breakable prefill CUDA
            graph, SGLang calls the backend eagerly at replay with the attributes the
            last capture set, so a tensor computed here would be a stale graph buffer.
            The backend derives what the kernels take.

    Returns:
        ``(T, H, V)``, zeros in the padding rows.
    """
    # object.__setattr__: nn.Module would register the parameters as the layer's.
    object.__setattr__(layer, "conv_weights", conv_weights)
    object.__setattr__(layer, "A_log", A_log)
    object.__setattr__(layer, "dt_bias", dt_bias)
    forward_batch = get_forward_batch()
    num_tokens = forward_batch.out_cache_loc.shape[0]
    num_padding = mixed_qkv_TC.shape[0] - num_tokens
    out_THV = layer(
        forward_batch, mixed_qkv_TC[:num_tokens], a[:num_tokens], b[:num_tokens]
    ).squeeze(0)
    if num_padding:
        out_THV = F.pad(out_THV, (0, 0, 0, 0, 0, num_padding))
    return out_THV


def linear_attention_weights(
    layer: SGLangRadixLinearAttention,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The layer's ``(conv_weight_CW, A_log, dt_bias)`` as the kernels take them: one
    ``[C, W]`` convolution weight, and the gate parameters in float32."""
    conv_weight_CW = torch.cat(layer.conv_weights, dim=0).squeeze(1)
    return conv_weight_CW, layer.A_log.float(), layer.dt_bias.float()


class TorchTitanLinearAttnBackend(MambaAttnBackendBase):
    """The base of SGLang's linear attention backends for TorchTitan's layers.

    SGLang's base backend builds the step's state slot of each request
    (``mamba_cache_indices``) and its query offsets, CUDA-graph buffers included; the
    subclasses run the trainer's Attention Gym convolution and recurrence on those
    slots, in place, as the vLLM engine's adapters do. Slot 0 is SGLang's padding slot,
    which the kernels skip. Speculative decoding is not supported.
    """

    def convolve_extend(
        self,
        forward_batch,
        mixed_qkv_TC: torch.Tensor,
        conv_weight_CW: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convolve each request's new tokens, resuming the conv history of its cached
        prefix.

        Returns:
            ``(convolved_1TC, state_indices, cu_seqlens, has_initial_state)``.
        """
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError(
                "TorchTitan linear attention does not support speculative decoding"
            )
        state_indices = self.forward_metadata.mamba_cache_indices.to(torch.int32)
        cu_seqlens = self.forward_metadata.query_start_loc
        has_initial_state = forward_batch.extend_prefix_lens > 0
        convolved_1TC = paged_causal_conv1d(
            mixed_qkv_TC.unsqueeze(0),
            conv_weight_CW,
            conv_state,
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )
        return convolved_1TC, state_indices, cu_seqlens, has_initial_state
