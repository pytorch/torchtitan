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
    if num_tokens == 0:
        # An idle attention-DP rank: only padding rows, and no attention metadata.
        return mixed_qkv_TC.new_zeros(
            mixed_qkv_TC.shape[0], layer.num_v_heads, layer.head_v_dim
        )
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

    def __init__(self, model_runner) -> None:
        super().__init__(model_runner)
        # TODO(https://github.com/sgl-project/sglang/issues/48463): SGLang has no
        # public hooks for the extra_buffer prefix cache. The backends rely on its
        # internals: conv_states_shape below, the forward_metadata track fields read
        # in track_extend_states, and the private _track_mamba_state_decode in decode.
        # SGLang's extend tracking reads the conv window length (W - 1) as the last
        # dim of conv_states_shape; the pool keeps [W - 1, C] per slot, as SGLang's KDA
        # backend does, which exposes the transposed shape too.
        self.conv_states_shape = (
            self.state_pool.mamba_cache.conv[0].transpose(-1, -2).shape
        )

    def track_extend_states(
        self, forward_batch, mixed_qkv_TC: torch.Tensor, conv_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Prepare the prefix-cache snapshots for SGLang's ``extra_buffer`` strategy.

        For each request SGLang tracks in this extend, the snapshot is taken at the
        last 64-token boundary of its new tokens and stored in the request's track
        slot:

        - the conv history is copied there directly;
        - the recurrent state is written there by the ``paged_gdn`` / ``paged_kda``
          kernels, from the returned ``checkpoint_slots`` and ``checkpoint_lens``.

        Returns ``{}`` when no request is tracked.
        """
        metadata = self.forward_metadata
        if not metadata.has_mamba_track_mask:
            return {}
        track_mask = forward_batch.mamba_track_mask
        track_indices = metadata.mamba_track_indices
        if metadata.mamba_track_mask_indices is None:
            # Once per forward, as SGLang's KDA backend does: nonzero syncs.
            metadata.mamba_track_mask_indices = track_mask.nonzero(as_tuple=True)[0]
        # The conv history is the [W - 1, C] inputs before the position.
        conv_state[track_indices[metadata.mamba_track_mask_indices]] = mixed_qkv_TC[
            metadata.track_conv_indices
        ].to(conv_state.dtype)
        return {
            # Untracked rows take no snapshot, so their (garbage) lengths are unused.
            "checkpoint_slots": torch.where(track_mask, track_indices, -1).to(
                torch.int32
            ),
            "checkpoint_lens": forward_batch.mamba_track_aligned_lens().to(torch.int32),
        }

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
