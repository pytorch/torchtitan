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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.kv.router import STATE_PART
from sglang.srt.model_executor.runner_utils.forward_batch import get_forward_batch
from sglang.srt.runtime_context import get_parallel, get_schedule
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
            storage stays put across model state dict loads: when the layer breaks a
            breakable prefill CUDA graph, SGLang calls the backend eagerly at replay with
            the attributes the last capture set, so a tensor computed here would be a
            stale graph buffer.
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


@dataclass(frozen=True, kw_only=True)
class PrefillGraphMetadata:
    """The tables a captured prefill bucket's linear attention layers read, sized by
    the bucket and refilled for each batch before the graph replays.

    Example: a 256-token bucket with rows for 4 requests replays 2 (W = 4). Request 0
    has 100 new tokens and is snapshotted after 64 of them, into track slot 12. Request
    1 resumes a cached prefix with 30 new tokens. "-" marks values no kernel reads.

        cu_seqlens          [0, 100, 130, 130, 130]
        state_indices       [5, 9, -1, -1]           -1: the kernels skip the row
        has_initial_state   [False, True, False, False]
        checkpoint_slots    [12, -1, -1, -1]
        checkpoint_lens     [64, -, 0, 0]
        conv_history_slots  [12, 0, 0, 0]            0: SGLang's dummy slot
        conv_history_rows   [[61, 62, 63], -, [0, 0, 0], [0, 0, 0]]
        padding_tokens      True from token 130 on
    """

    cu_seqlens: torch.Tensor  # int32 [max_seqs + 1]
    state_indices: torch.Tensor  # int32 [max_seqs]
    has_initial_state: torch.Tensor  # bool [max_seqs]
    checkpoint_slots: torch.Tensor  # int32 [max_seqs]
    checkpoint_lens: torch.Tensor  # int32 [max_seqs]
    conv_history_slots: torch.Tensor  # int64 [max_seqs]
    conv_history_rows: torch.Tensor  # int64 [max_seqs, W - 1]
    padding_tokens: torch.Tensor  # bool [num_tokens]: past the batch's tokens


class TorchTitanLinearAttnBackend(MambaAttnBackendBase):
    """The base of SGLang's linear attention backends for TorchTitan's layers.

    SGLang's base backend builds the step's state slot of each request
    (``mamba_cache_indices``) and its query offsets, CUDA-graph buffers included; the
    subclasses run the trainer's Attention Gym convolution and recurrence on those
    slots, in place, as the vLLM engine's adapters do. Slot 0 is SGLang's padding slot,
    which the kernels skip. Speculative decoding is not supported.

    Where ``supports_prefill_graph_extend`` holds, the extend runs inside SGLang's
    breakable prefill CUDA graphs, on each bucket's ``PrefillGraphMetadata``, and each
    subclass zeroes its output at ``padding_tokens``, which the kernels leave unwritten.
    """

    # Set by a bucket's capture or replay; the next eager forward (init_forward_metadata)
    # clears it.
    prefill_graph_metadata: PrefillGraphMetadata | None = None

    def __init__(self, model_runner) -> None:
        super().__init__(model_runner)
        # A prefill batch holds at most one request per request slot, and at most
        # SGLang's prefill_max_requests. Each table row costs GPU time in every layer,
        # even when no request fills it.
        self.prefill_graph_max_seqs = min(
            self.req_to_token_pool.size,
            get_schedule().prefill_max_requests or self.req_to_token_pool.size,
        )
        # TODO(https://github.com/sgl-project/sglang/issues/48463): SGLang has no
        # public hooks for the extra_buffer prefix cache. The backends rely on its
        # internals: conv_states_shape below, the forward_metadata track fields read
        # in track_extend_states, the forward_batch track fields and the private
        # _translate_mamba_indices in refresh_prefill_graph_metadata, and the private
        # _track_mamba_state_decode in decode.
        # SGLang's extend tracking reads the conv window length (W - 1) as the last
        # dim of conv_states_shape; the pool keeps [W - 1, C] per slot, as SGLang's KDA
        # backend does, which exposes the transposed shape too.
        self.conv_states_shape = (
            self.state_pool.mamba_cache.conv[0].transpose(-1, -2).shape
        )

    def init_forward_metadata(self, forward_batch) -> None:
        # An eager forward reads SGLang's metadata of the batch, not a bucket's tables.
        self.prefill_graph_metadata = None
        super().init_forward_metadata(forward_batch)

    def supports_prefill_graph_extend(self) -> bool:
        """Whether SGLang's breakable prefill CUDA graphs capture the extend.

        Off, so each layer breaks the graph and runs eagerly:
        (a) on SGLang's draft runners (MTP), as with SGLang's own GDN backend;
        (b) under attention DP > 1.
        """
        # TODO: capture the extend under attention DP too. SGLang replays prefill graphs
        # there only on steps where no DP rank decodes (a decoding rank votes against
        # them for SSM models). Mask an idle rank's made-up request (rows past
        # ForwardBatch._original_batch_size), as SGLang's GDN backend does. Needs a
        # DP 2 test: no GDN class runs at attention DP, and the KDA DP-EP class runs
        # no CUDA graphs.
        return not self.is_draft_worker and get_parallel().attn_dp_size == 1

    def prefill_graph_extend_active(self) -> bool:
        return self.prefill_graph_metadata is not None

    def init_prefill_graph_metadata(self, forward_batch) -> PrefillGraphMetadata:
        """Allocate the tables of the bucket SGLang captures with ``forward_batch``, and
        fill them."""
        num_tokens = forward_batch.out_cache_loc.shape[0]
        max_seqs = min(num_tokens, self.prefill_graph_max_seqs)

        def zeros(*shape: int, dtype: torch.dtype = torch.int32) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=self.device)

        metadata = PrefillGraphMetadata(
            cu_seqlens=zeros(max_seqs + 1),
            state_indices=zeros(max_seqs),
            has_initial_state=zeros(max_seqs, dtype=torch.bool),
            checkpoint_slots=zeros(max_seqs),
            checkpoint_lens=zeros(max_seqs),
            conv_history_slots=zeros(max_seqs, dtype=torch.int64),
            conv_history_rows=zeros(
                max_seqs, self.conv_states_shape[-1], dtype=torch.int64
            ),
            padding_tokens=zeros(num_tokens, dtype=torch.bool),
        )
        self.refresh_prefill_graph_metadata(metadata, forward_batch)
        return metadata

    def refresh_prefill_graph_metadata(
        self, metadata: PrefillGraphMetadata, forward_batch
    ) -> None:
        """Fill ``metadata`` with the batch's requests, as the eager metadata would hold
        them, with device copies only (no host sync). The rows past the batch are
        padding."""
        self.prefill_graph_metadata = metadata
        # HiCache loads a request's states back from host memory layer by layer. A
        # captured extend can't wait for its own layer, so the replay waits for all of
        # them: a stream wait, the host does not block.
        self.token_to_kv_pool.layer_router.wait_for_part(STATE_PART)

        batch_size = forward_batch.batch_size
        num_tokens = sum(forward_batch.extend_seq_lens_cpu)

        # Each request's tokens and state slot; padding rows are empty sequences.
        metadata.cu_seqlens[:batch_size].copy_(
            forward_batch.extend_start_loc[:batch_size]
        )
        metadata.cu_seqlens[batch_size:].fill_(num_tokens)
        state_indices = self._translate_mamba_indices(
            self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
        )
        metadata.state_indices[:batch_size].copy_(state_indices[:batch_size])
        metadata.state_indices[batch_size:].fill_(self.pad_slot_id)
        torch.gt(
            forward_batch.extend_prefix_lens[:batch_size],
            0,
            out=metadata.has_initial_state[:batch_size],
        )
        metadata.has_initial_state[batch_size:].fill_(False)
        metadata.padding_tokens[:num_tokens].fill_(False)
        metadata.padding_tokens[num_tokens:].fill_(True)

        # The prefix-cache snapshots (see track_extend_states). Untracked and padding
        # rows copy their conv history into SGLang's dummy slot 0: a torch index
        # write would wrap the kernels' -1 to the last slot, a real one.
        metadata.checkpoint_slots.fill_(self.pad_slot_id)
        metadata.checkpoint_lens.zero_()
        metadata.conv_history_slots.zero_()
        metadata.conv_history_rows.zero_()

        # In place, as SGLang's eager metadata translates them once per forward.
        if forward_batch.mamba_track_indices is not None:
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
        aligned_lens = forward_batch.mamba_track_aligned_lens()
        if aligned_lens is None:
            return

        track_mask = forward_batch.mamba_track_mask[:batch_size]
        track_indices = forward_batch.mamba_track_indices[:batch_size]
        metadata.checkpoint_slots[:batch_size].copy_(
            torch.where(track_mask, track_indices, self.pad_slot_id)
        )
        metadata.checkpoint_lens[:batch_size].copy_(aligned_lens[:batch_size])
        metadata.conv_history_slots[:batch_size].copy_(
            torch.where(track_mask, track_indices, 0)
        )

        # The W - 1 inputs before each snapshot, clamped as SGLang's eager track
        # clamps them: untracked rows hold garbage lengths.
        window = self.conv_states_shape[-1]
        starts = (
            forward_batch.extend_start_loc[:batch_size]
            + aligned_lens[:batch_size]
            - window
        )
        torch.clamp(
            starts[:, None] + torch.arange(window, device=starts.device),
            min=0,
            max=num_tokens - 1,
            out=metadata.conv_history_rows[:batch_size],
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

        Returns ``{}`` when no request is tracked. In a prefill graph it returns the
        bucket's tables: untracked and padding rows copy into slot 0 and take no snapshot.
        """
        graph_metadata = self.prefill_graph_metadata
        if graph_metadata is not None:
            conv_state[graph_metadata.conv_history_slots] = mixed_qkv_TC[
                graph_metadata.conv_history_rows
            ].to(conv_state.dtype)
            # Every row passes a snapshot slot, so paged_gdn takes its snapshot path for
            # every batch.
            return {
                "checkpoint_slots": graph_metadata.checkpoint_slots,
                "checkpoint_lens": graph_metadata.checkpoint_lens,
            }

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
        graph_metadata = self.prefill_graph_metadata
        if graph_metadata is None:
            state_indices = self.forward_metadata.mamba_cache_indices.to(torch.int32)
            cu_seqlens = self.forward_metadata.query_start_loc
            has_initial_state = forward_batch.extend_prefix_lens > 0
        else:
            state_indices = graph_metadata.state_indices
            cu_seqlens = graph_metadata.cu_seqlens
            has_initial_state = graph_metadata.has_initial_state
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
