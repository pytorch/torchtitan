# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Kimi Delta Attention recurrence over a paged state pool, shared by the engines'
KDA adapters: the trainer's Attention Gym kernels, run on the engine's state slots.

Legend: B = rows, T = tokens per row, C = rank-local q|k|v channels, H = heads,
K = key head dim, V = value head dim (V == K), N = state slots.
"""

import torch

from torchtitan.models.common.attention.linear_attention import chunk_kernel_options
from torchtitan.models.common.attn_gym import (
    bound_gate,
    copy_replay_tails,
    KernelOptions,
    l2norm,
    paged_chunk_kda,
    ReplayState,
    StateCheckpoints,
)

# Attention Gym's replay state: the q, k, v, gate and beta of each slot's unfinished
# chunk, and its token count.
# TODO(https://github.com/meta-pytorch/attention-gym/issues/692): Attention Gym
# documents this layout (ReplayState) but exposes no constants or constructor for it;
# take them from Attention Gym.
REPLAY_CHUNK_SIZE = 64
REPLAY_STATE_DTYPES = (
    torch.bfloat16,
    torch.bfloat16,
    torch.bfloat16,
    torch.float32,
    torch.float32,
    torch.int32,
)


def replay_state_shapes(
    local_num_heads: int, head_dim: int
) -> tuple[tuple[int, ...], ...]:
    """Return the per-slot shapes of ``REPLAY_STATE_DTYPES``'s tensors."""
    return (
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads, head_dim),
        (REPLAY_CHUNK_SIZE, local_num_heads),
        (1,),
    )


def paged_kda(
    convolved_BTC: torch.Tensor,
    raw_gate_BTHK: torch.Tensor,
    raw_beta_BTH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_HK: torch.Tensor,
    state_pool_NHVK: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    replay_state: tuple[torch.Tensor, ...] | None,
    head_dim: int,
    lower_bound: float,
    checkpoint_slots: torch.Tensor | None = None,
    checkpoint_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the recurrence of each sequence from its slot in ``state_indices``,
    advancing ``state_pool_NHVK`` (and ``replay_state``) in place.

    Packed prefill passes ``B == 1`` with ``cu_seqlens``; decode passes one token per
    row and no ``cu_seqlens``. With ``replay_state`` (batch-invariant mode), the slot's
    state is the one after its last complete chunk plus the replayed unfinished chunk,
    which matches the trainer's chunk kernel bitwise.

    Args:
        convolved_BTC: The post-convolution ``q | k | v`` channels, rank-local.
        A_log_H, dt_bias_HK: The gate parameters, in float32.
        has_initial_state: Per sequence, whether its slot holds a state to resume;
            None resumes every slot.
        checkpoint_slots: The index of the state-pool slot that receives each
            sequence's snapshot, for the prefix cache. A non-positive index means that
            sequence takes no snapshot.
            With ``replay_state`` (batch-invariant mode), the slot also receives the
            unfinished chunk's tokens, so the snapshot can resume mid-chunk. Prefill only.
        checkpoint_lens: The number of each sequence's new tokens in this call to
            process before its snapshot is taken. It must be a multiple of 64, the
            kernels' chunk size.

    Returns:
        ``(B, T, H, K)``.
    """
    batch_size, num_tokens = convolved_BTC.shape[:2]
    query_BTHK, key_BTHK, value_BTHK = (
        tensor.reshape(batch_size, num_tokens, -1, head_dim)
        for tensor in convolved_BTC.chunk(3, dim=-1)
    )
    inputs = (
        l2norm(query_BTHK),
        l2norm(key_BTHK),
        value_BTHK,
        bound_gate(raw_gate_BTHK, A_log_H, dt_bias_HK, lower_bound=lower_bound),
        raw_beta_BTH.float().sigmoid(),
    )
    checkpoints = None
    if checkpoint_slots is not None:
        assert cu_seqlens is not None
        checkpoints = StateCheckpoints(
            state_pool_NHVK, checkpoint_lens, checkpoint_slots
        )
        if replay_state is not None:
            # Before paged_chunk_kda, which advances the replay counts this reads.
            copy_replay_tails(
                ReplayState(*replay_state),
                inputs,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                checkpoint_positions=checkpoint_lens,
                checkpoint_slots=checkpoint_slots,
                has_initial_state=has_initial_state,
            )
    return paged_chunk_kda(
        *inputs,
        state_pool_NHVK,
        state_indices,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial_state,
        replay_state=replay_state,
        # TODO(acisseJZhong, https://github.com/meta-pytorch/attention-gym/issues/980):
        # warpdrive kernels can't write state checkpoints yet, so prefills that take
        # a snapshot use the fused kernels.
        kernel_options=(
            None
            if cu_seqlens is None
            else chunk_kernel_options(query_BTHK)
            if checkpoints is None
            else KernelOptions(backend="fused")
        ),
        state_checkpoints=checkpoints,
    )
