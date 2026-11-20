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
from torchtitan.models.common.attn_gym import bound_gate, l2norm, paged_chunk_kda

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

    Returns:
        ``(B, T, H, K)``.
    """
    batch_size, num_tokens = convolved_BTC.shape[:2]
    query_BTHK, key_BTHK, value_BTHK = (
        tensor.reshape(batch_size, num_tokens, -1, head_dim)
        for tensor in convolved_BTC.chunk(3, dim=-1)
    )
    return paged_chunk_kda(
        l2norm(query_BTHK),
        l2norm(key_BTHK),
        value_BTHK,
        bound_gate(raw_gate_BTHK, A_log_H, dt_bias_HK, lower_bound=lower_bound),
        raw_beta_BTH.float().sigmoid(),
        state_pool_NHVK,
        state_indices,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial_state,
        replay_state=replay_state,
        kernel_options=(
            None if cu_seqlens is None else chunk_kernel_options(query_BTHK)
        ),
    )
