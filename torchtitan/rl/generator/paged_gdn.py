# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Gated DeltaNet recurrence over a paged state pool, shared by the engines' GDN
adapters: the trainer's Attention Gym kernels, run on the engine's state slots.

Legend: T = tokens of the step, C = rank-local q|k|v channels, H = value heads,
HK = key heads, K = key head dim, V = value head dim, N = state slots.
"""

import torch

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.attention.linear_attention import chunk_kernel_options
from torchtitan.models.common.attn_gym import (
    gate_transform,
    l2norm,
    paged_chunk_gdn,
    recurrent_gdn,
)


def paged_gdn(
    conv_output_TC: torch.Tensor,
    a_TH: torch.Tensor,
    b_TH: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_H: torch.Tensor,
    state_pool_NHVK: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> torch.Tensor:
    """Run the recurrence of each ``cu_seqlens`` interval from its slot in
    ``state_indices``, advancing ``state_pool_NHVK`` in place.

    Batch-invariant mode runs the trainer's sequential scan; otherwise the chunk kernel
    runs. Null (non-positive) slots skip state writes and produce zero output.

    Args:
        conv_output_TC: The post-convolution ``q | k | v`` channels, rank-local.
        A_log_H, dt_bias_H: The gate parameters, in float32.
        state_pool_NHVK: The float32 state pool.
        has_initial_state: Per interval, whether its slot holds a state to resume;
            False starts from zero.
        num_k_heads, num_v_heads: Rank-local head counts.

    Returns:
        ``(T, H, V)``.
    """
    num_tokens = conv_output_TC.shape[0]
    key_dim = num_k_heads * head_k_dim
    # Query and key are strided views, which l2norm reads and writes compact. Value is
    # copied because paged_chunk_gdn would otherwise copy a strided value itself, more
    # slowly.
    query = conv_output_TC[:, :key_dim].view(1, num_tokens, num_k_heads, head_k_dim)
    key = conv_output_TC[:, key_dim : 2 * key_dim].view(
        1, num_tokens, num_k_heads, head_k_dim
    )
    value = (
        conv_output_TC[:, 2 * key_dim :]
        .contiguous()
        .view(1, num_tokens, num_v_heads, head_v_dim)
    )
    # Keep the existing FP32 arithmetic while reusing the GDN gate contract.
    decay = gate_transform(
        a_TH.unsqueeze(0),
        A_log_H,
        dt_bias_H,
        kind="softplus",
        fastmath=False,
        impl="fused",
    )
    update_gate = torch.sigmoid(b_TH).unsqueeze(0)
    query = l2norm(query, cu_seqlens=cu_seqlens)
    key = l2norm(key, cu_seqlens=cu_seqlens)

    if is_in_batch_invariant_mode():
        recurrent_output, _ = recurrent_gdn(
            query,
            key,
            value,
            decay,
            update_gate,
            state_pool_NHVK,
            cu_seqlens=cu_seqlens,
            scale=head_k_dim**-0.5,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            # Triton autotuning breaks batch invariance.
            autotune=False,
        )
    else:
        recurrent_output = paged_chunk_gdn(
            query,
            key,
            value,
            decay,
            update_gate,
            state_pool_NHVK,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            scale=head_k_dim**-0.5,
            kernel_options=chunk_kernel_options(query),
        )
    return recurrent_output[0]
