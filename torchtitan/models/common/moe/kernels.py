# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


__all__ = ["generate_permute_indices", "fill_indices_wrapper", "apply_router_scores"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "top_k"],
)
@triton.jit
def _apply_router_scores_fwd_kernel(
    routed_output_ptr,  # (N, D) bfloat16
    inv_perm_ptr,  # (N,) int32  —  inv_perm[j] = i  iff  tidxs[i] = j
    top_scores_ptr,  # (T, top_k) float32
    out_ptr,  # (T, D) bfloat16
    T,
    D,
    top_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    d_block = tl.program_id(1)
    d_off = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(top_k):
        # inv_perm[t*top_k+k] is the row in routed_output for token t's k-th slot
        sorted_idx = tl.load(inv_perm_ptr + t * top_k + k)
        val = tl.load(
            routed_output_ptr + sorted_idx.to(tl.int64) * D + d_off,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.load(top_scores_ptr + t * top_k + k)
        acc += score * val

    tl.store(out_ptr + t * D + d_off, acc.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _apply_router_scores_bwd_routed_kernel(
    d_out_ptr,  # (T, D) bfloat16
    top_scores_ptr,  # (T, top_k) float32
    tidxs_ptr,  # (N,) int64  —  token_indices_experts_sorted
    d_routed_ptr,  # (N, D) bfloat16  output
    D,
    top_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # grad wrt routed_output: d_routed[i] = score[t,k] * d_out[t]
    # where t = tidxs[i] // top_k, k = tidxs[i] % top_k
    i = tl.program_id(0)
    d_block = tl.program_id(1)
    d_off = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D

    j = tl.load(tidxs_ptr + i)
    t = j // top_k
    k = j % top_k
    score = tl.load(top_scores_ptr + t * top_k + k)
    d_out = tl.load(d_out_ptr + t * D + d_off, mask=mask, other=0.0).to(tl.float32)
    tl.store(d_routed_ptr + i * D + d_off, (score * d_out).to(tl.bfloat16), mask=mask)


@triton.jit
def _apply_router_scores_bwd_scores_kernel(
    routed_output_ptr,  # (N, D) bfloat16
    d_out_ptr,  # (T, D) bfloat16
    inv_perm_ptr,  # (N,) int32
    d_scores_ptr,  # (T, top_k) float32
    D,
    top_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # grad wrt top_scores: d_scores[t,k] = dot(routed_output[inv_perm[t*top_k+k]], d_out[t])
    t = tl.program_id(0)
    k = tl.program_id(1)

    sorted_idx = tl.load(inv_perm_ptr + t * top_k + k)
    # Accumulate element-wise products in a [BLOCK_D] buffer; reduce to scalar at end.
    # This avoids the tl.zeros([1]) → scalar-ptr store mismatch that breaks
    # torch.compile's identify_mutated_tensors analysis.
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        mask = d_off < D
        ro = tl.load(
            routed_output_ptr + sorted_idx.to(tl.int64) * D + d_off,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        do = tl.load(d_out_ptr + t * D + d_off, mask=mask, other=0.0).to(tl.float32)
        acc = acc + ro * do

    tl.store(d_scores_ptr + t * top_k + k, tl.sum(acc))


def apply_router_scores(
    routed_output: torch.Tensor,
    token_indices_experts_sorted: torch.Tensor,
    top_scores: torch.Tensor,
    inv_perm: torch.Tensor,
) -> torch.Tensor:
    """Fused scatter + FP32 bmm for MoE output combine.

    Replaces the two-step scatter + bmm in the MoE combine path:

        out_unsorted = zeros(T*top_k, D)
        out_unsorted[token_indices_experts_sorted] = routed_output   # IndexFuncLargeIndex
        out_unsorted = out_unsorted.reshape(T, top_k, D)
        out = bmm(top_scores.reshape(T,1,top_k), out_unsorted.float()).squeeze(1)  # SM80 FP32

    with a single Triton kernel that gathers top_k rows per output token and accumulates
    the weighted sum in FP32, writing BF16 output. No intermediate (T*top_k, D) tensor
    is allocated.

    Args:
        routed_output: (T*top_k, D) bfloat16 — expert outputs, sorted by expert.
        token_indices_experts_sorted: (T*top_k,) int64 — argsort of expert assignments,
            maps sorted position i → original flat index j = token*top_k + slot.
        top_scores: (T, top_k) float32 — routing scores.
        inv_perm: (T*top_k,) int32 — inverse permutation of token_indices_experts_sorted,
            pre-computed in TokenReorderer via O(N) scatter to avoid a second argsort.

    Returns:
        (T, D) bfloat16 — per-token weighted sum of expert outputs.
    """
    return _ApplyRouterScoresFunction.apply(
        routed_output, token_indices_experts_sorted, top_scores, inv_perm
    )


class _ApplyRouterScoresFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        routed_output: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        top_scores: torch.Tensor,
        inv_perm: torch.Tensor,
    ) -> torch.Tensor:
        N, D = routed_output.shape
        T, top_k = top_scores.shape

        out = torch.empty(
            (T, D), dtype=routed_output.dtype, device=routed_output.device
        )
        grid = lambda meta: (T, triton.cdiv(D, meta["BLOCK_D"]))  # noqa: E731
        _apply_router_scores_fwd_kernel[grid](
            routed_output,
            inv_perm,
            top_scores,
            out,
            T,
            D,
            top_k=top_k,
        )

        ctx.save_for_backward(
            routed_output, token_indices_experts_sorted, inv_perm, top_scores
        )
        ctx.top_k = top_k
        return out

    @staticmethod
    def backward(
        ctx, d_out: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
        routed_output, tidxs, inv_perm, top_scores = ctx.saved_tensors
        top_k = ctx.top_k
        N, D = routed_output.shape
        T = top_scores.shape[0]

        d_out = d_out.contiguous()

        d_routed = torch.empty_like(routed_output)
        grid_r = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))  # noqa: E731
        _apply_router_scores_bwd_routed_kernel[grid_r](
            d_out,
            top_scores,
            tidxs,
            d_routed,
            D,
            top_k=top_k,
        )

        d_scores = torch.empty_like(top_scores)
        _apply_router_scores_bwd_scores_kernel[(T, top_k)](
            routed_output,
            d_out,
            inv_perm,
            d_scores,
            D,
            top_k=top_k,
            BLOCK_D=128,
        )

        return d_routed, None, d_scores, None


# parallelized kernel
@triton.jit
def _fill_indices_kernel(
    tokens_per_expert_group_ptr,
    start_index_values_ptr,
    write_offsets_ptr,
    output_ptr,
    experts_per_rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # map programs (blocks) to the experts and loop (grid stride) if needed
    # pyrefly: ignore [no-matching-overload]
    for expert_id in range(pid, experts_per_rank, num_programs):
        # read this experts write offset
        write_offset = tl.load(write_offsets_ptr + expert_id)

        for r in range(num_ranks):
            # index into tokens_per_expert_group array
            i = r * experts_per_rank + expert_id

            # load start index and number of tokens for this expert-rank pair
            start_index = tl.load(start_index_values_ptr + i)
            length = tl.load(tokens_per_expert_group_ptr + i)

            # each thread in block processes tokens in parallel
            offsets = tl.arange(0, BLOCK_SIZE)

            # tokens are processed in chunks of BLOCK_SIZE
            # pyrefly: ignore [no-matching-overload]
            for chunk_start in range(0, length, BLOCK_SIZE):
                chunk_offsets = chunk_start + offsets

                # mask valid indices
                mask = chunk_offsets < length

                values = start_index + chunk_offsets

                # destination
                dest_indices = write_offset + chunk_offsets

                # store
                tl.store(output_ptr + dest_indices, values, mask=mask)

            # update write offset for next rank
            write_offset += length


# ==============
# wrapper
# ==============


def _fill_indices_impl(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,  # cap on total number of blocks to launch
) -> torch.Tensor:
    # preallocate output
    permuted_indices = torch.full(
        (max_len,), -1, dtype=torch.int64, device=tokens_per_expert_group.device
    )

    # write offsets is per local expert...
    num_blocks = min(experts_per_rank, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)

    # launch kernel
    _fill_indices_kernel[grid](
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        permuted_indices,
        # pyrefly: ignore [bad-argument-type]
        experts_per_rank,
        # pyrefly: ignore [bad-argument-type]
        num_ranks,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_SIZE=block_size,
    )
    return permuted_indices


@torch.library.custom_op("torchtitan::fill_indices", mutates_args=())
def fill_indices_wrapper(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
) -> torch.Tensor:
    return _fill_indices_impl(
        tokens_per_expert_group,
        start_index_values,
        write_offsets,
        experts_per_rank,
        num_ranks,
        max_len,
        block_size,
        max_blocks,
    )


@fill_indices_wrapper.register_fake
def _fill_indices_fake(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    block_size: int = 128,
    max_blocks: int = 1024,
) -> torch.Tensor:
    return torch.empty(
        max_len, dtype=torch.int64, device=tokens_per_expert_group.device
    )


# reference
def fill_indices_cpu(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
):
    # We need to preallocate the output - we ignore device and force it on cpu
    # device = tokens_per_expert_group.device
    permuted_indices = torch.full(
        (max_len,),
        -1,
        dtype=torch.int64,
    )  # device=device)
    # Fill the permuted indices
    # For each local expert
    for e in range(experts_per_rank):
        write_start = write_offsets[e].item()
        # For each remote rank
        for r in range(num_ranks):
            i = r * experts_per_rank + e
            start_index = start_index_values[i].item()
            length = tokens_per_expert_group[i].item()
            # Fill in the indices
            if length > 0:
                end_idx = min(write_start + length, max_len)
                permuted_indices[write_start:end_idx] = torch.arange(
                    start_index,
                    start_index + (end_idx - write_start),
                    dtype=torch.int64,
                    # device=device,
                )
            write_start += length
    return permuted_indices


def generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
    alignment: int,
    use_cpu: bool = False,
):
    """
    Prepare permutation indices and the number of tokens for each expert.

    Args:
        tokens_per_expert_group: number of tokens for each expert from all ranks.
        experts_per_rank: number of experts per rank.
        num_ranks: number of ranks.
        max_len: maximum length of the output index vector.
        alignment: alignment for each returned element in `m_sizes` and padding min for zero token experts.
        use_cpu: whether to use CPU implementation.


    Returns:
        permuted_indices: Tensor of indices that map original token order to the expert-grouped order.
        m_sizes: aligned number of tokens for each expert (padded to alignment boundary).
        m_offsets: Cumulative sum of m_sizes. The exclusive ending position for each expert's tokens.

    Explanatory details:
        `tokens_per_expert_group` is of shape (num_ranks * experts_per_rank,), for example:
        From: |       rank 0      |       rank 1      |
        To:   | E0 | E1 | E2 | E3 | E0 | E1 | E2 | E3 |
              |  4 |  2 |  1 |  3 |  1 |  2 |  3 |  4 |
    """

    # prefix sum to get start index of each expert (parallel scan kernel in future?)
    start_index_values = (
        torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group
    )

    # total tokens for each expert (sum over ranks)
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # pad out empty experts to alignment requirement
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)

    # align the chunk sizes (cdiv)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(
        torch.int32
    )

    # additional prefix sum to get write offset of each expert in permuted_indices
    # write offsets is per local expert, not global
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    # Select the implementation to use
    if use_cpu:
        permuted_indices = fill_indices_cpu(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )
    else:
        permuted_indices = fill_indices_wrapper(
            tokens_per_expert_group,
            start_index_values,
            write_offsets,
            experts_per_rank,
            num_ranks,
            max_len,
        )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)
