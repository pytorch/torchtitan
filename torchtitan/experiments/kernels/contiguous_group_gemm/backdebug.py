# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from cg_backward import early_config_prune, STANDARD_CONFIGS


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_GROUP", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_backward_weights_fixed(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    inputs_ptr,  # [M_TOTAL, K]
    grad_weights_ptr,  # [num_experts, N, K]
    # Pointer to indices array
    indices_ptr,  # [M_TOTAL]
    # Expert ID for this kernel
    expert_idx: tl.constexpr,
    # Group parameters
    group_start: tl.constexpr,  # Start index of group
    group_size: tl.constexpr,  # Size of group
    # Matrix dimensions
    M_GROUP: tl.constexpr,  # Group size (equals group_size)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes the gradient with respect to weights for a single expert group.
    Fixed version with improved memory access patterns and computational stability.
    """
    pid = tl.program_id(0)

    # Number of tiles in N and K dimensions
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # 2D tile index within the expert's grid
    tile_n = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    # Starting indices for this tile within the expert's weight matrix
    n_start = tile_n * BLOCK_SIZE_N
    k_start = tile_k * BLOCK_SIZE_K

    # Only process if the indices are in bounds
    if n_start < N and k_start < K:

        # Offsets for this tile
        offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

        # Masks for bounds checking
        mask_n = offs_n < N
        mask_k = offs_k < K

        # Initialize accumulator for the gradient
        grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

        # Process the group in tiles along M dimension
        for m_offset in range(0, group_size, BLOCK_SIZE_M):
            # Compute actual block size (might be smaller at the end)
            actual_block_size = min(BLOCK_SIZE_M, group_size - m_offset)

            # Load mask for this M block
            mask_m = tl.arange(0, BLOCK_SIZE_M) < actual_block_size

            # Global offsets for group's data
            global_m_offset = group_start + m_offset
            offs_m = tl.arange(0, BLOCK_SIZE_M) + global_m_offset

            # Load grad_output [M, N]
            go_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            mask_go = mask_m[:, None] & mask_n[None, :]
            go = tl.load(go_ptrs, mask=mask_go, other=0.0)

            # Load inputs [M, K]
            in_ptrs = inputs_ptr + offs_m[:, None] * K + offs_k[None, :]
            mask_in = mask_m[:, None] & mask_k[None, :]
            inp = tl.load(in_ptrs, mask=mask_in, other=0.0)

            # Compute gradient contribution
            # Explicitly reshape for better numerical stability
            go_t = tl.trans(go)
            grad_weights += tl.dot(go_t, inp)

        # Store results to expert's weight gradient matrix
        grad_w_ptrs = (
            grad_weights_ptr
            + expert_idx * N * K
            + offs_n[:, None] * K
            + offs_k[None, :]
        )
        mask_gw = mask_n[:, None] & mask_k[None, :]
        tl.atomic_add(grad_w_ptrs, grad_weights, mask=mask_gw)


def cg_grouped_gemm_backward_weights_fixed(
    grad_output: torch.Tensor,  # [M_total, N]
    inputs: torch.Tensor,  # [M_total, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Backward pass for contiguous grouped GEMM with respect to expert weights.
    Fixed version with improved stability.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        inputs: Input tensor, shape [M_total, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        num_experts: Number of experts
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert inputs.is_contiguous(), "Inputs tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, N = grad_output.shape
    _, K = inputs.shape

    # Check if dimensions match
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Number of groups
    num_groups = M_total // group_size_m

    # Create output tensor for gradients (initialized to zeros)
    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Process each group separately
    for group_idx in range(num_groups):
        # Get group boundaries
        group_start = group_idx * group_size_m

        # Get expert ID for this group - get expert index from the group start
        # Using .item() can detach from computation graph, which could cause issues
        expert_idx = int(expert_indices[group_start].item())

        # Make sure the expert index is valid
        if expert_idx < 0 or expert_idx >= num_experts:
            raise ValueError(f"Invalid expert index {expert_idx} for group {group_idx}")

        # Check that all tokens in this group use the same expert
        group_end = (group_idx + 1) * group_size_m
        group_indices = expert_indices[group_start:group_end]
        if not (group_indices == expert_idx).all():
            print(
                f"Warning: Not all tokens in group {group_idx} use expert {expert_idx}"
            )

        # Grid for this kernel launch
        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_SIZE_N"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]),
        )

        # Launch a kernel instance for this group
        _kernel_cg_backward_weights_fixed[grid](
            grad_output,
            inputs,
            grad_weights,
            expert_indices,  # Pass indices even though we're not using them in the kernel
            expert_idx=expert_idx,
            group_start=group_start,
            group_size=group_size_m,
            M_GROUP=group_size_m,
            N=N,
            K=K,
        )

    return grad_weights


# Update the autograd function to use the fixed weight gradient implementation
class ContiguousGroupedGEMMFixed(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM with improved backward pass.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, group_size_m=128):
        """Forward pass for contiguous grouped GEMM."""
        from cg_forward import cg_grouped_gemm_forward

        # Save for backward
        ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.group_size_m = group_size_m

        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for contiguous grouped GEMM."""
        inputs, expert_weights, expert_indices = ctx.saved_tensors
        group_size_m = ctx.group_size_m

        # Get number of experts
        num_experts = expert_weights.shape[0]

        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Compute gradients
        grad_inputs = cg_grouped_gemm_backward_inputs(
            grad_output=grad_output,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

        # Use the fixed weight gradient implementation
        grad_weights = cg_grouped_gemm_backward_weights_fixed(
            grad_output=grad_output,
            inputs=inputs,
            expert_indices=expert_indices,
            num_experts=num_experts,
            group_size_m=group_size_m,
        )

        # No gradient for expert_indices (it's just an index tensor)
        grad_indices = None

        # No gradient for group_size_m (it's just a parameter)
        grad_group_size_m = None

        return grad_inputs, grad_weights, grad_indices, grad_group_size_m


def cg_grouped_gemm_fixed(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Interface for contiguous grouped GEMM with improved backward pass.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMMFixed.apply(
        inputs, expert_weights, expert_indices, group_size_m
    )


def manual_pytorch_weight_gradient(
    grad_output: torch.Tensor,  # [M_total, N]
    inputs: torch.Tensor,  # [M_total, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Compute the weight gradient manually using PyTorch operations.
    This serves as a detailed reference for comparison and debugging.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        inputs: Input tensor, shape [M_total, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        num_experts: Number of experts
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Get dimensions
    M_total, N = grad_output.shape
    _, K = inputs.shape

    # Number of groups
    num_groups = M_total // group_size_m

    # Create output tensor for gradients
    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Process each group separately
    for group_idx in range(num_groups):
        # Get group boundaries
        group_start = group_idx * group_size_m
        group_end = min(group_start + group_size_m, M_total)

        # Get expert ID for this group
        expert_idx = expert_indices[group_start].item()

        # Extract group data
        group_grad_output = grad_output[group_start:group_end]  # [group_size, N]
        group_inputs = inputs[group_start:group_end]  # [group_size, K]

        # Compute gradient contribution for this group
        # Formula: dW = grad_output.T @ inputs
        # Shape: [N, group_size] @ [group_size, K] = [N, K]
        grad_contribution = group_grad_output.t() @ group_inputs

        # Accumulate gradient for the expert
        grad_weights[expert_idx] += grad_contribution

    return grad_weights


def inspect_gradient_calculation(
    M_total=512, N=256, K=256, num_experts=4, group_size_m=128, device="cuda"
):
    """
    Detailed inspection of the gradient calculation process to identify issues.
    This function compares step-by-step the gradient calculation between different methods.
    """
    print("==== Gradient Calculation Inspection ====")

    # Ensure M_total is a multiple of group_size_m
    M_total = (M_total // group_size_m) * group_size_m
    num_groups = M_total // group_size_m

    print(
        f"Configuration: M_total={M_total}, N={N}, K={K}, num_experts={num_experts}, group_size_m={group_size_m}"
    )

    # Create test tensors with specific patterns to help with debugging
    torch.manual_seed(0)
    inputs = torch.randn((M_total, K), device=device)
    grad_output = torch.randn((M_total, N), device=device)

    # Create expert indices - each token in a group has the same expert
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    chosen_experts = []
    for g in range(num_groups):
        expert_idx = g % num_experts  # Use a deterministic pattern for debugging
        chosen_experts.append(expert_idx)
        start_idx = g * group_size_m
        end_idx = (g + 1) * group_size_m
        expert_indices[start_idx:end_idx] = expert_idx

    print(f"Expert assignment pattern: {chosen_experts}")

    # 1. Compute manually with PyTorch - the ground truth reference
    manual_grad = manual_pytorch_weight_gradient(
        grad_output, inputs, expert_indices, num_experts, group_size_m
    )

    # 2. Import the fixed triton implementation
    # from comprehensive_backward_fix import cg_grouped_gemm_backward_weights_fixed

    triton_grad = cg_grouped_gemm_backward_weights_fixed(
        grad_output, inputs, expert_indices, num_experts, group_size_m
    )

    # 3. Import the original triton implementation
    from cg_backward import cg_grouped_gemm_backward_weights

    original_grad = cg_grouped_gemm_backward_weights(
        grad_output, inputs, expert_indices, num_experts, group_size_m
    )

    # Compare results
    match_fixed = torch.allclose(triton_grad, manual_grad, rtol=1e-1, atol=1e-1)
    match_original = torch.allclose(original_grad, manual_grad, rtol=1e-1, atol=1e-1)

    print(f"Fixed implementation matches manual: {match_fixed}")
    print(f"Original implementation matches manual: {match_original}")

    # Print detailed statistics for each expert
    for e in range(num_experts):
        used = e in chosen_experts

        manual_expert = manual_grad[e]
        triton_expert = triton_grad[e]
        original_expert = original_grad[e]

        # Only compare if this expert was used
        if used:
            expert_match_fixed = torch.allclose(
                triton_expert, manual_expert, rtol=1e-3, atol=1e-3
            )
            expert_match_original = torch.allclose(
                original_expert, manual_expert, rtol=1e-3, atol=1e-3
            )

            # Calculate statistics
            max_diff_fixed = torch.max(torch.abs(triton_expert - manual_expert)).item()
            max_diff_original = torch.max(
                torch.abs(original_expert - manual_expert)
            ).item()

            avg_diff_fixed = torch.mean(torch.abs(triton_expert - manual_expert)).item()
            avg_diff_original = torch.mean(
                torch.abs(original_expert - manual_expert)
            ).item()

            manual_norm = torch.norm(manual_expert).item()
            triton_norm = torch.norm(triton_expert).item()
            original_norm = torch.norm(original_expert).item()

            print(f"\nExpert {e} (Used: {used}):")
            print(
                f"  Fixed impl match: {expert_match_fixed}, Max diff: {max_diff_fixed:.6f}, Avg diff: {avg_diff_fixed:.6f}"
            )
            print(
                f"  Original impl match: {expert_match_original}, Max diff: {max_diff_original:.6f}, Avg diff: {avg_diff_original:.6f}"
            )
            print(
                f"  Norm - Manual: {manual_norm:.4f}, Fixed: {triton_norm:.4f}, Original: {original_norm:.4f}"
            )

            # If differences are large, look at a sample
            if max_diff_fixed > 0.1 or max_diff_original > 0.1:
                # Find a sample position with large difference
                if max_diff_fixed > max_diff_original:
                    diff_tensor = torch.abs(triton_expert - manual_expert)
                    method = "fixed"
                else:
                    diff_tensor = torch.abs(original_expert - manual_expert)
                    method = "original"

                flat_idx = torch.argmax(diff_tensor)
                row = flat_idx // K
                col = flat_idx % K

                print(f"  Sample at position [{row}, {col}]:")
                print(f"    Manual: {manual_expert[row, col].item():.6f}")
                print(f"    Fixed: {triton_expert[row, col].item():.6f}")
                print(f"    Original: {original_expert[row, col].item():.6f}")

                # Manually trace the calculation for this specific position
                print("\n  Tracing calculation for this position:")
                for g in range(num_groups):
                    if chosen_experts[g] == e:
                        group_start = g * group_size_m
                        group_end = (g + 1) * group_size_m

                        # Extract group data
                        group_grad_output = grad_output[
                            group_start:group_end
                        ]  # [group_size, N]
                        group_inputs = inputs[group_start:group_end]  # [group_size, K]

                        # Contribution of this group to the specific position
                        group_contribution = torch.sum(
                            group_grad_output[:, row] * group_inputs[:, col]
                        ).item()

                        print(f"    Group {g} contribution: {group_contribution:.6f}")

    return manual_grad, triton_grad, original_grad


# Implementation of a completely new approach using a simpler kernel design
@triton.jit
def _kernel_cg_backward_weights_simple(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    inputs_ptr,  # [M_TOTAL, K]
    grad_weights_ptr,  # [num_experts, N, K]
    indices_ptr,  # [M_total]
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Group parameters
    GROUP_SIZE_M: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Significantly simplified kernel for weight gradient computation.
    This kernel processes one N-K tile across all groups that use the same expert.
    """
    pid = tl.program_id(0)

    # Determine expert and position within the matrix
    expert_id = pid // ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))
    position_id = pid % ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))

    # Only process if expert is valid
    if expert_id < NUM_EXPERTS:
        # Calculate positions in N and K dimensions
        n_tiles = K // BLOCK_SIZE_K
        tile_n = position_id // n_tiles
        tile_k = position_id % n_tiles

        n_start = tile_n * BLOCK_SIZE_N
        k_start = tile_k * BLOCK_SIZE_K

        # Only process if in bounds
        if n_start < N and k_start < K:
            # Create offset arrays
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

            # Create masks for bounds checking
            mask_n = offs_n < N
            mask_k = offs_k < K

            # Initialize accumulator for the gradient
            grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

            # Go through all groups to find those using this expert
            for group_idx in range(0, M_TOTAL // GROUP_SIZE_M):
                group_start = group_idx * GROUP_SIZE_M

                # Get expert ID for this group
                group_expert = tl.load(indices_ptr + group_start)

                # Only process if this group uses the current expert
                if group_expert == expert_id:
                    # Process the group in blocks
                    for m_offset in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
                        # Global offsets for group's data
                        m_start = group_start + m_offset
                        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start

                        # Create mask for M dimension
                        mask_m = offs_m < min(group_start + GROUP_SIZE_M, M_TOTAL)

                        # Load grad_output [M, N]
                        go_ptrs = (
                            grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
                        )
                        mask_go = mask_m[:, None] & mask_n[None, :]
                        go = tl.load(go_ptrs, mask=mask_go, other=0.0)

                        # Load inputs [M, K]
                        in_ptrs = inputs_ptr + offs_m[:, None] * K + offs_k[None, :]
                        mask_in = mask_m[:, None] & mask_k[None, :]
                        inp = tl.load(in_ptrs, mask=mask_in, other=0.0)

                        # Compute gradient contribution
                        go_t = tl.trans(go)  # Transpose: [N, M]
                        grad_weights += tl.dot(go_t, inp)

            # Store results to the appropriate part of the expert's weight gradients
            grad_w_ptrs = (
                grad_weights_ptr
                + expert_id * N * K
                + offs_n[:, None] * K
                + offs_k[None, :]
            )
            mask_gw = mask_n[:, None] & mask_k[None, :]
            tl.store(grad_w_ptrs, grad_weights, mask=mask_gw)


def cg_grouped_gemm_backward_weights_simple(
    grad_output: torch.Tensor,  # [M_total, N]
    inputs: torch.Tensor,  # [M_total, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Simple version of backward pass for weights using a single kernel launch.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        inputs: Input tensor, shape [M_total, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        num_experts: Number of experts
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert inputs.is_contiguous(), "Inputs tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, N = grad_output.shape
    _, K = inputs.shape

    # Ensure expert_indices has the right dtype
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Create output tensor for gradients
    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Define block sizes based on dimensions
    # These are chosen to balance parallelism and shared memory usage
    block_size_n = min(128, N)
    block_size_k = min(32, K)
    block_size_m = min(32, group_size_m)

    # Calculate grid size for the kernel
    # Each thread block handles one expert's N-K tile
    n_tiles = triton.cdiv(N, block_size_n)
    k_tiles = triton.cdiv(K, block_size_k)
    grid = (num_experts * n_tiles * k_tiles,)

    # Launch kernel
    _kernel_cg_backward_weights_simple[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_M=block_size_m,
    )

    return grad_weights


def verify_simple_approach(
    M_total=1024,
    N=256,
    K=256,
    num_experts=8,
    group_size_m=128,
    device="cuda",
    atol=1e-1,
    rtol=1e-1,
):
    """
    Verify the simple approach against the PyTorch reference implementation.
    """
    print("==== Verifying Simple Approach ====")

    # Ensure M_total is a multiple of group_size_m
    M_total = (M_total // group_size_m) * group_size_m
    num_groups = M_total // group_size_m

    # Create test tensors
    torch.manual_seed(0)
    inputs = torch.randn((M_total, K), device=device)
    grad_output = torch.randn((M_total, N), device=device)

    # Create expert indices - each token in a group has the same expert
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    chosen_experts = []
    for g in range(num_groups):
        expert_idx = g % num_experts  # Use a deterministic pattern
        chosen_experts.append(expert_idx)
        start_idx = g * group_size_m
        end_idx = (g + 1) * group_size_m
        expert_indices[start_idx:end_idx] = expert_idx

    # Compute gradients with PyTorch reference
    manual_grad = manual_pytorch_weight_gradient(
        grad_output, inputs, expert_indices, num_experts, group_size_m
    )

    # Compute with simple approach
    simple_grad = cg_grouped_gemm_backward_weights_simple(
        grad_output, inputs, expert_indices, num_experts, group_size_m
    )

    # Compare results
    match = torch.allclose(simple_grad, manual_grad, rtol=rtol, atol=atol)
    print(f"Simple approach matches manual: {match}")

    # Detailed comparison per expert
    for e in range(num_experts):
        used = e in chosen_experts

        if used:
            expert_match = torch.allclose(
                simple_grad[e], manual_grad[e], rtol=rtol, atol=atol
            )
            max_diff = torch.max(torch.abs(simple_grad[e] - manual_grad[e])).item()
            avg_diff = torch.mean(torch.abs(simple_grad[e] - manual_grad[e])).item()

            print(
                f"Expert {e}: Match: {expert_match}, Max diff: {max_diff:.6f}, Avg diff: {avg_diff:.6f}"
            )

    return match, manual_grad, simple_grad


if __name__ == "__main__":
    # inspect_gradient_calculation()
    verify_simple_approach()
    inspect_gradient_calculation()
