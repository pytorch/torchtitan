import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton import Config

# ===== Configuration and Utilities =====

# Define standard configurations with optimizations for backward pass
BACKWARD_CONFIGS = [
    # Small matrix configurations
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    # Medium matrix configurations
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    # Large matrix configurations
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    # New: Specialized configurations for backward pass
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=4,
    ),
]


# Enhanced configuration pruning for backward pass
def backward_config_prune(configs, args, **kwargs):
    """Filter configurations based on matrix dimensions and hardware limits."""
    M = kwargs.get("M", 0)
    N = kwargs.get("N", 0)
    K = kwargs.get("K", 0)

    pruned_configs = []
    for config in configs:
        block_m = config.kwargs.get("BLOCK_SIZE_M", 0)
        block_n = config.kwargs.get("BLOCK_SIZE_N", 0)
        block_k = config.kwargs.get("BLOCK_SIZE_K", 0)

        # Skip if block size exceeds dimensions
        if block_k > K:
            continue

        # For very small matrices, prefer smaller blocks
        if M < 128 and N < 128 and (block_m > 128 or block_n > 128):
            continue

        pruned_configs.append(config)

    # Ensure we don't prune all configs
    if not pruned_configs and configs:
        return configs[:1]

    return pruned_configs


# ===== Gradient with respect to inputs (dX) kernel =====


@triton.autotune(
    configs=BACKWARD_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": backward_config_prune},
)
@triton.jit
def contiguous_grouped_gemm_backward_dx_kernel(
    # Pointers to matrices
    grad_output_ptr,  # Gradient of output [M, N]
    weight_ptr,  # Expert weights [num_experts, N, K]
    grad_input_ptr,  # Output: Gradient of input [M, K]
    # Pointer to indices array
    indices_ptr,  # Expert indices [M]
    # Matrix dimensions
    M: tl.constexpr,  # Number of tokens
    N: tl.constexpr,  # Hidden dimension of output
    K: tl.constexpr,  # Hidden dimension of input
    NUM_EXPERTS: tl.constexpr,  # Number of experts
    # Kernel configuration
    NUM_SMS: tl.constexpr,  # Number of SMs
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Optional parameters
    USE_TENSOR_CORES: tl.constexpr = True,
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Backward pass kernel to compute gradients with respect to inputs.

    For each token i, computes:
    grad_input[i] = grad_output[i] @ expert_weights[indices[i]]
    """
    # Get thread block index
    pid = tl.program_id(0)

    # Get data types
    grad_output_dtype = grad_output_ptr.dtype.element_ty
    weight_dtype = weight_ptr.dtype.element_ty
    grad_input_dtype = grad_input_ptr.dtype.element_ty

    # Use higher precision for accumulation
    compute_dtype = tl.float32

    # Number of tiles and work distribution
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_m_tiles * num_k_tiles

    # Process tiles in a strided pattern
    for tile_idx in range(pid, total_tiles, NUM_SMS):
        # Convert linear index to tile coordinates
        tile_m = tile_idx % num_m_tiles
        tile_k = tile_idx // num_m_tiles

        # Calculate start indices for this tile
        m_start = tile_m * BLOCK_SIZE_M
        k_start = tile_k * BLOCK_SIZE_K

        # Create offsets for this tile
        offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        # Create masks for bounds checking
        mask_m = offs_m < M
        mask_k = offs_k < K

        # Get the expert index for this block
        block_start_idx = (m_start // GROUP_SIZE_M) * GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + block_start_idx)

        # Initialize accumulator for this tile
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=compute_dtype)

        # Process N dimension in tiles
        for n_start in range(0, N, BLOCK_SIZE_N):
            n_end = tl.minimum(n_start + BLOCK_SIZE_N, N)
            offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
            mask_n = offs_n < N

            # Load grad_output block with shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
            grad_output_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            mask_grad_output = mask_m[:, None] & mask_n[None, :]
            grad_output_block = tl.load(
                grad_output_ptrs, mask=mask_grad_output, other=0.0
            ).to(compute_dtype)

            # Load weights block with shape [BLOCK_SIZE_N, BLOCK_SIZE_K]
            # Note: Weights stored as [num_experts, N, K]
            weight_ptrs = (
                weight_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            )
            mask_weight = mask_n[:, None] & mask_k[None, :]
            weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0).to(
                compute_dtype
            )

            # Compute partial result: grad_output @ weights
            # regular matmul (not transposed)
            if USE_TENSOR_CORES:  #
                accumulator += tl.dot(grad_output_block, weight_block, allow_tf32=True)
            else:
                accumulator += tl.dot(grad_output_block, weight_block)

        # Store the accumulated gradients
        grad_input_ptrs = grad_input_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_grad_input = mask_m[:, None] & mask_k[None, :]
        tl.store(
            grad_input_ptrs, accumulator.to(grad_input_dtype), mask=mask_grad_input
        )


# ===== Gradient with respect to weights (dW) kernel =====


@triton.autotune(
    configs=BACKWARD_CONFIGS,
    key=["NUM_EXPERTS", "N", "K"],
    prune_configs_by={"early_config_prune": backward_config_prune},
)
@triton.jit
def contiguous_grouped_gemm_backward_dw_kernel(
    # Pointers to matrices
    grad_output_ptr,  # Gradient of output [M, N]
    input_ptr,  # Input tensor [M, K]
    grad_weight_ptr,  # Output: Gradient of weights [num_experts, N, K]
    indices_ptr,  # Expert indices [M]
    # Optional pointer to workspace for atomic operations
    workspace_ptr,  # Workspace for atomic operations
    # Matrix dimensions
    M: tl.constexpr,  # Number of tokens
    N: tl.constexpr,  # Hidden dimension of output
    K: tl.constexpr,  # Hidden dimension of input
    NUM_EXPERTS: tl.constexpr,  # Number of experts
    # Kernel configuration
    NUM_SMS: tl.constexpr,  # Number of SMs
    # Atomic params
    USE_ATOMICS: tl.constexpr,  # Whether to use atomic operations
    # Tiling parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    # Optional parameters
    USE_TENSOR_CORES: tl.constexpr = True,
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Backward pass kernel to compute gradients with respect to expert weights.

    For each expert e, accumulates:
    grad_weight[e] += sum(grad_output[i] @ input[i].T) for all i where indices[i] == e
    """
    pid = tl.program_id(0)

    # For weight gradients, we process tiles of the expert weights
    # Each thread block is responsible for computing gradient for a portion
    # of one expert's weights.

    # Compute which expert and which portion of its weights this thread block handles
    expert_idx = pid // (tl.cdiv(N, BLOCK_SIZE_N) * tl.cdiv(K, BLOCK_SIZE_K))
    remaining_idx = pid % (tl.cdiv(N, BLOCK_SIZE_N) * tl.cdiv(K, BLOCK_SIZE_K))

    # Break down remaining_idx to n_tile and k_tile
    n_tile = remaining_idx % tl.cdiv(N, BLOCK_SIZE_N)
    k_tile = remaining_idx // tl.cdiv(N, BLOCK_SIZE_N)

    # Ensure we're not exceeding number of experts
    if expert_idx >= NUM_EXPERTS:
        return

    # Calculate start indices for this expert's weight tile
    n_start = n_tile * BLOCK_SIZE_N
    k_start = k_tile * BLOCK_SIZE_K

    # Create offsets and masks for this tile
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

    mask_n = offs_n < N
    mask_k = offs_k < K

    # Initialize accumulator for this weight gradient tile
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # If using atomic accumulation, we need to load current values first
    if USE_ATOMICS:
        # Get pointer to this expert's weight gradient
        grad_weight_ptrs = (
            grad_weight_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
        )

        # Load current values if using atomics
        mask_weight = mask_n[:, None] & mask_k[None, :]
        current_values = tl.load(grad_weight_ptrs, mask=mask_weight, other=0.0)

        # Update accumulator with current values
        accumulator += current_values

    # Process input tokens that use this expert
    # We'll process in tiles of BLOCK_SIZE_M tokens
    for m_start in range(0, M, BLOCK_SIZE_M):
        m_end = tl.minimum(m_start + BLOCK_SIZE_M, M)

        # Load expert indices for this block
        block_start_idx = (m_start // GROUP_SIZE_M) * GROUP_SIZE_M
        block_expert_idx = tl.load(indices_ptr + block_start_idx)

        # Only process this block if it belongs to the current expert
        process_block = block_expert_idx == expert_idx

        if process_block:
            # Create offsets and masks for this token block
            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offs_m < M

            # Load grad_output block with shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
            grad_output_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            mask_grad_output = mask_m[:, None] & mask_n[None, :]
            grad_output_block = tl.load(
                grad_output_ptrs, mask=mask_grad_output, other=0.0
            )

            # Load input block with shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
            input_ptrs = input_ptr + offs_m[:, None] * K + offs_k[None, :]
            mask_input = mask_m[:, None] & mask_k[None, :]
            input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)

            # Compute partial gradient: grad_output.T @ input
            if USE_TENSOR_CORES:  #
                accumulator += tl.dot(
                    grad_output_block.to(tl.float32).T,
                    input_block.to(tl.float32),
                    allow_tf32=True,
                )
            else:
                accumulator += tl.dot(
                    grad_output_block.to(tl.float32).T, input_block.to(tl.float32)
                )

    # Store the accumulated gradients
    grad_weight_ptrs = (
        grad_weight_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
    )

    mask_weight = mask_n[:, None] & mask_k[None, :]

    # Use atomics if specified, otherwise just store
    if USE_ATOMICS:
        # Store with atomic add
        tl.atomic_add(
            grad_weight_ptrs,
            accumulator.to(grad_weight_ptr.dtype.element_ty),
            mask=mask_weight,
        )
    else:
        # Regular store
        tl.store(
            grad_weight_ptrs,
            accumulator.to(grad_weight_ptr.dtype.element_ty),
            mask=mask_weight,
        )


# ===== Backward pass implementation =====


def contiguous_grouped_gemm_backward(
    grad_output: torch.Tensor,  # [M, N]
    inputs: torch.Tensor,  # [M, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M]
    use_tensor_cores: bool = True,
    use_atomics: bool = True,
    group_size_m: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for contiguous grouped GEMM.

    Args:
        grad_output: Gradient of the output tensor [M, N]
        inputs: Input tensor [M, K]
        expert_weights: Expert weights tensor [num_experts, N, K]
        expert_indices: Expert indices tensor [M]
        use_tensor_cores: Whether to use tensor cores for better performance
        use_atomics: Whether to use atomic operations for weight gradients
        group_size_m: Size of contiguous token blocks for experts

    Returns:
        Tuple of (grad_inputs, grad_weights)
    """
    # Validate inputs
    if not grad_output.is_contiguous():  # "grad_output tensor must be contiguous"
        grad_output = grad_output.contiguous()
    assert inputs.is_contiguous(), "inputs tensor must be contiguous"
    assert expert_weights.is_contiguous(), "expert_weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "expert_indices tensor must be contiguous"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    M, N = grad_output.shape
    M_inputs, K = inputs.shape
    num_experts, N_weights, K_weights = expert_weights.shape

    # Validate dimensions
    assert M == M_inputs, f"grad_output M ({M}) must match inputs M ({M_inputs})"
    assert N == N_weights, f"grad_output N ({N}) must match weights N ({N_weights})"
    assert K == K_weights, f"inputs K ({K}) must match weights K ({K_weights})"
    assert expert_indices.shape[0] == M, "expert_indices length must match M"

    # Create output tensors
    grad_inputs = torch.zeros_like(inputs)
    grad_weights = torch.zeros_like(expert_weights)

    # Get number of SMs
    try:
        num_sms = torch.cuda.get_device_properties(inputs.device).multi_processor_count
    except:
        # Fallback to reasonable default
        num_sms = 108

    # Workspace for atomic operations if needed
    workspace = torch.zeros(1, device=grad_output.device, dtype=torch.int32)

    # 1. Compute gradients with respect to inputs (dx)
    def grid_dx(META):
        # Calculate grid size for input gradients
        block_size_m = META["BLOCK_SIZE_M"]
        block_size_k = META["BLOCK_SIZE_K"]

        # Calculate number of tiles
        num_m_tiles = triton.cdiv(M, block_size_m)
        num_k_tiles = triton.cdiv(K, block_size_k)
        total_tiles = num_m_tiles * num_k_tiles

        # Return grid size
        return (min(total_tiles, num_sms * 4),)

    contiguous_grouped_gemm_backward_dx_kernel[grid_dx](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        M=M,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        NUM_SMS=num_sms,
        USE_TENSOR_CORES=use_tensor_cores,
        GROUP_SIZE_M=group_size_m,
    )

    # 2. Compute gradients with respect to weights (dw)
    def grid_dw(META):
        # Calculate grid size for weight gradients
        block_size_n = META["BLOCK_SIZE_N"]
        block_size_k = META["BLOCK_SIZE_K"]

        # Calculate number of tiles per expert
        num_n_tiles = triton.cdiv(N, block_size_n)
        num_k_tiles = triton.cdiv(K, block_size_k)
        tiles_per_expert = num_n_tiles * num_k_tiles

        # Total grid size - one set of tiles for each expert
        total_tiles = num_experts * tiles_per_expert

        # Return grid size
        return (total_tiles,)

    contiguous_grouped_gemm_backward_dw_kernel[grid_dw](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        workspace,
        M=M,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        NUM_SMS=num_sms,
        USE_ATOMICS=use_atomics,
        USE_TENSOR_CORES=use_tensor_cores,
        GROUP_SIZE_M=group_size_m,
    )

    return grad_inputs, grad_weights


# ===== PyTorch-friendly Autograd Function =====


class ContiguousGroupedGEMMFunction(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM with backward pass.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        use_tma: bool = True,
        use_tensor_cores: bool = True,
    ):
        """Forward pass for contiguous grouped GEMM."""
        # Import forward function from the forward module
        from cg_forward import (
            cg_grouped_gemm as optimized_contiguous_grouped_gemm_forward,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.use_tensor_cores = use_tensor_cores

        # Call the optimized forward function
        output = optimized_contiguous_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            use_tma=use_tma,
            # use_tensor_cores=use_tensor_cores,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for contiguous grouped GEMM."""
        # Retrieve saved tensors
        inputs, expert_weights, expert_indices = ctx.saved_tensors
        use_tensor_cores = ctx.use_tensor_cores

        # Compute gradients
        grad_inputs, grad_weights = contiguous_grouped_gemm_backward(
            grad_output=grad_output,
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            # use_tensor_cores=use_tensor_cores,
        )

        # Return gradients for all inputs (None for indices and boolean args)
        return grad_inputs, grad_weights, None, None, None


# ===== User-friendly interface =====


def moe_contiguous_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    use_tma: bool = True,
    use_tensor_cores: bool = True,
) -> torch.Tensor:
    """
    User-friendly interface for contiguous grouped GEMM with backward support.

    Args:
        inputs: Input tensor of shape [M, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M] mapping each token to its expert
        use_tma: Whether to use TMA optimization (for Hopper GPUs)
        use_tensor_cores: Whether to use tensor cores

    Returns:
        Output tensor of shape [M, N]
    """
    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Call the autograd function
    return ContiguousGroupedGEMMFunction.apply(
        inputs, expert_weights, expert_indices, use_tma, use_tensor_cores
    )


# ===== Test backward pass =====


def test_backward_pass():
    """Test the backward pass implementation with PyTorch autograd."""
    import torch.nn as nn
    from cg_reference import reference_moe_forward

    # Define model parameters
    batch_size = 4
    seq_len = 8
    hidden_dim = 64  # K
    output_dim = 128  # N
    num_experts = 4

    # Create dummy inputs
    inputs = torch.randn(
        (batch_size * seq_len, hidden_dim),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    # Create dummy expert weights
    expert_weights = torch.randn(
        (num_experts, output_dim, hidden_dim),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )

    # Create expert assignment - ensure contiguous blocks
    expert_indices = torch.zeros(batch_size * seq_len, dtype=torch.int32, device="cuda")
    tokens_per_expert = (batch_size * seq_len) // num_experts
    for i in range(num_experts):
        expert_indices[i * tokens_per_expert : (i + 1) * tokens_per_expert] = i

    # Run forward and backward with custom implementation
    output_custom = moe_contiguous_grouped_gemm(
        inputs, expert_weights, expert_indices, use_tma=False
    )

    # Create a loss function and compute gradients
    loss_custom = output_custom.sum()
    loss_custom.backward()

    # Save gradients
    grad_inputs_custom = inputs.grad.clone()
    grad_weights_custom = expert_weights.grad.clone()

    # Reset gradients
    inputs.grad.zero_()
    expert_weights.grad.zero_()

    # Run with reference implementation
    output_ref = reference_moe_forward(inputs, expert_weights, expert_indices)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    # Compare results
    forward_match = torch.allclose(output_custom, output_ref, rtol=1e-2, atol=1e-2)
    print(f"Forward outputs match: {forward_match}")
    if not forward_match:
        print("Forward outputs don't match")
        print(f"Output custom: {output_custom}")
        print(f"Output ref: {output_ref}")
        return

    torch.testing.assert_close(
        grad_inputs_custom,
        inputs.grad,
        rtol=1e-2,
        atol=1e-2,
        msg="Input gradients don't match",
    )

    torch.testing.assert_close(
        grad_weights_custom,
        expert_weights.grad,
        rtol=1e-2,
        atol=1e-2,
        msg="Weight gradients don't match",
    )

    print("Backward pass test passed!")

    return grad_inputs_custom, inputs.grad, grad_weights_custom, expert_weights.grad


if __name__ == "__main__":
    test_backward_pass()
