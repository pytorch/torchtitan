from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Define standard configurations for various matrix sizes
STANDARD_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    # Medium sizes
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    # Larger sizes with more warps
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
]


def early_config_prune(configs, args, **kwargs):
    """Filter out configurations that would exceed shared memory capacity."""
    k = kwargs.get("K", 0)
    configs = [
        config for config in configs if config.kwargs.get("BLOCK_SIZE_K", 0) <= k
    ]
    return configs


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["NUM_EXPERTS", "M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_forward_sms(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    NUM_EXPERTS: tl.constexpr,  # Number of expert groups
    # Kernel configuration
    NUM_SMS: tl.constexpr,  # Number of SMs to use
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Contiguous grouped GEMM kernel for MoE forward pass with SMS-based work distribution.
    Follows a flat index approach similar to MG GEMM.

    Computes: C[i] = A[i] @ B[indices[i]].T for each token i.
    Assumes tokens are grouped in contiguous blocks of size GROUP_SIZE_M,
    with all tokens in a block assigned to the same expert.
    """
    # Get thread block index (0 to NUM_SMS-1)
    pid = tl.program_id(0)

    # Output data type for conversion
    c_dtype = c_ptr.dtype.element_ty

    # Calculate the total number of tiles needed
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles

    # Process multiple tiles per thread block in a grid-strided loop
    # Each SM processes tiles spaced NUM_SMS apart
    for tile_idx in range(pid, total_tiles, NUM_SMS):
        # Convert linear index to 2D tile coordinates (row-major ordering)
        tile_m = tile_idx // num_n_tiles
        tile_n = tile_idx % num_n_tiles

        # Starting indices for this tile
        m_start = tile_m * BLOCK_SIZE_M
        n_start = tile_n * BLOCK_SIZE_N

        # Process if this tile is within bounds

        if m_start < M_TOTAL:

            # Calculate which expert group this tile belongs to
            # Each group of tokens of size GROUP_SIZE_M uses the same expert
            group_idx = m_start // GROUP_SIZE_M

            # Load the expert index for this group
            expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

            # Create offset arrays for input, output coordinates
            offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start

            # Create masks for bounds checking
            mask_m = offs_m < M_TOTAL
            mask_n = offs_n < N

            # Initialize accumulator for output
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # Process the matrix multiplication in tiles along K dimension
            for k_offset in range(0, K, BLOCK_SIZE_K):
                # Create offsets and mask for K dimension
                offs_k = tl.arange(0, BLOCK_SIZE_K) + k_offset
                mask_k = offs_k < K

                # Combined masks for inputs
                mask_a = mask_m[:, None] & mask_k[None, :]
                mask_b = mask_n[:, None] & mask_k[None, :]

                # Load inputs with bounds checking
                # a[BLOCK_SIZE_M, BLOCK_SIZE_K]
                a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)

                # Load weights for the assigned expert
                # b[BLOCK_SIZE_N, BLOCK_SIZE_K]
                b_ptrs = (
                    b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)

                # Compute matrix multiplication for this K tile
                # A[M,K] @ B[N,K].T -> C[M,N]
                accumulator += tl.dot(a, b.T)

            # Store results with bounds checking
            # c[BLOCK_SIZE_M, BLOCK_SIZE_N]
            c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            mask_c = mask_m[:, None] & mask_n[None, :]
            tl.store(c_ptrs, accumulator.to(c_dtype), mask=mask_c)


def cg_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
    num_sms: Optional[int] = 132,  # None,
) -> torch.Tensor:
    """
    SMS-based contiguous grouped GEMM forward pass.

    For each token group i, computes out[i] = inputs[i] @ expert_weights[indices[i]].T
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)
        num_sms: Number of SMs to use for computation (default: auto-detect)

    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, K = inputs.shape
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Auto-detect number of SMs if not specified
    if num_sms is None:
        if torch.cuda.is_available():
            num_sms = torch.cuda.get_device_properties(
                inputs.device
            ).multi_processor_count
        else:
            # Default to a reasonable number for CPU or other devices
            num_sms = 108

    # Grid function for kernel launch
    def grid(META):
        return (num_sms,)

    # Launch kernel
    _kernel_cg_forward_sms[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        NUM_SMS=num_sms,
        GROUP_SIZE_M=group_size_m,
    )

    return output


class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM.
    This allows for integration with PyTorch's autograd system.
    Note: This implementation only provides the forward pass.
    """

    @staticmethod
    def forward(
        ctx, inputs, expert_weights, expert_indices, group_size_m=128, num_sms=None
    ):
        """Forward pass for contiguous grouped GEMM."""
        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
            num_sms=num_sms,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass
        """
        raise NotImplementedError("Backward pass not implemented")


def cg_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """
    Interface for contiguous grouped GEMM.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert
        num_sms: Number of SMs to use for computation (default: auto-detect)

    Returns:
        Output tensor, [M_total, N]
    """
    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(
        inputs, expert_weights, expert_indices, group_size_m, num_sms
    )


def test_contiguous_grouped_gemm(
    batch_size=4, seq_len=128, hidden_dim=128, output_dim=256, num_experts=4
):
    """
    Test function for contiguous grouped GEMM implementation.
    Creates synthetic data and verifies correctness against a PyTorch reference.
    """
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return None

    # Create dummy input
    m_total = batch_size * seq_len
    inputs = torch.randn((m_total, hidden_dim), dtype=torch.bfloat16, device="cuda")

    # Create dummy expert weights
    expert_weights = torch.randn(
        (num_experts, output_dim, hidden_dim), dtype=torch.bfloat16, device="cuda"
    )

    # Create expert indices with contiguous groups
    expert_indices = torch.zeros(m_total, dtype=torch.int32, device="cuda")
    group_size_m = 128

    # Ensure m_total is a multiple of group_size_m
    if m_total % group_size_m != 0:
        pad_size = group_size_m - (m_total % group_size_m)
        inputs = torch.cat(
            [
                inputs,
                torch.zeros(pad_size, hidden_dim, dtype=inputs.dtype, device="cuda"),
            ]
        )
        expert_indices = torch.cat(
            [expert_indices, torch.zeros(pad_size, dtype=torch.int32, device="cuda")]
        )
        m_total += pad_size

    # Assign experts in contiguous blocks
    num_groups = m_total // group_size_m
    for g in range(num_groups):
        start_idx = g * group_size_m
        end_idx = start_idx + group_size_m
        expert_idx = g % num_experts
        expert_indices[start_idx:end_idx] = expert_idx

    # Run our implementation
    output_custom = cg_grouped_gemm(
        inputs=inputs,
        expert_weights=expert_weights,
        expert_indices=expert_indices,
        group_size_m=group_size_m,
    )

    # Run reference implementation
    output_ref = torch.zeros_like(output_custom)
    for g in range(num_groups):
        start_idx = g * group_size_m
        end_idx = start_idx + group_size_m
        expert_idx = expert_indices[start_idx].item()
        output_ref[start_idx:end_idx] = (
            inputs[start_idx:end_idx] @ expert_weights[expert_idx].T
        )

    # Compare results
    max_diff = torch.max(torch.abs(output_custom - output_ref)).item()
    is_match = torch.allclose(output_custom, output_ref, rtol=1e-2, atol=1e-2)

    print(f"Max difference: {max_diff}")
    print(f"Outputs match: {is_match}")

    return output_custom


if __name__ == "__main__":
    test_contiguous_grouped_gemm()
