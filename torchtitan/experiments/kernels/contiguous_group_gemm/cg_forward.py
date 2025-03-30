# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from triton import Config
from triton.runtime import driver  # @manual

# Configuration for autotuning

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===== Supporting utils, CUDA and TMA =====


class CudaUtils:
    @staticmethod
    def is_cuda() -> bool:
        """Check if Triton is running on CUDA backend."""
        return driver.active.get_current_target().backend == "cuda"

    @staticmethod
    def verify_tma() -> bool:
        """Check if TMA is supported on the current device."""
        return (
            CudaUtils.is_cuda()
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
        )

    @staticmethod
    def get_num_sms() -> int:
        """Get the number of streaming multiprocessors on the current device."""
        if not CudaUtils.is_cuda():
            raise RuntimeError("Triton is not running on CUDA backend")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.cuda.get_device_properties("cuda").multi_processor_count


class TmaDescriptorHelper:
    """Helper class for managing TMA descriptors in Triton kernels."""

    class KernelParamWrapper:
        """Wrapper to implement the TmaDescKernelParam interface."""

        def __init__(self, desc: torch.Tensor):
            self.desc = desc

        def tma_desc_cpu_ptr(self) -> int:
            """Return the CPU pointer to the TMA descriptor."""
            return self.desc.data_ptr()

    def __init__(self, tma_size: int = 128):
        """Initialize the TMA descriptor helper.

        Args:
            tma_size: Size of the TMA descriptor in bytes
        """
        if not CudaUtils.verify_tma():
            raise RuntimeError(
                "TMA not supported on this device (requires Hopper or newer)"
            )
        if "nv_tma_desc_type" not in dir(tl):
            raise RuntimeError(
                "TMA grid constant descriptors not supported in your Triton version"
            )

        self.tma_size = tma_size
        self.fill_1d_tma_descriptor_inner = driver.active.utils.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor_inner = driver.active.utils.fill_2d_tma_descriptor
        self.descriptors: Dict[str, torch.Tensor] = {}

    def init_tma_descriptor(self, name: str) -> None:
        """Initialize a TMA descriptor with the given name.

        Call this method outside of the lambda function for grid size.
        """
        self.descriptors[name] = torch.empty(
            self.tma_size, device="cpu", dtype=torch.int8
        )

    def fill_1d_tma_descriptor(
        self, name: str, ptr: int, dim: int, block_dim: int, element_size: int
    ) -> None:
        """Fill a 1D TMA descriptor.

        Call this method inside the lambda function for grid size.
        """
        if name not in self.descriptors:
            raise ValueError(f"TMA descriptor '{name}' not initialized")

        desc_x = self.descriptors[name]
        if desc_x.data_ptr() % 64 != 0:
            raise ValueError("TMA descriptor must be 64-byte aligned")
        self.fill_1d_tma_descriptor_inner(
            ptr, dim, block_dim, element_size, desc_x.data_ptr()
        )

    def fill_2d_tma_descriptor(
        self,
        name: str,
        ptr: int,
        dim1: int,
        dim0: int,
        block_dim1: int,
        block_dim0: int,
        element_size: int,
    ) -> None:
        """Fill a 2D TMA descriptor.

        Call this method inside the lambda function for grid size.
        """
        if name not in self.descriptors:
            raise ValueError(f"TMA descriptor '{name}' not initialized")

        desc_x = self.descriptors[name]
        if desc_x.data_ptr() % 64 != 0:
            raise ValueError("TMA descriptor must be 64-byte aligned")
        self.fill_2d_tma_descriptor_inner(
            ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
        )

    def get_tma_descriptor_kernel_param(self, name: str) -> KernelParamWrapper:
        """Get the TMA descriptor kernel parameter for the given name."""
        if name not in self.descriptors or self.descriptors[name] is None:
            raise ValueError(f"TMA descriptor '{name}' not initialized")
        return self.KernelParamWrapper(self.descriptors[name])


# ================== End of supporting functions ==================


def early_config_prune(configs, args, **kwargs):
    """Filter out configurations that would exceed shared memory capacity."""
    sms = kwargs.get("NUM_SMS", 108)  # Default value if not provided
    k = kwargs.get("K", 0)
    configs = [
        config for config in configs if config.kwargs.get("BLOCK_SIZE_K", 0) <= k
    ]
    return configs


# Define standard configurations for Hopper GPUs
HOPPER_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    # Configurations for medium to large matrices
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
]


# =====
@triton.autotune(
    configs=HOPPER_CONFIGS,
    key=["NUM_GROUPS", "M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_forward(
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
    NUM_GROUPS: tl.constexpr,  # Number of expert groups
    # Kernel configuration
    NUM_SMS: tl.constexpr,  # Number of SMs to use
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    # Use prefetching
    USE_PREFETCHING: tl.constexpr = False,
):
    """
    non-TMA contiguous grouped GEMM kernel for MoE forward pass.
    Computes: C[i] = A[i] @ B[indices[i]].T for each token i.
    Assumes tokens are grouped in contiguous blocks of size GROUP_SIZE_M,
    with all tokens in a block assigned to the same expert.
    """
    # Get thread block index
    pid = tl.program_id(0)

    # data type of output
    c_dtype = c_ptr.dtype.element_ty

    # Number of tiles in each dimension
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    total_tiles = num_m_tiles * num_n_tiles

    # Process tiles in a grid strided pattern (each SM handles multiple tiles)
    for tile_idx in range(pid, total_tiles, NUM_SMS):
        # Convert linear index to 2D tile coordinates
        tile_n = tile_idx % num_n_tiles
        tile_m = tile_idx // num_n_tiles

        # Starting indices for this tile
        m_start = tile_m * BLOCK_SIZE_M
        n_start = tile_n * BLOCK_SIZE_N

        # Offsets for this tile
        offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

        # Masks for out-of-bounds checking
        mask_m = offs_m < M_TOTAL
        mask_n = offs_n < N

        # Calculate the group index based on the start of the M block
        # This is determined by the GROUP_SIZE_M alignment constraint
        group_idx = m_start // GROUP_SIZE_M

        # Get the expert index for this group's expert
        # We load a single expert index for the entire group since all tokens
        # in the same GROUP_SIZE_M block use the same expert
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        # Accumulator for this tile
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Process the matmul in tiles along K dimension
        for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_start = k_idx * BLOCK_SIZE_K
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < K

            # Combined masks for bounds checking
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_n[:, None] & mask_k[None, :]

            # Load input matrix A - shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
            a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            # Load expert weight matrix B - shape [BLOCK_SIZE_N, BLOCK_SIZE_K]
            # Use the correct expert index for the entire block
            b_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            # Perform matrix multiplication for this K tile
            # A[M,K] @ B[N,K].T -> C[M,N]
            accumulator += tl.dot(a, b.T)

        # Store results back to C
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, accumulator.to(c_dtype), mask=mask_c)


# Improved contiguous grouped GEMM forward function
def cg_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    use_tma: bool = True,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Optimized contiguous grouped GEMM forward pass for Mixture of Experts.

    For each token group i, computes out[i] = inputs[i] @ expert_weights[indices[i]].T
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        use_tma: Whether to use TMA optimization (if on Hopper)
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    M_total, K = inputs.shape
    num_experts, N, K_weights = expert_weights.shape
    print(f"Input shape: {inputs.shape}")
    print(f"Expert weights shape: {expert_weights.shape}")
    print(f"Expert indices shape: {expert_indices.shape}")
    print(f"M_total: {M_total}")
    # Validate group size
    assert (
        M_total % group_size_m == 0
    ), f"M_total must be a multiple of group_size_m ({group_size_m})"

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Get number of SMs
    num_sms = CudaUtils.get_num_sms() if torch.cuda.is_available() else 108

    # Grid function for the standard kernel with proper autotuning
    def grid(META):
        # Calculate optimal grid size based on matrix dimensions and block sizes
        block_size_m = META["BLOCK_SIZE_M"]
        block_size_n = META["BLOCK_SIZE_N"]

        # Calculate tiles in each dimension - NOTE: We've flipped the tile ordering
        # to ensure better memory locality
        num_n_tiles = triton.cdiv(N, block_size_n)
        num_m_tiles = triton.cdiv(M_total, block_size_m)

        # For autotuning, make sure we have enough work for each SM
        total_tiles = num_m_tiles * num_n_tiles
        grid_size = min(total_tiles, num_sms)

        return (grid_size,)

    # Launch standard kernel
    _kernel_cg_forward[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_GROUPS=num_experts,
        NUM_SMS=num_sms,
        GROUP_SIZE_M=group_size_m,
    )

    return output


# ======


class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM.
    This allows for integration with PyTorch's autograd system.
    Note: This implementation only provides the forward pass.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, use_tma=True):
        """Forward pass for contiguous grouped GEMM."""
        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            use_tma=use_tma,
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
    use_tma: bool = True,
) -> torch.Tensor:
    """
    interface for contiguous grouped GEMM.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        use_tma: Whether to use TMA optimization (if on Hopper)

    Returns:
        Output tensor, [M_total, N]
    """
    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(inputs, expert_weights, expert_indices, use_tma)


# Example usage and verify correctness:


def test_contiguous_grouped_gemm():
    # Import reference implementation
    from cg_reference import reference_moe_forward

    # Create test data
    batch_size = 4
    seq_len = 32
    hidden_dim = 128  # K dimension
    output_dim = 256  # N dimension
    num_experts = 4

    # Create dummy input
    inputs = torch.randn(
        (batch_size * seq_len, hidden_dim), dtype=torch.bfloat16, device="cuda"
    )

    # Create dummy expert weights
    expert_weights = torch.randn(
        (num_experts, output_dim, hidden_dim), dtype=torch.bfloat16, device="cuda"
    )

    # Create dummy expert assignment (one expert per token)
    expert_indices = torch.randint(
        0, num_experts, (batch_size * seq_len,), dtype=torch.int32, device="cuda"
    )

    has_hopper = True  # is_hopper_gpu()
    print(f"Running on Hopper GPU: {has_hopper}")

    # Run with TMA if on Hopper
    output_custom = cg_grouped_gemm(
        inputs=inputs,
        expert_weights=expert_weights,
        expert_indices=expert_indices,
        use_tma=False,  # Will automatically fall back if not on Hopper
    )

    # Run reference implementation
    output_ref = reference_moe_forward(inputs, expert_weights, expert_indices)

    # Compare results
    forward_match = torch.allclose(output_custom, output_ref, rtol=1e-2, atol=1e-2)
    print(f"Forward outputs match: {forward_match}")

    # Verify output shape
    assert output_custom.shape == (batch_size * seq_len, output_dim)
    print(f"Output shape: {output_custom.shape}")

    return output_custom


if __name__ == "__main__":
    test_contiguous_grouped_gemm()
