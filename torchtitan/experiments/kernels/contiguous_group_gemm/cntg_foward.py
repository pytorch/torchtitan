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


def early_config_prune(configs, args):
    """Filter out configurations that would exceed shared memory capacity."""
    sms = args["NUM_SMS"]
    k = args["K"]
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


# Standard non-TMA kernel version
@triton.autotune(
    configs=HOPPER_CONFIGS,
    key=["NUM_GROUPS", "M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def contiguous_grouped_gemm_kernel(
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
):
    """
    Standard non-TMA contiguous grouped GEMM kernel for MoE forward pass.

    Computes: C[i] = A[i] @ B[indices[i]].T for each token i.
    """
    # Get thread block index
    pid = tl.program_id(0)

    # Get data type of output
    c_dtype = c_ptr.dtype.element_ty

    # Number of tiles in output
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles

    # Process tiles in a strided pattern (each SM handles multiple tiles)
    for tile_idx in range(pid, total_tiles, NUM_SMS):
        # Convert linear index to 2D tile coordinates
        tile_m = tile_idx % num_m_tiles
        tile_n = tile_idx // num_m_tiles

        # Calculate starting indices for this tile
        m_start = tile_m * BLOCK_SIZE_M
        n_start = tile_n * BLOCK_SIZE_N

        # Initialize accumulator for this tile
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Get the expert index for this block
        # Note: All rows in a block must use the same expert (128-row alignment)
        block_start_idx = m_start
        expert_idx = tl.load(indices_ptr + block_start_idx)

        # Create offsets for this tile
        offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_n = offs_n < N

        # Process the matmul in tiles along K dimension
        for k_start in range(0, K, BLOCK_SIZE_K):
            # Create offsets and mask for K dimension
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < K

            # Create combined masks
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_n[:, None] & mask_k[None, :]

            # Compute pointers
            a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
            b_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]

            # Load the tiles with bounds checking
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            # Perform matrix multiplication for this K tile
            accumulator += tl.dot(a, b.T)

        # Store results
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, accumulator.to(c_dtype), mask=mask_c)


# TMA-optimized kernel version for Hopper
@triton.autotune(
    configs=HOPPER_CONFIGS,
    key=["NUM_GROUPS", "M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def contiguous_grouped_gemm_tma_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Workspace for TMA descriptors
    workspace,
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    NUM_GROUPS: tl.constexpr,  # Number of expert groups
    # Kernel configuration
    NUM_SMS: tl.constexpr,  # Number of SMs to use
    TMA_SIZE: tl.constexpr,  # Size of TMA descriptor
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    TMA-optimized contiguous grouped GEMM kernel for MoE forward pass.

    Computes: C[i] = A[i] @ B[indices[i]].T for each token i.
    Uses TMA for efficient memory access on Hopper GPUs.
    """
    # Get thread block index
    pid = tl.program_id(0)

    # Get data type of output
    c_dtype = c_ptr.dtype.element_ty

    # Create local ptr for TMA descriptor (we'll reuse it for all operations)
    tma_desc_ptr = workspace + (pid * TMA_SIZE)

    # Number of tiles in output
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_m_tiles * num_n_tiles

    # Process tiles in a strided pattern (each SM handles multiple tiles)
    for tile_idx in range(pid, total_tiles, NUM_SMS):
        # Convert linear index to 2D tile coordinates
        tile_m = tile_idx % num_m_tiles
        tile_n = tile_idx // num_m_tiles

        # Calculate starting indices for this tile
        m_start = tile_m * BLOCK_SIZE_M
        n_start = tile_n * BLOCK_SIZE_N

        # Calculate actual sizes accounting for boundaries
        m_size = tl.minimum(BLOCK_SIZE_M, M_TOTAL - m_start)
        n_size = tl.minimum(BLOCK_SIZE_N, N - n_start)

        # Only process if we have actual work
        if m_size <= 0 or n_size <= 0:
            continue

        # Initialize accumulator for this tile
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Get the expert index for this block
        block_start_idx = m_start
        expert_idx = tl.load(indices_ptr + block_start_idx)

        # Create TMA descriptor for output matrix
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=tma_desc_ptr,
            global_address=c_ptr + m_start * N + n_start,
            load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            global_size=[m_size, n_size],
            element_ty=c_dtype,
        )

        # Acquire exclusive access to TMA descriptor
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(tma_desc_ptr)

        # Process the matmul in tiles along K dimension
        for k_start in range(0, K, BLOCK_SIZE_K):
            k_size = tl.minimum(BLOCK_SIZE_K, K - k_start)

            # Skip empty tiles
            if k_size <= 0:
                continue

            # Load A tile (inputs) - use manual loading
            offs_m = tl.arange(0, BLOCK_SIZE_M)
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_m = offs_m < m_size
            mask_k = offs_k < K
            mask_a = mask_m[:, None] & mask_k[None, :]
            a_ptrs = a_ptr + (m_start + offs_m)[:, None] * K + offs_k[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            # Load B tile (expert weights) - use manual loading
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            mask_n = offs_n < n_size
            mask_b = mask_n[:, None] & mask_k[None, :]
            b_ptrs = (
                b_ptr
                + expert_idx * N * K
                + (n_start + offs_n)[:, None] * K
                + offs_k[None, :]
            )
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            # Perform matrix multiplication for this K tile
            # Transpose b for the matmul: b.T
            accumulator += tl.dot(a, b.T)

        # Store results using TMA
        tl._experimental_descriptor_store(tma_desc_ptr, accumulator.to(c_dtype), [0, 0])


def get_num_sms():
    """Get the number of streaming multiprocessors on the device."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count


def is_hopper_gpu():
    """Check if the current GPU is a Hopper architecture (SM90 or newer)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    compute_capability = props.major * 10 + props.minor
    return compute_capability >= 90


def contiguous_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    use_tma: bool = True,
) -> torch.Tensor:
    """
    Contiguous grouped GEMM forward pass for Mixture of Experts.

    For each token i, computes out[i] = inputs[i] @ expert_weights[indices[i]].T
    All tokens mapped to the same expert must be in contiguous blocks of size 128.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        use_tma: Whether to use TMA optimization (if on Hopper)

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

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Get number of SMs
    num_sms = get_num_sms()

    # Check if we're on a Hopper GPU
    has_hopper = is_hopper_gpu()

    # Use TMA only if on Hopper and user wants it
    can_use_tma = use_tma and has_hopper

    # reference:
'''
def grouped_gemm_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    tma_size: int = 128,
    using_fp8: bool = False,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    M*G style grouped GEMM with TMA and Float8 support.
    # Removed for now - FP8 support is triggered by passing x_scale and w_scale tensors.

    """
    if not CudaUtils.verify_tma():
        raise NotImplementedError("Grouped GEMM without TMA is not supported yet")

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    # Total input size is now [M_total, K] where M_total is the sum of all group sizes
    M_total, K = x.shape
    N = w.shape[0]  # N is now the same for all groups

    assert K == w.shape[1], f"Input K ({K}) must match weight K ({w.shape[1]})"

    # Verify that the sum of m_sizes matches M_total
    sum_m_sizes = m_sizes.sum().item()
    assert (
        M_total == sum_m_sizes
    ), f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"

    # Create output tensor with correct shape [M_total, N]
    y = torch.empty((M_total, N), device=x.device, dtype=x.dtype)

    NUM_SMS = CudaUtils.get_num_sms()
    USE_TMA_LOAD = True
    USE_TMA_STORE = True
    USE_EPILOGUE_SUBTILING = False



    """
    # print(f"{x_scale=}")
    desc_helper = None
    desc_x = x
    desc_w = w
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = TmaDescriptorHelper(tma_size=tma_size)
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if USE_TMA_STORE:
        workspace = torch.empty(
            NUM_SMS * desc_helper.tma_size,
            device=x.device,
            dtype=torch.uint8,
        )

    def grid(META):
        if USE_TMA_LOAD:
            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M_total,
                K,
                META["BLOCK_SIZE_M"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N,
                K,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w.element_size(),
            )
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)
    # print(f"{M_BUCKET=}")
    _kernel_mg_forward_hopper_bf16[grid](  #
        # _kernel_grouped_gemm_flat_indexing[grid](  # _kernel_grouped_gemm[grid](
        desc_x,
        desc_w,
        y,
        workspace,
        m_sizes,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        TMA_SIZE=tma_size,
        USE_EPILOGUE_SUBTILING=USE_EPILOGUE_SUBTILING,
    )

    return y
    '''
    # Choose kernel based on hardware capabilities
    if can_use_tma:
        # Grid function for the TMA kernel
        # TODO
        def grid(META):
            return (num_sms,)

        # Allocate workspace for TMA descriptors (one per SM)
        tma_desc_size = 128
        workspace = torch.empty(
            num_sms * tma_desc_size, device=inputs.device, dtype=torch.uint8
        )

        # Launch TMA kernel
        contiguous_grouped_gemm_tma_kernel[grid](
            inputs,
            expert_weights,
            output,
            workspace,
            expert_indices,
            M_TOTAL=M_total,
            N=N,
            K=K,
            NUM_GROUPS=num_experts,
            NUM_SMS=num_sms,
            TMA_SIZE=tma_desc_size,
        )
    else:
        # Grid function for the standard kernel
        # TODO
        def grid(META):
            return (num_sms,)

        # Launch standard kernel
        contiguous_grouped_gemm_kernel[grid](
            inputs,
            expert_weights,
            output,
            expert_indices,
            M_TOTAL=M_total,
            N=N,
            K=K,
            NUM_GROUPS=num_experts,
            NUM_SMS=num_sms,
        )

    return output


class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM.
    This allows for integration with PyTorch's autograd system.
    Note: This implementation only provides the forward pass.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, use_tma=True):
        """Forward pass for contiguous grouped GEMM."""
        return contiguous_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            use_tma=use_tma,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass would go here for a complete implementation.
        Not implemented for this example.
        """
        raise NotImplementedError("Backward pass not implemented")


def moe_contiguous_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    use_tma: bool = True,
) -> torch.Tensor:
    """
    User-friendly interface for contiguous grouped GEMM.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        use_tma: Whether to use TMA optimization (if on Hopper)

    Returns:
        Output tensor of shape [M_total, N]
    """
    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(inputs, expert_weights, expert_indices, use_tma)


# Example usage:
def test_contiguous_grouped_gemm():
    # Create test data
    batch_size = 4
    seq_len = 10
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

    has_hopper = is_hopper_gpu()
    print(f"Running on Hopper GPU: {has_hopper}")

    # Run with TMA if on Hopper
    output = moe_contiguous_grouped_gemm(
        inputs=inputs,
        expert_weights=expert_weights,
        expert_indices=expert_indices,
        use_tma=False,  # Will automatically fall back if not on Hopper
    )

    # Verify output shape
    assert output.shape == (batch_size * seq_len, output_dim)
    print(f"Output shape: {output.shape}")

    return output


if __name__ == "__main__":
    test_contiguous_grouped_gemm()
