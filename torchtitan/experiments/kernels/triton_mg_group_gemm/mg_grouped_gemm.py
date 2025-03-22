# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import functools
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

import triton
import triton.language as tl

from tma_cuda_wrapper import CudaUtils, TmaDescriptorHelper
from triton.runtime import driver  # @manual

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


# this class derived from FBGemm -https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm/utils.py
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


# ======  Autotuning utilities ======
# Autotuning from FBGemm - https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm/grouped_gemm.py

_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
]


def early_config_prune(configs, named_args, dtsize=None, dtype=None, **kwargs):
    device = torch.cuda.current_device()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    if dtsize is None:
        dtsize = named_args["c_ptr"].element_size()
    if dtype is None:
        dtype = named_args["c_ptr"].dtype

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
        )
        G, M, N, K = (
            named_args["G"],
            named_args["M_BUCKET"],
            named_args["N"],
            named_args["K"],
        )

        # 1. make sure we have enough smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]

        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        M_PER_GROUP = M // G
        MIN_M_TILES = 64
        # 2. make sure we don't load M tiles that are too big
        if BLOCK_M > MIN_M_TILES and BLOCK_M > (M_PER_GROUP * 2):
            continue
        # 3. make sure we don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = N // BLOCK_N
        MIN_N_TILES = 64
        # 4. make sure we don't load N tiles that are too big
        if BLOCK_N > MIN_N_TILES and M * N_TILES < num_sm:
            continue
        # 5. make sure we don't load N tiles that are too small
        if BLOCK_N < 128 and M * N_TILES > 2 * num_sm:
            continue
        # 6. make sure K can be evenly divided
        if K % BLOCK_K != 0:
            continue

        pruned_configs.append(config)

    return pruned_configs


# ======== End Autotuning utilities ========


# =============== Start Triton Kernels ===============
"""
Forward pass for grouped GEMM with Triton, where grouping is M*G
"""


# Flat Global Indexing Kernel (previously gride stride loop)
@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_flat_indexing(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    # Use flat global indexing
    tidx = tl.program_id(0)

    dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0  # Track total tiles processed so far

    for g in range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            # Compute for this group
            n_size = N

            # Calculate the number of tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # Set up TMA descriptor for output
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Process tiles using the flat indexing approach
            # Only threads with IDs in the range [iterated_tiles, iterated_tiles + num_tiles) work on this group
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = (
                    tidx - iterated_tiles
                )  # Convert to local tile index within this group

                # Split M first and N second
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)

                if USE_TMA_LOAD:
                    # Use TMA to load input and weight blocks
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        # Load input block [M, K]
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )

                        # Load weight block [N, K]
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )

                        # Compute matrix multiplication
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    # Manual load without TMA
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)

                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )

                    b_ptrs = b_desc_ptr + (offs_bn[:, None]) * K + offs_k[None, :]

                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        # Load with bounds checking
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)

                        # Compute matrix multiplication
                        accumulator += tl.dot(a, b.T)

                        # Update pointers for next block
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                # Store result
                if USE_TMA_STORE:
                    # Store using TMA
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                else:
                    # Manual store
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                    c = accumulator.to(c_ptr.dtype.element_ty)

                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N  # Row stride is N
                        + offs_bn[None, :],  # Column offset
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )

                # Move to the next tile using the stride equal to NUM_SMS
                tidx += NUM_SMS

            # Update the total tiles count for the next group
            iterated_tiles += num_tiles


"""
Backward pass for grouped GEMM with Triton, where grouping is M*G
We compute gradients with respect to both input (`grad_x`) and weights (`grad_w`).
"""


@triton.jit
def _kernel_grouped_gemm_backward_dx_scheduled(
    grad_output_ptr,  # grad of dl/dY [M_total, N]
    w_ptr,  # weights [N, K]
    grad_input_ptr,  # output of kernel [M_total, K]
    group_offsets_ptr,  # Pre-computed group offsets [G+1]
    workspace_ptr,  # Workspace for TMA descriptors (if needed)
    G,  # Number of groups
    M_TOTAL,  # Total M dimension size
    N,  # N dimension size (same for all groups)
    K,  # K dimension size
    stride_go_m,  # Stride of grad_output in M dimension
    stride_go_n,  # Stride of grad_output in N dimension
    stride_w_n,  # Stride of w in N dimension
    stride_w_k,  # Stride of w in K dimension
    stride_gi_m,  # Stride of grad_input in M dimension
    stride_gi_k,  # Stride of grad_input in K dimension
    NUM_SMS,  # Number of SMs on the GPU
    USE_TMA: tl.constexpr = False,  # Whether to use TMA (compile-time constant)
    BLOCK_SIZE_M: tl.constexpr = 64,  # Block size in M dimension
    BLOCK_SIZE_N: tl.constexpr = 64,  # Block size in N dimension
    BLOCK_SIZE_K: tl.constexpr = 64,  # Block size in K dimension
    EVEN_K: tl.constexpr = True,  # Whether K is even (for TF32 optimization)
) -> None:
    """
    Scheduled grouped GEMM backward pass for input gradients with optional TMA support.

    For the forward pass Y = X @ W.T, the backward for input is:
    grad_X = grad_Y @ W

    Where:
    - grad_Y is [M_total, N]
    - W is [N, K]
    - grad_X is [M_total, K]
    """
    # Get coordinates for the current program
    pid = tl.program_id(axis=0)

    # Calculate work distribution parameters
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_m * num_pid_k

    # Process work items
    while pid < G * num_pid_in_group:
        # work distribution for this pid
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        pid_m = pid_in_group % num_pid_m
        pid_k = pid_in_group // num_pid_m

        # group boundaries
        valid_group = group_id < G
        group_start = tl.where(valid_group, tl.load(group_offsets_ptr + group_id), 0)
        group_end = tl.where(valid_group, tl.load(group_offsets_ptr + group_id + 1), 0)
        group_size = group_end - group_start

        # mask for valid processing (valid group and non-empty)
        valid_work = valid_group & (group_size > 0)

        if valid_work:
            # Block dimensions
            m_block_offset = pid_m * BLOCK_SIZE_M
            k_block_offset = pid_k * BLOCK_SIZE_K

            # offsets for this block
            offs_m = group_start + m_block_offset + tl.arange(0, BLOCK_SIZE_M)
            offs_k = k_block_offset + tl.arange(0, BLOCK_SIZE_K)

            # bounds checking
            m_mask = offs_m < group_end
            k_mask = offs_k < K

            # output mask
            output_mask = m_mask[:, None] & k_mask[None, :]

            # init accumulator
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

            # Loop over the reduction dimension (N)
            for n_offset in range(0, N, BLOCK_SIZE_N):
                # boundary conditions for the reduction dimension
                n_size = tl.minimum(BLOCK_SIZE_N, N - n_offset)
                offs_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
                n_mask = offs_n < N

                # combined masks
                m_n_mask = m_mask[:, None] & n_mask[None, :]
                n_k_mask = n_mask[:, None] & k_mask[None, :]

                # Load grad_output block
                grad_output_block = tl.load(
                    grad_output_ptr
                    + offs_m[:, None] * stride_go_m
                    + offs_n[None, :] * stride_go_n,
                    mask=m_n_mask,
                    other=0.0,
                )

                # Load weights block
                w_block = tl.load(
                    w_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k,
                    mask=n_k_mask,
                    other=0.0,
                )

                # matrix multiplication: grad_input = grad_output @ w
                # Allow TF32 if K is even and divisible by 8
                if EVEN_K:
                    accumulator += tl.dot(
                        grad_output_block.to(tl.float32),
                        w_block.to(tl.float32),
                        allow_tf32=True,
                    )
                else:
                    accumulator += tl.dot(
                        grad_output_block.to(tl.float32),
                        w_block.to(tl.float32),
                        allow_tf32=False,
                    )

            # Store result to grad_input with explicit strides
            # TODO: We don't use TMA for now regardless of the flag - can be conditionally added later
            tl.store(
                grad_input_ptr
                + offs_m[:, None] * stride_gi_m
                + offs_k[None, :] * stride_gi_k,
                accumulator.to(grad_input_ptr.dtype.element_ty),
                mask=output_mask,
            )

        # Move to next work item
        pid = pid + NUM_SMS


@triton.jit
def _kernel_grouped_gemm_backward_dw_scheduled(
    x_ptr,  # input tensor [M_total, K]
    grad_output_ptr,  # grad of dl/dY [M_total, N]
    grad_weight_ptr,  # output of kernel (grad_w) [N, K]
    group_offsets_ptr,  # Pre-computed group offsets [G+1]
    G,  # Number of groups
    M_TOTAL,  # Total M dimension size
    N,  # N dimension size (same for all groups)
    K,  # K dimension size
    stride_x_m,  # Stride of x in M dimension
    stride_x_k,  # Stride of x in K dimension
    stride_go_m,  # Stride of grad_output in M dimension
    stride_go_n,  # Stride of grad_output in N dimension
    stride_gw_n,  # Stride of grad_weight in N dimension
    stride_gw_k,  # Stride of grad_weight in K dimension
    BLOCK_SIZE_N: tl.constexpr = 64,  # Block size in N dimension
    BLOCK_SIZE_K: tl.constexpr = 64,  # Block size in K dimension
    BLOCK_SIZE_M: tl.constexpr = 64,  # Block size in M dimension for reduction
    EVEN_K: tl.constexpr = True,  # Whether K is even (for TF32 optimization)
) -> None:
    """
    Scheduled grouped GEMM backward for weights

    For the forward pass Y = X @ W.T, the backward for weights is:
    grad_W = grad_Y.T @ X

    Where:
    - X is [M_total, K]
    - grad_Y is [M_total, N]
    - grad_W is [N, K]

    This kernel has one thread block per output tile and accumulates
    contributions from all groups.
    """
    # Get coordinates for the current program - each thread computes one tile of grad_w
    n_idx = tl.program_id(0)
    k_idx = tl.program_id(1)

    # offsets for this block
    n_offset = n_idx * BLOCK_SIZE_N
    k_offset = k_idx * BLOCK_SIZE_K

    # block indices
    offs_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

    # bounds checking
    n_mask = offs_n < N
    k_mask = offs_k < K

    # Combined mask for output
    output_mask = n_mask[:, None] & k_mask[None, :]

    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Process each group and accumulate contributions
    for g in tl.range(G):
        # Get group boundaries
        group_start = tl.load(group_offsets_ptr + g)
        group_end = tl.load(group_offsets_ptr + g + 1)
        group_size = group_end - group_start

        process_group = group_size > 0

        # Process the current group in chunks (to avoid large memory usage)
        for m_offset in range(0, group_size, BLOCK_SIZE_M):
            # Only process this chunk if we should process this group
            if process_group:
                # Calculate actual block size (handling boundary)
                m_size = tl.minimum(BLOCK_SIZE_M, group_size - m_offset)

                # Create offsets and mask for this chunk
                offs_m = group_start + m_offset + tl.arange(0, BLOCK_SIZE_M)
                m_mask = offs_m < group_end

                # Combined masks for loading
                m_n_mask = m_mask[:, None] & n_mask[None, :]
                m_k_mask = m_mask[:, None] & k_mask[None, :]

                # Load input chunk [M_chunk, K]
                x_block = tl.load(
                    x_ptr + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k,
                    mask=m_k_mask,
                    other=0.0,
                )

                # Load grad_output chunk [M_chunk, N]
                grad_output_block = tl.load(
                    grad_output_ptr
                    + offs_m[:, None] * stride_go_m
                    + offs_n[None, :] * stride_go_n,
                    mask=m_n_mask,
                    other=0.0,
                )

                # Compute partial contribution: grad_W += grad_Y.T @ X
                # Need to transpose grad_output for the matmul
                if EVEN_K:
                    accumulator += tl.dot(
                        grad_output_block.to(tl.float32).T,  # [N, M_chunk]
                        x_block.to(tl.float32),  # [M_chunk, K]
                        allow_tf32=True,
                    )
                else:
                    accumulator += tl.dot(
                        grad_output_block.to(tl.float32).T,  # [N, M_chunk]
                        x_block.to(tl.float32),  # [M_chunk, K]
                        allow_tf32=False,
                    )

    # Convert to output dtype
    grad_weight = accumulator.to(grad_weight_ptr.dtype.element_ty)

    # Store computed gradient block
    tl.store(
        grad_weight_ptr + offs_n[:, None] * stride_gw_n + offs_k[None, :] * stride_gw_k,
        grad_weight,
        mask=output_mask,
    )


# ======== End Triton kernels ========

# ======== PyTorch wrapper functions ========


def _grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """
    M*G style grouped GEMM with TMA support.
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
    USE_TMA_STORE = False  # TODO: Sometimes results in compile error and not seeing perf win with it yet...

    desc_helper = None
    desc_x = x
    desc_w = w
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = TmaDescriptorHelper()
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

    assert x_scale is None
    assert w_scale is None

    _kernel_grouped_gemm_flat_indexing[grid](  # _kernel_grouped_gemm[grid](
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
        USE_TMA_LOAD,
        USE_TMA_STORE,
        USE_FAST_ACCUM=use_fast_accum,
    )

    return y


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor
) -> torch.Tensor:
    return _grouped_gemm(x, w, m_sizes)


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_tma: bool = False,  # Optional flag to enable/disable TMA
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for grouped matrix multiplication with M*G distribution.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        x: Input tensor from forward pass, shape [M_total, K]
        w: Weight tensor from forward pass, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        use_tma: Whether to try using TMA acceleration (if available)

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    logging.info("Starting grouped_gemm_backward with optimized scheduling")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.error("CUDA not available for backward pass")
        raise RuntimeError("CUDA not available for backward pass")

    # Get GPU parameters
    device_props = torch.cuda.get_device_properties("cuda")
    NUM_SMS = device_props.multi_processor_count

    # Determine TMA support
    can_use_tma = False
    if use_tma:
        try:
            # Check if we're on SM90+ (Hopper) and have TMA support in Triton
            if device_props.major >= 9:
                # Check for available TMA functions in Triton - we'd add actual checks here
                # but for now just assume they don't exist to avoid compile errors
                logging.info(
                    "Hardware supports TMA (SM90+), but disabling for compatibility"
                )
                # can_use_tma = True
            else:
                logging.info(
                    f"TMA not supported on SM{device_props.major}{device_props.minor}"
                )
        except Exception as e:
            logging.warning(f"Error checking TMA support: {e}. TMA will be disabled.")

    # Validate input dimensions
    G = m_sizes.shape[0]
    M_total, K_x = x.shape
    N, K_w = w.shape

    # Check that K dimensions match
    if K_x != K_w:
        logging.warning(f"K dimension mismatch: x has K={K_x}, w has K={K_w}")
        raise ValueError("K dimensions must match for grouped GEMM backward")

    K = K_x  # Use common K dimension
    logging.info(f"K dimension: {K}")
    try:
        # Ensure contiguous tensors
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        w = w.contiguous()
        m_sizes = m_sizes.contiguous()

        # Allocate output tensors
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(w)

        # Set stride values
        stride_go_m = grad_output.stride(0)  # Stride of grad_output in M dimension
        stride_go_n = grad_output.stride(1)  # Stride of grad_output in N dimension

        stride_x_m = x.stride(0)  # Stride of x in M dimension
        stride_x_k = x.stride(1)  # Stride of x in K dimension

        stride_w_n = w.stride(0)  # Stride of w in N dimension
        stride_w_k = w.stride(1)  # Stride of w in K dimension

        stride_gx_m = grad_x.stride(0)  # Stride of grad_x in M dimension
        stride_gx_k = grad_x.stride(1)  # Stride of grad_x in K dimension

        stride_gw_n = grad_w.stride(0)  # Stride of grad_w in N dimension
        stride_gw_k = grad_w.stride(1)  # Stride of grad_w in K dimension

        # Pre-compute group offsets for indexing
        group_offsets = torch.zeros(G + 1, device=m_sizes.device, dtype=torch.int32)
        m_offset = 0
        for g in range(G):
            group_offsets[g] = m_offset
            m_offset += m_sizes[g].item()
        group_offsets[G] = m_offset  # Total M

        # Check if K dimension is even for TF32 optimization
        EVEN_K = (K % 8) == 0
        logging.info(f"EVEN_K optimization enabled: {EVEN_K} (K={K})")

        # Allocate workspace if needed for TMA
        if can_use_tma:
            # TMA needs workspace, allocate it - size would depend on implementation
            workspace = torch.empty((NUM_SMS * 128), device=x.device, dtype=torch.uint8)
        else:
            # No workspace needed for non-TMA version
            workspace = torch.empty(0, device=x.device, dtype=torch.uint8)

        # Set block sizes based on K dimension
        if K <= 64:
            BLOCK_SIZE_K = 64
            BLOCK_SIZE_M = 64
            BLOCK_SIZE_N = 64
        else:
            # For larger K, use smaller blocks to avoid register pressure
            BLOCK_SIZE_K = 32
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 32

        try:
            logging.info(f"Computing grad_x with optimized kernel (TMA={can_use_tma})")

            # Fixed grid size based on SM count
            grid = (NUM_SMS,)

            _kernel_grouped_gemm_backward_dx_scheduled[grid](
                grad_output,
                w,
                grad_x,
                group_offsets,
                workspace,
                G,
                M_total,
                N,
                K,
                stride_go_m,
                stride_go_n,
                stride_w_n,
                stride_w_k,
                stride_gx_m,
                stride_gx_k,
                NUM_SMS,
                USE_TMA=can_use_tma,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                EVEN_K=EVEN_K,
            )

            logging.info("Kernel run success: grad_x computation successful")
        except Exception as e:
            logging.error(f"Error in backward_dx kernel: {e}")
            raise RuntimeError(f"Error in backward_dx kernel: {e}")

        try:
            logging.info("Computing grad_w with optimized kernel")

            # For grad_w, use a grid with one thread block per output tile
            grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(K, BLOCK_SIZE_K))

            _kernel_grouped_gemm_backward_dw_scheduled[grid](
                x,
                grad_output,
                grad_w,
                group_offsets,
                G,
                M_total,
                N,
                K,
                stride_x_m,
                stride_x_k,
                stride_go_m,
                stride_go_n,
                stride_gw_n,
                stride_gw_k,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                EVEN_K=EVEN_K,
            )
            logging.info("Kernel run success: grad_w computation successful")
        except Exception as e:
            logging.error(f"Error in backward_dw kernel: {e}")
            raise RuntimeError(f"Error in backward_dw kernel: {e}")

        return grad_x, grad_w
    except Exception as e:
        logging.error(f"Error in grouped_gemm_backward: {e}")
        raise RuntimeError(f"Error in grouped_gemm_backward: {e}")


class GroupedGEMM_mg(torch.autograd.Function):
    """
    Autograd function for GroupedGEMM operation with M*G distribution.

    This enables automatic differentiation of the GroupedGEMM operation.
    """

    @staticmethod
    def forward(ctx, x, w, m_sizes, use_tma=False):
        """
        Forward pass of GroupedGEMM.

        Args:
            x: Input tensor, shape [M_total, K]
            w: Weight tensor, shape [N, K]
            m_sizes: Tensor of shape [G] containing the size of each group
            use_tma: Whether to try using TMA acceleration (if available)

        Returns:
            Output tensor, shape [M_total, N]
        """
        # Import here to avoid circular import
        from mg_forward import group_gemm_forward

        output = group_gemm_forward(x, w, m_sizes)

        # Save inputs for backward pass
        ctx.save_for_backward(x, w, m_sizes)
        ctx.use_tma = use_tma

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of M*G GroupedGEMM.

        Args:
            grad_output: Gradient of output, shape [M_total, N]

        Returns:
            Tuple of gradients:
                - grad_x: Gradient with respect to x, shape [M_total, K]
                - grad_w: Gradient with respect to w, shape [N, K]
                - None: Gradient with respect to m_sizes (not differentiable)
                - None: Gradient with respect to use_tma (not differentiable)
        """
        x, w, m_sizes = ctx.saved_tensors
        use_tma = ctx.use_tma

        # Compute gradients using the optimized implementation
        grad_x, grad_w = grouped_gemm_backward(
            grad_output, x, w, m_sizes, use_tma=use_tma
        )

        return grad_x, grad_w, None, None


def mg_grouped_gemm_full(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, use_tma: bool = True
) -> torch.Tensor:
    """
    Differentiable grouped GEMM operation with M*G distribution.

    Args:
        x: Input tensor, shape [M_total, K]
        w: Weight tensor, shape [N, K]
        m_sizes: Tensor of shape [G] containing the size of each group
        use_tma: Whether to try using TMA acceleration (if available)

    Returns:
        Output tensor, shape [M_total, N]
    """
    # logging.info("Starting grouped_gemm_full")
    return GroupedGEMM_mg.apply(x, w, m_sizes, use_tma)
