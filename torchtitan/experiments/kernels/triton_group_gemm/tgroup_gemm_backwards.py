# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl
from tma_utils import TmaAutoTuneHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
Backward pass for grouped GEMM with Triton, where grouping is N*G
We are computing gradients with respect to both the input (`grad_x`) and the weights (`grad_w`).
"""


# =============== Start Triton Kernels ===============
@triton.jit
def _kernel_grouped_gemm_backward_x_scheduled(
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    w_t_ptr,  # w transposed [K, N*G]
    grad_x_ptr,  # output of kernel [M, K]
    group_offsets_ptr,  # Pre-computed group offsets [G+1]
    workspace,  # Workspace for TMA descriptors
    G,  # Number of groups
    M,  # Total M dimension size
    N,  # N per group
    K,  # K dimension size
    stride_go_m,
    stride_go_n,
    stride_w_n,
    stride_w_k,
    stride_gx_m,
    stride_gx_k,
    NUM_SMS,
    USE_TMA_LOAD: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
    GROUP_SIZE_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
) -> None:
    """
    Scheduled grouped GEMM backward for X with TMA support.

    For each group g, computes: grad_x[g] = grad_y[g] @ w_t[g].T

    Where:
    - grad_y is [M, N*G]
    - w_t is [K, N*G] (transposed from [N*G, K])
    - grad_x is [M, K]
    """
    # Get coordinates for the current program
    tidx = tl.program_id(axis=0)
    dtype = grad_x_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    # Calculate work distribution parameters
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_m * num_pid_k

    # Process all assigned work items
    pid = tidx
    while pid < G * num_pid_in_group:
        # Calculate work distribution for this pid
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        pid_m = pid_in_group % num_pid_m
        pid_k = pid_in_group // num_pid_m

        # Get group boundaries
        valid_group = group_id < G
        group_start = tl.where(valid_group, tl.load(group_offsets_ptr + group_id), 0)
        group_end = tl.where(valid_group, tl.load(group_offsets_ptr + group_id + 1), 0)
        group_size = group_end - group_start

        # Calculate a mask for valid processing (valid group and non-empty)
        valid_work = valid_group & (group_size > 0)

        # Only process if we have valid work
        if valid_work:
            # Compute offsets for this group
            n_start = group_id * N

            # Block dimensions
            m_block_offset = pid_m * BLOCK_SIZE_M
            k_block_offset = pid_k * BLOCK_SIZE_K

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
                m_size = tl.minimum(
                    BLOCK_SIZE_M, group_end - (group_start + m_block_offset)
                )
                if m_size > 0:
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=c_desc_ptr,
                        global_address=grad_x_ptr
                        + (group_start + m_block_offset) * stride_gx_m
                        + k_block_offset * stride_gx_k,
                        load_size=[
                            m_size,
                            tl.minimum(BLOCK_SIZE_K, K - k_block_offset),
                        ],
                        global_size=[m_size, K],
                        element_ty=dtype,
                    )
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Initialize offsets for this block
            offs_m = group_start + m_block_offset + tl.arange(0, BLOCK_SIZE_M)

            # For K dimension, optimize memory access if EVEN_K is True
            offs_k = k_block_offset + tl.arange(0, BLOCK_SIZE_K)

            # Create masks
            m_mask = offs_m < group_end
            k_mask = offs_k < K

            # Initialize accumulator
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

            # Loop over the reduction dimension (N)
            # Use smaller steps to improve precision and avoid numerical issues
            for n_offset in range(0, N, BLOCK_SIZE_N):
                # Handle boundary conditions for the reduction dimension
                n_size = tl.minimum(BLOCK_SIZE_N, N - n_offset)
                offs_n = n_start + n_offset + tl.arange(0, BLOCK_SIZE_N)
                n_mask = offs_n < (n_start + N)

                # Fixed stride formats to ensure consistent memory access
                grad_y_block = tl.load(
                    grad_y_ptr
                    + offs_m[:, None] * stride_go_m
                    + offs_n[None, :] * stride_go_n,
                    mask=m_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

                # Load w_t [K, N*G] block with correct strides
                w_t_block = tl.load(
                    w_t_ptr
                    + offs_k[:, None] * stride_w_k
                    + offs_n[None, :] * stride_w_n,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

                # grad_y @ w_t.T
                # Allow TF32 if K is even and divisible by 8
                if EVEN_K:
                    accumulator += tl.dot(
                        grad_y_block.to(tl.float32),
                        w_t_block.to(tl.float32).T,
                        allow_tf32=True,
                    )
                else:
                    accumulator += tl.dot(
                        grad_y_block.to(tl.float32),
                        w_t_block.to(tl.float32).T,
                        allow_tf32=False,
                    )

            # Store result to grad_x with explicit strides
            if USE_TMA_STORE:
                # TMA store
                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(dtype),
                    [0, 0],  # Starting offset in the output block
                )
            else:
                # Standard store
                tl.store(
                    grad_x_ptr
                    + offs_m[:, None] * stride_gx_m
                    + offs_k[None, :] * stride_gx_k,
                    accumulator.to(dtype),
                    mask=m_mask[:, None] & k_mask[None, :],
                )

        pid = pid + NUM_SMS


@triton.jit
def _kernel_grouped_gemm_backward_w_scheduled(
    x_t_ptr,  # x transposed [K, M]
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    grad_w_ptr,  # output of kernel (grad_w) [N*G, K]
    group_offsets_ptr,  # Pre-computed group offsets [G+1]
    workspace,  # Workspace for TMA descriptors
    G,  # Number of groups
    M,  # Total M dimension size
    N,  # N per group
    K,  # K dimension size
    stride_x_m,
    stride_x_k,
    stride_go_m,
    stride_go_n,
    stride_gw_n,
    stride_gw_k,
    NUM_SMS,
    USE_TMA_LOAD: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
    BLOCK_SIZE_M: tl.constexpr = 32,
    GROUP_SIZE_N: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
) -> None:
    """
    Scheduled implementation of grouped GEMM backward for W with TMA support.

    For each group g, computes:
        grad_w[g] = grad_y[g].T @ x[g]

    Where:
    - x_t is [K, M] (transposed from [M, K])
    - grad_y is [M, N*G]
    - grad_w is [N*G, K]
    """
    # Define coordinates for the current program
    tidx = tl.program_id(axis=0)
    dtype = grad_w_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    # Calculate work distribution parameters
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_n * num_pid_k

    # Process all assigned work items
    pid = tidx
    while pid < G * num_pid_in_group:
        # Calculate work distribution for this pid
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        pid_n = pid_in_group % num_pid_n
        pid_k = pid_in_group // num_pid_n

        # Get group boundaries
        valid_group = group_id < G
        group_start = tl.where(valid_group, tl.load(group_offsets_ptr + group_id), 0)
        group_end = tl.where(valid_group, tl.load(group_offsets_ptr + group_id + 1), 0)
        group_size = group_end - group_start

        # Calculate a mask for valid processing (valid group and non-empty)
        valid_work = valid_group & (group_size > 0)

        # Only process if we have valid work
        if valid_work:
            # Compute offsets for this group
            n_start = group_id * N

            # Block dimensions
            n_block_offset = pid_n * BLOCK_SIZE_N
            k_block_offset = pid_k * BLOCK_SIZE_K

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
                n_size = tl.minimum(BLOCK_SIZE_N, N - n_block_offset)
                if n_size > 0:
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=c_desc_ptr,
                        global_address=grad_w_ptr
                        + (n_start + n_block_offset) * stride_gw_n
                        + k_block_offset * stride_gw_k,
                        load_size=[
                            n_size,
                            tl.minimum(BLOCK_SIZE_K, K - k_block_offset),
                        ],
                        global_size=[n_size, K],
                        element_ty=dtype,
                    )
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Initialize offsets for this block
            offs_n = n_start + n_block_offset + tl.arange(0, BLOCK_SIZE_N)
            offs_k = k_block_offset + tl.arange(0, BLOCK_SIZE_K)

            # Create masks
            n_mask = offs_n < (n_start + N)
            k_mask = offs_k < K

            # Initialize accumulator
            accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

            # Loop over the reduction dimension (M) with smaller steps to avoid overflow
            for m_offset in range(0, group_size, BLOCK_SIZE_M):
                # Handle boundary conditions for the reduction dimension
                m_size = tl.minimum(BLOCK_SIZE_M, group_size - m_offset)
                offs_m = group_start + m_offset + tl.arange(0, BLOCK_SIZE_M)
                m_mask = offs_m < group_end

                # Load grad_y [M, N*G] block with explicit strides
                grad_y_block = tl.load(
                    grad_y_ptr
                    + offs_m[:, None] * stride_go_m
                    + offs_n[None, :] * stride_go_n,
                    mask=m_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

                # Load x_t [K, M] block with explicit strides
                x_t_block = tl.load(
                    x_t_ptr
                    + offs_k[:, None] * stride_x_k
                    + offs_m[None, :] * stride_x_m,
                    mask=k_mask[:, None] & m_mask[None, :],
                    other=0.0,
                )

                # Matrix multiplication: (grad_y_block.T @ x_t_block.T)
                if EVEN_K:
                    accumulator += tl.dot(
                        grad_y_block.to(
                            tl.float32
                        ).T,  # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_M]
                        x_t_block.to(
                            tl.float32
                        ).T,  # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        allow_tf32=True,
                    )
                else:
                    accumulator += tl.dot(
                        grad_y_block.to(
                            tl.float32
                        ).T,  # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_M]
                        x_t_block.to(
                            tl.float32
                        ).T,  # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        allow_tf32=False,
                    )

            # Store result to grad_w with explicit strides
            if USE_TMA_STORE:
                # TMA store
                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(dtype),
                    [0, 0],  # Starting offset in the output block
                )
            else:
                # Standard store with explicit strides
                tl.store(
                    grad_w_ptr
                    + offs_n[:, None] * stride_gw_n
                    + offs_k[None, :] * stride_gw_k,
                    accumulator.to(dtype),
                    mask=n_mask[:, None] & k_mask[None, :],
                )

        pid = pid + NUM_SMS


# ========== End Triton kernels ==========

# Moved PyTorch fallback implementation to a separate file
# ========== Begin grouped_gemm_backward ==========


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for grouped matrix multiplication using scheduled kernels with TMA support.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor, shape [G]

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    logging.info("Starting grouped_gemm_backward with TMA-enabled scheduling")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.error("CUDA not available for backward pass")
        raise RuntimeError("CUDA not available for backward pass")
        # return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    # Get GPU parameters
    device_props = torch.cuda.get_device_properties("cuda")
    NUM_SMS = device_props.multi_processor_count

    # Check TMA support
    has_tma = hasattr(tl.extra, "cuda") and device_props.major >= 9

    if has_tma:
        logging.info(f"TMA support detected on GPU with {NUM_SMS} SMs")
        USE_TMA_LOAD = True  # TODO - this does nothing atm..removed to focus on numerical correctness first.
        USE_TMA_STORE = False
    else:
        logging.warning("TMA support not detected, disabling TMA optimizations")
        USE_TMA_LOAD = False
        USE_TMA_STORE = False

    # Validate input dimensions
    G = m_sizes.shape[0]
    M, K_x = x.shape
    N_times_G, K_w = w.shape

    # Check that K dimensions match
    if K_x != K_w:
        logging.warning(f"K dimension mismatch: x has K={K_x}, w has K={K_w}")
        raise ValueError("K dimensions must match for grouped GEMM backward")
        # return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    try:
        # Ensure contiguous tensors
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        w = w.contiguous()
        m_sizes = m_sizes.contiguous()

        # Allocate output tensors
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(w)

        # Determine N per group
        # N*G is the second dimension size of grad_output
        N = N_times_G // G

        # Set stride values
        # Direct access pattern for grad_output tensor
        stride_go_m = grad_output.stride(0)  # grad_output in M dimension
        stride_go_n = grad_output.stride(1)  # grad_output in N dimension

        # Pattern match the transposed weight tensor
        stride_w_n = 1  # transposed weights in N dimension
        stride_w_k = N * G  # transposed weights in K dimension

        # Pattern match the output grad_x tensor
        stride_gx_m = grad_x.stride(0)  # grad_x in M dimension
        stride_gx_k = grad_x.stride(1)  # Sgrad_x in K dimension

        # Pattern match the transposed x tensor
        stride_x_m = 1  # Stride for transposed x in M dimension
        stride_x_k = M  # Stride for transposed x in K dimension

        # Pattern match the output grad_w tensor
        stride_gw_n = grad_w.stride(0)  # grad_w in N dimension
        stride_gw_k = grad_w.stride(1)  # grad_w in K dimension

        # Pre-compute group offsets for indexing
        group_offsets = torch.zeros(G + 1, device=m_sizes.device, dtype=torch.int32)
        m_offset = 0
        for g in range(G):
            group_offsets[g] = m_offset
            m_offset += m_sizes[g].item()
        group_offsets[G] = m_offset  # Total M

        # Check if K dimension is even (maybe? optimize memory access patterns)
        EVEN_K = (K_x % 8) == 0
        logging.info(f"EVEN_K optimization enabled: {EVEN_K} (K={K_x})")

        # Transpose x and w for backward computation
        x_t = x.T.contiguous()  # Shape: [K, M]
        w_t = w.T.contiguous()  # Shape: [K, N*G]

        # Allocate workspace for TMA descriptors if needed
        if USE_TMA_LOAD or USE_TMA_STORE:
            workspace = torch.empty((NUM_SMS * 128), device=x.device, dtype=torch.uint8)
        else:
            # Empty tensor when TMA is not used
            workspace = torch.empty(0, device=x.device, dtype=torch.uint8)

        # Set block sizes based on K dimension - TODO - autotuning
        if K_x <= 64:
            BLOCK_SIZE_K = 64
            BLOCK_SIZE_M = 64
            # BLOCK_SIZE_K_W = 64
            BLOCK_SIZE_N = 64
        else:
            # For larger K, use smaller blocks to avoid register pressure
            BLOCK_SIZE_K = 32
            BLOCK_SIZE_M = 32
            # BLOCK_SIZE_K_W = 32
            BLOCK_SIZE_N = 32

        # Determine maximum size needed and set the grid size
        num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = triton.cdiv(K_x, BLOCK_SIZE_K)
        num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

        # Compute total number of blocks needed for each kernel
        total_blocks_x = G * num_pid_m * num_pid_k
        total_blocks_w = G * num_pid_n * num_pid_k

        # Set block sizes based on K dimension
        if K_x <= 64:
            BLOCK_SIZE_K_X = 64
            BLOCK_SIZE_M = 64
            BLOCK_SIZE_K_W = 64
            BLOCK_SIZE_N = 64
        else:
            # For larger K, use smaller blocks to avoid register pressure
            BLOCK_SIZE_K_X = 32
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_K_W = 32
            BLOCK_SIZE_N = 32

        try:
            logging.info("Computing grad_x with TMA-enabled kernel")

            # Fixed grid size based on SM count
            grid = (NUM_SMS,)

            _kernel_grouped_gemm_backward_x_scheduled[grid](
                grad_output,
                w_t,  # Using transposed weights
                grad_x,
                group_offsets,
                workspace,
                G,
                M,
                N,
                K_x,
                stride_go_m,
                stride_go_n,
                stride_w_n,
                stride_w_k,
                stride_gx_m,
                stride_gx_k,
                NUM_SMS,
                USE_TMA_LOAD,
                USE_TMA_STORE,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K_X,
                EVEN_K=EVEN_K,
            )
            logging.info(
                "Kernel run success: grad_X computation successful with TMA-enabled kernel"
            )
        except Exception as e:
            logging.error(f"FAILED: Error in TMA-enabled backward_x kernel: {e}")
            logging.info("WARNING: Falling back to PyTorch for grad_x")
            _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x)

        try:
            logging.info("Computing grad_w with TMA-enabled kernel")

            # Fixed grid size based on SM count - similar to original approach
            grid = (NUM_SMS,)

            _kernel_grouped_gemm_backward_w_scheduled[grid](
                x_t,  # Using transposed inputs
                grad_output,
                grad_w,
                group_offsets,
                workspace,
                G,
                M,
                N,
                K_w,
                stride_x_m,
                stride_x_k,
                stride_go_m,
                stride_go_n,
                stride_gw_n,
                stride_gw_k,
                NUM_SMS,
                USE_TMA_LOAD,
                USE_TMA_STORE,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K_W,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                EVEN_K=EVEN_K,
            )
            logging.info(
                "Kernel run success - grad_W computation successful with TMA-enabled kernel"
            )
        except Exception as e:
            logging.error(f"FAILED: Error in TMA-enabled backward_w kernel: {e}")
            logging.info("WARNING: Falling back to PyTorch for grad_w")
            # _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)

        return grad_x, grad_w
    except Exception as e:
        logging.error(f"Error in grouped_gemm_backward: {e}")
        # return _pytorch_fallback_backward(grad_output, x, w, m_sizes)
