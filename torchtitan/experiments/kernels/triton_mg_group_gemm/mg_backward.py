# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.runtime import driver

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
Backward pass for grouped GEMM with Triton, where grouping is M*G
We compute gradients with respect to both input (`grad_x`) and weights (`grad_w`).
"""


# =============== Start Triton Kernels ===============
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
    tidx = tl.program_id(axis=0)

    # Calculate work distribution parameters
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
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
            # Block dimensions
            m_block_offset = pid_m * BLOCK_SIZE_M
            k_block_offset = pid_k * BLOCK_SIZE_K

            # Initialize offsets for this block
            offs_m = group_start + m_block_offset + tl.arange(0, BLOCK_SIZE_M)
            offs_k = k_block_offset + tl.arange(0, BLOCK_SIZE_K)

            # Create masks for bounds checking
            m_mask = offs_m < group_end
            k_mask = offs_k < K

            # Create output mask
            output_mask = m_mask[:, None] & k_mask[None, :]

            # Initialize accumulator
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

            # Loop over the reduction dimension (N)
            for n_offset in range(0, N, BLOCK_SIZE_N):
                # Handle boundary conditions for the reduction dimension
                n_size = tl.minimum(BLOCK_SIZE_N, N - n_offset)
                offs_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
                n_mask = offs_n < N

                # Create combined masks
                m_n_mask = m_mask[:, None] & n_mask[None, :]
                n_k_mask = n_mask[:, None] & k_mask[None, :]

                # Load grad_output block with correct strides
                grad_output_block = tl.load(
                    grad_output_ptr
                    + offs_m[:, None] * stride_go_m
                    + offs_n[None, :] * stride_go_n,
                    mask=m_n_mask,
                    other=0.0,
                )

                # Load weights block with correct strides
                w_block = tl.load(
                    w_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k,
                    mask=n_k_mask,
                    other=0.0,
                )

                # Compute matrix multiplication: grad_input = grad_output @ w
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
            # We don't use TMA for now regardless of the flag - can be conditionally added later
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
    Scheduled grouped GEMM backward for weights with independent processing.

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
    n_idx = tl.program_id(0)  # N dimension
    k_idx = tl.program_id(1)  # K dimension

    # Calculate offsets for this block
    n_offset = n_idx * BLOCK_SIZE_N
    k_offset = k_idx * BLOCK_SIZE_K

    # Create block indices
    offs_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

    # Create masks for bounds checking
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

        # Process the current group in chunks to avoid large memory usage
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

    # Store the computed gradient block to global memory
    tl.store(
        grad_weight_ptr + offs_n[:, None] * stride_gw_n + offs_k[None, :] * stride_gw_k,
        grad_weight,
        mask=output_mask,
    )


# ======== End Triton kernels ========


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


def group_gemm_full(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor, use_tma: bool = False
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
    logging.info("Starting grouped_gemm_full with optimized scheduling")
    return GroupedGEMM_mg.apply(x, w, m_sizes, use_tma)
