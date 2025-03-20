# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

# This is a series of helper functions for grouped GEMM backward that compute the gradients
# using eager PyTorch operations. They are used as a verification reference for the Triton kernels.
# They can also used as a fallback when the Triton kernels cannot be used, though lets hope that is not needed.


def _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x):
    """
    Compute grad_x using pure PyTorch operations with FP32 precision
    """
    G = m_sizes.shape[0]
    M, K = grad_x.shape
    N = w.shape[0] // G

    # Zero out the output tensor first
    grad_x.zero_()

    # Store original dtype and convert to float32 for computation
    orig_dtype = grad_x.dtype
    grad_output_fp32 = grad_output.float()
    w_fp32 = w.float()
    grad_x_fp32 = torch.zeros_like(grad_x, dtype=torch.float32)

    # Process each group separately
    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Get slices for this group
            grad_output_slice = grad_output_fp32[m_start:m_end, n_start:n_end]
            w_slice = w_fp32[n_start:n_end]

            # Process in chunks for better precision on large matrices
            CHUNK_SIZE = 256
            for chunk_start in range(0, m_size, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, m_size)
                chunk_size = chunk_end - chunk_start

                # Compute matrix multiplication with higher precision
                grad_output_chunk = grad_output_slice[chunk_start:chunk_end]
                result_chunk = torch.matmul(
                    grad_output_chunk.double(), w_slice.double()
                )

                # Store the result
                grad_x_fp32[m_start + chunk_start : m_start + chunk_end].copy_(
                    result_chunk.float()
                )

        m_start = m_end

    # Convert back to original dtype
    grad_x.copy_(grad_x_fp32.to(orig_dtype))


def _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w):
    """
    Compute grad_w using pure PyTorch operations with FP64 precision for better accuracy.
    """
    G = m_sizes.shape[0]
    N_times_G, K = grad_w.shape
    N = N_times_G // G

    # Zero out the output tensor first
    grad_w.zero_()

    # Store original dtype and convert to float32 for computation
    orig_dtype = grad_w.dtype
    grad_output_fp32 = grad_output.float()
    x_fp32 = x.float()
    grad_w_fp32 = torch.zeros_like(grad_w, dtype=torch.float32)

    # Handle potential K dimension mismatches
    K_x = x.shape[1]
    min_K = min(K, K_x)

    # Process each group separately
    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Get slices for this group
            grad_output_slice = grad_output_fp32[m_start:m_end, n_start:n_end]
            x_slice = x_fp32[m_start:m_end, :min_K]

            # Process in chunks for better precision
            CHUNK_SIZE = 32
            result = torch.zeros(
                (grad_output_slice.shape[1], min_K),
                dtype=torch.float64,
                device=grad_output_slice.device,
            )

            for chunk_start in range(0, m_size, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, m_size)

                # Get chunks
                grad_output_chunk = grad_output_slice[chunk_start:chunk_end].double()
                x_chunk = x_slice[chunk_start:chunk_end].double()

                # Matrix multiplication in FP64
                chunk_result = torch.matmul(grad_output_chunk.t(), x_chunk)
                result += chunk_result

            # Handle K dimension padding if needed
            if K > min_K:
                temp_result = torch.zeros(
                    (grad_output_slice.shape[1], K),
                    dtype=torch.float32,
                    device=grad_output_slice.device,
                )
                temp_result[:, :min_K] = result.float()
                grad_w_fp32[n_start:n_end].copy_(temp_result)
            else:
                grad_w_fp32[n_start:n_end].copy_(result.float())

        m_start = m_end

    # Convert back to original dtype
    grad_w.copy_(grad_w_fp32.to(orig_dtype))


def _pytorch_fallback_backward(grad_output, x, w, m_sizes):
    """
    Pure PyTorch implementation of grouped GEMM backward with high precision.
    Used as a fallback when the Triton kernels cannot be used.
    """
    logging.info(
        "WARNING:  Using PyTorch fallback for grouped GEMM backward with high precision"
    )

    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    grad_output = grad_output.contiguous()
    m_sizes = m_sizes.contiguous()

    # Allocate output tensors
    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)

    # Compute gradients using the helper functions
    _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x)
    _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)

    return grad_x, grad_w


def _pytorch_reference_backward(grad_output, x, w, m_sizes):
    """
    Pure PyTorch implementation of grouped GEMM backward for validation.
    Simple version that's easy to verify but may be less numerically accurate
    for large matrices.
    """
    # Create output gradients
    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)

    # Compute group-by-group
    G = m_sizes.shape[0]
    N = w.shape[0] // G

    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Compute gradients
            grad_x[m_start:m_end] = torch.matmul(
                grad_output[m_start:m_end, n_start:n_end], w[n_start:n_end]
            )
            grad_w[n_start:n_end] = torch.matmul(
                grad_output[m_start:m_end, n_start:n_end].t(), x[m_start:m_end]
            )

        m_start += m_size

    return grad_x, grad_w


# ========== End helper functions ==========
