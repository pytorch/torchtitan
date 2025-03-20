import logging
from typing import List, Optional, Tuple, Union

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try to import the custom GEMM implementations
try:
    from NG_backward import grouped_gemm_backward
    from NG_forward import grouped_gemm_forward

    _HAS_CUSTOM_GEMM = True
except ImportError:
    logging.warning("Triton GEMM implementations not found, falling back to PyTorch")
    _HAS_CUSTOM_GEMM = False


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    group_sizes: Union[torch.Tensor, List[int]],
    use_custom_kernel: bool = True,
) -> torch.Tensor:
    """
    Simple cover function for grouped matrix multiplication.

    Args:
        x: Input tensor of shape [M, K]
        w: Weight tensor of shape [N*G, K]
        group_sizes: Tensor or list specifying the size of each group, shape [G]
        use_custom_kernel: Whether to use the custom CUDA kernel (if available)

    Returns:
        Result tensor of shape [M, N*G]
    """
    # Convert group_sizes to tensor if it's a list
    if isinstance(group_sizes, list):
        group_sizes = torch.tensor(group_sizes, device=x.device, dtype=torch.int32)

    # Ensure group_sizes has the right type
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)

    # Make sure tensors are contiguous
    x = x.contiguous()
    w = w.contiguous()
    group_sizes = group_sizes.contiguous()

    # Check dimensions
    G = group_sizes.shape[0]  # Number of groups
    M, K_x = x.shape
    N_times_G, K_w = w.shape

    if K_x != K_w:
        raise ValueError(f"K dimensions must match: x has K={K_x}, w has K={K_w}")

    if N_times_G % G != 0:
        raise ValueError(
            f"Weight rows ({N_times_G}) must be divisible by number of groups ({G})"
        )

    # Use custom kernel if available and requested
    if _HAS_CUSTOM_GEMM and use_custom_kernel:
        try:
            result = grouped_gemm_forward(x, w, group_sizes)
            return result
        except Exception as e:
            logging.warning(
                f"Error in custom kernel: {e}. Falling back to PyTorch implementation."
            )

    # PyTorch fallback implementation
    N = N_times_G // G
    result = torch.zeros(M, N_times_G, dtype=x.dtype, device=x.device)

    m_start = 0
    for g in range(G):
        m_size = group_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Standard matrix multiplication for this group
            result[m_start:m_end, n_start:n_end] = torch.matmul(
                x[m_start:m_end], w[n_start:n_end].t()
            )

        m_start += m_size

    return result


class GroupedGEMMWithGrad(torch.autograd.Function):
    """
    Autograd function wrapper for grouped GEMM with custom backward.
    This class enables direct use of the custom kernels with automatic gradient.
    """

    @staticmethod
    def forward(ctx, x, w, group_sizes, use_custom_kernel=True):
        # Save for backward
        ctx.save_for_backward(x, w, group_sizes)
        ctx.use_custom_kernel = use_custom_kernel

        # Call the cover function
        return grouped_gemm(x, w, group_sizes, use_custom_kernel)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, group_sizes = ctx.saved_tensors
        use_custom_kernel = ctx.use_custom_kernel

        # Use custom kernel if available and requested
        if _HAS_CUSTOM_GEMM and use_custom_kernel:
            try:
                grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, group_sizes)
                return grad_x, grad_w, None, None
            except Exception as e:
                logging.warning(
                    f"Error in custom backward kernel: {e}. Falling back to PyTorch."
                )

        # PyTorch fallback
        G = group_sizes.shape[0]
        N = w.shape[0] // G

        # Create output gradients
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(w)

        # Compute gradients group by group
        m_start = 0
        for g in range(G):
            m_size = group_sizes[g].item()
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

        return grad_x, grad_w, None, None


def grouped_gemm_with_grad(
    x: torch.Tensor,
    w: torch.Tensor,
    group_sizes: Union[torch.Tensor, List[int]],
    use_custom_kernel: bool = True,
) -> torch.Tensor:
    """
    Grouped matrix multiplication with gradient support for full training use.

    Args:
        x: Input tensor of shape [M, K]
        w: Weight tensor of shape [N*G, K]
        group_sizes: Tensor or list specifying the size of each group, shape [G]
        use_custom_kernel: Whether to use custom CUDA kernels (if available)

    Returns:
        Result tensor of shape [M, N*G]
    """
    # Convert group_sizes to tensor if it's a list
    if isinstance(group_sizes, list):
        group_sizes = torch.tensor(group_sizes, device=x.device, dtype=torch.int32)

    # Ensure correct dtype
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)

    return GroupedGEMMWithGrad.apply(x, w, group_sizes, use_custom_kernel)
