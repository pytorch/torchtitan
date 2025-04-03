# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_reference_forward(x, w, m_sizes):
    """
    Compute reference forward pass using PyTorch operations.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K)
        w (torch.Tensor): Weight tensor of shape (N, K)
        m_sizes (torch.Tensor): Group sizes tensor of shape (G)

    Returns:
        torch.Tensor: Reference output tensor of shape (M, N)
    """
    result = torch.zeros((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)

    m_start = 0
    for g in range(len(m_sizes)):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size

            # Extract group input
            x_g = x[m_start:m_end]

            # Compute group output: y_g = x_g @ w.T
            y_g = torch.matmul(x_g, w.T)

            # Store result
            result[m_start:m_end] = y_g

            # Update start index
            m_start = m_end

    return result


def compute_reference_backward(x, w, m_sizes, grad_output):
    """
    Compute reference backward pass using PyTorch autograd.

    Args:
        x (torch.Tensor): Input tensor of shape (M, K)
        w (torch.Tensor): Weight tensor of shape (N, K)
        m_sizes (torch.Tensor): Group sizes tensor of shape (G)
        grad_output (torch.Tensor): Gradient tensor of shape (M, N)

    Returns:
        tuple: (grad_x, grad_w) gradient tensors
    """
    # Create autograd-enabled copies
    x_autograd = x.detach().clone().requires_grad_(True)
    w_autograd = w.detach().clone().requires_grad_(True)

    # Compute forward pass
    output = compute_reference_forward(x_autograd, w_autograd, m_sizes)

    # Backpropagate
    output.backward(grad_output)

    return x_autograd.grad, w_autograd.grad


def analyze_tensor_differences(actual, expected, name):
    """
    Analyze differences between actual and expected tensors.

    Args:
        actual (torch.Tensor): Actual tensor
        expected (torch.Tensor): Expected tensor
        name (str): Name of the tensor for logging

    Returns:
        bool: True if tensors are close enough
    """
    rtol = 0.5  # Relative tolerance for float16
    atol = 0.5  # Absolute tolerance for float16

    # Analyze differences
    diff = (actual - expected).abs()
    max_idx = diff.argmax().item()
    idx = np.unravel_index(max_idx, actual.shape)
    max_diff = diff.max().item()

    logging.info(f"Largest {name} difference: {max_diff} at {idx}")
    logging.info(f"Values: {actual[idx].item()} vs {expected[idx].item()}")

    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    if is_close:
        logging.info(f"✓ SUCCESS: {name} matches PyTorch reference")
    else:
        logging.error(f"✗ FAILURE: {name} mismatch detected")

        # Count zeros
        zeros_actual = (actual == 0).sum().item()
        zeros_expected = (expected == 0).sum().item()
        logging.info(
            f"Zeros in {name} (actual): {zeros_actual}/{actual.numel()} ({zeros_actual/actual.numel()*100:.2f}%)"
        )
        logging.info(
            f"Zeros in {name} (expected): {zeros_expected}/{expected.numel()} ({zeros_expected/expected.numel()*100:.2f}%)"
        )

        # Check for NaNs
        nan_actual = torch.isnan(actual).sum().item()
        if nan_actual > 0:
            logging.error(f"NaN values detected in {name}: {nan_actual}")

    return is_close
