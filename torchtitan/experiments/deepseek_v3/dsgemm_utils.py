# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This module contains code to support working with DeepGEMM:
# DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling
# https://github.com/deepseek-ai/DeepGEMM

# A subset of functions (marked below) are directly copied over from DeepGemm testing and utils
# in order to support grouped gemm creation and testing of our integration.
# Please see their respective MIT license agreement.

from typing import Tuple

import torch

_num_sms = None


def compare_fp8_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compare two FP8 tensors by converting them to float32 first.

    Args:
        a: First tensor (float8_e4m3fn)
        b: Second tensor (float8_e4m3fn)

    Returns:
        Maximum absolute difference between tensors
    """
    # Convert to float32 before comparison
    a_float = a.to(torch.float32)
    b_float = b.to(torch.float32)

    # Calculate difference
    diff = torch.abs(a_float - b_float)

    # Return maximum difference
    return diff.max().item()


def prepare_fp8_input(x):
    """
    Convert input tensor to FP8 format with per-token scaling for DeepSeek GEMM.

    Args:
        x: Input tensor

    Returns:
        Tuple of (FP8 tensor, scale factors) in the format expected by DeepSeek GEMM
    """
    # Apply per-token quantization
    x_fp8, x_scales = per_token_cast_to_fp8(x)

    # Get TMA-aligned format for scales
    x_scales_aligned = get_col_major_tma_aligned_tensor(x_scales)

    return (x_fp8, x_scales_aligned)


def prepare_fp8_weight(w):
    """
    Convert weight tensor to FP8 format with per-block scaling for DeepSeek GEMM.

    Args:
        w: Weight tensor (3D)

    Returns:
        Tuple of (FP8 tensor, scale factors) in the format expected by DeepSeek GEMM
    """
    num_experts = w.shape[0]
    n = w.shape[1]
    k = w.shape[2]

    # Create containers for FP8 weights and scales
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    w_scales = torch.empty(
        (num_experts, (n + 127) // 128, (k + 127) // 128),
        device=w.device,
        dtype=torch.float32,
    )

    # Process each expert's weights
    for i in range(num_experts):
        w_fp8[i], w_scales[i] = per_block_cast_to_fp8(w[i])

    return (w_fp8, w_scales)


def create_indices_from_offsets_nosync(m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Create m_indices tensor from m_offsets tensor without CPU-GPU sync points.

    Args:
        m_offsets: Tensor containing cumulative offsets for each group
            e.g., [128, 128, 256, 384, 640, ...]

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    # Get total size from the last offset
    total_size = m_offsets[-1]

    # Pre-allocate output tensor
    indices = torch.empty(total_size, device=m_offsets.device, dtype=torch.int32)

    # Create a range tensor for each section
    prev_offset = torch.zeros(1, device=m_offsets.device, dtype=m_offsets.dtype)

    for i in range(len(m_offsets)):
        # Calculate current section size
        section_size = m_offsets[i] - prev_offset

        # Only fill if section has elements
        if section_size > 0:
            indices[prev_offset : m_offsets[i]] = i

        # Update prev_offset for next iteration
        prev_offset = m_offsets[i]

    return indices


def create_m_indices_from_offsets(m_offsets: torch.Tensor) -> torch.Tensor:
    """
    Create m_indices tensor from m_offsets tensor.

    Args:
        m_offsets: Tensor containing cumulative offsets for each group
            e.g., [128, 128, 256, 384, 640, ...]

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    # Get total size from the last offset
    total_size = m_offsets[-1].item()

    # Pre-allocate output tensor
    indices = torch.empty(total_size, device=m_offsets.device, dtype=torch.int32)

    # Fill indices directly using offsets
    prev_offset = 0
    for i, offset in enumerate(m_offsets):
        offset_val = offset.item()
        if offset_val > prev_offset:
            indices[prev_offset:offset_val] = i
        prev_offset = offset_val

    return indices


def create_m_indices_from_sizes(m_sizes: torch.Tensor) -> torch.Tensor:
    """
    Fast implementation to create m_indices when tokens are already contiguous.

    Args:
        m_sizes: Tensor containing the number of rows for each group

    Returns:
        m_indices: Tensor mapping each row to its group index
    """
    # Use cumulative sum to create offsets
    total_size = m_sizes.sum().item()

    # Pre-allocate output tensor
    indices = torch.empty(total_size, device=m_sizes.device, dtype=torch.int32)

    # Fill indices directly
    offset = 0
    for i, size in enumerate(m_sizes):
        size_val = size.item()
        if size_val > 0:
            indices[offset : offset + size_val] = i
            offset += size_val

    return indices


# ------ Quant kernel from DeepSeek: --------------------------


# ----- Functions below are copied from DeepGeMM testing and utilities
# https://github.com/deepseek-ai/DeepGEMM


def get_m_indices(num_groups: int, m: int) -> torch.Tensor:
    m_indices = torch.arange(0, num_groups, device="cuda", dtype=torch.int)
    m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
    return m_indices


def make_grouped_layout(num_groups: int, x: torch.Tensor, y: torch.Tensor):

    m = x.size(0)
    n = y.size(1)
    k = x.size(1)

    # create fp8 containers
    out = torch.empty((num_groups, m, n), device="cuda", dtype=torch.bfloat16)
    x_fp8 = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, m, k // 128), device="cuda", dtype=torch.float),
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, (n + 127) // 128, k // 128), device="cuda", dtype=torch.float
        ),
    )

    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
    out = out.view(-1, n)  # collapse the group and M dims

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out


def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert (
        0
        < num_sms
        <= torch.cuda.get_device_properties(device="cuda").multi_processor_count
    )
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    return _num_sms


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def get_m_alignment_for_contiguous_layout():
    """
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.

    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    """
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    """
    Global memory address of TMA must be 16-byte aligned.
    Since we use column-major layout for the LHS scaling tensor,
        the M-axis of the LHS scaling tensor needs to be padded to a multiple of 16 bytes.

    Arguments:
        x: original M-axis shape of the LHS scaling tensor.
        element_size: element size of the LHS scaling tensor.

    Returns:
        M-axis shape of the LHS scaling tensor after padding.
    """
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns TMA-aligned transposed format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already column-major layout and 16-byte aligned along the M axis
        (thus meets the requirement of LHS scaling tensor in DeepGEMM), this function will do nothing.

    Arguments:
        x: usually the LHS scaling tensor in GEMM.

    Returns:
        The LHS scaling tensor of TMA-aligned transposed format.
    """
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dim() in (2, 3)
    remove_dim = False
    m, n = x.shape[-2], x.shape[-1]
    aligned_m = get_tma_aligned_size(m, x.element_size())
    if x.dim() == 2:
        if x.stride(0) == 1 and x.stride(1) == aligned_m:
            return x
        x, remove_dim = x.unsqueeze(0), True

    b = x.shape[0]

    # The last kernel gives a column-major TMA aligned layout
    if x.stride(0) == aligned_m * n and x.stride(1) == 1 and x.stride(2) == aligned_m:
        return x.squeeze(0) if remove_dim else x

    # Normal layout requires transposing
    aligned_x = torch.transpose(
        torch.empty((b, n, aligned_m), device=x.device, dtype=x.dtype), 1, 2
    )
    aligned_x[:, :m, :] = x
    aligned_x = aligned_x[:, :m, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def construct_grouped(
    num_groups: int, x: torch.Tensor, y: torch.Tensor, is_masked: bool
):

    out = torch.empty((num_groups, m, n), device="cuda", dtype=torch.bfloat16)

    assert m % 4 == 0, f"TMA alignment error: {m}"
    x_fp8 = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, m, k // 128), device="cuda", dtype=torch.float),
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, (n + 127) // 128, k // 128), device="cuda", dtype=torch.float
        ),
    )

    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out = (out.view(-1, n),)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return (
        x_fp8,
        y_fp8,
        out,
    )
