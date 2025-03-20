# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import os
import sys
from typing import Optional

import torch

import triton
import triton.language as tl

from tma_cuda_wrapper import CudaUtils, TmaDescriptorHelper
from triton.runtime import driver  # @manual

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@triton.jit
def _kernel_grouped_gemm(
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
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + pid * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    for g in range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            # Compute for this group
            # N is now the same for all groups
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
                    global_address=c_ptr
                    + M_start_offset * N,  # Offset to this group's output
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Process tiles in a grid-strided loop
            for tile_idx in range(pid, num_tiles, num_pid):
                # Split M first and N second.
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles

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


TT_FP8_DTYPE = tl.float8e4nv


def _grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
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

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

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

    # TODO - Skip autotuning - use fixed grid size
    grid_size = (min(NUM_SMS, 4),)  # Use smaller grid for small inputs
    M_BUCKET = triton.next_power_of_2(M_total)

    try:
        if USE_TMA_LOAD and desc_helper is not None:
            # Fixed block sizes that work well for most cases
            BLOCK_SIZE_M = 64
            BLOCK_SIZE_N = 64
            BLOCK_SIZE_K = 32

            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M_total,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_K,
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N,
                K,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                w.element_size(),
            )
    except Exception as e:
        print(f"Error in TMA descriptor setup: {e}")

    assert x_scale is None
    assert w_scale is None
    # TODO -  autotuning
    _kernel_grouped_gemm[grid_size](
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
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )

    # Verify the output shape
    expected_output_shape = (M_total, N)
    assert y.shape == expected_output_shape, (
        f"Output shape mismatch: got {y.shape}, " f"expected {expected_output_shape}"
    )

    return y


def group_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor
) -> torch.Tensor:
    return _grouped_gemm(x, w, m_sizes)
