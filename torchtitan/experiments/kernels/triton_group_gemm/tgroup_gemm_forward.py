# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
from typing import Optional

import tma_utils as utils

import torch

import triton
import triton.language as tl
from triton.runtime import driver  # @manual

"""
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

_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [32, 64, 128, 256]
    for block_size_k in [128, 256]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2), (16, 4)]
    for matrix_instr_nonkdim in [16]
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
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        M_PER_GROUP = M // G
        MIN_M_TILES = 32 if torch.version.hip else 64
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
        MIN_N_TILES = 32 if torch.version.hip else 64
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


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
"""


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
    N: tl.constexpr,  # N is per group
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            # Compute for this group
            N_start_offset = g * N
            n_size = N  # N is already per group

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
                    + M_start_offset * (N * G)
                    + N_start_offset,  # Offset to this group's output
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)

                if USE_TMA_LOAD:
                    # Use TMA to load input and weight blocks
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

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

                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )

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
                        + (M_start_offset + offs_am[:, None])
                        * (N * G)  # Row stride is N*G
                        + (
                            N_start_offset + offs_bn[None, :]
                        ),  # Column offset to this group's N
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )

                tidx += NUM_SMS  # Move to next tile

            iterated_tiles += num_tiles


TT_FP8_DTYPE = tl.float8e4b8 if torch.version.hip else tl.float8e4nv


"""@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
)
"""


@triton.jit
def _kernel_grouped_gemm_fp8_rowwise(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,  # N is per group
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    dtype = TT_FP8_DTYPE
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            # Compute for this group
            N_start_offset = g * N
            n_size = N  # N is already per group

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
                    + M_start_offset * (N * G)
                    + N_start_offset,  # Offset to this group's output
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)

                if USE_TMA_LOAD:
                    # Use TMA to load input and weight blocks with FP8 support
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        # Load input block [M, K] with FP8
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )

                        # Load weight block [N, K] with FP8
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )

                        # Compute matrix multiplication
                        accumulator += tl.dot(a, b.T)
                else:
                    # Manual load without TMA for FP8
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)

                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )

                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )

                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        # Load with bounds checking
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)

                        # Compute matrix multiplication
                        accumulator += tl.dot(a, b.T)

                        # Update pointers for next block
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                # Load FP8 scales
                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )

                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )

                # Apply scales to result
                c = accumulator.to(tl.float32) * a_scale * b_scale

                # Store result
                if USE_TMA_STORE:
                    # Store using TMA
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        c.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                else:
                    # Manual store
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None])
                        * (N * G)  # Row stride is N*G
                        + (
                            N_start_offset + offs_bn[None, :]
                        ),  # Column offset to this group's N
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )

                tidx += NUM_SMS  # Move to next tile

            iterated_tiles += num_tiles


def _grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not utils.HAS_TMA_DESC:
        raise NotImplementedError("Grouped GEMM without TMA is not supported yet")

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N_times_G = w.shape[0]

    # Ensure N is per group
    assert (
        N_times_G % G == 0
    ), f"Weight dimension ({N_times_G}) must be divisible by groups ({G})"
    N = N_times_G // G

    assert K == w.shape[1], f"Input K ({K}) must match weight K ({w.shape[1]})"

    # Create output tensor with correct shape [M, N*G]
    y = torch.empty((M, N_times_G), device=x.device, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    USE_TMA_LOAD = True  # not torch.version.hip
    USE_TMA_STORE = True

    desc_helper = None
    desc_x = x
    desc_w = w
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = utils.TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if USE_TMA_STORE:
        workspace = torch.empty(
            NUM_SMS * utils.TmaAutoTuneHelper.TMA_SIZE,
            device=x.device,
            dtype=torch.uint8,
        )

    # Skip autotuning - use fixed grid size
    grid_size = (min(NUM_SMS, 4),)  # Use smaller grid for small inputs
    M_BUCKET = triton.next_power_of_2(M)

    try:

        if USE_TMA_LOAD and desc_helper is not None:
            # Fixed block sizes that work well for most cases
            BLOCK_SIZE_M = 64
            BLOCK_SIZE_N = 64
            BLOCK_SIZE_K = 32

            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_K,
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N_times_G,
                K,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                w.element_size(),
            )
    except Exception as e:
        print(f"Error in TMA descriptor setup: {e}")

    if x_scale is not None and w_scale is not None:
        assert x_scale.is_contiguous()
        assert w_scale.is_contiguous()
        # Call kernel directly without autotuning
        _kernel_grouped_gemm_fp8_rowwise[grid_size](
            desc_x,
            x_scale,
            desc_w,
            w_scale,
            y,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,  # N is per group
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
            BLOCK_SIZE_M=64,  # Fixed block sizes
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
        )
    else:
        assert x_scale is None
        assert w_scale is None
        # Call kernel directly without autotuning
        _kernel_grouped_gemm[grid_size](
            desc_x,
            desc_w,
            y,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,  # N is per group
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
            BLOCK_SIZE_M=64,  # Fixed block sizes
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
        )

    # Verify the output shape
    expected_output_shape = (M, N_times_G)
    assert y.shape == expected_output_shape, (
        f"Output shape mismatch: got {y.shape}, " f"expected {expected_output_shape}"
    )

    return y


"""
def _grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not utils.HAS_TMA_DESC:
        raise NotImplementedError("Grouped GEMM without TMA is not supported yet")

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N_times_G = w.shape[0]

    # Ensure N is per group
    assert (
        N_times_G % G == 0
    ), f"Weight dimension ({N_times_G}) must be divisible by groups ({G})"
    N = N_times_G // G

    assert K == w.shape[1], f"Input K ({K}) must match weight K ({w.shape[1]})"

    # Create output tensor with correct shape [M, N*G]
    y = torch.empty((M, N_times_G), device=x.device, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    USE_TMA_LOAD = not torch.version.hip
    USE_TMA_STORE = False

    desc_helper = None
    desc_x = x
    desc_w = w
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = utils.TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if USE_TMA_STORE:
        workspace = torch.empty(
            NUM_SMS * utils.TmaAutoTuneHelper.TMA_SIZE,
            device=x.device,
            dtype=torch.uint8,
        )

    def grid(META):
        try:
            if USE_TMA_LOAD:
                nonlocal desc_helper
                desc_helper.fill_2d_tma_descriptor(
                    "x",
                    x.data_ptr(),
                    M,
                    K,
                    META["BLOCK_SIZE_M"],
                    META["BLOCK_SIZE_K"],
                    x.element_size(),
                )

                desc_helper.fill_2d_tma_descriptor(
                    "w",
                    w.data_ptr(),
                    N_times_G,
                    K,
                    META["BLOCK_SIZE_N"],
                    META["BLOCK_SIZE_K"],
                    w.element_size(),
                )
        except Exception as e:
            print(f"Error in TMA descriptor setup: {e}")
            # Use fallback values if descriptor setup fails
            pass

        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M)
    if x_scale is not None and w_scale is not None:
        assert x_scale.is_contiguous()
        assert w_scale.is_contiguous()
        _kernel_grouped_gemm_fp8_rowwise[grid](
            desc_x,
            x_scale,
            desc_w,
            w_scale,
            y,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,  # N is per group
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
        )
    else:
        assert x_scale is None
        assert w_scale is None
        _kernel_grouped_gemm[grid](
            desc_x,
            desc_w,
            y,
            workspace,
            m_sizes,
            G,
            M_BUCKET,
            N,  # N is per group
            K,
            NUM_SMS,
            USE_TMA_LOAD,
            USE_TMA_STORE,
        )

    # Verify the output shape
    expected_output_shape = (M, N_times_G)
    assert y.shape == expected_output_shape, (
        f"Output shape mismatch: got {y.shape}, " f"expected {expected_output_shape}"
    )

    return y
"""


def grouped_gemm_forward(
    x: torch.Tensor, w: torch.Tensor, m_sizes: torch.Tensor
) -> torch.Tensor:
    return _grouped_gemm(x, w, m_sizes)


def grouped_gemm_fp8_rowwise(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return _grouped_gemm(x, w, m_sizes, x_scale, w_scale)
