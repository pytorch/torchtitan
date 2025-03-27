# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# credit - TMAHelper class, AutoTuning, FP8 row quantization and flat style indexed forward kernel are derived from FBGemm:
# https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm

# pyre-unsafe
import functools
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

import triton
import triton.language as tl
from triton import Config as TConfig

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

# ================== Start of FP8 row quantization ==================


@triton.autotune(
    configs=[
        TConfig({"BLOCK_SIZE": 512}),
        TConfig({"BLOCK_SIZE": 1024}),
        TConfig({"BLOCK_SIZE": 2048}),
        TConfig({"BLOCK_SIZE": 4096}),
        TConfig({"BLOCK_SIZE": 8192}),
    ],
    key=["K"],
)
@triton.jit
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    zero_start_index_M,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_an,
    stride_ak,
    stride_ob,
    stride_om,
    stride_on,
    stride_ok,
    stride_zb,  # not used
    stride_zm,  # not used
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): higher precision input tensor of 4 dimension.
        A_scale (Tensor): [B * M * N] reciprocal scale tensor per row.
        A_fp8 (Tensor): fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        B (int): Size of dimenion 0
        M (int): Size of dimenion 1
        N (int): Size of dimenion 2
        K (int): Size of dimenion 3
        stride_ab (int): Stride of b dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_ob (int): Stride of b dimension of output.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_zb (int): Stride of b dimension of jagged index.
        stride_zm (int): Stride of m dimension of jagged index.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        Removed:  JAGGED (bool): Whether to use jagged indexing.
        BLOCK_SIZE (int): Block size for reduction.
        USE_INT64 (bool): Whether to use int64 indexing for large inputs.
    """
    pid = tl.program_id(0)
    # Use int64 indexing for large inputs. This is slower, but
    # needed to avoid index overflows.
    if USE_INT64:
        pid = pid.to(tl.int64)
    n_offset = tl.arange(0, BLOCK_SIZE)
    a_offset_base = (
        pid // (M * N) * stride_ab
        + (pid % (M * N)) // N * stride_am
        + (pid % (M * N)) % N * stride_an
    )
    a_fp8_offset_base = (
        pid // (M * N) * stride_ob
        + (pid % (M * N)) // N * stride_om
        + (pid % (M * N)) % N * stride_on
    )

    K_in = K

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(K_in, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)
        n_offset += BLOCK_SIZE

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS)
    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max
    tl.store(A_scale + pid, 1.0 / a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE)):
        a = tl.load(
            A + a_offset_base + n_offset * stride_ak,
            mask=n_offset < K_in,
            other=0.0,
        )
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(
            A_fp8 + a_fp8_offset_base + n_offset * stride_ok,
            a_fp8,
            mask=n_offset < K,
        )
        n_offset += BLOCK_SIZE


def triton_quantize_fp8_row(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    zero_start_index_M: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): higher precision input tensor of 4 dimension.
        scale_ub (Tensor): Maximum allowed value for scale.
        zero_start_index_M (Tensor): Indicates number of nonzero elements in each row.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
    """
    assert a.dim() <= 4, "Triton only supports up to 4 dimension input tensor."
    a_shape = a.shape
    while a.dim() < 4:
        a = a.unsqueeze(0)
    # if zero_start_index_M is not None:
    # There should be one value of zero_start_index_M per NxK matrix.
    #     zero_start_index_M = zero_start_index_M.view(a.shape[0], a.shape[1])

    # Set constant values.
    pt_fp8_dtype = torch.float8_e4m3fn
    tl_dtype = tl.float8e4nv
    max_fp8 = torch.finfo(pt_fp8_dtype).max
    eps = 1e-12

    num_rows = a.numel() // a.shape[-1]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    a_fp8 = torch.empty(a.shape, device=a.device, dtype=pt_fp8_dtype)

    # If input tensor is sufficiently large, we need to use int64 indexing.
    use_int64 = a.numel() > (2**31 - 1)
    grid = (num_rows,)

    _kernel_quantize_fp8_row[grid](
        a,
        a_scale,
        a_fp8,
        scale_ub,
        zero_start_index_M,
        a.shape[0],
        a.shape[1],
        a.shape[2],
        a.shape[3],
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a.stride(3),
        a_fp8.stride(0),
        a_fp8.stride(1),
        a_fp8.stride(2),
        a_fp8.stride(3),
        None,
        None,
        # zero_start_index_M.stride(0) if zero_start_index_M is not None else None,
        # zero_start_index_M.stride(1) if zero_start_index_M is not None else None,
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
        # JAGGED=zero_start_index_M is not None,
        USE_INT64=use_int64,
    )

    return a_fp8.view(a_shape), a_scale.view(a_shape[:-1])


# ================== End of FP8 row quantization ==================


# ======  Autotuning utilities ======

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
    # Check for all possible pointer parameter names
    if "grad_input_ptr" in named_args:
        ptr_name = "grad_input_ptr"
    elif "c_ptr" in named_args:
        ptr_name = "c_ptr"
    elif "grad_weight_ptr" in named_args:
        ptr_name = "grad_weight_ptr"
    else:
        raise KeyError("No recognized pointer parameter found in kernel arguments")

    if dtsize is None:
        dtsize = named_args[ptr_name].element_size()
    if dtype is None:
        dtype = named_args[ptr_name].dtype

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


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_forward_hopper_bf16(
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
    # config
    NUM_SMS: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    USE_EPILOGUE_SUBTILING: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Flat index style forward kernel for Hopper.
    For simplicity, we always use TMA Load and TMA Store
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty  # output dtype

    c_desc_ptr = workspace + (tbidx * TMA_SIZE)  # for TMA Store

    M_end = 0
    M_start = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups
        # reset to new M offset
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            n_size = N

            # Acquire hold on c_desc_ptr for TMA Store
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start * N,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_dtype,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                # columnwise
                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    # input block [M,K]
                    a = tl._experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        c_dtype,
                    )
                    # weight block [N, K]
                    b = tl._experimental_descriptor_load(
                        b_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        c_dtype,
                    )

                    accumulator += tl.dot(a, b.T)

                # Store using TMA

                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)

                if USE_EPILOGUE_SUBTILING:
                    acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(c_dtype)
                    tl._experimental_descriptor_store(
                        c_desc_ptr, c0, [m_offset, n_offset]
                    )
                    c1 = acc1.to(c_dtype)
                    tl._experimental_descriptor_store(
                        c_desc_ptr, c1, [m_offset, n_offset + BLOCK_SIZE_N // 2]
                    )
                else:
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(c_dtype),
                        [m_offset, n_offset],
                    )
                # move to next tile in group
                tbidx += NUM_SMS
            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_forward_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
    a_scale_ptr,
    b_scale_ptr,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    USE_FP8: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Flat index style forward kernel.
    For simplicity, we always use TMA Load and TMA Store
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty

    c_desc_ptr = workspace + (tbidx * TMA_SIZE)

    M_end = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups
        # reset to new M offset
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            n_size = N

            # TMA Store prep
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start * N,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_dtype,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    # input block [M,K]
                    a = tl._experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        c_dtype,
                    )
                    # weight block [N, K]
                    b = tl._experimental_descriptor_load(
                        b_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        c_dtype,
                    )

                    accumulator += tl.dot(a, b.T)

                # Store using TMA

                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                # n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(c_dtype),
                    [m_offset, n_offset],
                )

                # Move to the next tile
                tbidx += NUM_SMS
            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_forward_no_tma(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Flat index style forward kernel.
    For bc and Ampere, we never use TMA Load and TMA Store
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty
    c_desc_ptr = None

    M_end = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups
        # reset to new M offset
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            n_size = N

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

                offs_am = tile_m_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_index * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_ptr + (M_start + offs_am[:, None]) * K + offs_k[None, :]
                b_ptrs = b_ptr + (offs_bn[:, None]) * K + offs_k[None, :]

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    # Load with bounds checking
                    a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                    b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)

                    # Main matmul
                    accumulator += tl.dot(a, b.T)

                    # Update pointers for next block
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K

                # Store without TMA
                offs_am = tile_m_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_index * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                c = accumulator.to(c_dtype)

                tl.store(
                    c_ptr
                    + (M_start + offs_am[:, None]) * N  # Row stride is N
                    + offs_bn[None, :],  # Column offset
                    c,
                    mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                )
                # Move to the next tile
                tbidx += NUM_SMS
            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


"""
Backward pass for grouped GEMM with Triton, where grouping is M*G
We compute gradients with respect to both input (`grad_x`) and weights (`grad_w`).
"""


# ---- dx flat linear indexed ----
@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_dx_tma(
    grad_output_desc_ptr,  # grad_output descriptor [M_total, N]
    w_desc_ptr,  # weight descriptor [N, K]
    grad_input_ptr,  # output grad_x [M_total, K]
    workspace,  # workspace for TMA store
    m_sizes,  # group sizes [G]
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    TMA-optimized kernel for computing gradients with respect to input (dx).
    For the forward pass Y = X @ W.T, the backward for input is:
    grad_X = grad_Y @ W

    Key differences from forward:
    1. W is used directly (not transposed)
    2. The reduction dimension is now N (not K)
    3. Output is [M, K] instead of [M, N]
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = grad_input_ptr.dtype.element_ty
    c_desc_ptr = workspace + (tbidx * TMA_SIZE)

    M_end = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups - same as forward
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            # tiles for this group - now producing [M, K] output
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            group_num_tiles = num_m_tiles * num_k_tiles

            # TMA Store prep for [M, K] output
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=grad_input_ptr + M_start * K,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                global_size=[m_size, K],
                element_ty=c_dtype,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                # Different tiling scheme for [M, K] output
                tile_m_index = group_index % num_m_tiles
                tile_k_index = group_index // num_m_tiles

                # Initialize accumulator for grad_input block [M, K]
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

                # Position in full matrix
                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                k_offset = (tile_k_index * BLOCK_SIZE_K).to(tl.int32)

                # reduce along N dimension (instead of K in forward)
                for n_offset in range(0, N, BLOCK_SIZE_N):
                    # grad_output block [M, N]
                    grad_output = tl._experimental_descriptor_load(
                        grad_output_desc_ptr,
                        [m_offset, n_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_N],
                        c_dtype,
                    )

                    # weight block [N, K] - no transpose needed
                    w = tl._experimental_descriptor_load(
                        w_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        c_dtype,
                    )

                    # grad_x = grad_output @ w
                    # reducing along N dimension
                    accumulator += tl.dot(grad_output, w)

                # Store using TMA
                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                k_offset = (tile_k_index * BLOCK_SIZE_K).to(tl.int32)

                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(c_dtype),
                    [m_offset, k_offset],
                )

                # Move to the next tile
                tbidx += NUM_SMS

            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


# ---- dw flat linear indexed ----
@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_dw_tma(
    x_desc_ptr,  # input descriptor [M_total, K]
    grad_output_desc_ptr,  # grad_output descriptor [M_total, N]
    grad_weight_ptr,  # output grad_w [N, K]
    workspace,  # workspace for TMA store
    m_sizes,  # group sizes [G]
    # x_scale_ptr,  # Optional scale for x in FP8
    # grad_output_scale_ptr,  # Optional scale for grad_output in FP8
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # tiles
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # block size for reduction dimension
) -> None:
    """
    TMA-optimized kernel for computing gradients with respect to weights (dw).
    For the forward pass Y = X @ W.T, the backward for weights is:
    grad_W = grad_Y.T @ X

    Key differences from forward and dx:
    1. The output is now [N, K] instead of [M, N] or [M, K]
    2. We reduce along the M dimension
    3. The computation pattern is different (transpose of grad_output)
    """
    # Get the block indices - here we use a different grid layout
    # Instead of being assigned a tile within a group, each block computes
    # a single [BLOCK_SIZE_N, BLOCK_SIZE_K] tile of grad_weight
    pid_n = tl.program_id(0)  # N dimension
    pid_k = tl.program_id(1)  # K dimension

    c_dtype = grad_weight_ptr.dtype.element_ty

    # Pre-compute number of tiles in K dimension for workspace indexing
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K).to(tl.int32)
    c_desc_ptr = workspace + ((pid_n * k_tiles + pid_k) * TMA_SIZE)

    # Compute global indices for this block's output tile
    n_offset = pid_n * BLOCK_SIZE_N
    k_offset = pid_k * BLOCK_SIZE_K

    # Create the offsets for this output tile
    offs_n = n_offset + tl.arange(0, BLOCK_SIZE_N)
    offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

    # Create masks for bounds checking
    n_mask = offs_n < N
    k_mask = offs_k < K

    # Combined mask for output
    output_mask = n_mask[:, None] & k_mask[None, :]

    # Create TMA store descriptor for [N, K] output
    if USE_TMA_STORE:
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=c_desc_ptr,
            global_address=grad_weight_ptr + n_offset * K + k_offset,
            load_size=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            global_size=[N, K],
            element_ty=c_dtype,
        )
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

    # Initialize accumulator for this output tile
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Process each group
    M_end = 0
    for g in range(G):
        # Get group boundaries
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        # Only process if group is non-empty
        if m_size > 0:
            # Process this group in chunks along the M dimension
            for m_offset in range(0, m_size, BLOCK_SIZE_M):
                # Calculate actual block size (handling boundary)
                m_block_size = tl.minimum(BLOCK_SIZE_M, m_size - m_offset)

                # Only process if we have actual work to do
                if m_block_size > 0:
                    # Global offset for this chunk
                    m_global_offset = M_start + m_offset

                    # Load input chunk [M_chunk, K] using TMA if available
                    if USE_TMA_LOAD:
                        # Load using TMA descriptor - always load full blocks with TMA
                        x_block = tl._experimental_descriptor_load(
                            x_desc_ptr,
                            [m_global_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],  # Always load full block size
                            c_dtype,
                        )

                        # Create mask for the valid part of the block
                        offs_m = tl.arange(0, BLOCK_SIZE_M)
                        m_mask = offs_m < m_block_size

                        # Apply mask after loading to zero out invalid elements
                        x_block = tl.where(m_mask[:, None], x_block, 0.0)
                    else:
                        # Manual load with bounds checking
                        offs_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
                        m_valid = offs_m < m_size

                        # Combined mask
                        mk_mask = m_valid[:, None] & k_mask[None, :]

                        # Ensure offset types for pointer arithmetic
                        m_ptr_offset = (M_start + offs_m[:, None]).to(tl.int32)
                        k_ptr_offset = offs_k[None, :].to(tl.int32)

                        # Load from x
                        # Note: using different variable name as we need different shape
                        x_block_partial = tl.load(
                            x_desc_ptr + m_ptr_offset * K + k_ptr_offset,
                            mask=mk_mask,
                            other=0.0,
                        )

                    # Ensure correct types for pointer arithmetic when using raw pointers
                    if not USE_TMA_LOAD:
                        x_ptr = x_desc_ptr
                        grad_output_ptr = grad_output_desc_ptr
                    if USE_TMA_LOAD:
                        # Load using TMA descriptor
                        grad_output_block = tl._experimental_descriptor_load(
                            grad_output_desc_ptr,
                            [m_global_offset, n_offset],
                            [m_block_size, BLOCK_SIZE_N],
                            c_dtype,
                        )
                        # Create mask for the valid part of the block
                        offs_m = tl.arange(0, BLOCK_SIZE_M)
                        m_mask = offs_m < m_block_size

                        # Apply mask after loading to zero out invalid elements
                        grad_output_block = tl.where(
                            m_mask[:, None], grad_output_block, 0.0
                        )
                    else:
                        # Manual load with bounds checking
                        offs_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
                        m_valid = offs_m < m_size

                        # Combined mask
                        mn_mask = m_valid[:, None] & n_mask[None, :]

                        # Ensure offset types for pointer arithmetic
                        m_ptr_offset = (M_start + offs_m[:, None]).to(tl.int32)
                        n_ptr_offset = offs_n[None, :].to(tl.int32)

                        # Load from grad_output
                        grad_output_block_partial = tl.load(
                            grad_output_desc_ptr + m_ptr_offset * N + n_ptr_offset,
                            mask=mn_mask,
                            other=0.0,
                        )

                    # Compute contribution to grad_W: grad_Y.T @ X
                    # Need to transpose grad_output for the matmul
                    # Note: we use different x_block variable names based on TMA/non-TMA
                    if USE_TMA_LOAD:
                        # If we used TMA load, use the direct variables
                        contribution = tl.dot(
                            grad_output_block.to(tl.float32).T,  # [N, M_chunk]
                            x_block.to(tl.float32),  # [M_chunk, K]
                        )
                    else:
                        # If we used manual load, use the partial variables
                        contribution = tl.dot(
                            grad_output_block_partial.to(tl.float32).T,  # [N, M_chunk]
                            x_block_partial.to(tl.float32),  # [M_chunk, K]
                        )

                    # Accumulate the contribution
                    accumulator += contribution

    # Apply FP8 scaling if needed (removed for now)

    # Store the result
    if USE_TMA_STORE:
        # Store using TMA
        tl._experimental_descriptor_store(
            c_desc_ptr,
            accumulator.to(c_dtype),
            [0, 0],  # Offset within the tile
        )
    else:
        # Store manually with bounds checking
        tl.store(
            grad_weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            accumulator.to(c_dtype),
            mask=output_mask,
        )


# ======== End Triton kernels ========

# ======== Triton wrapper functions ========


def grouped_gemm_dx_optimized(
    grad_output: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    num_sms: int,
    tma_size: int = 128,
    using_fp8: bool = False,
) -> torch.Tensor:
    """
    Optimized backward pass for computing gradient with respect to input (dx)
    using TMA patterns similar to the forward pass.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        w: Weight tensor, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor
        # using_fp8: Whether to use FP8 quantization
        # grad_output_scale: Scale for grad_output in FP8 mode
        # w_scale: Scale for w in FP8 mode

    Returns:
        grad_x: Gradient with respect to x, shape [M_total, K]
    """
    """
    Optimized backward pass for computing gradient with respect to input (dx)
    using TMA patterns similar to the forward pass.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        w: Weight tensor, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor
        using_fp8: Whether to use FP8 quantization
        # grad_output_scale: Scale for grad_output in FP8 mode
        # w_scale: Scale for w in FP8 mode

    Returns:
        grad_x: Gradient with respect to x, shape [M_total, K]
    """
    if not CudaUtils.verify_tma():
        raise NotImplementedError("Optimized dx computation requires TMA support")

    G = m_sizes.shape[0]

    assert grad_output.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M_total, N_grad = grad_output.shape
    N_w, K = w.shape

    # Check dimensions
    assert N_grad == N_w, f"Grad_output N ({N_grad}) must match weight N ({N_w})"

    # Verify that the sum of m_sizes matches M_total
    sum_m_sizes = m_sizes.sum().item()
    assert (
        M_total == sum_m_sizes
    ), f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"

    # Create output tensor (grad_x) with shape [M_total, K]
    grad_x = torch.empty(
        (M_total, K), device=grad_output.device, dtype=grad_output.dtype
    )

    NUM_SMS = num_sms  # CudaUtils.get_num_sms()
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

    # Set up TMA descriptors
    desc_helper = TmaDescriptorHelper(tma_size=tma_size)
    desc_helper.init_tma_descriptor("grad_output")
    desc_helper.init_tma_descriptor("w")
    desc_grad_output = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    # Allocate workspace for TMA store
    workspace = torch.empty(
        NUM_SMS * desc_helper.tma_size,
        device=grad_output.device,
        dtype=torch.uint8,
    )

    def grid(META):
        # Fill TMA descriptors with appropriate dimensions
        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M_total,
            N_grad,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_N"],
            grad_output.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "w",
            w.data_ptr(),
            N_w,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w.element_size(),
        )
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)

    # Launch the optimized kernel for computing grad_x
    _kernel_mg_dx_tma[grid](
        desc_grad_output,
        desc_w,
        grad_x,
        workspace,
        m_sizes,
        G,
        M_BUCKET,
        N_grad,  # N dimension is now the reduction dimension
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
    )

    return grad_x


def grouped_gemm_dw_optimized(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    m_sizes: torch.Tensor,
    num_sms: int,
    tma_size: int = 128,
    using_fp8: bool = False,
    x_scale: Optional[torch.Tensor] = None,
    grad_output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Optimized computation of gradients with respect to weights (dw) using TMA.

    Args:
        x: Input tensor, shape [M_total, K]
        grad_output: Gradient of output, shape [M_total, N]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor in bytes
        using_fp8: Whether to use FP8 quantization
        x_scale: Scale for x in FP8 mode
        grad_output_scale: Scale for grad_output in FP8 mode

    Returns:
        grad_w: Gradient with respect to weights, shape [N, K]
    """
    # Check TMA support
    # can_use_tma = CudaUtils.verify_tma()

    G = m_sizes.shape[0]

    # Ensure contiguous tensors
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    m_sizes = m_sizes.contiguous()

    # Get dimensions
    M_total, K_x = x.shape
    M_grad, N = grad_output.shape

    # Check dimensions
    assert M_total == M_grad, f"x M ({M_total}) must match grad_output M ({M_grad})"

    # Verify that the sum of m_sizes matches M_total
    sum_m_sizes = m_sizes.sum().item()
    assert (
        M_total == sum_m_sizes
    ), f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"

    # Create output tensor (grad_w) with shape [N, K]
    grad_w = torch.zeros((N, K_x), device=x.device, dtype=x.dtype)

    # Get number of SMs
    NUM_SMS = num_sms  # CudaUtils.get_num_sms()

    # Set up TMA flags
    USE_TMA_LOAD = False  # can_use_tma
    USE_TMA_STORE = False  # can_use_tma

    # Handle FP8 scaling if needed
    if using_fp8 and (x_scale is None or grad_output_scale is None):
        print("FP8 scaling in progress...")
        x_fp8, x_scales = triton_quantize_fp8_row(x)
        grad_output_fp8, grad_output_scales = triton_quantize_fp8_row(grad_output)
        x = x_fp8
        grad_output = grad_output_fp8
        x_scale = x_scales
        grad_output_scale = grad_output_scales

    # Set up TMA descriptors
    if USE_TMA_LOAD:
        desc_helper = TmaDescriptorHelper(tma_size=tma_size)
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("grad_output")
        x_desc = desc_helper.get_tma_descriptor_kernel_param("x")
        grad_output_desc = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    else:
        # If not using TMA, just use the tensors directly
        x_desc = x
        grad_output_desc = grad_output

    """# Choose block sizes based on dimensions
    if max(N, K_x) <= 512:
        BLOCK_SIZE_N = min(64, N)
        BLOCK_SIZE_K = min(64, K_x)
        BLOCK_SIZE_M = 128  # Reduction dimension can be larger
    else:
        BLOCK_SIZE_N = min(128, N)
        BLOCK_SIZE_K = min(128, K_x)
        BLOCK_SIZE_M = 64  # Smaller for large tensors to avoid register pressure

    # Make block sizes powers of 2 for better performance
    BLOCK_SIZE_N = triton.next_power_of_2(BLOCK_SIZE_N)
    BLOCK_SIZE_K = triton.next_power_of_2(BLOCK_SIZE_K)
    BLOCK_SIZE_M = triton.next_power_of_2(BLOCK_SIZE_M)
    """
    # Allocate workspace for TMA store
    if USE_TMA_STORE:
        num_tiles = triton.cdiv(N, BLOCK_SIZE_N) * triton.cdiv(K_x, BLOCK_SIZE_K)
        workspace = torch.empty(
            num_tiles * tma_size,
            device=x.device,
            dtype=torch.uint8,
        )
    else:
        workspace = torch.empty(0, device=x.device, dtype=torch.uint8)

    # Define grid for kernel launch
    def grid(META):
        if USE_TMA_LOAD:
            nonlocal desc_helper
            # Fill TMA descriptors with appropriate dimensions
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M_total,
                K_x,
                META["BLOCK_SIZE_M"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "grad_output",
                grad_output.data_ptr(),
                M_total,
                N,
                META["BLOCK_SIZE_M"],
                META["BLOCK_SIZE_N"],
                grad_output.element_size(),
            )

        # Return grid size - one block per output tile
        return (
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
            triton.cdiv(K_x, META["BLOCK_SIZE_K"]),
        )

    M_BUCKET = triton.next_power_of_2(M_total)

    # Launch kernel
    _kernel_mg_dw_tma[grid](
        x_desc,
        grad_output_desc,
        grad_w,
        workspace,
        m_sizes,
        # x_scale,
        # grad_output_scale,
        G,
        M_BUCKET,
        N,
        K_x,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
        # USE_FP8=using_fp8,
        # BLOCK_SIZE_N=BLOCK_SIZE_N,
        # BLOCK_SIZE_K=BLOCK_SIZE_K,
        # BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return grad_w


# ======== End Backwards Wrapper Functions =============

# ======== PyTorch wrapper functions ========


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

    if x_scale is not None and w_scale is not None:
        using_fp8 = True

    # TODO: not clear if we should integrate FP8 by handling here in the wrapper
    # or if we should expect scales to be passed in.
    """if using_fp8 and x_scale is None and w_scale is None:
        print(f"FP8 scaling in progress...")
        x_fp8, x_scales = triton_quantize_fp8_row(x)
        w_fp8, w_scales = triton_quantize_fp8_row(w)
        x = x_fp8
        w = w_fp8
        x_scale = x_scales
        w_scale = w_scales
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


# ======== Improved Backward =============
def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_tma: bool = False,
    tma_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified backward pass for grouped matrix multiplication with M*G distribution.
    Uses optimized TMA-based implementations for both dx and dw when available.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        x: Input tensor from forward pass, shape [M_total, K]
        w: Weight tensor from forward pass, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        use_tma: Whether to try using TMA acceleration (if available)
        tma_size: Size of TMA descriptor in bytes
        using_fp8: Whether to use FP8 quantization
        x_scale: Scale for x in FP8 mode
        w_scale: Scale for w in FP8 mode
        grad_output_scale: Scale for grad_output in FP8 mode

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    logging.info("Starting unified grouped_gemm_backward")

    # do this once, seems expensive
    NUM_SMS = CudaUtils.get_num_sms()

    # Basic validation
    G = m_sizes.shape[0]
    M_total, K_x = x.shape
    M_grad, N = grad_output.shape
    N_w, K_w = w.shape

    # Check dimensions
    if K_x != K_w:
        raise ValueError(f"K dimension mismatch: x has K={K_x}, w has K={K_w}")
    if M_total != M_grad:
        raise ValueError(
            f"M dimension mismatch: x has M={M_total}, grad_output has M={M_grad}"
        )

    # Check total M matches sum of group sizes
    sum_m_sizes = m_sizes.sum().item()
    if M_total != sum_m_sizes:
        raise ValueError(
            f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"
        )

    # Make sure inputs are contiguous
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    # Check TMA support
    can_use_tma = use_tma and CudaUtils.verify_tma()
    if use_tma and not can_use_tma:
        logging.info("TMA requested but not supported on this device")
        use_tma = False

    # Compute grad_x using optimized implementation
    try:
        logging.info(f"Computing grad_x with optimized kernel (TMA={can_use_tma})")

        # Use TMA-optimized implementation
        grad_x = grouped_gemm_dx_optimized(
            grad_output=grad_output,
            w=w,
            m_sizes=m_sizes,
            num_sms=NUM_SMS,
            tma_size=tma_size,
        )

    except Exception as e:
        logging.error(f"Error in grad_x computation: {e}")
        raise

    # Compute grad_w using optimized implementation
    try:
        logging.info(f"Computing grad_w with optimized kernel (TMA={can_use_tma})")

        grad_w = grouped_gemm_dw_optimized(
            x=x,
            grad_output=grad_output,
            m_sizes=m_sizes,
            num_sms=NUM_SMS,
            tma_size=tma_size,
        )
    except Exception as e:
        logging.error(f"Error in grad_w computation: {e}")
        raise

    return grad_x, grad_w


class GroupedGEMM_mg(torch.autograd.Function):
    """
    Autograd function for GroupedGEMM with M*G grouping.
    Supports both standard and FP8 quantized operations.
    """

    @staticmethod
    def forward(ctx, x, w, m_sizes, use_tma=True, tma_size=128, using_fp8=False):
        """
        Forward pass of GroupedGEMM.

        Args:
            x: Input tensor, shape [M_total, K]
            w: Weight tensor, shape [N, K]
            m_sizes: Tensor of shape [G] containing the size of each group
            use_tma: Whether to try using TMA acceleration (if available)
            tma_size: Size of TMA descriptor in bytes
            using_fp8: Whether to use FP8 quantization

        Returns:
            Output tensor, shape [M_total, N]
        """
        # Handle FP8 quantization if needed
        x_scale = None
        w_scale = None

        if using_fp8:
            # Quantize inputs to FP8 and save scales for backward
            x_fp8, x_scale = triton_quantize_fp8_row(x)
            w_fp8, w_scale = triton_quantize_fp8_row(w)

            # Use quantized tensors for forward pass
            output = grouped_gemm_forward(
                x=x_fp8,
                w=w_fp8,
                m_sizes=m_sizes,
                tma_size=tma_size,
                using_fp8=True,
                x_scale=x_scale,
                w_scale=w_scale,
            )
        else:
            # Use regular forward without quantization
            output = grouped_gemm_forward(
                x=x, w=w, m_sizes=m_sizes, tma_size=tma_size, using_fp8=False
            )

        # Save inputs and parameters for backward pass
        ctx.save_for_backward(x, w, m_sizes)
        ctx.use_tma = use_tma
        ctx.tma_size = tma_size
        ctx.using_fp8 = using_fp8

        # If using FP8, also save the scales
        if using_fp8:
            ctx.save_for_backward(x, w, m_sizes, x_scale, w_scale)
            ctx.x_scale = x_scale
            ctx.w_scale = w_scale
        else:
            ctx.save_for_backward(x, w, m_sizes)
            ctx.x_scale = None
            ctx.w_scale = None

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
                - None: Gradient with respect to tma_size (not differentiable)
                - None: Gradient with respect to using_fp8 (not differentiable)
        """
        # Retrieve saved tensors and parameters
        if ctx.using_fp8:
            x, w, m_sizes, x_scale, w_scale = ctx.saved_tensors
        else:
            x, w, m_sizes = ctx.saved_tensors
            x_scale = None
            w_scale = None

        use_tma = ctx.use_tma
        tma_size = ctx.tma_size
        using_fp8 = ctx.using_fp8

        # Handle FP8 for grad_output if needed
        grad_output_scale = None
        if using_fp8:
            grad_output_fp8, grad_output_scale = triton_quantize_fp8_row(grad_output)
            grad_output = grad_output_fp8

        # Compute gradients using the unified implementation
        grad_x, grad_w = grouped_gemm_backward(
            grad_output=grad_output,
            x=x,
            w=w,
            m_sizes=m_sizes,
            use_tma=use_tma,
            tma_size=tma_size,
            using_fp8=using_fp8,
            x_scale=x_scale,
            w_scale=w_scale,
            grad_output_scale=grad_output_scale,
        )

        # Return gradients for all inputs (None for non-differentiable parameters)
        return grad_x, grad_w, None, None, None, None


def mg_grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_tma: bool = True,
    tma_size: int = 128,
    using_fp8: bool = False,
) -> torch.Tensor:
    """
    Unified differentiable grouped GEMM operation for M*G grouped GEMM.
    Supports both standard precision and FP8 quantized operations.

    Args:
        x: Input tensor, shape [M_total, K]
        w: Weight tensor, shape [N, K]
        m_sizes: Tensor of shape [G] containing the size of each group
        use_tma: Whether to try using TMA acceleration (if available)
        tma_size: Size of TMA descriptor in bytes
        using_fp8: Whether to use FP8 quantization

    Returns:
        Output tensor, shape [M_total, N]
    """
    return GroupedGEMM_mg.apply(x, w, m_sizes, use_tma, tma_size, using_fp8)
