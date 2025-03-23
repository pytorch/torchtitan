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

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            # TMA Store prep
            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start * N,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_dtype,
            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

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
                if USE_FP8:
                    # Scale the accumulator
                    # Load the scales and apply
                    offs_am = tile_m_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_index * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                    a_scale = tl.load(
                        a_scale_ptr + M_start + offs_am[:, None],
                        mask=offs_am[:, None] < m_size,
                    )
                    b_scale = tl.load(
                        b_scale_ptr + offs_bn[None, :],
                        mask=offs_bn[None, :] < n_size,
                    )

                    accumulator = accumulator.to(tl.float32) * a_scale * b_scale

                # Store using TMA

                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

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
    For bc, we never use TMA Load and TMA Store
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


# ---- dx ----
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
    a_scale_ptr,  # Optional scale for grad_output in FP8
    b_scale_ptr,  # Optional scale for w in FP8
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

                # Apply FP8 scaling if needed
                if USE_FP8:
                    # Load the scales and apply
                    offs_am = tile_m_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bk = tile_k_index * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                    a_scale = tl.load(
                        a_scale_ptr + M_start + offs_am[:, None],
                        mask=offs_am[:, None] < m_size,
                    )
                    b_scale = tl.load(
                        b_scale_ptr + offs_bk[None, :],
                        mask=offs_bk[None, :] < K,
                    )

                    accumulator = accumulator.to(tl.float32) * a_scale * b_scale

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


def grouped_gemm_dx_optimized(
    grad_output: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    tma_size: int = 128,
    using_fp8: bool = False,
    grad_output_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Optimized backward pass for computing gradient with respect to input (dx)
    using TMA patterns similar to the forward pass.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        w: Weight tensor, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor
        using_fp8: Whether to use FP8 quantization
        grad_output_scale: Scale for grad_output in FP8 mode
        w_scale: Scale for w in FP8 mode

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
        grad_output_scale: Scale for grad_output in FP8 mode
        w_scale: Scale for w in FP8 mode

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

    NUM_SMS = CudaUtils.get_num_sms()
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

    # Handle FP8 scaling if needed
    if using_fp8 and (grad_output_scale is None or w_scale is None):
        print("FP8 scaling in progress...")
        grad_output_fp8, grad_output_scales = triton_quantize_fp8_row(grad_output)
        w_fp8, w_scales = triton_quantize_fp8_row(w)
        grad_output = grad_output_fp8
        w = w_fp8
        grad_output_scale = grad_output_scales
        w_scale = w_scales

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
        grad_output_scale,
        w_scale,
        G,
        M_BUCKET,
        N_grad,  # N dimension is now the reduction dimension
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
        USE_FP8=using_fp8,
    )

    return grad_x


# ======== End Triton kernels ========


# ======== DW Experiment =================
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
    x_scale_ptr,  # Optional scale for x in FP8
    grad_output_scale_ptr,  # Optional scale for grad_output in FP8
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

    # Apply FP8 scaling if needed
    if USE_FP8:
        # Load the scales and apply
        offs_n_scale = offs_n + tl.arange(0, BLOCK_SIZE_N)
        offs_k_scale = offs_k + tl.arange(0, BLOCK_SIZE_K)

        x_scale = tl.load(
            x_scale_ptr + offs_k_scale[None, :],
            mask=k_mask[None, :],
        )

        grad_output_scale = tl.load(
            grad_output_scale_ptr + offs_n_scale[:, None],
            mask=n_mask[:, None],
        )

        accumulator = accumulator.to(tl.float32) * x_scale * grad_output_scale

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


def grouped_gemm_dw_optimized(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    m_sizes: torch.Tensor,
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
    can_use_tma = CudaUtils.verify_tma()
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
    NUM_SMS = CudaUtils.get_num_sms()

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
        x_scale,
        grad_output_scale,
        G,
        M_BUCKET,
        N,
        K_x,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
        USE_FP8=using_fp8,
        # BLOCK_SIZE_N=BLOCK_SIZE_N,
        # BLOCK_SIZE_K=BLOCK_SIZE_K,
        # BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return grad_w


# ======== End DW Experiment =============

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
    FP8 support is triggered by passing x_scale and w_scale tensors.

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

    if x_scale is not None and w_scale is not None:
        using_fp8 = True

    # TODO: not clear if we should integrate FP8 by handling here in the wrapper
    # or if we should expect scales to be passed in.
    if using_fp8 and x_scale is None and w_scale is None:
        print(f"FP8 scaling in progress...")
        x_fp8, x_scales = triton_quantize_fp8_row(x)
        w_fp8, w_scales = triton_quantize_fp8_row(w)
        x = x_fp8
        w = w_fp8
        x_scale = x_scales
        w_scale = w_scales

    print(f"{x_scale=}")
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
    print(f"{M_BUCKET=}")
    _kernel_mg_forward_tma[grid](  #
        # _kernel_grouped_gemm_flat_indexing[grid](  # _kernel_grouped_gemm[grid](
        desc_x,
        desc_w,
        y,
        workspace,
        m_sizes,
        x_scale,
        w_scale,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
        USE_FP8=using_fp8,
    )

    return y


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
            grad_x = grouped_gemm_dx_optimized(
                grad_output,
                w,
                m_sizes,
            )
            # Fixed grid size based on SM count
            """grid = (NUM_SMS,)

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
            """

            logging.info("Kernel run success: grad_x computation successful")
        except Exception as e:
            logging.error(f"Error in backward_dx kernel: {e}")
            raise RuntimeError(f"Error in backward_dx kernel: {e}")

        try:
            logging.info("Computing grad_w with experimental dw tma kernel")
            grad_w = grouped_gemm_dw_optimized(
                x,
                grad_output,
                m_sizes,
            )

            # For grad_w, use a grid with one thread block per output tile
            # grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(K, BLOCK_SIZE_K))

            """_kernel_grouped_gemm_backward_dw_scheduled[grid](
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
            """

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
    Autograd function for GroupedGEMM with M*G grouping.

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
        # from mg_forward import group_gemm_forward

        output = grouped_gemm_forward(x, w, m_sizes)

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
    Differentiable grouped GEMM operation for M*G grouped GeMM.

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
