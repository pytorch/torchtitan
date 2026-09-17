# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any, Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .boundary import lane_boundary
from .utils import get_fake_cute_tensor


@cute.jit
def _load(
    gX: cute.Tensor, rC: cute.Tensor, thr_copy, copy_atom: cute.CopyAtom, block_coord, is_within_boundary
) -> cute.TensorSSA:
    bX = gX[block_coord]
    tX = thr_copy.partition_S(bX)
    rX = cute.make_rmem_tensor_like(tX)

    if is_within_boundary:
        cute.copy(copy_atom, tX, rX)
    else:
        cute.copy(copy_atom, tX, rX, pred=rC)

    return rX.load()


@cute.jit
def _store(
    gY: cute.Tensor,
    y: cute.TensorSSA,
    rC: cute.Tensor,
    thr_copy,
    copy_atom: cute.CopyAtom,
    block_coord,
    is_within_boundary,
) -> None:
    bY = gY[block_coord]
    tY = thr_copy.partition_D(bY)
    rY = cute.make_rmem_tensor_like(tY)

    rY.store(y)

    if is_within_boundary:
        cute.copy(copy_atom, rY, tY)
    else:
        cute.copy(copy_atom, rY, tY, pred=rC)


class ElementwiseCUDAKernel:
    def __init__(self, BLOCK_SIZE: int, M: int) -> ElementwiseCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE
        self.M = M

    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gXs: list[cute.Tensor],
        gYs: list[cute.Tensor],
        gC: cute.Tensor,
        copy_atom_Xs: list[cute.CopyAtom],
        copy_atom_Ys: list[cute.CopyAtom],
        tiled_copy_Xs: list[cute.TiledCopy],
        shape: cute.Shape,
        TOTAL_TILES: Int32,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        NUM_BLOCKS, _, _ = cute.arch.grid_dim()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        while BLOCK_ID < TOTAL_TILES:
            block_coord = ((None, None), BLOCK_ID)

            thr_copy, rC, is_within_boundary = lane_boundary(
                gC=gC, tiled_copy=tiled_copy_Xs[0], block_coord=block_coord, THREAD_ID=THREAD_ID, shape=shape
            )

            xs = [
                _load(
                    gX=gX,
                    rC=rC,
                    thr_copy=thr_copy,
                    copy_atom=copy_atom,
                    block_coord=block_coord,
                    is_within_boundary=is_within_boundary,
                )
                for gX, copy_atom in zip(gXs, copy_atom_Xs)
            ]

            ys = self.compute(xs)

            for y, gY, copy_atom in zip(ys, gYs, copy_atom_Ys):
                _store(
                    gY=gY,
                    y=y,
                    rC=rC,
                    thr_copy=thr_copy,
                    copy_atom=copy_atom,
                    block_coord=block_coord,
                    is_within_boundary=is_within_boundary,
                )

            BLOCK_ID += NUM_BLOCKS

    @cute.jit
    def __call__(self, mXs: list[cute.Tensor], mYs: list[cute.Tensor], stream: cuda.CUstream) -> None:
        vector_size = min([128 // i.element_type.width for i in mXs + mYs])

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((self.M, vector_size), order=(1, 0))
        tiler_mn, _ = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mXs[0].shape)
        gC = cute.zipped_divide(mC, tiler_mn)

        gXs = [cute.zipped_divide(i, tiler_mn) for i in mXs]
        gYs = [cute.zipped_divide(i, tiler_mn) for i in mYs]

        copy_atom_Xs = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type) for i in gXs]
        copy_atom_Ys = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type) for i in gYs]

        tiled_copy_Xs = [cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout) for copy_atom in copy_atom_Xs]

        NUM_BLOCKS = torch.cuda.get_device_properties(0).multi_processor_count
        TOTAL_TILES = cute.size(gC, mode=[1])

        self.kernel(
            gXs=gXs,
            gYs=gYs,
            gC=gC,
            copy_atom_Xs=copy_atom_Xs,
            copy_atom_Ys=copy_atom_Ys,
            tiled_copy_Xs=tiled_copy_Xs,
            shape=mXs[0].shape,
            TOTAL_TILES=TOTAL_TILES,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


def get_compiled_elementwise_cuda_kernel(
    caller_op: Callable,
    key: Any,
    kernel_class: type,
    example_tensors_list: tuple[tuple[torch.Tensor]],
    divisibility_list_list: tuple[tuple[int]],
    stream: cuda.CUstream,
) -> ElementwiseCUDAKernel:
    if not hasattr(caller_op, "cache"):
        caller_op.cache = {}

    kernel = caller_op.cache.get(key)

    if kernel is None:
        fake_tensors = []
        for example_tensors, divisibility_list in zip(example_tensors_list, divisibility_list_list):
            if example_tensors is None:
                fake_tensors.append(None)
                continue

            _fake_tensors = []
            for tensor, div in zip(example_tensors, divisibility_list):
                if tensor is None:
                    _fake_tensors.append(None)
                    continue

                tensor = get_fake_cute_tensor(
                    dtype=tensor.dtype, shape=(cute.sym_int(), cute.sym_int(divisibility=div)), divisibility=div
                )

                _fake_tensors.append(tensor)

            fake_tensors.append(_fake_tensors)

        kernel = cute.compile(kernel_class(), *fake_tensors, stream, options="--enable-tvm-ffi")
        caller_op.cache[key] = kernel

    return kernel
