# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .boundary import lane_boundary
from .elementwise import _load, _store


class ElementwisePackedCUDAKernel:
    def __init__(self, BLOCK_SIZE: int, M: int) -> ElementwisePackedCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE
        self.M = M

    @cute.jit
    def compute(self, *inputs):
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gXs_1: list[cute.Tensor],
        gXs_2: list[cute.Tensor],
        gYs_1: list[cute.Tensor],
        gYs_2: list[cute.Tensor],
        gC_1: cute.Tensor,
        gC_2: cute.Tensor,
        copy_atoms: list[cute.CopyAtom],
        tiled_copys_1: list[cute.TiledCopy],
        tiled_copys_2: list[cute.TiledCopy],
        shape_1: cute.Shape,
        shape_2: cute.Shape,
        TOTAL_TILES: Int32,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        NUM_BLOCKS, _, _ = cute.arch.grid_dim()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        while BLOCK_ID < TOTAL_TILES:
            block_coord = ((None, None), BLOCK_ID)

            _, rC_1, is_within_boundary_1 = lane_boundary(
                gC=gC_1, tiled_copy=tiled_copys_1[0], block_coord=block_coord, THREAD_ID=THREAD_ID, shape=shape_1
            )

            _, rC_2, is_within_boundary_2 = lane_boundary(
                gC=gC_2, tiled_copy=tiled_copys_2[0], block_coord=block_coord, THREAD_ID=THREAD_ID, shape=shape_2
            )

            xs_1 = []
            xs_2 = []
            for xs, gXs, tiled_copys, rC, is_within_boundary in [
                (xs_1, gXs_1, tiled_copys_1, rC_1, is_within_boundary_1),
                (xs_2, gXs_2, tiled_copys_2, rC_2, is_within_boundary_2),
            ]:
                for gX, copy_atom, tiled_copy in zip(gXs, copy_atoms, tiled_copys):
                    xs.append(
                        _load(
                            gX=gX,
                            rC=rC,
                            thr_copy=tiled_copy.get_slice(THREAD_ID),
                            copy_atom=copy_atom,
                            block_coord=block_coord,
                            is_within_boundary=is_within_boundary,
                        )
                    )

            ys_1, ys_2 = self.compute(xs_1, xs_2)

            for ys, gYs, tiled_copys, rC, is_within_boundary in [
                (ys_1, gYs_1, tiled_copys_1, rC_1, is_within_boundary_1),
                (ys_2, gYs_2, tiled_copys_2, rC_2, is_within_boundary_2),
            ]:
                for y, gY, copy_atom, tiled_copy in zip(ys, gYs, copy_atoms, tiled_copys):
                    _store(
                        gY=gY,
                        y=y,
                        rC=rC,
                        thr_copy=tiled_copy.get_slice(THREAD_ID),
                        copy_atom=copy_atom,
                        block_coord=block_coord,
                        is_within_boundary=is_within_boundary,
                    )

            BLOCK_ID += NUM_BLOCKS

    @cute.jit
    def __call__(
        self,
        mXs_1: list[cute.Tensor],
        mXs_2: list[cute.Tensor],
        mYs_1: list[cute.Tensor],
        mYs_2: list[cute.Tensor],
        stream: cuda.CUstream,
    ) -> None:
        vector_size = min([128 // i.element_type.width for i in mXs_1 + mXs_2 + mYs_1 + mYs_2])

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))

        val_layout_1 = cute.make_ordered_layout((self.M, vector_size >> 1), order=(1, 0))
        val_layout_2 = cute.make_ordered_layout((self.M, vector_size), order=(1, 0))

        tiler_mn_1, _ = cute.make_layout_tv(thr_layout, val_layout_1)
        tiler_mn_2, _ = cute.make_layout_tv(thr_layout, val_layout_2)

        shape_1 = (mYs_1 if const_expr(len(mXs_1) == 0) else mXs_1)[0].shape
        shape_2 = (mYs_2 if const_expr(len(mXs_2) == 0) else mXs_2)[0].shape

        mC_1 = cute.make_identity_tensor(shape_1)
        mC_2 = cute.make_identity_tensor(shape_2)

        gC_1 = cute.zipped_divide(mC_1, tiler_mn_1)
        gC_2 = cute.zipped_divide(mC_2, tiler_mn_2)

        gXs_1 = [cute.zipped_divide(i, tiler_mn_1) for i in mXs_1]
        gXs_2 = [cute.zipped_divide(i, tiler_mn_2) for i in mXs_2]
        gYs_1 = [cute.zipped_divide(i, tiler_mn_1) for i in mYs_1]
        gYs_2 = [cute.zipped_divide(i, tiler_mn_2) for i in mYs_2]

        copy_atoms = [
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type)
            for i in (gYs_1 if const_expr(len(gXs_1) == 0) else gXs_1)
        ]

        tiled_copys_1 = [cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_1) for copy_atom in copy_atoms]
        tiled_copys_2 = [cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_2) for copy_atom in copy_atoms]

        NUM_BLOCKS = torch.cuda.get_device_properties(0).multi_processor_count
        TOTAL_TILES = cute.size(gYs_1[0] if const_expr(len(gXs_1) == 0) else gXs_1[0], mode=[1])

        self.kernel(
            gXs_1=gXs_1,
            gXs_2=gXs_2,
            gYs_1=gYs_1,
            gYs_2=gYs_2,
            gC_1=gC_1,
            gC_2=gC_2,
            copy_atoms=copy_atoms,
            tiled_copys_1=tiled_copys_1,
            tiled_copys_2=tiled_copys_2,
            shape_1=shape_1,
            shape_2=shape_2,
            TOTAL_TILES=TOTAL_TILES,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)
