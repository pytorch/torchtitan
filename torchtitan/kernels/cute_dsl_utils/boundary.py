# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import cutlass.cute as cute
from cutlass import Boolean, range_constexpr


@cute.jit
def lane_boundary(gC: cute.Tensor, tiled_copy: cute.TiledCopy, block_coord, THREAD_ID: int, shape: cute.Shape):
    thr_copy = tiled_copy.get_slice(THREAD_ID)

    bC = gC[block_coord]
    tC = thr_copy.partition_S(bC)

    rC = cute.make_rmem_tensor(tC.shape, Boolean)
    is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

    if not is_within_boundary:
        for i in range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

    return thr_copy, rC, is_within_boundary
