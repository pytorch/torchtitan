# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from ..math import get_powers_of_2
from ..cute_dsl_utils import get_cute_dtype_from_torch_dtype


def get_fake_cute_tensor(
    dtype: torch.dtype, shape: tuple[int], divisibility: int = 1, leading_dim: int = -1
) -> cute.Tensor:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim

    stride = tuple(1 if i == leading_dim else cute.sym_int64(divisibility=divisibility) for i in range(len(shape)))

    tensor = cute.runtime.make_fake_tensor(
        get_cute_dtype_from_torch_dtype(dtype), shape, stride=stride, assumed_align=divisibility * dtype.itemsize
    )

    return tensor


def get_alignment(x: torch.Tensor) -> int:
    x = x.data_ptr()

    alignment = 4
    for i in get_powers_of_2(4, 16):
        if x % i != 0:
            break
        else:
            alignment = i

    return alignment


def torch_tensor_to_cute_tensor(x: torch.Tensor, leading_dim: int) -> cute.Tensor:
    if leading_dim < 0:
        leading_dim += x.dim()

    x = x.detach()
    x = from_dlpack(x, assumed_align=get_alignment(x))

    # not sure if there is a better way to check PyTorch's broadcasting
    if x.stride[leading_dim] == 0:
        leading_dim = None

    x = x.mark_layout_dynamic(leading_dim=leading_dim)

    return x
