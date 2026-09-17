# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Numeric, Uint32


_TORCH_DTYPE_TO_CUTE_DTYPE_MAPPING = {
    # floating point dtypes
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    # integer dtypes
    torch.int32: Int32,
    torch.int64: Int64,
    torch.uint32: Uint32,
}


def get_cute_dtype_from_torch_dtype(dtype: torch.dtype) -> type[Numeric]:
    return _TORCH_DTYPE_TO_CUTE_DTYPE_MAPPING[dtype]
