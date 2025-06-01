import argparse
import functools
from inspect import isclass
from typing import List, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils

import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from dense_gemm import DenseGemmKernel


a = torch.randn(512, 256, device="cuda", dtype=torch.float32)
b = torch.randn(512, 256, device="cuda", dtype=torch.float32)
c = torch.zeros(512, 256, device="cuda", dtype=torch.float32)

mnkl = (512, 512, 256, 1)
cluster_shape = (1, 1)
mma_tiler_mn = (128, 128)
acc_dtype = cutlass.Float32

# Configure gemm kernel
gemm = DenseGemmKernel(
    acc_dtype=acc_dtype,
    use_2cta_instrs=True,
    mma_tiler_mn=mma_tiler_mn,
    cluster_shape_mn=cluster_shape,
    use_tma_store=True,
)

torch_stream = torch.cuda.Stream()
stream = cuda.CUstream(torch_stream.cuda_stream)

# Compile gemm kernel
# compiled_gemm = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream)
