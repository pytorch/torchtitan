import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--gpu-architecture=sm_90",  #  H100
        "--use_fast_math",
        "--extended-lambda",
    ],
}

# Source files
sources = [
    "token_sorting_kernels.cu",
]  # "moe_kernel_utils.h"]

setup(
    name="token_sorting_cuda",
    version="0.1",
    description="CUDA-accelerated token sorting for Mixture of Experts models",
    ext_modules=[
        CUDAExtension(
            name="token_sorting_cuda",
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
