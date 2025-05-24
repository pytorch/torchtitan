import os

import pybind11
import torch
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import Extension, setup
from torch.utils import cpp_extension

# Get CUDA and PyTorch paths
cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
torch_include = torch.utils.cpp_extension.include_paths()

# CUTLASS path
cutlass_include = os.environ.get("CUTLASS_PATH", "/path/to/cutlass")  # /include")

print(f"cuda_home: {cuda_home}")
print(f"torch_include: {torch_include}")
print(f"cutlass_include: {cutlass_include}")


# Compiler flags
cxx_flags = [
    "-O3",
    "-std=c++17",
    "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-fPIC",
]

nvcc_flags = [
    "-O3",
    "--std=c++17",
    "-gencode=arch=compute_100,code=sm_100",  # Blackwell architecture
    "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-Xcompiler=-fPIC",
]

include_dirs = [
    pybind11.get_include(),
    cutlass_include,
    f"{cuda_home}/include",
] + torch_include

library_dirs = [
    f"{cuda_home}/lib64",
]

libraries = [
    "cuda",
    "cudart",
    "cublas",
]

ext_modules = [
    cpp_extension.CUDAExtension(
        name="fp8_grouped_gemm_cuda",
        sources=[
            "c4_grouped_gemm.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        language="c++",
    )
]

setup(
    name="fp8-grouped-gemm",
    version="0.1.0",
    author="Me",
    author_email="less@meta.com",
    description="FP8 E4M3 Grouped GEMM using CUTLASS 4.0 Blackwell",
    long_description="PyTorch extension for FP8 E4M3 grouped GEMM operations via CUTLASS on NVIDIA Blackwell",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
