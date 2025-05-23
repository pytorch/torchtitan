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

# CUTLASS path - adjust this to your CUTLASS installation
cutlass_include = os.environ.get("CUTLASS_PATH", "/path/to/cutlass/include")

print(f"cuda_home: {cuda_home}")
print(f"torch_include: {torch_include}")
print(f"cutlass_include: {cutlass_include}")
