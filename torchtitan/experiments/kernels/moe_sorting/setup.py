# Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
#  LICENSE file in the root directory of this source tree.


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="token_sorting",
    ext_modules=[
        CUDAExtension(
            name="token_sorting_cuda",
            sources=["token_sorting_kernels.cu"],
            # include_dirs=["."],  # Include current directory for header files
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
