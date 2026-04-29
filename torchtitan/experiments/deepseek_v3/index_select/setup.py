from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fast_permute_tokens",
    ext_modules=[
        CUDAExtension(
            name="fast_permute_tokens_cuda",
            sources=["fast_permute_tokens_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_90a,code=sm_90a",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
