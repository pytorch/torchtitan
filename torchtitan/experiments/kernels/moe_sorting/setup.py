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
                "nvcc": ["-O3"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    # install_requires=["torch>=1.7.0"],
)
