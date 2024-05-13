
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name=f"titan_cuda",
    description="Fused RMSNorm",
    keywords="norm ",
    version="1.5.14.2024",
    url="https://github.com/lessw2020/4Bit_AdamW_Triton",
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        CUDAExtension(
            name='titan_cuda',
            sources=['rmsnorm_cuda.cpp', 'rmsnorm_cuda_kernel.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-lineinfo']}
        ),



    ],
)
