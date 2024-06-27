from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# int4kernel - scale - zeros
setup(
    name='Quant',
    ext_modules=[
        CUDAExtension('Quant', 
            sources = ['Quant.cu',],
            extra_compile_args={
            "cxx": ["-O3", 
                    "-std=c++17"
                   ],
            "nvcc": [
                "-O3",
                "-use_fast_math",
                "-std=c++17",
                "--gpu-architecture=sm_70",
            ],},
            # extra_link_args=["-o./complex2int8_vec"],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
