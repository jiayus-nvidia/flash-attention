# Setup script for magi_to_hstu CUDA extension
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="magi_to_hstu_cuda",
    version="0.1",
    description="CUDA kernel for converting MagiAttention mask to HSTU format",
    ext_modules=[
        CUDAExtension(
            name="magi_to_hstu_cuda",
            sources=[
                "magi_to_hstu_api.cpp",
                "magi_to_hstu.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-DMAX_MAGI_ATTN_SLICES=16",
                    "-gencode", "arch=compute_80,code=sm_80",
                    "-gencode", "arch=compute_90,code=sm_90",
                    "-gencode", "arch=compute_100,code=sm_100",
                    "-gencode", "arch=compute_120,code=sm_120",
                ],
            },
            include_dirs=[this_dir],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
