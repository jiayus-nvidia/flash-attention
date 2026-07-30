# Setup script for create_block_mask CUDA extension
import os
import runpy
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
# Project root is kept on the include path for repo-local headers.
project_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))

# ============================================================================
# Optimization Control via Environment Variables (for benchmarking)
# ============================================================================
# Set these environment variables to enable/disable specific optimizations:
#
# DISABLE_REG_CACHE=1         - Disable register caching for func_tensor values
#                               Expected performance impact: ~4% slower on dense patterns
#
# DISABLE_KV_RANGE_OPT=1      - Disable kv_range optimization for both Q2K and K2Q
#                               Default: Q2K uses inline kv_range, K2Q uses precompute
#                               With this flag: both iterate all blocks (no optimization)
#                               Expected: significant slowdown on sparse patterns
#
# DISABLE_BLOCK_SIZE_TEMPLATE=1 - Disable block size template specialization
#                                  Uses runtime block sizes instead of compile-time constants
#
# ENABLE_WARP_LEVEL_OPT=1     - Enable warp-level optimization (EXPERIMENTAL)
#                               Uses 32 threads per block, each processing Q_BLOCK_SIZE/32 q_tokens
#                               Avoids block-level sync and shared memory for state reduction
#                               Requirements: Q_BLOCK_SIZE % 32 == 0, n_func in [1,33] (odd)
#
# Example usage:
#   # Build with all optimizations (default: Q2K inline + K2Q precompute)
#   python setup.py build_ext --inplace
#
#   # Build without register caching (baseline for reg cache benchmark)
#   DISABLE_REG_CACHE=1 python setup.py build_ext --inplace
#
#   # Build without kv range optimization (baseline for kv range benchmark)
#   DISABLE_KV_RANGE_OPT=1 python setup.py build_ext --inplace
#
#   # Build without block size template (baseline for block size template benchmark)
#   DISABLE_BLOCK_SIZE_TEMPLATE=1 python setup.py build_ext --inplace
#
#   # Build with warp-level optimization (experimental)
#   ENABLE_WARP_LEVEL_OPT=1 python setup.py build_ext --inplace
#
#   # Build without any optimizations (full baseline)
#   DISABLE_REG_CACHE=1 DISABLE_KV_RANGE_OPT=1 DISABLE_BLOCK_SIZE_TEMPLATE=1 python setup.py build_ext --inplace
# ============================================================================

import torch

def get_cuda_gencode_flags():
    """Detect current GPU architecture and return appropriate -gencode flags."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot determine target architecture")
    capability = torch.cuda.get_device_capability()
    arch = capability[0] * 10 + capability[1]
    return ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]

cuda_gencode_flags = get_cuda_gencode_flags()

extra_defines = []
disable_reg_cache = os.environ.get("DISABLE_REG_CACHE")
disable_kv_range_opt = os.environ.get("DISABLE_KV_RANGE_OPT")
disable_block_size_template = os.environ.get("DISABLE_BLOCK_SIZE_TEMPLATE")
enable_warp_level_opt = os.environ.get("ENABLE_WARP_LEVEL_OPT")

# Keep the auto K2Q tile selector in lockstep with a nearby Hopper extension
# build, including intentionally cropped head-dimension builds.  Explicit
# environment values take precedence; a generated config is used when present.
flash_build_flags = {}
flash_config_path = os.path.join(project_root, "hopper", "flash_attn_config.py")
if os.path.exists(flash_config_path):
    flash_build_flags = runpy.run_path(flash_config_path).get("CONFIG", {}).get(
        "build_flags", {}
    )
for head_dim in (64, 96, 128, 192, 256):
    env_name = f"FLASH_ATTENTION_DISABLE_HDIM{head_dim}"
    config_name = f"FLASHATTENTION_DISABLE_HDIM{head_dim}"
    env_value = os.environ.get(env_name)
    disabled = (
        env_value.upper() == "TRUE"
        if env_value is not None
        else flash_build_flags.get(config_name, False)
    )
    if disabled:
        extra_defines.append(f"-DFLASHATTENTION_DISABLE_HDIM{head_dim}")

if disable_reg_cache:
    extra_defines.append("-DDISABLE_REG_CACHE")
if disable_kv_range_opt:
    extra_defines.append("-DDISABLE_KV_RANGE_OPT")
if disable_block_size_template:
    extra_defines.append("-DDISABLE_BLOCK_SIZE_TEMPLATE")
if enable_warp_level_opt:
    extra_defines.append("-DENABLE_WARP_LEVEL_OPT")

# Print configuration summary
print("=" * 70)
print("Building create_block_mask_cuda with configuration:")
print("=" * 70)
print(f"  [{'x' if not disable_reg_cache else ' '}] Register caching: {'enabled' if not disable_reg_cache else 'DISABLED'}")
print(f"  [{'x' if not disable_kv_range_opt else ' '}] KV range optimization: {'enabled (Q2K inline + K2Q precompute)' if not disable_kv_range_opt else 'DISABLED'}")
print(f"  [{'x' if not disable_block_size_template else ' '}] Block size template: {'enabled' if not disable_block_size_template else 'DISABLED'}")
print(f"  [{' ' if not enable_warp_level_opt else 'x'}] Warp-level optimization: {'disabled' if not enable_warp_level_opt else 'ENABLED (experimental)'}")
print("=" * 70)

setup(
    name="create_block_mask_cuda",
    version="0.1",
    description="CUDA kernels for creating block masks (arbitrary function encoding)",
    ext_modules=[
        CUDAExtension(
            name="create_block_mask_cuda",
            sources=[
                "create_block_mask_api.cpp",
                "create_block_mask.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + extra_defines,
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ] + cuda_gencode_flags + extra_defines,
            },
            include_dirs=[this_dir, project_root],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
