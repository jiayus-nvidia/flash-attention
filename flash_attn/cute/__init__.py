"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .arbitrary_block_sparsity import create_arbitrary_block_sparse_tensors
from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "create_arbitrary_block_sparse_tensors",
    "flash_attn_func",
    "flash_attn_varlen_func",
]
