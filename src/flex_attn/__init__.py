"""FlexAttention CuTe DSL backend."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flex-attn")
except PackageNotFoundError:
    __version__ = "0.0.0"

from flex_attn.interface import flex_attn_func, flex_attn_varlen_func
from flex_attn.plan import MaskPlan, create_mask_plan

__all__ = [
    "MaskPlan",
    "create_mask_plan",
    "flex_attn_func",
    "flex_attn_varlen_func",
]
