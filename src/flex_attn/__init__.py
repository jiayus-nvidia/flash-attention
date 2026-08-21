"""FlexAttention CuTe DSL backend."""

from importlib.metadata import PackageNotFoundError, version

from flex_attn.runtime.dsl_compat import ensure_quack_compat as _ensure_quack_compat


_ensure_quack_compat()
del _ensure_quack_compat

try:
    __version__ = version("flex-attn")
except PackageNotFoundError:
    __version__ = "0.0.0"

# These imports must follow the Quack compatibility initialization above.
from flex_attn.interface import flex_attn_func, flex_attn_varlen_func  # noqa: E402
from flex_attn.plan import MaskPlan, create_mask_plan  # noqa: E402

__all__ = [
    "MaskPlan",
    "create_mask_plan",
    "flex_attn_func",
    "flex_attn_varlen_func",
]
