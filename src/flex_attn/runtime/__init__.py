"""Runtime support for FlexAttention CuTe DSL kernels."""

from .arch import SUPPORTED_ARCHES, get_device_arch

__all__ = ["SUPPORTED_ARCHES", "get_device_arch"]
