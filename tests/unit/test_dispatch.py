import cutlass.cute as cute
import pytest
import torch

from flex_attn.runtime.arch import SUPPORTED_ARCHES, get_device_arch
from flex_attn.runtime.dsl_compat import ensure_quack_compat
from flex_attn.runtime.dsl_utils import maybe_contiguous
from flex_attn import interface


def test_supported_dispatch_architectures():
    assert SUPPORTED_ARCHES == (90, 100, 103)
    assert get_device_arch.__name__ == "get_device_arch"


@pytest.mark.parametrize("type_name", ("ThrMma", "ThrCopy"))
def test_quack_cute_dsl_type_compatibility_is_initialized(monkeypatch, type_name):
    monkeypatch.delattr(cute.core, type_name)
    ensure_quack_compat()
    ensure_quack_compat()
    assert getattr(cute.core, type_name) is getattr(cute, type_name)


def test_public_return_lse_contract(monkeypatch):
    class PlanStub:
        def _validate_runtime(self, q, k, v, *, mode):
            return None

    out = object()
    lse = object()
    plan = PlanStub()
    monkeypatch.setattr(interface, "_validate_plan", lambda value: None)
    monkeypatch.setattr(interface.FlexAttnFunc, "apply", lambda *args: (out, lse))
    monkeypatch.setattr(interface.FlexAttnVarlenFunc, "apply", lambda *args: (out, lse))

    for function in (interface.flex_attn_func, interface.flex_attn_varlen_func):
        assert function(None, None, None, mask_plan=plan) is out
        assert function(None, None, None, mask_plan=plan, return_lse=True) == (out, lse)


def test_maybe_contiguous_alignment_contract():
    contiguous = torch.empty((2, 4, 8), dtype=torch.bfloat16)
    assert maybe_contiguous(contiguous) is contiguous

    misaligned = torch.empty(65, dtype=torch.bfloat16)[1:]
    aligned_clone = maybe_contiguous(misaligned)
    assert aligned_clone is not misaligned
    assert aligned_clone.is_contiguous()
    assert aligned_clone.data_ptr() % 16 == 0

    aligned_strided = torch.empty((2, 4, 8), dtype=torch.bfloat16).transpose(0, 1)
    assert maybe_contiguous(aligned_strided) is aligned_strided

    unaligned_strided = torch.empty((2, 4, 10), dtype=torch.bfloat16)[:, :, :8]
    canonical = maybe_contiguous(unaligned_strided)
    assert canonical is not unaligned_strided
    assert canonical.is_contiguous()

    hd256_unaligned_stride = torch.empty((2, 4, 264), dtype=torch.bfloat16)[:, :, :256]
    hd256_canonical = maybe_contiguous(hd256_unaligned_stride, align_bytes=128)
    assert hd256_canonical is not hd256_unaligned_stride
    assert hd256_canonical.is_contiguous()
