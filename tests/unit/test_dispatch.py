import cutlass.cute as cute
import pytest

from flex_attn.runtime.arch import SUPPORTED_ARCHES, get_device_arch
from flex_attn.runtime.dsl_compat import ensure_quack_compat
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
