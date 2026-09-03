from __future__ import annotations

import inspect

import pytest

from flex_attn.runtime.dsl_utils import (
    _cute_dsl_bulk_copy_self_elects,
    _cute_dsl_nvvm_fmax_has_explicit_result_type,
)


def test_nvvm_fmax_result_type_matches_installed_signature():
    from cutlass._mlir.dialects import nvvm

    positional_parameters = tuple(
        parameter
        for parameter in inspect.signature(nvvm.fmax).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    expects_result_type = len(positional_parameters) >= 3
    assert _cute_dsl_nvvm_fmax_has_explicit_result_type() is expects_result_type


@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ((4, 5, 2), False),
        ((4, 6, 0), True),
        ((4, 6, 1), True),
        ((4, 6, 2), False),
        ((4, 6, 3), False),
        ((4, 7, 0), False),
    ),
)
def test_bulk_copy_internal_election_version_window(version, expected):
    assert _cute_dsl_bulk_copy_self_elects(version) is expected
