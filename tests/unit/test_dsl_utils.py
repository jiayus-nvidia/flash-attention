from __future__ import annotations

import pytest

from flex_attn.runtime.dsl_utils import _cute_dsl_bulk_copy_self_elects


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
