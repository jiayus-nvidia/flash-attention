import pytest

from flex_attn.plan.validation import (
    SM100_STANDARD_HEAD_DIMS,
    is_supported_head_dims,
    validate_call_options,
)


def test_supported_head_dimensions():
    assert len(SM100_STANDARD_HEAD_DIMS) == 256
    assert is_supported_head_dims(8, 8)
    assert is_supported_head_dims(64, 96)
    assert is_supported_head_dims(128, 8)
    assert is_supported_head_dims(64, 64)
    assert is_supported_head_dims(192, 128)
    assert is_supported_head_dims(256, 256)
    assert not is_supported_head_dims(8, 136)
    assert not is_supported_head_dims(72, 76)
    assert not is_supported_head_dims(192, 192)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), "bad"])
def test_call_option_validation(value):
    with pytest.raises(ValueError):
        validate_call_options(
            softmax_scale=value,
            deterministic=False,
            return_lse=False,
        )
