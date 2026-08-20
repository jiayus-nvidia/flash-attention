import pytest
import torch

from flex_attn.plan.validation import (
    PlanGeometry,
    SM100_STANDARD_HEAD_DIMS,
    is_supported_head_dims,
    make_internal_mask_func,
    validate_call_options,
)


def test_local_endpoints_are_internalized_without_public_padding():
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
    geometry = PlanGeometry(
        is_varlen=False,
        arch=100,
        batch_size=2,
        seqlen_q=2,
        seqlen_k=4,
        total_q=4,
        total_k=8,
        max_seqlen_q=2,
        max_seqlen_k=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=128,
        head_dim_v=128,
        hmask=1,
        nfunc=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
    )
    public = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.int32)
    internal = make_internal_mask_func(public, geometry)
    assert internal.shape == (1, 1, 260)
    assert internal[0, 0, :4].tolist() == [1, 2, 7, 8]
    assert internal[0, 0, 4:].count_nonzero() == 0


@pytest.mark.parametrize("value", [float("inf"), float("nan"), "bad"])
def test_call_option_validation(value):
    with pytest.raises(ValueError):
        validate_call_options(
            softmax_scale=value,
            deterministic=False,
            return_lse=False,
        )
