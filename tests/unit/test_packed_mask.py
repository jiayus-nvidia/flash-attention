import pytest
import torch

from flex_attn.kernels.common.packed_mask import interval_endpoints_to_dense
from tests.datas.mask_func_cases import make_benchmark_mask_func


def test_interval_endpoint_semantics():
    endpoints = torch.tensor([2, 4, 7, 9, 10], dtype=torch.int32)
    dense = interval_endpoints_to_dense(endpoints, 12)
    assert dense.nonzero().flatten().tolist() == [0, 1, 4, 5, 6, 9]


def test_interval_endpoint_validation():
    with pytest.raises(ValueError):
        interval_endpoints_to_dense(torch.tensor([3, 2, 4], dtype=torch.int32), 8)


def test_benchmark_local_window_endpoints():
    endpoints = make_benchmark_mask_func(
        mask_kind="local",
        batch_size=1,
        seqlen_q=1024,
        seqlen_k=1024,
        device="cpu",
    )
    q_index = 700
    assert endpoints[0, :, q_index].tolist() == [0, q_index - 512, q_index + 1]
