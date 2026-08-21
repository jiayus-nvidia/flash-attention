from __future__ import annotations

import math

import pytest
import torch

import benchmarks.benchmark_flex_attn as benchmark
from benchmarks.benchmark_flex_attn import (
    BackendInputs,
    DOCUMENT_LENGTHS_128K,
    HSTU_DOCUMENT_MAX,
    HSTU_DOCUMENT_MIN,
    MASK_NAMES,
    STANDARD_SEQLEN,
    Workload,
    _causal_block_stats,
    _make_fa4_runner,
    _phase_flops,
    endpoint_visible,
    main,
    make_mask_spec,
    make_torch_mask_mod,
    visible_pair_count,
)


def test_standard_mask_specs_are_valid_interval_unions():
    for name in MASK_NAMES:
        spec = make_mask_spec(name, 2048)
        endpoints = spec.endpoints
        assert endpoints.dtype == torch.int32
        assert endpoints.shape[1] == 2048
        assert endpoints.shape[0] % 2 == 1
        assert endpoints.shape[0] < 33
        assert endpoints.is_contiguous()
        assert torch.all((0 <= endpoints) & (endpoints <= 2048))
        assert torch.all(endpoints[1:] >= endpoints[:-1])
        assert spec.visible_pairs == visible_pair_count(endpoints)
        assert 0 < spec.density <= 1


def test_visible_pair_count_matches_elementwise_predicate():
    seqlen = 257
    for name in MASK_NAMES:
        spec = make_mask_spec(name, seqlen)
        expected = sum(
            endpoint_visible(spec.endpoints, q_idx, kv_idx)
            for q_idx in range(seqlen)
            for kv_idx in range(seqlen)
        )
        assert spec.visible_pairs == expected


def test_torch_mask_mod_matches_endpoint_encoding():
    seqlen = 257
    q_idx = torch.arange(seqlen, dtype=torch.int32)[:, None]
    kv_idx = torch.arange(seqlen, dtype=torch.int32)[None, :]
    for name in MASK_NAMES:
        spec = make_mask_spec(name, seqlen)
        rows = spec.endpoints.t()
        expected = kv_idx < rows[:, 0, None]
        for endpoint_idx in range(1, spec.nfunc, 2):
            expected |= (kv_idx >= rows[:, endpoint_idx, None]) & (
                kv_idx < rows[:, endpoint_idx + 1, None]
            )
        mask_mod = make_torch_mask_mod(spec)
        actual = torch.vmap(lambda q: torch.vmap(lambda k: mask_mod(None, None, q, k))(kv_idx[0]))(
            q_idx[:, 0]
        )
        torch.testing.assert_close(actual, expected)


def test_document_causal_uses_fixed_standard_lengths():
    spec = make_mask_spec("document_causal", STANDARD_SEQLEN)
    assert tuple(spec.details["document_lengths"]) == DOCUMENT_LENGTHS_128K
    assert sum(DOCUMENT_LENGTHS_128K) == STANDARD_SEQLEN
    second_begin = DOCUMENT_LENGTHS_128K[0]
    assert not endpoint_visible(spec.endpoints, second_begin, second_begin - 1)
    assert endpoint_visible(spec.endpoints, second_begin, second_begin)


def _tree_starts(spec):
    offset = 0
    starts = {}
    lengths = spec.details["node_lengths"]
    for node in spec.details["node_order"]:
        starts[node] = offset
        offset += lengths[node]
    return starts


def test_tree_dfs_and_bfs_preserve_logical_ancestry():
    dfs = make_mask_spec("tree_dfs", 8192)
    bfs = make_mask_spec("tree_bfs", 8192)
    assert dfs.visible_pairs == bfs.visible_pairs
    assert dfs.details["node_lengths"] == bfs.details["node_lengths"]
    node = 100
    parent = (node - 1) // 2
    unrelated = 1
    for spec in (dfs, bfs):
        starts = _tree_starts(spec)
        lengths = spec.details["node_lengths"]
        q_idx = starts[node] + lengths[node] - 1
        assert endpoint_visible(spec.endpoints, q_idx, starts[0])
        assert endpoint_visible(spec.endpoints, q_idx, starts[parent])
        assert endpoint_visible(spec.endpoints, q_idx, q_idx)
        assert not endpoint_visible(spec.endpoints, q_idx, starts[unrelated])


def test_hstu_context_target_and_document_isolation():
    spec = make_mask_spec("hstu", STANDARD_SEQLEN)
    contexts = spec.details["context_lengths"]
    targets = spec.details["target_lengths"]
    document_lengths = spec.details["document_lengths"]
    assert contexts == targets
    assert sum(document_lengths) == STANDARD_SEQLEN
    assert all(
        HSTU_DOCUMENT_MIN <= length <= HSTU_DOCUMENT_MAX
        for length in document_lengths
    )
    assert spec.details["document_length_bounds"] == [
        HSTU_DOCUMENT_MIN,
        HSTU_DOCUMENT_MAX,
    ]
    assert spec.details["target_context_ratio"] == 1
    first_context_end = contexts[0]
    first_document_end = first_context_end + targets[0]
    first_target = first_context_end
    second_target = first_target + 1
    assert endpoint_visible(spec.endpoints, first_target, 0)
    assert endpoint_visible(spec.endpoints, first_target, first_target)
    assert not endpoint_visible(spec.endpoints, first_target, second_target)
    assert endpoint_visible(spec.endpoints, second_target, 0)
    assert not endpoint_visible(spec.endpoints, second_target, first_target)
    assert not endpoint_visible(spec.endpoints, first_document_end, 0)
    assert endpoint_visible(spec.endpoints, first_document_end, first_document_end)


def test_active_flop_formulas():
    workload = Workload(seqlen=128)
    pairs = 1234
    head_pairs = workload.num_q_heads * pairs
    assert _phase_flops(workload, pairs, "forward") == 4 * head_pairs * 128
    assert _phase_flops(workload, pairs, "backward") == 10 * head_pairs * 128
    assert _phase_flops(workload, pairs, "combined") == 14 * head_pairs * 128
    assert math.isclose(
        _phase_flops(workload, pairs, "combined") / _phase_flops(workload, pairs, "forward"),
        3.5,
    )


def test_fa4_causal_uses_native_path():
    workload = Workload(seqlen=256)
    shape = (1, workload.seqlen, workload.num_q_heads, workload.head_dim)
    q, k, v = (torch.zeros(shape) for _ in range(3))
    inputs = BackendInputs(q=q, k=k, v=v, dout=torch.zeros_like(v))
    calls = []

    class FakeFa4:
        @staticmethod
        def flash_attn_func(*args, **kwargs):
            calls.append((args, kwargs))
            return args[0]

    def unexpected_block_mask(*args, **kwargs):
        raise AssertionError("native causal must not construct a BlockMask")

    spec = make_mask_spec("causal", workload.seqlen)
    runner = _make_fa4_runner(
        inputs,
        spec,
        spec.endpoints,
        workload,
        FakeFa4(),
        unexpected_block_mask,
    )
    assert runner.build_metadata() is None
    assert runner.block_stats == _causal_block_stats(workload.seqlen, (256, 128))
    assert runner.forward(return_lse=True) is q
    _, kwargs = calls.pop()
    assert kwargs["causal"] is True
    assert kwargs["return_lse"] is True
    assert "mask_mod" not in kwargs
    assert "block_sparse_tensors" not in kwargs
    assert "block_sparse_tensors_bwd" not in kwargs


def test_benchmark_dry_run_does_not_require_cuda_runtime(capsys):
    main(["--dry-run", "--seqlen", "256", "--mask", "causal,hstu"])
    output = capsys.readouterr().out
    assert "masks=2" in output
    assert "causal: nfunc=1" in output
    assert "hstu: nfunc=5" in output


def test_flex_benchmark_accepts_cutlass_dsl_47_without_relaxing_magi(monkeypatch):
    versions = {
        "nvidia-cutlass-dsl": "4.7.0",
        "quack-kernels": "0.5.0",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 3))
    monkeypatch.setattr(benchmark, "_safe_version", versions.__getitem__)

    benchmark._validate_environment(("flex",))
    with pytest.raises(RuntimeError, match="Magi benchmark requires"):
        benchmark._validate_environment(("magi",))
