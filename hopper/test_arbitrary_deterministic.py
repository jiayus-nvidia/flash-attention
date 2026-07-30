import math
import multiprocessing
import os
import traceback

import pytest
import torch

import flash_attn_interface as fai
from flash_attn_interface import (
    LinearBlockSparseTensors,
    compute_dq_write_order_from_linear_csr,
    flash_attn_func,
    prepare_deterministic_k2q_metadata,
)


def _is_supported_arch():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((8, 0), (8, 6), (8, 9), (9, 0))


supported_arch_only = pytest.mark.skipif(
    not _is_supported_arch(),
    reason="arbitrary deterministic tests require an SM80, SM86, SM89, or SM90 GPU",
)


try:
    from flash_attn_config import CONFIG as BUILD_CONFIG
except ImportError:
    BUILD_CONFIG = {"build_flags": {}}


def _build_flags():
    return BUILD_CONFIG.get("build_flags", {})


def _env_true(*names):
    return any(os.getenv(name, "FALSE").upper() == "TRUE" for name in names)


def _require_test_build():
    flags = _build_flags()
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (9, 0):
        if _env_true("FLASHATTENTION_DISABLE_SM90", "FLASH_ATTENTION_DISABLE_SM90"):
            pytest.skip("SM90 kernels were not compiled")
        if flags.get("FLASHATTENTION_DISABLE_SM90", False):
            pytest.skip("SM90 kernels were not compiled")
    else:
        if _env_true(
            "FLASH_ATTENTION_DISABLE_SM80",
            "FLASHATTENTION_DISABLE_SM80",
            "FLASHATTENTION_DISABLE_SM8x",
        ):
            pytest.skip("SM8x kernels were not compiled")
        if flags.get("FLASHATTENTION_DISABLE_SM8x", False):
            pytest.skip("SM8x kernels were not compiled")
    if _env_true("FLASHATTENTION_DISABLE_ARBITRARY", "FLASH_ATTENTION_DISABLE_ARBITRARY"):
        pytest.skip("arbitrary kernels were not compiled")
    if flags.get("FLASHATTENTION_DISABLE_ARBITRARY", False):
        pytest.skip("arbitrary kernels were not compiled")
    if _env_true("FLASHATTENTION_DISABLE_BACKWARD", "FLASH_ATTENTION_DISABLE_BACKWARD"):
        pytest.skip("backward kernels were not compiled")
    if flags.get("FLASHATTENTION_DISABLE_BACKWARD", False):
        pytest.skip("backward kernels were not compiled")
    compiled_nfunc = None
    for env_name in ("FLASH_ATTENTION_NUM_FUNC", "FLASHATTENTION_NUM_FUNC"):
        if env_name in os.environ:
            compiled_nfunc = {
                int(value.strip())
                for value in os.environ[env_name].split(",")
                if value.strip()
            }
            break
    if compiled_nfunc is None:
        compiled_nfunc = flags.get("FLASHATTENTION_NUM_FUNC")
    if compiled_nfunc is not None and 1 not in compiled_nfunc:
        pytest.skip("arbitrary func_num=1 kernels were not compiled")


def _compiled_head_dim():
    flags = _build_flags()
    for head_dim in (64, 96, 128, 192):
        disabled = _env_true(
            f"FLASHATTENTION_DISABLE_HDIM{head_dim}",
            f"FLASH_ATTENTION_DISABLE_HDIM{head_dim}",
        ) or flags.get(f"FLASHATTENTION_DISABLE_HDIM{head_dim}", False)
        if not disabled:
            return head_dim
    pytest.skip("no compatible arbitrary deterministic head-dimension bucket was compiled")


def _bwd_block_size_for_arch(head_dim):
    major, minor = torch.cuda.get_device_capability()
    if major == 8:
        return fai._dense_bwd_block_size_sm8x(
            head_dim, sm86_or_89=minor in (6, 9)
        )
    return fai._dense_bwd_block_size_sm90(head_dim)


def _arch_label():
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def test_raw_bwd_schema_keeps_validation_flag_last():
    """K2Q tile/SPT are a caller-side contract, not duplicated scalar inputs."""
    arguments = torch.ops.flash_attn_3.bwd.default._schema.arguments
    tail = [
        argument.name
        for argument in arguments[-3:]
    ]
    assert tail == [
        "block_sparse_dq_write_order",
        "block_sparse_dq_write_order_full",
        "unsafe_skip_block_sparse_semantic_validation",
    ]
    names = {argument.name for argument in arguments}
    assert names.isdisjoint(
        {"block_sparse_block_m", "block_sparse_block_n", "block_sparse_spt"}
    )


def test_python_metadata_contract_omits_tile_certificate():
    assert LinearBlockSparseTensors._fields[-2:] == (
        "dq_write_order",
        "dq_write_order_full",
    )
    assert tuple(
        argument.name
        for argument in torch.ops.flash_attn_3._flash_attn_forward.default._schema.arguments[-3:]
    ) == (
        "k2q_block_sparse_dq_write_order",
        "k2q_block_sparse_dq_write_order_full",
        "deterministic",
    )
    assert set(
        argument.name
        for argument in torch.ops.flash_attn_3._flash_attn_forward.default._schema.arguments
    ).isdisjoint(
        {
            "k2q_block_sparse_block_m",
            "k2q_block_sparse_block_n",
            "k2q_block_sparse_spt",
        }
    )
    assert prepare_deterministic_k2q_metadata.__code__.co_argcount == 1


def _linear_csr(partial_rows, full_rows, device):
    assert len(partial_rows) == len(full_rows)

    def _pack(rows):
        counts = [len(row) for row in rows]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        indices = [index for row in rows for index in row]
        return (
            torch.tensor(counts, dtype=torch.int32, device=device).view(1, 1, -1),
            torch.tensor(offsets, dtype=torch.int32, device=device),
            torch.tensor(indices, dtype=torch.int32, device=device),
        )

    mask_cnt, mask_offset, mask_idx = _pack(partial_rows)
    full_cnt, full_offset, full_idx = _pack(full_rows)
    return LinearBlockSparseTensors(
        mask_block_cnt=mask_cnt,
        mask_block_offset=mask_offset,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_offset=full_offset,
        full_block_idx=full_idx,
    )


def _make_prefix_mask(seqlen_q, block_n, device):
    """n0 is full, n1 is partial, and n2 is empty for every q tile."""
    arbitrary_func = torch.zeros(
        1, 1, 1, seqlen_q + 256, dtype=torch.int32, device=device
    )
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device=device)
    partial_width = block_n // 2
    arbitrary_func[0, 0, 0, :seqlen_q] = (
        block_n + block_n // 4 + q_idx.remainder(partial_width)
    )
    return arbitrary_func


def _make_runtime_sparse_pair(q, v, seqlen_q, seqlen_k):
    fwd_block_m, fwd_block_n = fai._dense_fwd_block_size(q, v)
    bwd_block_m, bwd_block_n = fai._dense_bwd_block_size(q, v)
    assert 2 * bwd_block_n < seqlen_k <= 3 * bwd_block_n

    num_fwd_m = math.ceil(seqlen_q / fwd_block_m)
    num_fwd_n = math.ceil(seqlen_k / fwd_block_n)
    num_bwd_m = math.ceil(seqlen_q / bwd_block_m)
    num_n = math.ceil(seqlen_k / bwd_block_n)
    assert num_n == 3

    # Forward and backward can use different N tiles on SM8x.
    # Marking every forward edge partial is conservative and lets the element
    # mask decide validity, while backward retains the mixed/full/hole shape
    # that exercises the deterministic turnstile.
    q2k = _linear_csr(
        partial_rows=[list(range(num_fwd_n)) for _ in range(num_fwd_m)],
        full_rows=[[] for _ in range(num_fwd_m)],
        device=q.device,
    )
    k2q = _linear_csr(
        partial_rows=[[], list(range(num_bwd_m)), []],
        full_rows=[list(range(num_bwd_m)), [], []],
        device=q.device,
    )
    return q2k, prepare_deterministic_k2q_metadata(k2q)


def _attention_ref(q, k, v, arbitrary_func):
    q_ref, k_ref, v_ref = q.float(), k.float(), v.float()
    if q_ref.shape[2] != k_ref.shape[2]:
        repeats = q_ref.shape[2] // k_ref.shape[2]
        k_ref = k_ref.repeat_interleave(repeats, dim=2)
        v_ref = v_ref.repeat_interleave(repeats, dim=2)
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", q_ref * (1.0 / math.sqrt(q.shape[-1])), k_ref
    )
    cols = torch.arange(k.shape[1], device=q.device, dtype=torch.int32)
    valid = cols.view(1, 1, 1, -1) < arbitrary_func[
        :, :, 0, : q.shape[1]
    ].unsqueeze(-1)
    scores = scores.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v_ref).to(q.dtype)


def _raw_forward(q, k, v, arbitrary_func, q2k, k2q):
    with torch.no_grad():
        out, softmax_lse, *_ = fai._flash_attn_forward(
            q,
            k,
            v,
            block_sparse_mask_cnt=q2k.mask_block_cnt,
            block_sparse_mask_offset=q2k.mask_block_offset,
            block_sparse_mask_idx=q2k.mask_block_idx,
            block_sparse_full_cnt=q2k.full_block_cnt,
            block_sparse_full_offset=q2k.full_block_offset,
            block_sparse_full_idx=q2k.full_block_idx,
            arbitrary_func=arbitrary_func,
            k2q_block_sparse_mask_cnt=k2q.mask_block_cnt,
            k2q_block_sparse_mask_offset=k2q.mask_block_offset,
            k2q_block_sparse_mask_idx=k2q.mask_block_idx,
            k2q_block_sparse_full_cnt=k2q.full_block_cnt,
            k2q_block_sparse_full_offset=k2q.full_block_offset,
            k2q_block_sparse_full_idx=k2q.full_block_idx,
            k2q_block_sparse_dq_write_order=k2q.dq_write_order,
            k2q_block_sparse_dq_write_order_full=k2q.dq_write_order_full,
            deterministic=True,
        )
    return out, softmax_lse


def _run_raw_accum_repeatability(
    q, k, v, dout, arbitrary_func, q2k, k2q, repeats
):
    """Check the valid FP32 accumulation regions exposed by the raw C++ op."""
    out, softmax_lse = _raw_forward(q, k, v, arbitrary_func, q2k, k2q)
    baseline = None
    for repeat in range(repeats):
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _, _, dq_accum, dk_accum, dv_accum = torch.ops.flash_attn_3.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            deterministic=True,
            arbitrary_func=arbitrary_func,
            block_sparse_mask_cnt=k2q.mask_block_cnt,
            block_sparse_mask_offset=k2q.mask_block_offset,
            block_sparse_mask_idx=k2q.mask_block_idx,
            block_sparse_full_cnt=k2q.full_block_cnt,
            block_sparse_full_offset=k2q.full_block_offset,
            block_sparse_full_idx=k2q.full_block_idx,
            block_sparse_dq_write_order=k2q.dq_write_order,
            block_sparse_dq_write_order_full=k2q.dq_write_order_full,
            # Exercise the native C++ semantic validator on the first launch.
            # Later repetitions can use the already validated metadata.
            unsafe_skip_block_sparse_semantic_validation=repeat != 0,
        )
        head_dim = q.shape[-1]
        names = ["dQaccum"]
        valid = [
            dq_accum.view(q.shape[0], q.shape[2], -1, head_dim)[
                :, :, : q.shape[1]
            ]
        ]
        if dk_accum is not None:
            names.append("dKaccum")
            valid.append(
                dk_accum.view(k.shape[0], k.shape[2], -1, head_dim)[
                    :, :, : k.shape[1]
                ]
            )
        if dv_accum is not None:
            names.append("dVaccum")
            valid.append(
                dv_accum.view(v.shape[0], v.shape[2], -1, head_dim)[
                    :, :, : v.shape[1]
                ]
            )
        names.extend(("dQ", "dK", "dV"))
        valid.extend((dq, dk, dv))
        if baseline is None:
            baseline = tuple(tensor.clone() for tensor in valid)
        else:
            for name, actual, expected in zip(names, valid, baseline):
                assert torch.equal(actual, expected), (
                    f"{name} changed on raw deterministic repeat {repeat}"
                )


def _make_case(kv_mode):
    _require_test_build()
    torch.manual_seed(20260729 + {"mha": 0, "gqa": 1, "mqa": 2}[kv_mode])
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen_q, heads = 1, 257, 4
    kv_heads = {"mha": heads, "gqa": heads // 2, "mqa": 1}[kv_mode]
    head_dim = _compiled_head_dim()
    _, block_n = _bwd_block_size_for_arch(head_dim)
    seqlen_k = 2 * block_n + 65
    q = torch.randn(
        batch,
        seqlen_q,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        seqlen_k,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    arbitrary_func = _make_prefix_mask(seqlen_q, block_n, device)
    q2k, k2q = _make_runtime_sparse_pair(q, v, seqlen_q, seqlen_k)
    return q, k, v, arbitrary_func, q2k, k2q


@supported_arch_only
def test_rank_combines_partial_full_across_sparse_n_holes():
    """Ranks are compact per m across both CSR lists, not per list or raw n."""
    _require_test_build()
    k2q = _linear_csr(
        # n0 contributes through full, n1 is an internal hole, and n2
        # contributes through partial.  m1/m2 have only one contributor,
        # while m0/m3 combine contributors from both lists.
        partial_rows=[[], [], [0, 2, 3]],
        full_rows=[[0, 1, 3], [], []],
        device="cuda",
    )
    partial_rank, full_rank = compute_dq_write_order_from_linear_csr(k2q)
    torch.testing.assert_close(
        partial_rank, torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        full_rank, torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
    )


@pytest.mark.parametrize("capability", [(7, 5), (8, 7), (10, 0)])
def test_python_entry_rejects_unsupported_deterministic_arch(monkeypatch, capability):
    q = torch.empty(
        1, 1, 1, 64, dtype=torch.bfloat16, device="meta", requires_grad=True
    )
    k = torch.empty_like(q, requires_grad=True)
    v = torch.empty_like(q, requires_grad=True)
    arbitrary_func = torch.zeros(
        1, 1, 1, 257, dtype=torch.int32, device="meta"
    )
    monkeypatch.setattr(fai, "_device_capability_major", lambda _device: capability)
    with pytest.raises(
        NotImplementedError, match="only on SM80, SM86, SM89, and SM90"
    ):
        fai._prepare_arbitrary_block_sparse(
            q,
            k,
            v,
            arbitrary_func,
            None,
            None,
            1,
            1,
            deterministic=True,
        )


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0)])
def test_python_entry_accepts_supported_deterministic_arch(monkeypatch, capability):
    q = torch.empty(
        1, 1, 1, 64, dtype=torch.bfloat16, device="meta", requires_grad=True
    )
    k = torch.empty_like(q, requires_grad=True)
    v = torch.empty_like(q, requires_grad=True)
    arbitrary_func = torch.zeros(
        1, 1, 1, 257, dtype=torch.int32, device="meta"
    )
    monkeypatch.setattr(fai, "_device_capability_major", lambda _device: capability)
    # Reaching the fixed-length check proves that the architecture gate passed.
    with pytest.raises(NotImplementedError, match="fixed-length tensors only"):
        fai._prepare_arbitrary_block_sparse(
            q,
            k,
            v,
            arbitrary_func,
            None,
            None,
            1,
            1,
            deterministic=True,
            is_varlen=True,
        )


@pytest.mark.parametrize("capability", [(7, 5), (8, 7), (10, 0)])
def test_fake_backward_rejects_unsupported_deterministic_arch(
    monkeypatch, capability
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda")
        arbitrary_func = torch.empty(
            1, 1, 1, 257, dtype=torch.int32, device="cuda"
        )
        softmax_lse = torch.empty(
            1, 1, 1, dtype=torch.float32, device="cuda"
        )
        # Construct CUDA FakeTensors before simulating a target architecture;
        # on CPU-only hosts, advertising CUDA availability earlier makes
        # FakeTensorMode attempt real device initialization.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda _device: capability
        )
        with pytest.raises(
            NotImplementedError, match="only on SM80, SM86, SM89, and SM90"
        ):
            fai._flash_attn_backward_fake(
                q,
                q,
                q,
                q,
                q,
                softmax_lse,
                deterministic=True,
                arbitrary_func=arbitrary_func,
            )


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0)])
def test_fake_backward_accepts_supported_deterministic_arch(
    monkeypatch, capability
):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda")
        arbitrary_func = torch.empty(
            1, 1, 1, 257, dtype=torch.int32, device="cuda"
        )
        softmax_lse = torch.empty(
            1, 1, 1, dtype=torch.float32, device="cuda"
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda _device: capability
        )
        # Reaching metadata validation proves that the architecture gate passed.
        with pytest.raises(ValueError, match="requires all six"):
            fai._flash_attn_backward_fake(
                q,
                q,
                q,
                q,
                q,
                softmax_lse,
                deterministic=True,
                arbitrary_func=arbitrary_func,
            )


def test_fake_backward_allows_unknown_cuda_arch_until_runtime(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (_ for _ in ()).throw(AssertionError("must not query CUDA")),
    )
    with FakeTensorMode():
        q = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda")
        arbitrary_func = torch.empty(
            1, 1, 1, 257, dtype=torch.int32, device="cuda"
        )
        softmax_lse = torch.empty(
            1, 1, 1, dtype=torch.float32, device="cuda"
        )
        mask_cnt = torch.empty(1, 1, 1, dtype=torch.int32, device="cuda")
        mask_offset = torch.empty(2, dtype=torch.int32, device="cuda")
        mask_idx = torch.empty(1, dtype=torch.int32, device="cuda")
        full_cnt = torch.empty_like(mask_cnt)
        full_offset = torch.empty_like(mask_offset)
        full_idx = torch.empty(0, dtype=torch.int32, device="cuda")
        softmax_d = fai._flash_attn_backward_fake(
            q,
            q,
            q,
            q,
            q,
            softmax_lse,
            deterministic=True,
            arbitrary_func=arbitrary_func,
            block_sparse_mask_cnt=mask_cnt,
            block_sparse_mask_offset=mask_offset,
            block_sparse_mask_idx=mask_idx,
            block_sparse_full_cnt=full_cnt,
            block_sparse_full_offset=full_offset,
            block_sparse_full_idx=full_idx,
            block_sparse_dq_write_order=torch.empty_like(mask_idx),
            block_sparse_dq_write_order_full=torch.empty_like(full_idx),
        )
        assert softmax_d.shape == (0,)
        assert softmax_d.dtype == torch.float32


def test_fake_backward_requires_deterministic_metadata(monkeypatch):
    q = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="meta")
    arbitrary_func = torch.empty(1, 1, 1, 257, dtype=torch.int32, device="meta")
    softmax_lse = torch.empty(1, 1, 1, dtype=torch.float32, device="meta")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _device: (8, 9)
    )
    with pytest.raises(ValueError, match="requires all six"):
        fai._flash_attn_backward_fake(
            q,
            q,
            q,
            q,
            q,
            softmax_lse,
            deterministic=True,
            arbitrary_func=arbitrary_func,
        )


def test_fake_backward_meta_dispatch_returns_fixed_token():
    q = torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="meta")
    arbitrary_func = torch.empty(1, 1, 1, 257, dtype=torch.int32, device="meta")
    softmax_lse = torch.empty(1, 1, 1, dtype=torch.float32, device="meta")
    mask_cnt = torch.empty(1, 1, 1, dtype=torch.int32, device="meta")
    mask_offset = torch.empty(2, dtype=torch.int32, device="meta")
    mask_idx = torch.empty(1, dtype=torch.int32, device="meta")
    full_cnt = torch.empty_like(mask_cnt)
    full_offset = torch.empty_like(mask_offset)
    full_idx = torch.empty(0, dtype=torch.int32, device="meta")
    mask_rank = torch.empty_like(mask_idx)
    full_rank = torch.empty_like(full_idx)
    softmax_d = fai._flash_attn_backward(
        q,
        q,
        q,
        q,
        q,
        softmax_lse,
        dq=torch.empty_like(q),
        dk=torch.empty_like(q),
        dv=torch.empty_like(q),
        deterministic=True,
        arbitrary_func=arbitrary_func,
        block_sparse_mask_cnt=mask_cnt,
        block_sparse_mask_offset=mask_offset,
        block_sparse_mask_idx=mask_idx,
        block_sparse_full_cnt=full_cnt,
        block_sparse_full_offset=full_offset,
        block_sparse_full_idx=full_idx,
        block_sparse_dq_write_order=mask_rank,
        block_sparse_dq_write_order_full=full_rank,
    )
    assert softmax_d.shape == (0,)
    assert softmax_d.dtype == torch.float32
    assert softmax_d.device.type == "meta"


@supported_arch_only
@pytest.mark.parametrize(
    "malformation,match",
    [
        pytest.param("missing_full_rank", "requires dq_write_order", id="missing-full-rank"),
        pytest.param("rank_length", "parallel to its compact idx", id="rank-length"),
        pytest.param("wrong_combined_rank", "contiguous rank", id="combined-rank-order"),
        pytest.param("offset_count", "offset delta", id="offset-count"),
        pytest.param("duplicate_edge", "duplicate edge", id="duplicate-edge"),
        pytest.param("partial_full_overlap", "duplicate edge", id="partial-full-overlap"),
        pytest.param("out_of_range_m", "out-of-range m_block", id="out-of-range-m"),
        pytest.param("wrong_n_block_count", "n-block rows", id="wrong-n-block-count"),
    ],
)
def test_malformed_deterministic_metadata_fails_before_kernel(
    monkeypatch, malformation, match
):
    q, k, v, arbitrary_func, q2k, k2q = _make_case("mha")
    bwd_block_m, _ = fai._dense_bwd_block_size(q, v)
    if malformation == "missing_full_rank":
        bad_k2q = k2q._replace(dq_write_order_full=None)
    elif malformation == "rank_length":
        bad_k2q = k2q._replace(dq_write_order=k2q.dq_write_order[:-1])
    elif malformation == "wrong_combined_rank":
        bad_k2q = k2q._replace(
            dq_write_order_full=torch.zeros_like(k2q.dq_write_order_full)
        )
    elif malformation == "offset_count":
        bad_offset = k2q.mask_block_offset.clone()
        bad_offset[1] += 1
        bad_k2q = k2q._replace(mask_block_offset=bad_offset)
    elif malformation == "duplicate_edge":
        bad_idx = k2q.mask_block_idx.clone()
        bad_idx[1] = bad_idx[0]
        bad_k2q = k2q._replace(mask_block_idx=bad_idx)
    elif malformation == "partial_full_overlap":
        num_n_blocks = k2q.mask_block_cnt.shape[2]
        num_m_blocks = math.ceil(q.shape[1] / bwd_block_m)
        overlap = _linear_csr(
            partial_rows=[[], list(range(num_m_blocks)), []],
            full_rows=[list(range(num_m_blocks)), [0], []],
            device=q.device,
        )
        assert num_n_blocks == 3
        bad_k2q = overlap._replace(
            dq_write_order=torch.zeros_like(overlap.mask_block_idx),
            dq_write_order_full=torch.zeros_like(overlap.full_block_idx),
        )
    elif malformation == "out_of_range_m":
        bad_idx = k2q.mask_block_idx.clone()
        bad_idx[0] = math.ceil(q.shape[1] / bwd_block_m)
        bad_full_rank = k2q.dq_write_order_full.clone()
        # m=0 lost its n=1 partial contributor, so its remaining n=0
        # full contributor is rank 0.  Keep ranks otherwise valid so this
        # case reaches the launch-bound m-range check.
        bad_full_rank[0] = 0
        bad_k2q = k2q._replace(
            mask_block_idx=bad_idx,
            dq_write_order_full=bad_full_rank,
        )
    elif malformation == "wrong_n_block_count":
        num_m_blocks = math.ceil(q.shape[1] / bwd_block_m)
        bad_k2q = prepare_deterministic_k2q_metadata(
            _linear_csr(
                partial_rows=[[], list(range(num_m_blocks))],
                full_rows=[list(range(num_m_blocks)), []],
                device=q.device,
            ),
        )
    else:
        raise AssertionError(f"unknown malformation: {malformation}")

    reached_kernel = False

    def _unexpected_forward(*args, **kwargs):
        nonlocal reached_kernel
        reached_kernel = True
        raise AssertionError("malformed metadata reached the CUDA attention kernel")

    monkeypatch.setattr(fai, "_flash_attn_forward", _unexpected_forward)
    with pytest.raises(ValueError, match=match):
        flash_attn_func(
            q,
            k,
            v,
            deterministic=True,
            arbitrary_func=arbitrary_func,
            q2k_block_sparse=q2k,
            k2q_block_sparse=bad_k2q,
        )
    assert not reached_kernel


def _raw_semantic_validation_worker(result):
    try:
        q, k, v, arbitrary_func, q2k, k2q = _make_case("mha")
        out, softmax_lse = _raw_forward(
            q, k, v, arbitrary_func, q2k, k2q
        )
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        def _call_raw(metadata):
            torch.ops.flash_attn_3.bwd(
                torch.randn_like(out),
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                deterministic=True,
                arbitrary_func=arbitrary_func,
                block_sparse_mask_cnt=metadata.mask_block_cnt,
                block_sparse_mask_offset=metadata.mask_block_offset,
                block_sparse_mask_idx=metadata.mask_block_idx,
                block_sparse_full_cnt=metadata.full_block_cnt,
                block_sparse_full_offset=metadata.full_block_offset,
                block_sparse_full_idx=metadata.full_block_idx,
                block_sparse_dq_write_order=metadata.dq_write_order,
                block_sparse_dq_write_order_full=metadata.dq_write_order_full,
            )

        bad_offset = k2q.mask_block_offset.clone()
        bad_offset[1] += 1
        duplicate_idx = k2q.mask_block_idx.clone()
        duplicate_idx[1] = duplicate_idx[0]
        out_of_range_idx = k2q.mask_block_idx.clone()
        bwd_block_m, _ = fai._dense_bwd_block_size(q, v)
        out_of_range_idx[0] = math.ceil(q.shape[1] / bwd_block_m)
        num_m_blocks = math.ceil(q.shape[1] / bwd_block_m)
        overlap = _linear_csr(
            partial_rows=[[], list(range(num_m_blocks)), []],
            full_rows=[list(range(num_m_blocks)), [0], []],
            device=q.device,
        )._replace(
            dq_write_order=torch.zeros_like(k2q.mask_block_idx),
            dq_write_order_full=torch.zeros(
                num_m_blocks + 1, dtype=torch.int32, device=q.device
            ),
        )
        malformed = (
            (
                k2q._replace(
                    dq_write_order_full=torch.zeros_like(
                        k2q.dq_write_order_full
                    )
                ),
                "contiguous rank",
            ),
            (k2q._replace(mask_block_offset=bad_offset), "offset delta"),
            (k2q._replace(mask_block_idx=duplicate_idx), "duplicate edge"),
            (overlap, "duplicate edge"),
            (k2q._replace(mask_block_idx=out_of_range_idx), "out-of-range m_block"),
        )
        for metadata, match in malformed:
            with pytest.raises((RuntimeError, ValueError), match=match):
                _call_raw(metadata)
    except BaseException:
        result.send(("error", traceback.format_exc()))
    else:
        result.send(("ok", None))
    finally:
        result.close()


def _run_repeatability_case(kv_mode):
    q, k, v, arbitrary_func, q2k, k2q = _make_case(kv_mode)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = flash_attn_func(
        q,
        k,
        v,
        deterministic=True,
        arbitrary_func=arbitrary_func,
        q2k_block_sparse=q2k,
        k2q_block_sparse=k2q,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    torch.manual_seed(20260731)
    dout = torch.randn_like(out)
    reference_grads = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)
    baseline = None
    repeats = 50 if kv_mode in ("gqa", "mqa") else 20
    for repeat in range(repeats):
        grads = torch.autograd.grad(
            out,
            (q, k, v),
            dout,
            retain_graph=repeat + 1 < repeats,
        )
        if baseline is None:
            baseline = tuple(grad.clone() for grad in grads)
            for actual, expected in zip(grads, reference_grads):
                torch.testing.assert_close(actual, expected, atol=8e-2, rtol=8e-2)
            # K2Q row n=2 has no contributors.  In GQA this specifically
            # exercises the deterministic dV/dK zero-store semaphore chain.
            _, bwd_block_n = fai._dense_bwd_block_size(q, v)
            empty_k_start = 2 * bwd_block_n
            assert torch.count_nonzero(grads[1][:, empty_k_start:]).item() == 0
            assert torch.count_nonzero(grads[2][:, empty_k_start:]).item() == 0
        else:
            for name, actual, expected in zip(("dQ", "dK", "dV"), grads, baseline):
                assert torch.equal(actual, expected), (
                    f"{name} changed on deterministic repeat {repeat} "
                    f"for kv_mode={kv_mode}"
                )
    _run_raw_accum_repeatability(
        q, k, v, dout, arbitrary_func, q2k, k2q, repeats
    )
    torch.cuda.synchronize()


def _repeatability_worker(kv_mode, result):
    try:
        _run_repeatability_case(kv_mode)
    except BaseException:
        result.send(("error", traceback.format_exc()))
    else:
        result.send(("ok", None))
    finally:
        result.close()


def _run_long_grid_case():
    _require_test_build()
    torch.manual_seed(20260801)
    device = "cuda"
    head_dim = _compiled_head_dim()
    heads, kv_heads = 4, 2
    bwd_block_m, block_n = _bwd_block_size_for_arch(head_dim)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # SM8x backward uses 256-thread CTAs.  Six CTAs/SM is the architectural
    # thread-limit upper bound, so this grid is larger than full residency
    # even before register and shared-memory limits are considered.
    max_ctas_per_sm = 6
    num_n_blocks = max(
        48, math.ceil((sm_count * max_ctas_per_sm + 1) / heads)
    )
    if num_n_blocks % 2:
        num_n_blocks += 1
    assert num_n_blocks * heads > sm_count * max_ctas_per_sm
    seqlen_q = 8 * bwd_block_m + 1
    seqlen_k = num_n_blocks * block_n

    q = torch.randn(
        1,
        seqlen_q,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        1,
        seqlen_k,
        kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    arbitrary_func = torch.zeros(
        1, 1, 1, seqlen_q + 256, dtype=torch.int32, device=device
    )
    fwd_block_m, fwd_block_n = fai._dense_fwd_block_size(q, v)
    num_fwd_n_blocks = math.ceil(seqlen_k / fwd_block_n)
    # Visit every K tile.  This keeps Q2K correct when SM8x forward/backward N
    # tiles differ, while producing a turnstile chain longer than
    # full-grid residency for every m block.
    q2k = _linear_csr(
        partial_rows=[
            [] for _ in range(math.ceil(seqlen_q / fwd_block_m))
        ],
        full_rows=[
            list(range(num_fwd_n_blocks))
            for _ in range(math.ceil(seqlen_q / fwd_block_m))
        ],
        device=device,
    )
    num_m_blocks = math.ceil(seqlen_q / bwd_block_m)
    k2q = _linear_csr(
        partial_rows=[[] for _ in range(num_n_blocks)],
        full_rows=[
            list(range(num_m_blocks)) for _ in range(num_n_blocks)
        ],
        device=device,
    )
    k2q = prepare_deterministic_k2q_metadata(k2q)

    out = flash_attn_func(
        q,
        k,
        v,
        deterministic=True,
        arbitrary_func=arbitrary_func,
        q2k_block_sparse=q2k,
        k2q_block_sparse=k2q,
    )
    dout = torch.randn_like(out)
    baseline = None
    for repeat in range(50):
        grads = torch.autograd.grad(
            out, (q, k, v), dout, retain_graph=repeat != 49
        )
        if baseline is None:
            baseline = tuple(grad.clone() for grad in grads)
        else:
            for name, actual, expected in zip(
                ("dQ", "dK", "dV"), grads, baseline
            ):
                assert torch.equal(actual, expected), (
                    f"{name} changed on long-grid deterministic repeat {repeat}"
                )
    torch.cuda.synchronize()


def _long_grid_worker(result):
    try:
        _run_long_grid_case()
    except BaseException:
        result.send(("error", traceback.format_exc()))
    else:
        result.send(("ok", None))
    finally:
        result.close()


def _run_with_watchdog(target, args, name, timeout_s):
    context = multiprocessing.get_context("spawn")
    result, child_result = context.Pipe(duplex=False)
    process = context.Process(
        target=target,
        args=(*args, child_result),
        name=name,
    )
    process.start()
    child_result.close()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        result.close()
        pytest.fail(f"{name} hung for {timeout_s:g}s")
    if not result.poll():
        result.close()
        pytest.fail(
            f"{name} exited with code {process.exitcode} without a result"
        )
    status, detail = result.recv()
    result.close()
    assert process.exitcode == 0
    if status != "ok":
        pytest.fail(detail)


@supported_arch_only
def test_aot_autograd_preserves_arbitrary_deterministic_metadata():
    """The registered custom-op path must carry CSR, ranks, and deterministic."""
    q, k, v, arbitrary_func, q2k, k2q = _make_case("gqa")

    def attention(q_, k_, v_):
        return flash_attn_func(
            q_,
            k_,
            v_,
            deterministic=True,
            arbitrary_func=arbitrary_func,
            q2k_block_sparse=q2k,
            k2q_block_sparse=k2q,
        )

    compiled_attention = torch.compile(
        attention, backend="aot_eager", fullgraph=True
    )
    out = compiled_attention(q, k, v)
    grads = torch.autograd.grad(
        out, (q, k, v), torch.randn_like(out)
    )
    assert all(torch.isfinite(grad).all() for grad in grads)
    torch.cuda.synchronize()


@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@supported_arch_only
def test_arbitrary_deterministic_backward_is_bitwise_repeatable(kv_mode):
    """Run spin-wait kernels in a spawned process so a deadlock cannot wedge pytest."""
    _require_test_build()
    _compiled_head_dim()
    timeout_s = float(os.getenv("FLASH_ATTENTION_DETERMINISM_TIMEOUT_S", "120"))
    _run_with_watchdog(
        _repeatability_worker,
        (kv_mode,),
        f"{_arch_label()}-arbitrary-deterministic-{kv_mode}",
        timeout_s,
    )


@supported_arch_only
def test_raw_cpp_op_rejects_malformed_semantics_before_launch():
    _require_test_build()
    _compiled_head_dim()
    _run_with_watchdog(
        _raw_semantic_validation_worker,
        (),
        f"{_arch_label()}-raw-cpp-semantic-validation",
        30,
    )


@supported_arch_only
def test_arbitrary_deterministic_long_grid_watchdog():
    _require_test_build()
    _compiled_head_dim()
    timeout_s = float(os.getenv("FLASH_ATTENTION_DETERMINISM_TIMEOUT_S", "120"))
    _run_with_watchdog(
        _long_grid_worker,
        (),
        f"{_arch_label()}-arbitrary-deterministic-long-grid",
        timeout_s,
    )
