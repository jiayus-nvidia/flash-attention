import math

import pytest
import torch
from einops import rearrange

from flash_attn.cute.block_sparsity import (
    LinearBlockSparseTensorsTorch,
    compute_dq_write_order_from_linear_csr,
)
from flash_attn.cute.interface import (
    _tile_size_bwd_sm90,
    _tile_size_fwd_sm90,
    flash_attn_func,
    flash_attn_varlen_func,
)

try:
    import create_block_mask_cuda
except ImportError:
    create_block_mask_cuda = None


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_arbitrary_func(
    batch,
    heads,
    seqlen_q,
    seqlen_k,
    device,
    *,
    vary_by_batch=False,
    vary_by_head=False,
):
    func = torch.zeros(batch, heads, 3, seqlen_q + 256, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(heads):
            sink_base = 1 if (vary_by_head and h % 2) else 8 + (b if vary_by_batch else 0)
            sink = min(sink_base, seqlen_k)
            window = max(4, 16 - 4 * (h if vary_by_head else 0) - 2 * (b if vary_by_batch else 0))
            for q_idx in range(seqlen_q):
                func[b, h, 0, q_idx] = sink
                func[b, h, 1, q_idx] = max(0, q_idx - window)
                func[b, h, 2, q_idx] = min(seqlen_k, q_idx + 1)
    return func


def _make_head_dependent_arbitrary_func(batch, heads, seqlen_q, seqlen_k, device):
    return _make_arbitrary_func(
        batch,
        heads,
        seqlen_q,
        seqlen_k,
        device,
        vary_by_head=heads > 1,
    )


def _apply_arbitrary_mask(scores, arbitrary_func):
    batch, heads, seqlen_q, seqlen_k = scores.shape
    func = arbitrary_func[..., :seqlen_q]
    cols = torch.arange(seqlen_k, device=scores.device, dtype=torch.int32)
    base = cols.view(1, 1, 1, seqlen_k) < func[:, :, 0, :].unsqueeze(-1)
    interval = (
        (cols.view(1, 1, 1, seqlen_k) >= func[:, :, 1, :].unsqueeze(-1))
        & (cols.view(1, 1, 1, seqlen_k) < func[:, :, 2, :].unsqueeze(-1))
    )
    valid = (base | interval).expand(batch, heads, seqlen_q, seqlen_k)
    return scores.masked_fill(~valid, float("-inf"))


def _attention_ref(q, k, v, arbitrary_func):
    q_ref = q.float()
    k_ref = k.float()
    v_ref = v.float()
    if q_ref.shape[2] != k_ref.shape[2]:
        repeat = q_ref.shape[2] // k_ref.shape[2]
        k_ref = k_ref.repeat_interleave(repeat, dim=2)
        v_ref = v_ref.repeat_interleave(repeat, dim=2)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref * scale, k_ref)
    scores = _apply_arbitrary_mask(scores, arbitrary_func)
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v_ref).to(q.dtype)


SEQLEN_CONFIGS = [
    (64, 64),
    (96, 128),
    (129, 65),
]


def _device_major():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()[0]


def _cases_for_arch():
    major = _device_major()
    if major == 9:
        return [(192, 192), (256, 256)]
    if major == 10:
        return [(192, 128), (256, 256)]
    return []


def _generic_case_for_arch():
    major = _device_major()
    if major == 9:
        return (192, 192)
    if major == 10:
        return (192, 128)
    return None


def _kv_heads_for_mode(kv_mode, heads):
    if kv_mode == "mha":
        return heads
    if kv_mode == "gqa":
        return heads // 2
    if kv_mode == "mqa":
        return 1
    raise AssertionError(f"unknown kv_mode: {kv_mode}")


def _linear_block_sizes(head_dim, head_dim_v, seqlen_q, qhead_per_kvhead):
    major = _device_major()
    if major == 9:
        fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, False, False)
        bwd_cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            False,
            False,
            sparse_block_size_q=128,
        )
        return (fwd_cfg.m_block_size, fwd_cfg.n_block_size), (128, bwd_cfg.n_block_size)
    if major == 10:
        q_stage = 2 if seqlen_q * qhead_per_kvhead > 128 else 1
        if head_dim in (128, 192) and head_dim_v == 128:
            return (q_stage * 128, 128), (128, 256)
        return (q_stage * 128, 128), (256, 128)
    pytest.skip("linear arbitrary block-sparse test only runs on SM90/SM100")


def _require_create_block_mask_cuda():
    if create_block_mask_cuda is None:
        pytest.fail(
            "create_block_mask_cuda is required for CSR arbitrary-mask tests. "
            "Build it with `make create_block_mask` from the repo root."
        )
    return create_block_mask_cuda


def _linear_from_csr_tuple(csr_tensors, block_size):
    (
        mask_block_cnt,
        mask_block_offset,
        mask_block_idx,
        full_block_cnt,
        full_block_offset,
        full_block_idx,
    ) = csr_tensors
    return LinearBlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_offset=mask_block_offset,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_offset=full_block_offset,
        full_block_idx=full_block_idx,
        block_size=block_size,
    )


def _make_linear_block_sparse_pair(arbitrary_func, seqlen_q, seqlen_k, fwd_block_size, bwd_block_size):
    block_mask_cuda = _require_create_block_mask_cuda()
    q2k_csr = block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=fwd_block_size[0],
        KV_BLOCK_SIZE=fwd_block_size[1],
        check_q_boundary=True,
    )
    k2q_csr = block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=bwd_block_size[0],
        KV_BLOCK_SIZE=bwd_block_size[1],
    )
    linear_k = _linear_from_csr_tuple(q2k_csr, fwd_block_size)
    linear_q = _linear_from_csr_tuple(k2q_csr, bwd_block_size)
    return linear_k, linear_q


def _make_mixed_block_arbitrary_func(seqlen_q, seqlen_k, device):
    """Make a broadcast mask whose K2Q CSR contains mixed partial/full blocks."""
    func = torch.zeros(1, 1, 3, seqlen_q + 256, dtype=torch.int32, device=device)
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device=device)
    full_prefix = min(128, seqlen_k)
    interval_start = torch.maximum(
        torch.full_like(q_idx, full_prefix),
        q_idx - 48,
    )
    interval_end = torch.maximum(
        interval_start,
        torch.clamp(q_idx + 65, max=seqlen_k),
    )
    func[0, 0, 0, :seqlen_q] = full_prefix
    func[0, 0, 1, :seqlen_q] = interval_start
    func[0, 0, 2, :seqlen_q] = interval_end
    return func


def _with_deterministic_dq_write_order(linear_q, *, spt):
    dq_write_order, dq_write_order_full = compute_dq_write_order_from_linear_csr(
        linear_q,
        spt=spt,
    )
    return linear_q._replace(
        dq_write_order=dq_write_order,
        dq_write_order_full=dq_write_order_full,
        spt=spt,
    )


def _run_linear_block_sparse_case(kv_mode, head_dim, head_dim_v, seqlen_q, seqlen_k, seed):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen_q, seqlen_k, device)
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim,
        head_dim_v,
        seqlen_q,
        heads // kv_heads,
    )
    linear_k, linear_q = _make_linear_block_sparse_pair(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        fwd_block_size,
        bwd_block_size,
    )

    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        linear_q_block_sparse_tensors=linear_q,
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    if _device_major() == 9 and kv_mode != "mha":
        return
    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


def _run_sm90_deterministic_linear_block_sparse_case(
    kv_mode,
    head_dim,
    *,
    spt,
    seed,
    bwd_q_block_size=None,
    deterministic_repeats=5,
):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    seqlen_q, seqlen_k = 257, 321
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
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
    v = torch.randn(
        batch,
        seqlen_k,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_mixed_block_arbitrary_func(seqlen_q, seqlen_k, device)
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim,
        head_dim,
        seqlen_q,
        heads // kv_heads,
    )
    if bwd_q_block_size is not None:
        bwd_block_size = (bwd_q_block_size, bwd_block_size[1])
    linear_k, linear_q = _make_linear_block_sparse_pair(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        fwd_block_size,
        bwd_block_size,
    )

    # Exercise combined ranks: at least one Q block must receive contributors
    # from both the partial and full K2Q lists.
    assert linear_q.mask_block_idx.numel() > 0
    assert linear_q.full_block_idx is not None
    assert linear_q.full_block_idx.numel() > 0
    assert torch.isin(linear_q.mask_block_idx, linear_q.full_block_idx).any().item()
    linear_q = _with_deterministic_dq_write_order(linear_q, spt=spt)
    assert linear_q.dq_write_order_full is not None

    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        linear_q_block_sparse_tensors=linear_q,
        deterministic=True,
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    grads_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad)
    grads = torch.autograd.grad(out, (q, k, v), grad, retain_graph=True)
    for actual, expected in zip(grads, grads_ref):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    grad_names = ("dQ", "dK", "dV")
    for repeat_idx in range(1, deterministic_repeats):
        repeated_grads = torch.autograd.grad(
            out,
            (q, k, v),
            grad,
            retain_graph=repeat_idx + 1 < deterministic_repeats,
        )
        for name, actual, repeated in zip(grad_names, grads, repeated_grads):
            assert torch.equal(actual, repeated), (
                f"{name} changed on deterministic repeat {repeat_idx} "
                f"for head_dim={head_dim}, kv_mode={kv_mode}, spt={spt}"
            )


def _run_arbitrary_case(
    *,
    batch,
    seqlen_q,
    seqlen_k,
    heads,
    kv_heads,
    head_dim,
    head_dim_v,
    mask_batch,
    mask_heads,
    seed,
    dtype=torch.bfloat16,
    check_backward=True,
    atol=4e-2,
    rtol=4e-2,
):
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(
        mask_batch,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device,
        vary_by_batch=mask_batch > 1,
        vary_by_head=mask_heads > 1,
    )

    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    if not check_backward:
        return

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_dim, head_dim_v", _cases_for_arch())
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_CONFIGS)
def test_arbitrary_mask_fixedlen_output(seqlen_q, seqlen_k, kv_mode, head_dim, head_dim_v, dtype):
    if not _cases_for_arch():
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    heads = 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    mask_heads = heads if kv_mode != "mha" else 1
    check_backward = not (_device_major() == 9 and head_dim == 192 and kv_mode != "mha")
    _run_arbitrary_case(
        batch=2,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=2,
        mask_heads=mask_heads,
        seed=seqlen_q + seqlen_k + head_dim + 17 * kv_heads,
        dtype=dtype,
        check_backward=check_backward,
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "mask_batch, mask_heads",
    [
        pytest.param(1, 1, id="batch_head_broadcast"),
        pytest.param(2, 1, id="batch_specific_head_broadcast"),
        pytest.param(1, 2, id="batch_broadcast_head_specific"),
        pytest.param(2, 2, id="batch_head_specific"),
    ],
)
def test_arbitrary_mask_broadcast_patterns(mask_batch, mask_heads):
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_arbitrary_case(
        batch=2,
        seqlen_q=96,
        seqlen_k=128,
        heads=2,
        kv_heads=2,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=mask_batch,
        mask_heads=mask_heads,
        seed=mask_batch * 10 + mask_heads,
    )


def test_arbitrary_mask_generic_gqa():
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_arbitrary_case(
        batch=1,
        seqlen_q=96,
        seqlen_k=128,
        heads=4,
        kv_heads=2,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=1,
        mask_heads=4,
        seed=23,
        check_backward=not (_device_major() == 9 and head_dim == 192),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
def test_arbitrary_mask_linear_block_sparse(kv_mode):
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("linear arbitrary block-sparse test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_linear_block_sparse_case(
        kv_mode,
        head_dim,
        head_dim_v,
        seqlen_q=96,
        seqlen_k=160,
        seed=97 + len(kv_mode),
    )


@pytest.mark.parametrize(
    "head_dim,kv_mode,spt,bwd_q_block_size",
    [
        pytest.param(64, "mha", False, None, id="hd64_mha_spt_false"),
        pytest.param(96, "gqa", True, None, id="hd96_gqa_spt_true"),
        pytest.param(128, "gqa", False, 64, id="hd128_gqa_spt_false_qblock64"),
        pytest.param(128, "gqa", True, 64, id="hd128_gqa_spt_true_qblock64"),
        pytest.param(192, "mha", True, None, id="hd192_mha_spt_true"),
        pytest.param(256, "mha", False, None, id="hd256_mha_spt_false"),
    ],
)
def test_arbitrary_mask_linear_block_sparse_sm90_deterministic(
    head_dim, kv_mode, spt, bwd_q_block_size
):
    if _device_major() != 9:
        pytest.skip("SM90 deterministic linear CSR backward test only runs on SM90")
    _run_sm90_deterministic_linear_block_sparse_case(
        kv_mode,
        head_dim,
        spt=spt,
        seed=1301 + head_dim + 17 * (kv_mode == "gqa") + int(spt),
        bwd_q_block_size=bwd_q_block_size,
        deterministic_repeats=(
            50 if head_dim == 128 and kv_mode == "gqa" and not spt else 5
        ),
    )


@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
def test_arbitrary_mask_linear_block_sparse_sm100_hdim128_2cta(kv_mode):
    if _device_major() != 10:
        pytest.skip("SM100 hdim128 2CTA linear CSR backward test only runs on SM100")
    _run_linear_block_sparse_case(
        kv_mode,
        head_dim=128,
        head_dim_v=128,
        seqlen_q=384,
        seqlen_k=384,
        seed=811 + len(kv_mode),
    )


def test_arbitrary_mask_linear_block_sparse_sm100_hdim128_rejects_coarse_q_csr():
    if _device_major() != 10:
        pytest.skip("SM100 hdim128 2CTA linear CSR backward test only runs on SM100")
    torch.manual_seed(191)
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads, kv_heads = 1, 1, 1
    seqlen_q, seqlen_k = 128, 128
    head_dim, head_dim_v = 128, 128
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True
    )
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen_q, seqlen_k, device)
    linear_k, linear_q = _make_linear_block_sparse_pair(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        fwd_block_size=(128, 128),
        bwd_block_size=(256, 256),
    )
    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        linear_q_block_sparse_tensors=linear_q,
        return_lse=True,
    )
    with pytest.raises(ValueError, match=r"expects BLOCK_SIZE=\(128, 256\)"):
        out.backward(torch.randn_like(out))


@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 256), (256, 512), (384, 384), (96, 160)])
def test_arbitrary_mask_linear_block_sparse_hd256(kv_mode, seqlen_q, seqlen_k):
    """Forward-only CSR linear block sparsity for the SM100 dedicated head_dim=256 kernel.

    The dedicated 2CTA hd256 forward kernel processes a 256-row (2 * tile_m) Q tile
    per work item; the CSR Q block size therefore matches `_linear_block_sizes`.
    """
    if _device_major() != 10:
        pytest.skip("hd256 block-sparse forward only runs on SM100")
    torch.manual_seed(131 + seqlen_q + seqlen_k + len(kv_mode))
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    head_dim = head_dim_v = 256
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype)
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen_q, seqlen_k, device)
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim, head_dim_v, seqlen_q, heads // kv_heads
    )
    linear_k, _ = _make_linear_block_sparse_pair(
        arbitrary_func, seqlen_q, seqlen_k, fwd_block_size, bwd_block_size
    )
    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        return_lse=True,
    )
    out_ref = _attention_ref(q, k, v, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "kv_mode,mask_heads,seqlen_q,seqlen_k",
    [
        pytest.param("mha", 4, 384, 384, id="mha_head_specific"),
        pytest.param("gqa", 1, 384, 384, id="gqa_head_broadcast"),
        pytest.param("mqa", 1, 384, 384, id="mqa_head_broadcast"),
        pytest.param("mha", 4, 96, 160, id="mha_qstage1"),
    ],
)
def test_arbitrary_mask_linear_block_sparse_hd256_backward(
    kv_mode, mask_heads, seqlen_q, seqlen_k
):
    if _device_major() != 10:
        pytest.skip("hd256 block-sparse backward only runs on SM100")
    torch.manual_seed(733 + len(kv_mode))
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 2, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    head_dim = head_dim_v = 256
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, kv_heads, head_dim_v, device=device, dtype=dtype, requires_grad=True
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(
        1,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device,
        vary_by_head=mask_heads > 1,
    )
    fwd_block_size, bwd_block_size = _linear_block_sizes(
        head_dim, head_dim_v, seqlen_q, heads // kv_heads
    )
    linear_k, linear_q = _make_linear_block_sparse_pair(
        arbitrary_func, seqlen_q, seqlen_k, fwd_block_size, bwd_block_size
    )
    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        linear_k_block_sparse_tensors=linear_k,
        linear_q_block_sparse_tensors=linear_q,
        return_lse=True,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_arbitrary_mask_fp16_smoke(dtype):
    case = _generic_case_for_arch()
    if case is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    head_dim, head_dim_v = case
    _run_arbitrary_case(
        batch=1,
        seqlen_q=64,
        seqlen_k=96,
        heads=2,
        kv_heads=2,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        mask_batch=1,
        mask_heads=1,
        seed=71,
        dtype=dtype,
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.parametrize("head_dim, head_dim_v", _cases_for_arch())
@pytest.mark.parametrize("kv_mode", ["mha", "gqa"])
@pytest.mark.parametrize("seqlen", [64, 128])
def test_arbitrary_mask_varlen_output(seqlen, kv_mode, head_dim, head_dim_v):
    if not _cases_for_arch():
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    torch.manual_seed(seqlen + head_dim)
    device = "cuda"
    dtype = torch.bfloat16
    batch, heads = 3, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    q = torch.randn(batch, seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, kv_heads, head_dim_v, device=device, dtype=dtype)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_varlen = rearrange(q.detach(), "b s h d -> (b s) h d").requires_grad_(True)
    k_varlen = rearrange(k.detach(), "b s h d -> (b s) h d").requires_grad_(True)
    v_varlen = rearrange(v.detach(), "b s h d -> (b s) h d").requires_grad_(True)
    cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, device=device, dtype=torch.int32)
    arbitrary_func = _make_arbitrary_func(
        batch,
        heads,
        seqlen,
        seqlen,
        device,
        vary_by_batch=True,
        vary_by_head=kv_mode != "mha",
    )

    out_varlen, _ = flash_attn_varlen_func(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        arbitrary=True,
        aux_tensors=[arbitrary_func],
        return_lse=True,
    )
    out = rearrange(out_varlen, "(b s) h d -> b s h d", b=batch)
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    if _device_major() == 9 and head_dim == 192 and kv_mode != "mha":
        return

    grad = torch.randn_like(out)
    out_varlen.backward(rearrange(grad, "b s h d -> (b s) h d"))
    out_ref.backward(grad)
    torch.testing.assert_close(
        q_varlen.grad,
        rearrange(q_ref.grad, "b s h d -> (b s) h d"),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        k_varlen.grad,
        rearrange(k_ref.grad, "b s h d -> (b s) h d"),
        atol=5e-2,
        rtol=5e-2,
    )
    torch.testing.assert_close(
        v_varlen.grad,
        rearrange(v_ref.grad, "b s h d -> (b s) h d"),
        atol=5e-2,
        rtol=5e-2,
    )


@pytest.mark.parametrize(
    "aux_factory, error_match",
    [
        pytest.param(lambda valid: None, r"aux_tensors\[0\]", id="missing_aux"),
        pytest.param(lambda valid: [valid.to(torch.float32)], "torch.int32", id="wrong_dtype"),
        pytest.param(
            lambda valid: [torch.zeros(1, 1, 2, valid.shape[-1], dtype=torch.int32, device=valid.device)],
            "func_num",
            id="even_func_num",
        ),
        pytest.param(
            lambda valid: [torch.zeros(3, 1, 3, valid.shape[-1], dtype=torch.int32, device=valid.device)],
            "batch dimension",
            id="bad_batch",
        ),
        pytest.param(
            lambda valid: [torch.zeros(1, 3, 3, valid.shape[-1], dtype=torch.int32, device=valid.device)],
            "head dimension",
            id="bad_head",
        ),
        pytest.param(
            lambda valid: [valid[..., :-1]],
            "last dimension",
            id="short_seqlen_padding",
        ),
    ],
)
def test_arbitrary_mask_aux_tensor_validation(aux_factory, error_match):
    if _generic_case_for_arch() is None:
        pytest.skip("arbitrary mask target-dim test only runs on SM90/SM100")
    device = "cuda"
    dtype = torch.bfloat16
    head_dim, head_dim_v = _generic_case_for_arch()
    q = torch.randn(2, 64, 2, head_dim, device=device, dtype=dtype)
    k = torch.randn(2, 64, 2, head_dim, device=device, dtype=dtype)
    v = torch.randn(2, 64, 2, head_dim_v, device=device, dtype=dtype)
    valid = _make_arbitrary_func(1, 1, 64, 64, device)
    with pytest.raises(ValueError, match=error_match):
        flash_attn_func(q, k, v, arbitrary=True, aux_tensors=aux_factory(valid))


def test_arbitrary_mask_rejects_mla_qv():
    if _device_major() is None or _device_major() < 10:
        pytest.skip("MLA qv path is Blackwell-only")
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, 64, 2, 64, device=device, dtype=dtype)
    qv = torch.randn(1, 64, 2, 512, device=device, dtype=dtype)
    k = torch.randn(1, 64, 2, 64, device=device, dtype=dtype)
    v = torch.randn(1, 64, 2, 512, device=device, dtype=dtype)
    arbitrary_func = _make_arbitrary_func(1, 1, 64, 64, device)
    with pytest.raises(NotImplementedError, match="MLA qv"):
        flash_attn_func(q, k, v, qv=qv, arbitrary=True, aux_tensors=[arbitrary_func])


def test_arbitrary_mask_supports_hd256_gqa():
    if _device_major() != 10:
        pytest.skip("SM100 hd256 dedicated path is required")
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, 128, 2, 256, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 128, 1, 256, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, 128, 1, 256, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_head_dependent_arbitrary_func(1, 2, 128, 128, device)

    out, _ = flash_attn_func(q, k, v, arbitrary=True, aux_tensors=[arbitrary_func], return_lse=True)
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)
