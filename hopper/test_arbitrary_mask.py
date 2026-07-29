import math
import os

import pytest
import torch

import flash_attn_interface as fai
from flash_attn_interface import (
    LinearBlockSparseTensors,
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


try:
    from flash_attn_config import CONFIG as BUILD_CONFIG
except ImportError:
    BUILD_CONFIG = {"build_flags": {}}


def _build_flags():
    return BUILD_CONFIG.get("build_flags", {})


def _env_bool(name):
    if name not in os.environ:
        return None
    return os.environ[name].upper() == "TRUE"


def _build_flag(name, env_name=None, default=False):
    for candidate in dict.fromkeys((env_name or name, name)):
        value = _env_bool(candidate)
        if value is not None:
            return value
    flags = _build_flags()
    if name in flags:
        return flags[name]
    return default


def _compiled_hdim(head_dim):
    return not _build_flag(
        f"FLASHATTENTION_DISABLE_HDIM{head_dim}",
        f"FLASH_ATTENTION_DISABLE_HDIM{head_dim}",
    )


def _compiled_nfunc(nfunc):
    for env_name in ("FLASH_ATTENTION_NUM_FUNC", "FLASHATTENTION_NUM_FUNC"):
        if env_name in os.environ:
            values = os.environ[env_name].split(",")
            return str(nfunc) in {value.strip() for value in values}
    flags = _build_flags()
    if "FLASHATTENTION_NUM_FUNC" in flags:
        return nfunc in flags["FLASHATTENTION_NUM_FUNC"]
    return nfunc == 1


def _first_compiled_hdim():
    for head_dim in (64, 96, 128, 192, 256):
        if _compiled_hdim(head_dim):
            return head_dim
    pytest.skip("no arbitrary-mask head-dimension kernel was compiled")


def _require_arbitrary_hopper(head_dim=128, nfunc=1, backward=True):
    if torch.cuda.get_device_capability()[0] not in (8, 9):
        pytest.skip("C++ arbitrary mask tests require SM8x or SM90")
    if _build_flag("FLASHATTENTION_DISABLE_ARBITRARY", "FLASH_ATTENTION_DISABLE_ARBITRARY"):
        pytest.skip("arbitrary mask kernels were not compiled")
    if not _compiled_hdim(head_dim):
        pytest.skip(f"head_dim={head_dim} kernels were not compiled")
    if not _compiled_nfunc(nfunc):
        pytest.skip(f"arbitrary func_num={nfunc} kernels were not compiled")
    if backward and _build_flag("FLASHATTENTION_DISABLE_BACKWARD", "FLASH_ATTENTION_DISABLE_BACKWARD"):
        pytest.skip("backward kernels were not compiled")


def _ceildiv(x, y):
    return (x + y - 1) // y


def _dense_linear_block_sparse(num_outer_blocks, num_inner_blocks, device):
    mask_block_cnt = torch.full(
        (1, 1, num_outer_blocks), num_inner_blocks, dtype=torch.int32, device=device
    )
    mask_block_offset = torch.arange(
        num_outer_blocks + 1, dtype=torch.int32, device=device
    ) * num_inner_blocks
    mask_block_idx = torch.arange(num_inner_blocks, dtype=torch.int32, device=device).repeat(
        num_outer_blocks
    )
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_offset = torch.zeros(num_outer_blocks + 1, dtype=torch.int32, device=device)
    full_block_idx = torch.empty(0, dtype=torch.int32, device=device)
    return LinearBlockSparseTensors(
        mask_block_cnt,
        mask_block_offset,
        mask_block_idx,
        full_block_cnt,
        full_block_offset,
        full_block_idx,
    )


def _dense_fwd_block_size_sm90(head_dim):
    if head_dim <= 64:
        return 192, 128
    if head_dim <= 96:
        return 192, 128
    if head_dim <= 128:
        return 128, 128
    if head_dim <= 192:
        return 128, 96
    return 128, 64


def _dense_fwd_block_size_sm80(head_dim):
    if head_dim <= 64:
        return 128, 96
    if head_dim <= 96:
        return 128, 48
    if head_dim <= 128:
        return 128, 48
    return 128, 64


def _dense_bwd_block_size_sm90(head_dim):
    if head_dim <= 64:
        return 128, 128
    if head_dim <= 96:
        return 64, 128
    if head_dim <= 128:
        return 64, 128
    if head_dim <= 192:
        return 64, 96
    return 64, 80


def _dense_bwd_block_size_sm80(head_dim):
    if head_dim <= 64:
        return 128, 128
    if head_dim <= 96:
        return 64, 128
    if head_dim <= 128:
        return 64, 128
    if head_dim <= 192:
        return 64, 80
    return 64, 64


def _dense_block_sparse_pair(seqlen_q, seqlen_k, head_dim, device):
    major = torch.cuda.get_device_capability()[0]
    if major == 8:
        fwd_m, fwd_n = _dense_fwd_block_size_sm80(head_dim)
        bwd_m, bwd_n = _dense_bwd_block_size_sm80(head_dim)
    else:
        fwd_m, fwd_n = _dense_fwd_block_size_sm90(head_dim)
        bwd_m, bwd_n = _dense_bwd_block_size_sm90(head_dim)
    q2k = _dense_linear_block_sparse(
        _ceildiv(seqlen_q, fwd_m), _ceildiv(seqlen_k, fwd_n), device
    )
    k2q = _dense_linear_block_sparse(
        _ceildiv(seqlen_k, bwd_n), _ceildiv(seqlen_q, bwd_m), device
    )
    return q2k, k2q


def _make_native_mask_rejection_case():
    head_dim = _first_compiled_hdim()
    _require_arbitrary_hopper(head_dim=head_dim, backward=False)
    torch.manual_seed(20260729)
    device = "cuda"
    batch, seqlen_q, seqlen_k, heads = 1, 64, 64, 2
    q = torch.randn(
        batch, seqlen_q, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, seqlen_k, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    arbitrary_func = _make_arbitrary_func(
        1, 1, seqlen_q, seqlen_k, device
    )
    q2k, k2q = _dense_block_sparse_pair(
        seqlen_q, seqlen_k, head_dim, device
    )
    return q, k, v, arbitrary_func, q2k, k2q


def _kv_heads_for_mode(kv_mode, heads):
    if kv_mode == "mha":
        return heads
    if kv_mode == "gqa":
        return heads // 2
    if kv_mode == "mqa":
        return 1
    raise AssertionError(f"unknown kv_mode: {kv_mode}")


def _make_arbitrary_func(batch, heads, seqlen_q, seqlen_k, device, func_num=1):
    func = torch.zeros(batch, heads, func_num, seqlen_q + 256, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(heads):
            shift = (b if batch > 1 else 0) + (h if heads > 1 else 0)
            for q_idx in range(seqlen_q):
                func[b, h, 0, q_idx] = min(seqlen_k, q_idx + 1 + shift)
                for i in range(func_num // 2):
                    start = max(0, seqlen_k - 16 * (i + 1))
                    end = max(start, seqlen_k - 16 * i)
                    func[b, h, 2 * i + 1, q_idx] = start
                    func[b, h, 2 * i + 2, q_idx] = end
    return func


def _apply_arbitrary_mask(scores, arbitrary_func):
    batch, heads, seqlen_q, seqlen_k = scores.shape
    cols = torch.arange(seqlen_k, device=scores.device, dtype=torch.int32)
    cols = cols.view(1, 1, 1, seqlen_k)
    valid = cols < arbitrary_func[:, :, 0, :seqlen_q].unsqueeze(-1)
    for i in range(arbitrary_func.shape[2] // 2):
        start = arbitrary_func[:, :, 2 * i + 1, :seqlen_q].unsqueeze(-1)
        end = arbitrary_func[:, :, 2 * i + 2, :seqlen_q].unsqueeze(-1)
        valid = valid | ((cols >= start) & (cols < end))
    valid = valid.expand(batch, heads, seqlen_q, seqlen_k)
    return scores.masked_fill(~valid, float("-inf"))


def _attention_ref(q, k, v, arbitrary_func, softcap=0.0):
    q_ref = q.float()
    k_ref = k.float()
    v_ref = v.float()
    if q_ref.shape[2] != k_ref.shape[2]:
        repeat = q_ref.shape[2] // k_ref.shape[2]
        k_ref = k_ref.repeat_interleave(repeat, dim=2)
        v_ref = v_ref.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_ref * (1.0 / math.sqrt(q.shape[-1])), k_ref)
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    scores = _apply_arbitrary_mask(scores, arbitrary_func)
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v_ref).to(q.dtype)


def _varlen_attention_ref(q, k, v, cu_seqlens_q, cu_seqlens_k, arbitrary_func):
    outs = []
    batch = cu_seqlens_q.numel() - 1
    for b in range(batch):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        outs.append(
            _attention_ref(
                q[q_start:q_end].unsqueeze(0),
                k[k_start:k_end].unsqueeze(0),
                v[k_start:k_end].unsqueeze(0),
                arbitrary_func,
            ).squeeze(0)
        )
    return torch.cat(outs, dim=0)


def _run_case(
    *,
    kv_mode,
    mask_batch,
    mask_heads,
    head_dim=128,
    func_num=1,
    explicit_block_sparse=False,
    softcap=0.0,
    seed=0,
):
    _require_arbitrary_hopper(head_dim=head_dim, nfunc=func_num)
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen_q, seqlen_k, heads = 2, 64, 64, 4
    kv_heads = _kv_heads_for_mode(kv_mode, heads)
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype)
    if softcap > 0.0:
        q = q * (softcap / 4)
    q = q.detach().requires_grad_(True)
    k = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(
        mask_batch,
        mask_heads,
        seqlen_q,
        seqlen_k,
        device,
        func_num=func_num,
    )
    q2k_block_sparse = k2q_block_sparse = None
    if explicit_block_sparse:
        q2k_block_sparse, k2q_block_sparse = _dense_block_sparse_pair(
            seqlen_q, seqlen_k, head_dim, device
        )

    out = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        softcap=softcap,
        arbitrary_func=arbitrary_func,
        q2k_block_sparse=q2k_block_sparse,
        k2q_block_sparse=k2q_block_sparse,
    )
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func, softcap=softcap)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("func_num", [1, 3])
@pytest.mark.parametrize("head_dim", [64, 96, 128, 192, 256])
@pytest.mark.parametrize("kv_mode", ["mha", "gqa", "mqa"])
def test_arbitrary_mask_fixedlen_backward(kv_mode, head_dim, func_num):
    mask_heads = 1 if kv_mode == "mha" else 4
    _run_case(
        kv_mode=kv_mode,
        mask_batch=1,
        mask_heads=mask_heads,
        head_dim=head_dim,
        func_num=func_num,
        seed=17 + len(kv_mode) + head_dim + func_num,
    )


@pytest.mark.parametrize(
    "mask_batch,mask_heads",
    [
        pytest.param(1, 1, id="batch_head_broadcast"),
        pytest.param(2, 1, id="batch_specific"),
        pytest.param(1, 4, id="head_specific"),
        pytest.param(2, 4, id="batch_head_specific"),
    ],
)
def test_arbitrary_mask_broadcast_patterns(mask_batch, mask_heads):
    _run_case(
        kv_mode="gqa",
        mask_batch=mask_batch,
        mask_heads=mask_heads,
        seed=31 + mask_batch + mask_heads,
    )


def test_arbitrary_mask_explicit_linear_block_sparse():
    _run_case(
        kv_mode="gqa",
        mask_batch=1,
        mask_heads=4,
        explicit_block_sparse=True,
        seed=53,
    )


@pytest.mark.parametrize(
    "mode_kwargs",
    [
        pytest.param({"causal": True}, id="causal"),
        pytest.param({"window_size": (16, 16)}, id="local-window"),
        pytest.param({"attention_chunk": 16}, id="attention-chunk"),
    ],
)
def test_arbitrary_mask_rejects_native_causal_local_in_forward_only(mode_kwargs):
    q, k, v, arbitrary_func, q2k, _ = _make_native_mask_rejection_case()
    with pytest.raises(ValueError, match="cannot be combined with native causal/local"):
        flash_attn_func(
            q,
            k,
            v,
            deterministic=False,
            arbitrary_func=arbitrary_func,
            q2k_block_sparse=q2k,
            **mode_kwargs,
        )


def test_arbitrary_mask_fake_forward_rejects_native_causal_local():
    q, k, v, arbitrary_func, _, _ = _make_native_mask_rejection_case()
    for mode_kwargs in (
        {"causal": True},
        {"window_size_left": 16, "window_size_right": 16},
        {"attention_chunk": 16},
    ):
        with pytest.raises(
            ValueError, match="cannot be combined with native causal/local"
        ):
            fai._flash_attn_forward_fake(
                q,
                k,
                v,
                arbitrary_func=arbitrary_func,
                **mode_kwargs,
            )


def test_arbitrary_mask_kvcache_with_explicit_csr_rejects_native_causal_local():
    q, k_cache, v_cache, arbitrary_func, q2k, _ = (
        _make_native_mask_rejection_case()
    )
    with pytest.raises(ValueError, match="cannot be combined with native causal/local"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=k_cache.shape[1],
            causal=True,
            arbitrary_func=arbitrary_func,
            q2k_block_sparse=q2k,
            num_splits=1,
        )


def test_raw_cpp_forward_rejects_native_causal_local():
    q, k, v, arbitrary_func, q2k, _ = _make_native_mask_rejection_case()
    sparse_kwargs = {
        "block_sparse_mask_cnt": q2k.mask_block_cnt,
        "block_sparse_mask_offset": q2k.mask_block_offset,
        "block_sparse_mask_idx": q2k.mask_block_idx,
        "block_sparse_full_cnt": q2k.full_block_cnt,
        "block_sparse_full_offset": q2k.full_block_offset,
        "block_sparse_full_idx": q2k.full_block_idx,
    }
    for mode_kwargs in (
        {"is_causal": True},
        {"window_size_left": 16, "window_size_right": 16},
        {"attention_chunk": 16},
    ):
        with pytest.raises(
            (RuntimeError, ValueError),
            match="cannot be combined with native causal/local",
        ):
            torch.ops.flash_attn_3.fwd(
                q,
                k,
                v,
                num_splits=1,
                arbitrary_func=arbitrary_func,
                **sparse_kwargs,
                **mode_kwargs,
            )


def test_raw_cpp_backward_rejects_native_causal_local_when_nondeterministic():
    q, k, v, arbitrary_func, q2k, k2q = _make_native_mask_rejection_case()
    _require_arbitrary_hopper(head_dim=q.shape[-1], backward=True)
    out, softmax_lse, *_ = torch.ops.flash_attn_3.fwd(
        q,
        k,
        v,
        num_splits=1,
        block_sparse_mask_cnt=q2k.mask_block_cnt,
        block_sparse_mask_offset=q2k.mask_block_offset,
        block_sparse_mask_idx=q2k.mask_block_idx,
        block_sparse_full_cnt=q2k.full_block_cnt,
        block_sparse_full_offset=q2k.full_block_offset,
        block_sparse_full_idx=q2k.full_block_idx,
        arbitrary_func=arbitrary_func,
    )
    for mode_kwargs in (
        {"is_causal": True},
        {"window_size_left": 16, "window_size_right": 16},
    ):
        with pytest.raises(
            (RuntimeError, ValueError),
            match="cannot be combined with native causal/local",
        ):
            torch.ops.flash_attn_3.bwd(
                torch.randn_like(out),
                q,
                k,
                v,
                out,
                softmax_lse,
                torch.empty_like(q),
                torch.empty_like(k),
                torch.empty_like(v),
                deterministic=False,
                arbitrary_func=arbitrary_func,
                block_sparse_mask_cnt=k2q.mask_block_cnt,
                block_sparse_mask_offset=k2q.mask_block_offset,
                block_sparse_mask_idx=k2q.mask_block_idx,
                block_sparse_full_cnt=k2q.full_block_cnt,
                block_sparse_full_offset=k2q.full_block_offset,
                block_sparse_full_idx=k2q.full_block_idx,
                **mode_kwargs,
            )


def test_arbitrary_mask_softcap_backward():
    if _build_flag("FLASHATTENTION_DISABLE_SOFTCAP", "FLASH_ATTENTION_DISABLE_SOFTCAP"):
        pytest.skip("softcap kernels were not compiled")
    _run_case(
        kv_mode="mha",
        mask_batch=1,
        mask_heads=1,
        head_dim=128,
        softcap=15.0,
        seed=61,
    )


@pytest.mark.parametrize("head_dim", [64, 96, 128, 192, 256])
def test_arbitrary_mask_qkvpacked_backward(head_dim):
    _require_arbitrary_hopper(head_dim=head_dim)
    torch.manual_seed(67)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen, heads = 2, 64, 2
    qkv = torch.randn(
        batch, seqlen, 3, heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    qkv_ref = qkv.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen, seqlen, device)

    out = flash_attn_qkvpacked_func(qkv, causal=False, arbitrary_func=arbitrary_func)
    q_ref, k_ref, v_ref = qkv_ref.unbind(dim=2)
    out_ref = _attention_ref(q_ref, k_ref, v_ref, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(qkv.grad, qkv_ref.grad, atol=8e-2, rtol=8e-2)


def test_arbitrary_mask_varlen_backward():
    _require_arbitrary_hopper(head_dim=128)
    torch.manual_seed(79)
    device = "cuda"
    dtype = torch.bfloat16
    q_lens = [64, 48]
    k_lens = [64, 40]
    heads, head_dim = 2, 128
    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), dim=0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.cumsum(torch.tensor(k_lens), dim=0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    q = torch.randn(sum(q_lens), heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(sum(k_lens), heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(sum(k_lens), heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    arbitrary_func = _make_arbitrary_func(1, 1, max(q_lens), max(k_lens), device)
    arbitrary_func[..., : max(q_lens)].clamp_(max=min(k_lens))

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max(q_lens),
        max(k_lens),
        causal=False,
        arbitrary_func=arbitrary_func,
    )
    out_ref = _varlen_attention_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens_q,
        cu_seqlens_k,
        arbitrary_func,
    )
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=8e-2, rtol=8e-2)


def test_arbitrary_mask_kvcache_forward():
    _require_arbitrary_hopper(head_dim=128, backward=False)
    torch.manual_seed(83)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen_q, seqlen_k, heads, head_dim = 2, 16, 64, 2, 128
    q = torch.randn(batch, seqlen_q, heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch, seqlen_k, heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch, seqlen_k, heads, head_dim, device=device, dtype=dtype)
    arbitrary_func = _make_arbitrary_func(1, 1, seqlen_q, seqlen_k, device)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=seqlen_k,
        causal=False,
        arbitrary_func=arbitrary_func,
        num_splits=1,
    )
    out_ref = _attention_ref(q, k_cache, v_cache, arbitrary_func)
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
