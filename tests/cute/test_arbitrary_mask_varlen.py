import cutlass
import cutlass.cute as cute
import pytest
import torch
from flash_attn_cute import (
    create_arbitrary_block_sparse_tensors,
    flash_attn_func,
    flash_attn_varlen_func,
)
from flash_attn_cute.arbitrary_block_sparsity import (
    _CLASSIFY_COMPILE_CACHE,
    _MATERIALIZE_COMPILE_CACHE,
)
from flash_attn_cute.cute_dsl_utils import torch2cute_dtype_map
from flash_attn_cute.interface import _bwd_preprocess, _flash_attn_fwd
from flash_attn_cute.sm90_bwd_config import (
    resolve_sm90_bwd_consumer_config,
    sm90_native_bwd_can_implement,
)
from flash_attn_cute.sm90_fwd_config import (
    _num_sm90_fwd_mask_payload_groups,
    resolve_sm90_fwd_consumer_config,
    sm90_native_fwd_can_implement,
)


COMPUTE_CAPABILITY = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
)


def _causal_global_func(q_lengths, k_lengths, *, hmask=1, nfunc=1):
    total_q = sum(q_lengths)
    func = torch.zeros(
        hmask,
        nfunc,
        total_q + 256,
        dtype=torch.int32,
        device="cuda",
    )
    q_begin = 0
    k_begin = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        if q_len:
            local_end = torch.arange(
                1, q_len + 1, dtype=torch.int32, device="cuda"
            ).clamp(max=k_len)
            func[:, 0, q_begin : q_begin + q_len] = k_begin + local_end
        q_begin += q_len
        k_begin += k_len
    return func


def _varlen_reference(q, k, v, q_lengths, k_lengths):
    hq = q.shape[1]
    hkv = k.shape[1]
    scale = q.shape[-1] ** -0.5
    outputs = []
    lses = []
    q_begin = 0
    k_begin = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        if q_len == 0:
            k_begin += k_len
            continue
        q_cur = q[q_begin : q_begin + q_len].float().transpose(0, 1)
        k_cur = (
            k[k_begin : k_begin + k_len]
            .float()
            .repeat_interleave(hq // hkv, dim=1)
            .transpose(0, 1)
        )
        v_cur = (
            v[k_begin : k_begin + k_len]
            .float()
            .repeat_interleave(hq // hkv, dim=1)
            .transpose(0, 1)
        )
        if k_len == 0:
            outputs.append(torch.zeros(q_len, hq, v.shape[-1], device=q.device))
            lses.append(torch.full((hq, q_len), -torch.inf, device=q.device))
        else:
            scores = q_cur @ k_cur.transpose(-1, -2) * scale
            visible = (
                torch.arange(k_len, device=q.device)[None, :]
                <= torch.arange(q_len, device=q.device)[:, None]
            )
            scores.masked_fill_(~visible[None], -torch.inf)
            outputs.append((torch.softmax(scores, dim=-1) @ v_cur).transpose(0, 1))
            lses.append(torch.logsumexp(scores, dim=-1))
        q_begin += q_len
        k_begin += k_len
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


def _cu_seqlens(lengths):
    return torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )


def _resolve_fwd_config(
    *, head_dim=128, head_dim_v=128, dtype=torch.bfloat16, pack_gqa=False
):
    return resolve_sm90_fwd_consumer_config(
        arch=90,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=4,
        num_kv_heads=1,
        is_varlen=True,
        hmask=1,
        pack_gqa=pack_gqa,
    )


def _mask_payload_group_idx(consumer_tidx, qratio):
    warp_group_idx, tidx_in_warp_group = divmod(consumer_tidx, 128)
    a = tidx_in_warp_group % 4
    b = tidx_in_warp_group // 4 % 8
    c = tidx_in_warp_group // 32
    if qratio <= 8:
        return (
            warp_group_idx * (128 // qratio) + c * (32 // qratio) + b // qratio * 4 + a
        )
    logical_q = (warp_group_idx * 64 + c * 16 + b) // qratio
    return logical_q * 4 + a


def _mask_payload_representative_tidx(group_idx, qratio):
    a = group_idx % 4
    if qratio <= 8:
        groups_per_warp_group = 128 // qratio
        warp_group_idx, group_in_warp_group = divmod(group_idx, groups_per_warp_group)
        c, group_in_c = divmod(group_in_warp_group, 32 // qratio)
        b = group_in_c // 4 * qratio
        return warp_group_idx * 128 + c * 32 + b * 4 + a
    logical_q = group_idx // 4
    physical_q = logical_q * qratio
    warp_group_idx, q_in_warp_group = divmod(physical_q, 64)
    c = q_in_warp_group // 16
    return warp_group_idx * 128 + c * 32 + a


def _wgmma_logical_qk_signature(consumer_tidx, tile_n, qratio):
    warp_group_idx, tidx_in_warp_group = divmod(consumer_tidx, 128)
    a = tidx_in_warp_group % 4
    b = tidx_in_warp_group // 4 % 8
    c = tidx_in_warp_group // 32
    row_base = warp_group_idx * 64 + c * 16 + b
    values_per_thread = 64 * tile_n // 128
    signature = []
    for value_idx in range(values_per_thread):
        col_pair = value_idx % 2
        row_pair = value_idx // 2 % 2
        col_group = value_idx // 4
        row = row_base + row_pair * 8
        col = a * 2 + col_pair + col_group * 8
        signature.append((row // qratio, col))
    return tuple(signature)


def _reference_pack_gqa_payload(plan, func, q_len, k_len, config):
    qratio = config.qhead_per_kvhead
    func_cpu = func.cpu()
    expected = torch.zeros(
        (
            plan.mask_block_idx.numel(),
            1,
            config.num_mma_threads,
            config.payload_padded_words,
        ),
        dtype=torch.uint32,
    )
    offsets = plan.mask_block_offset.cpu()
    block_indices = plan.mask_block_idx.cpu()
    for plan_row in range(plan.mask_block_cnt.numel()):
        for payload_idx in range(int(offsets[plan_row]), int(offsets[plan_row + 1])):
            n_block = int(block_indices[payload_idx])
            for consumer_tidx in range(config.num_mma_threads):
                warp_group_idx, tidx_in_warp_group = divmod(consumer_tidx, 128)
                a = tidx_in_warp_group % 4
                b = tidx_in_warp_group // 4 % 8
                c = tidx_in_warp_group // 32
                row_base = warp_group_idx * 64 + c * 16 + b
                packed_words = [0] * config.payload_padded_words
                for value_idx in range(config.payload_values_per_thread):
                    col_pair = value_idx % 2
                    row_pair = value_idx // 2 % 2
                    col_group = value_idx // 4
                    physical_q = plan_row * config.tile_m + row_base + row_pair * 8
                    q_local = physical_q // qratio
                    k_local = n_block * config.tile_n + a * 2 + col_pair + col_group * 8
                    keep = False
                    if physical_q < q_len * qratio and k_local < k_len:
                        interval_begin = 0
                        for endpoint_idx in range(0, func_cpu.shape[1], 2):
                            interval_end = int(func_cpu[0, endpoint_idx, q_local])
                            if interval_begin <= k_local < interval_end:
                                keep = True
                            if endpoint_idx + 1 < func_cpu.shape[1]:
                                interval_begin = int(
                                    func_cpu[0, endpoint_idx + 1, q_local]
                                )
                    if keep:
                        word_idx, bit_idx = divmod(value_idx, 32)
                        packed_words[word_idx] |= 1 << bit_idx
                for word_idx, packed in enumerate(packed_words):
                    expected[payload_idx, 0, consumer_tidx, word_idx] = packed
    return expected


def _assert_compact_bwd_plan_invariants(plan):
    bwd = plan.bwd_tensors
    assert bwd is not None
    for counts, offsets, indices, write_order in (
        (
            bwd.mask_block_cnt,
            bwd.mask_block_offset,
            bwd.mask_block_idx,
            bwd.dq_write_order,
        ),
        (
            bwd.full_block_cnt,
            bwd.full_block_offset,
            bwd.full_block_idx,
            bwd.dq_write_order_full,
        ),
    ):
        assert counts is not None
        assert offsets is not None
        assert indices is not None
        assert write_order is not None
        assert torch.equal(counts.reshape(-1), offsets[1:] - offsets[:-1])
        assert int(offsets[-1]) == indices.numel() == write_order.numel()
        assert torch.equal(torch.bitwise_and(write_order, 0xFFFF), indices)


def _two_interval_global_func(q_lengths, k_lengths, *, hmask=1):
    total_q = sum(q_lengths)
    func = torch.zeros(
        hmask,
        3,
        total_q + 256,
        dtype=torch.int32,
        device="cuda",
    )
    q_begin = 0
    k_begin = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        for q_local in range(q_len):
            q_global = q_begin + q_local
            first_end = k_begin + min(k_len, 2 + q_local % 5)
            second_begin = k_begin + min(k_len, 8)
            second_end = k_begin + min(k_len, 9 + q_local // 2)
            func[:, 0, q_global] = first_end
            func[:, 1, q_global] = second_begin
            func[:, 2, q_global] = second_end
        q_begin += q_len
        k_begin += k_len
    return func


def _reference_attention(
    q,
    k,
    v,
    func,
    q_lengths,
    k_lengths,
    *,
    softcap=0.0,
    score_multiplier=1.0,
    kv_bias_scale=0.0,
):
    hq = q.shape[1]
    hkv = k.shape[1]
    scale = q.shape[-1] ** -0.5
    outputs = []
    q_begin = 0
    k_begin = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        if q_len == 0:
            k_begin += k_len
            continue
        q_cur = q[q_begin : q_begin + q_len].transpose(0, 1)
        k_cur = (
            k[k_begin : k_begin + k_len]
            .repeat_interleave(hq // hkv, dim=1)
            .transpose(0, 1)
        )
        v_cur = (
            v[k_begin : k_begin + k_len]
            .repeat_interleave(hq // hkv, dim=1)
            .transpose(0, 1)
        )
        scores = torch.matmul(q_cur, k_cur.transpose(-1, -2)) * scale
        scores = scores * score_multiplier
        if kv_bias_scale:
            kv_idx = torch.arange(k_len, dtype=scores.dtype, device=scores.device)
            scores = scores + kv_idx[None, None, :] * kv_bias_scale
        if softcap:
            scores = softcap * torch.tanh(scores / softcap)

        visible = torch.zeros(hq, q_len, k_len, dtype=torch.bool, device=q.device)
        nfunc = func.shape[1]
        for head in range(hq):
            mask_head = 0 if func.shape[0] == 1 else head
            for q_local in range(q_len):
                q_global = q_begin + q_local
                begin = 0
                for endpoint in range(0, nfunc, 2):
                    end = int(func[mask_head, endpoint, q_global].item())
                    lo = max(begin, k_begin) - k_begin
                    hi = min(end, k_begin + k_len) - k_begin
                    if hi > lo:
                        visible[head, q_local, lo:hi] = True
                    if endpoint + 1 < nfunc:
                        begin = int(func[mask_head, endpoint + 1, q_global].item())

        row_has_k = visible.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~visible, float("-inf"))
        safe_scores = torch.where(
            row_has_k, masked_scores, torch.zeros_like(masked_scores)
        )
        probs = torch.where(row_has_k, torch.softmax(safe_scores, dim=-1), 0.0)
        outputs.append(torch.matmul(probs, v_cur).transpose(0, 1))
        q_begin += q_len
        k_begin += k_len
    return torch.cat(outputs, dim=0) if outputs else q.new_empty((0, hq, v.shape[-1]))


def _run_arbitrary_backward(
    q,
    k,
    v,
    func,
    q_lengths,
    k_lengths,
    dout,
    *,
    deterministic=False,
    softcap=0.0,
    score_mod=None,
    score_mod_bwd=None,
    pack_gqa=True,
):
    cu_q = _cu_seqlens(q_lengths)
    cu_k = _cu_seqlens(k_lengths)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths, default=0),
        max_seqlen_k=max(k_lengths, default=0),
        pack_gqa=pack_gqa,
        build_backward=True,
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths, default=0),
        max_seqlen_k=max(k_lengths, default=0),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        softcap=softcap,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        return_lse=True,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, lse, dq, dk, dv, plan


def _run_arbitrary_fixed_backward(
    q,
    k,
    v,
    func,
    dout,
    *,
    deterministic=False,
):
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=deterministic,
        return_lse=True,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
    return out, lse, dq, dk, dv, plan


# Forward tests


def test_varlen_arbitrary_mask_plan_compile_keys_only_specialize_generated_layout():
    dv64 = _resolve_fwd_config(head_dim_v=64)
    dv128 = _resolve_fwd_config(head_dim_v=128)
    dv192 = _resolve_fwd_config(head_dim_v=192)
    fp16 = _resolve_fwd_config(dtype=torch.float16)
    packed = _resolve_fwd_config(pack_gqa=True)

    assert dv64.topology_planner_compile_key == dv128.topology_planner_compile_key
    assert dv64.payload_planner_compile_key == dv128.payload_planner_compile_key
    assert dv128.topology_planner_compile_key == dv192.topology_planner_compile_key
    assert dv128.payload_planner_compile_key == dv192.payload_planner_compile_key
    assert dv128.topology_planner_compile_key == fp16.topology_planner_compile_key
    assert dv128.payload_planner_compile_key != fp16.payload_planner_compile_key
    assert dv128.topology_planner_compile_key != packed.topology_planner_compile_key


@pytest.mark.parametrize("tile_m", [128, 192])
def test_sm90_arbitrary_mask_pack_gqa_payload_groups_match_wgmma_layout(tile_m):
    num_mma_threads = tile_m * 2
    for qratio in (1, 2, 4, 8, 16, 32, 64, 128):
        if tile_m % qratio != 0:
            continue
        num_groups = _num_sm90_fwd_mask_payload_groups(
            num_mma_threads=num_mma_threads,
            qhead_per_kvhead=qratio,
            pack_gqa=True,
        )
        expected_compression = qratio if qratio <= 8 else qratio // 2
        assert num_groups * expected_compression == num_mma_threads

        groups = [[] for _ in range(num_groups)]
        for consumer_tidx in range(num_mma_threads):
            group_idx = _mask_payload_group_idx(consumer_tidx, qratio)
            assert 0 <= group_idx < num_groups
            groups[group_idx].append(consumer_tidx)

        for group_idx, members in enumerate(groups):
            representative_tidx = _mask_payload_representative_tidx(group_idx, qratio)
            assert representative_tidx in members
            representative_signature = _wgmma_logical_qk_signature(
                representative_tidx, 128, qratio
            )
            assert all(
                _wgmma_logical_qk_signature(tidx, 128, qratio)
                == representative_signature
                for tidx in members
            )


@pytest.mark.parametrize(
    "head_dim,num_q_heads,num_kv_heads,requested_pack_gqa,expected_pack_gqa,error",
    [
        (128, 4, 1, None, True, None),
        (128, 6, 1, None, False, None),
        (128, 6, 1, True, None, "power of two"),
        (64, 128, 1, None, False, None),
        (64, 128, 1, True, None, "divisible by qratio"),
        (128, 4, 1, False, False, None),
    ],
)
def test_sm90_arbitrary_mask_pack_gqa_resolution(
    head_dim,
    num_q_heads,
    num_kv_heads,
    requested_pack_gqa,
    expected_pack_gqa,
    error,
):
    kwargs = dict(
        arch=90,
        dtype=torch.bfloat16,
        head_dim=head_dim,
        head_dim_v=128,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        is_varlen=True,
        hmask=1,
        pack_gqa=requested_pack_gqa,
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            resolve_sm90_fwd_consumer_config(**kwargs)
    else:
        assert resolve_sm90_fwd_consumer_config(**kwargs).pack_gqa is expected_pack_gqa


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_pack_gqa_payload_matches_per_thread_layout():
    torch.manual_seed(0)
    q_len, k_len, hq, hkv, head_dim = 65, 289, 4, 1, 128
    q = torch.randn(1, q_len, hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, k_len, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    func = torch.zeros(1, 3, q_len + 256, dtype=torch.int32, device="cuda")
    q_idx = torch.arange(q_len, dtype=torch.int32, device="cuda")
    func[0, 0, :q_len] = q_idx + 1
    func[0, 1, :q_len] = 128
    func[0, 2, :q_len] = 256 + q_idx % 33

    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=True)
    config = _resolve_fwd_config(pack_gqa=True)
    assert plan.mask_block_masks.shape[2] == config.num_mask_payload_groups
    assert plan.mask_block_masks.shape[2] * 4 == config.num_mma_threads

    group_indices = torch.tensor(
        [
            _mask_payload_group_idx(consumer_tidx, config.qhead_per_kvhead)
            for consumer_tidx in range(config.num_mma_threads)
        ],
    )
    expanded = plan.mask_block_masks.cpu()[:, :, group_indices, :]
    expected = _reference_pack_gqa_payload(plan, func, q_len, k_len, config)
    assert torch.equal(expanded, expected)


@pytest.mark.parametrize(
    "head_dim,head_dim_v,expected",
    [
        (64, 64, (192, 128, True, True, 2)),
        (64, 128, (192, 128, True, True, 2)),
        (128, 64, (128, 128, True, True, 2)),
        (128, 192, (128, 128, True, True, 2)),
        (192, 128, (128, 128, True, True, 2)),
        (192, 192, (128, 112, True, True, 2)),
        (256, 64, (128, 80, True, True, 2)),
        (256, 256, (128, 80, True, True, 2)),
    ],
)
def test_sm90_arbitrary_mask_fwd_config_resource_families(
    head_dim, head_dim_v, expected
):
    assert sm90_native_fwd_can_implement(head_dim, head_dim_v)
    config = _resolve_fwd_config(head_dim=head_dim, head_dim_v=head_dim_v)
    assert (
        config.tile_m,
        config.tile_n,
        config.mma_pv_is_rs,
        config.intra_wg_overlap,
        config.num_stages,
    ) == expected


@pytest.mark.parametrize(
    "head_dim,head_dim_v",
    [(64, 192), (64, 256), (128, 256), (192, 256)],
)
def test_sm90_arbitrary_mask_fwd_config_rejects_non_native_resource_families(
    head_dim, head_dim_v
):
    assert not sm90_native_fwd_can_implement(head_dim, head_dim_v)
    with pytest.raises(NotImplementedError, match="native"):
        _resolve_fwd_config(head_dim=head_dim, head_dim_v=head_dim_v)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(
    "head_dim,head_dim_v",
    [(64, 128), (128, 192), (192, 192), (256, 64)],
)
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_fwd_matches_causal_reference(
    dtype, pack_gqa, head_dim, head_dim_v
):
    torch.manual_seed(0)
    batch, seqlen, hq, hkv = 1, 128, 4, 1
    q = torch.randn(batch, seqlen, hq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen, hkv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, hkv, head_dim_v, device="cuda", dtype=dtype)
    func = _causal_global_func([seqlen], [seqlen])
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=pack_gqa)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=pack_gqa,
        return_lse=True,
    )
    ref, ref_lse = _varlen_reference(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        [seqlen],
        [seqlen],
    )
    torch.testing.assert_close(out.flatten(0, 1), ref.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse[None], atol=2e-4, rtol=2e-4)
    assert plan.pack_gqa is pack_gqa
    assert plan.mask_block_masks.dtype == torch.uint32
    assert plan.mask_block_masks.data_ptr() % 16 == 0


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_fwd_d192_full_only_row_uses_partial_anchor():
    torch.manual_seed(0)
    q = torch.randn(1, 128, 4, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 224, 1, 192, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    func = torch.full(
        (1, 1, q.shape[1] + 256),
        k.shape[1],
        dtype=torch.int32,
        device="cuda",
    )
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)

    assert plan.mask_block_cnt.tolist() == [[1]]
    assert plan.full_block_cnt.tolist() == [[1]]
    assert plan.mask_block_idx.tolist() == [1]
    assert plan.full_block_idx.tolist() == [0]

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    ref, ref_lse = flash_attn_func(
        q, k, v, causal=False, pack_gqa=False, return_lse=True
    )
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=0, rtol=0)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_fwd_unequal_and_zero_length():
    torch.manual_seed(0)
    q_lengths = [32, 0, 17]
    k_lengths = [64, 0, 25]
    cu_q = _cu_seqlens(q_lengths)
    cu_k = _cu_seqlens(k_lengths)
    hq, hkv, head_dim, head_dim_v = 4, 1, 128, 192
    q = torch.randn(sum(q_lengths), hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lengths), hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(
        sum(k_lengths), hkv, head_dim_v, device="cuda", dtype=torch.bfloat16
    )
    func = _causal_global_func(q_lengths, k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        pack_gqa=True,
    )

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=True,
        return_lse=True,
    )
    ref, ref_lse = _varlen_reference(q, k, v, q_lengths, k_lengths)
    torch.testing.assert_close(out, ref.to(out.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)
    assert plan.cu_total_m_blocks.tolist() == [0, 1, 1, 2]


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_fwd_ignores_other_sample_k_intervals():
    torch.manual_seed(0)
    q_lengths = [3, 2]
    k_lengths = [4, 3]
    cu_q = _cu_seqlens(q_lengths)
    cu_k = _cu_seqlens(k_lengths)
    hq, hkv, head_dim, head_dim_v = 2, 1, 64, 64
    q = torch.randn(sum(q_lengths), hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(k_lengths), hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(
        sum(k_lengths), hkv, head_dim_v, device="cuda", dtype=torch.bfloat16
    )
    func = torch.zeros(1, 1, sum(q_lengths) + 256, dtype=torch.int32, device="cuda")
    func[0, 0, : q_lengths[0]] = sum(k_lengths)
    func[0, 0, q_lengths[0] : sum(q_lengths)] = k_lengths[0]
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        pack_gqa=False,
    )

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )

    q_first = q[: q_lengths[0]].float().transpose(0, 1)
    k_first = k[: k_lengths[0]].float().repeat_interleave(hq, dim=1).transpose(0, 1)
    v_first = v[: k_lengths[0]].float().repeat_interleave(hq, dim=1).transpose(0, 1)
    scores_first = q_first @ k_first.transpose(-1, -2) * head_dim**-0.5
    out_first = (torch.softmax(scores_first, dim=-1) @ v_first).transpose(0, 1)
    ref = torch.cat(
        (
            out_first,
            torch.zeros(q_lengths[1], hq, head_dim_v, device="cuda"),
        )
    )
    ref_lse = torch.cat(
        (
            torch.logsumexp(scores_first, dim=-1),
            torch.full((hq, q_lengths[1]), -torch.inf, device="cuda"),
        ),
        dim=1,
    )
    torch.testing.assert_close(out, ref.to(out.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)
    assert torch.count_nonzero(out[q_lengths[0] :]) == 0
    assert torch.isneginf(lse[:, q_lengths[0] :]).all()


@pytest.mark.parametrize("mode", ["fixed", "varlen"])
@pytest.mark.parametrize("has_sink", [False, True])
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_arbitrary_mask_fwd_empty_rows_with_optional_sink(mode, has_sink):
    torch.manual_seed(0)
    hq, hkv, head_dim, head_dim_v = 2, 1, 64, 64
    sink = (
        torch.tensor([0.5, -0.25], dtype=torch.bfloat16, device="cuda")
        if has_sink
        else None
    )

    if mode == "fixed":
        batch, seqlen_q, seqlen_k = 2, 9, 13
        q = torch.randn(
            batch,
            seqlen_q,
            hq,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.randn(
            batch,
            seqlen_k,
            hkv,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        v = torch.randn(
            batch,
            seqlen_k,
            hkv,
            head_dim_v,
            dtype=torch.bfloat16,
            device="cuda",
        )
        func = torch.zeros(
            1, 1, batch * seqlen_q + 256, dtype=torch.int32, device="cuda"
        )
        plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)
        out, lse = flash_attn_func(
            q,
            k,
            v,
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=False,
            learnable_sink=sink,
            return_lse=True,
        )
    else:
        q_lengths = [9, 0, 7]
        k_lengths = [13, 0, 11]
        cu_q = _cu_seqlens(q_lengths)
        cu_k = _cu_seqlens(k_lengths)
        q = torch.randn(
            sum(q_lengths), hq, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        k = torch.randn(
            sum(k_lengths), hkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn(
            sum(k_lengths), hkv, head_dim_v, dtype=torch.bfloat16, device="cuda"
        )
        func = torch.zeros(1, 1, sum(q_lengths) + 256, dtype=torch.int32, device="cuda")
        plan = create_arbitrary_block_sparse_tensors(
            func,
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            pack_gqa=False,
        )
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=False,
            learnable_sink=sink,
            return_lse=True,
        )

    assert torch.count_nonzero(out) == 0
    if sink is None:
        assert torch.isneginf(lse).all()
    else:
        expected = (
            sink.float()[None, :, None] if mode == "fixed" else sink.float()[:, None]
        ).expand_as(lse)
        torch.testing.assert_close(lse, expected, atol=1e-6, rtol=0)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_fwd_gqa_multi_kv_head_and_different_value_dim():
    torch.manual_seed(0)
    batch, seqlen, hq, hkv, head_dim, head_dim_v = 1, 96, 4, 2, 128, 64
    q = torch.randn(batch, seqlen, hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, hkv, head_dim_v, device="cuda", dtype=torch.bfloat16)
    func = _causal_global_func([seqlen], [seqlen])
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=True)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=True,
        return_lse=True,
    )
    ref, ref_lse = _varlen_reference(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        [seqlen],
        [seqlen],
    )
    torch.testing.assert_close(
        out.flatten(0, 1), ref.to(out.dtype), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(lse, ref_lse[None], atol=2e-4, rtol=2e-4)
    assert plan.block_size == (128, 128)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_fwd_pack_gqa_with_partial_and_full_blocks():
    torch.manual_seed(0)
    seqlen, hq, hkv, head_dim, head_dim_v = 512, 4, 1, 128, 192
    q = torch.randn(1, seqlen, hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, seqlen, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, seqlen, hkv, head_dim_v, device="cuda", dtype=torch.bfloat16)

    q_idx = torch.arange(seqlen, device="cuda", dtype=torch.int32)
    func = torch.zeros(1, 3, seqlen + 256, device="cuda", dtype=torch.int32)
    func[0, 0, :seqlen] = q_idx.remainder(101) + 1
    func[0, 1, :seqlen] = 160
    func[0, 2, :seqlen] = 400 + q_idx.remainder(101)
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=True)

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=True,
        return_lse=True,
    )

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().repeat_interleave(hq, dim=2).transpose(1, 2)
    v_ref = v.float().repeat_interleave(hq, dim=2).transpose(1, 2)
    scores = q_ref @ k_ref.transpose(-1, -2) * head_dim**-0.5
    k_idx = torch.arange(seqlen, device="cuda")[None, :]
    first_end = (q_idx.remainder(101) + 1)[:, None]
    second_end = (400 + q_idx.remainder(101))[:, None]
    visible = (k_idx < first_end) | ((k_idx >= 160) & (k_idx < second_end))
    scores.masked_fill_(~visible[None, None], -torch.inf)
    ref = (torch.softmax(scores, dim=-1) @ v_ref).transpose(1, 2)
    ref_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, ref.to(out.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)
    assert plan.mask_block_idx.numel() > 0
    assert plan.full_block_idx.numel() > 0


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_fwd_is_batch_invariant():
    torch.manual_seed(0)
    hq, hkv, head_dim = 4, 1, 128
    target_q_len, target_k_len = 37, 53
    target_q = torch.randn(
        target_q_len, hq, head_dim, device="cuda", dtype=torch.bfloat16
    )
    target_k = torch.randn(
        target_k_len, hkv, head_dim, device="cuda", dtype=torch.bfloat16
    )
    target_v = torch.randn_like(target_k)

    def run(prefix_q_len, prefix_k_len):
        prefix_q = torch.randn(
            prefix_q_len, hq, head_dim, device="cuda", dtype=torch.bfloat16
        )
        prefix_k = torch.randn(
            prefix_k_len, hkv, head_dim, device="cuda", dtype=torch.bfloat16
        )
        prefix_v = torch.randn_like(prefix_k)
        q = torch.cat((prefix_q, target_q))
        k = torch.cat((prefix_k, target_k))
        v = torch.cat((prefix_v, target_v))
        q_lengths = [prefix_q_len, target_q_len]
        k_lengths = [prefix_k_len, target_k_len]
        cu_q = _cu_seqlens(q_lengths)
        cu_k = _cu_seqlens(k_lengths)
        func = _causal_global_func(q_lengths, k_lengths)
        plan = create_arbitrary_block_sparse_tensors(
            func,
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            pack_gqa=True,
        )
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(q_lengths),
            max_seqlen_k=max(k_lengths),
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=True,
        )
        return out[prefix_q_len:]

    torch.testing.assert_close(run(11, 29), run(47, 7), atol=0, rtol=0)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_plan_csr_is_stable_and_exact():
    q = torch.randn(1, 256, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    func = _causal_global_func([256], [256])
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)

    assert plan.block_size == (128, 128)
    assert plan.mask_block_cnt.tolist() == [[1, 1]]
    assert plan.full_block_cnt.tolist() == [[0, 1]]
    assert plan.mask_block_offset.tolist() == [0, 1, 2]
    assert plan.full_block_offset.tolist() == [0, 0, 1]
    assert plan.mask_block_idx.tolist() == [0, 1]
    assert plan.full_block_idx.tolist() == [0]
    assert plan.mask_block_masks.shape[0] == plan.mask_block_idx.numel()


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_plan_hmask_and_nfunc_are_runtime():
    torch.manual_seed(0)
    seqlen, hq, hkv, head_dim = 64, 4, 1, 128
    q = torch.randn(1, seqlen, hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, seqlen, hkv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    broadcast_func = _causal_global_func([seqlen], [seqlen])
    broadcast_plan = create_arbitrary_block_sparse_tensors(
        broadcast_func, q, k, v, pack_gqa=False
    )
    planner_cache_entries = (
        len(_CLASSIFY_COMPILE_CACHE),
        len(_MATERIALIZE_COMPILE_CACHE),
    )
    flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=broadcast_plan,
        pack_gqa=False,
        return_lse=True,
    )
    attention_cache_entries = len(_flash_attn_fwd.compile_cache.cache)

    func = torch.zeros(hq, 3, seqlen + 256, dtype=torch.int32, device="cuda")
    func[0, 0, :seqlen] = torch.arange(1, seqlen + 1, dtype=torch.int32, device="cuda")
    func[1, 0, :seqlen] = 16
    func[2, 1, :seqlen] = 32
    func[2, 2, :seqlen] = seqlen
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)
    assert (
        len(_CLASSIFY_COMPILE_CACHE),
        len(_MATERIALIZE_COMPILE_CACHE),
    ) == planner_cache_entries
    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    assert len(_flash_attn_fwd.compile_cache.cache) == attention_cache_entries

    q_ref = q.float().transpose(1, 2)
    k_ref = k.float().repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
    v_ref = v.float().repeat_interleave(hq // hkv, dim=2).transpose(1, 2)
    scores = q_ref @ k_ref.transpose(-1, -2) * head_dim**-0.5
    q_idx = torch.arange(seqlen, device="cuda")[:, None]
    k_idx = torch.arange(seqlen, device="cuda")[None, :]
    visible = torch.stack(
        (
            k_idx <= q_idx,
            (k_idx < 16).expand(seqlen, -1),
            (k_idx >= 32).expand(seqlen, -1),
            torch.zeros(seqlen, seqlen, dtype=torch.bool, device="cuda"),
        )
    )
    scores.masked_fill_(~visible[None], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1).nan_to_num()
    ref = (probabilities @ v_ref).transpose(1, 2)
    ref_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(out, ref.to(out.dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing_plan", "requires block_sparse_tensors"),
        ("causal", "cannot be combined"),
        ("wrong_pack", "does not match"),
    ],
)
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_interface_rejects_invalid_options(mutation, error):
    q = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    func = _causal_global_func([32], [32])
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)
    kwargs = dict(arbitrary=True, block_sparse_tensors=plan, pack_gqa=False)
    if mutation == "missing_plan":
        kwargs["block_sparse_tensors"] = None
    elif mutation == "causal":
        kwargs["causal"] = True
    else:
        kwargs["pack_gqa"] = True
    with pytest.raises((ValueError, NotImplementedError), match=error):
        flash_attn_func(q, k, v, **kwargs)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_bwd_requires_backward_plan():
    q = torch.randn(
        1, 32, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    func = _causal_global_func([32], [32])
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)
    out, _ = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
    )
    with pytest.raises(ValueError, match="build_backward=True"):
        out.sum().backward()


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_plan_builder_validation():
    q = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    even_nfunc = torch.zeros(1, 2, 288, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="positive odd"):
        create_arbitrary_block_sparse_tensors(even_nfunc, q, k, v)

    head_mask = torch.zeros(4, 1, 288, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires Hmask=1"):
        create_arbitrary_block_sparse_tensors(head_mask, q, k, v, pack_gqa=True)

    invalid_endpoint = _causal_global_func([32], [32])
    invalid_endpoint[0, 0, 0] = -1
    with pytest.raises(ValueError, match="endpoints must be in"):
        create_arbitrary_block_sparse_tensors(invalid_endpoint, q, k, v, pack_gqa=False)


# Backward tests


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_sm90_varlen_bwd_preprocess_small_dv_stays_within_each_tile():
    tile_m = 64
    head_dim = 184
    head_dim_v = 8
    head_dim_padded = 192
    lengths = [65, 63] * 32
    cu_seqlens = _cu_seqlens(lengths)
    total_q = sum(lengths)
    total_q_padded = (total_q + cu_seqlens.numel() * tile_m - 1) // tile_m * tile_m

    out = torch.ones(total_q, 4, head_dim_v, dtype=torch.bfloat16, device="cuda")
    dout = torch.ones_like(out)
    lse = torch.zeros(4, total_q, dtype=torch.float32, device="cuda")
    dpsum = torch.full((4, total_q_padded), -1.0, dtype=torch.float32, device="cuda")
    lse_log2 = torch.full_like(dpsum, -1.0)
    dq_accum = torch.full(
        (4, total_q_padded * head_dim_padded),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )

    _bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens,
        None,
        None,
        torch2cute_dtype_map[out.dtype],
        head_dim,
        head_dim_v,
        tile_m,
        use_padded_offsets=False,
        accum_hdim_multiple=16,
    )
    torch.cuda.synchronize()

    valid_rows = []
    offset = 0
    for batch_idx, length in enumerate(lengths):
        padded_offset = (offset + batch_idx * tile_m) // tile_m * tile_m
        valid_rows.append(
            torch.arange(
                padded_offset,
                padded_offset + length,
                dtype=torch.int64,
                device="cuda",
            )
        )
        offset += length
    valid_rows = torch.cat(valid_rows)
    torch.testing.assert_close(
        dpsum[:, valid_rows], torch.full_like(dpsum[:, valid_rows], 8.0)
    )
    torch.testing.assert_close(
        lse_log2[:, valid_rows], torch.zeros_like(lse_log2[:, valid_rows])
    )
    valid_dq = (
        valid_rows[:, None] * head_dim_padded
        + torch.arange(head_dim_padded, device="cuda")[None, :]
    ).reshape(-1)
    torch.testing.assert_close(
        dq_accum[:, valid_dq], torch.zeros_like(dq_accum[:, valid_dq])
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm90_arbitrary_mask_configs_follow_native_64_aligned_support(dtype):
    dims = (64, 128, 192, 256)
    fwd_supported_pairs = set()
    bwd_supported_pairs = {4: set(), 2: set(), 1: set()}
    for head_dim in dims:
        for head_dim_v in dims:
            fwd_supported = sm90_native_fwd_can_implement(head_dim, head_dim_v)
            for pack_gqa in (False, True):
                kwargs = dict(
                    arch=90,
                    dtype=dtype,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    num_q_heads=4,
                    num_kv_heads=1,
                    is_varlen=True,
                    hmask=1,
                    pack_gqa=pack_gqa,
                )
                if fwd_supported:
                    resolve_sm90_fwd_consumer_config(**kwargs)
                    fwd_supported_pairs.add((head_dim, head_dim_v))
                else:
                    with pytest.raises(NotImplementedError, match="native"):
                        resolve_sm90_fwd_consumer_config(**kwargs)
            for num_kv_heads in (4, 2, 1):
                kwargs = dict(
                    arch=90,
                    dtype=dtype,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    num_q_heads=4,
                    num_kv_heads=num_kv_heads,
                    is_varlen=True,
                )
                bwd_supported = sm90_native_bwd_can_implement(
                    head_dim,
                    head_dim_v,
                    4,
                    num_kv_heads,
                )
                if bwd_supported:
                    resolve_sm90_bwd_consumer_config(**kwargs)
                    bwd_supported_pairs[num_kv_heads].add((head_dim, head_dim_v))
                else:
                    with pytest.raises(NotImplementedError, match="native"):
                        resolve_sm90_bwd_consumer_config(**kwargs)

    assert len(fwd_supported_pairs) == 12
    assert {hkv: len(pairs) for hkv, pairs in bwd_supported_pairs.items()} == {
        4: 10,
        2: 4,
        1: 4,
    }


def test_sm90_arbitrary_mask_bwd_config_matches_native_resource_boundaries():
    supported = {
        (128, 192, 4): (64, 128, 2),
        (192, 128, 4): (64, 96, 2),
        (256, 128, 4): (64, 64, 1),
        (256, 256, 1): (64, 64, 1),
    }
    for (head_dim, head_dim_v, hkv), expected in supported.items():
        config = resolve_sm90_bwd_consumer_config(
            arch=90,
            dtype=torch.bfloat16,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_q_heads=4,
            num_kv_heads=hkv,
            is_varlen=True,
        )
        assert (config.tile_m, config.tile_n, config.num_stages_q) == expected

    for head_dim, head_dim_v, hkv in (
        (64, 128, 4),
        (64, 128, 1),
        (192, 256, 4),
        (256, 192, 4),
        (256, 128, 1),
    ):
        with pytest.raises(NotImplementedError, match="native"):
            resolve_sm90_bwd_consumer_config(
                arch=90,
                dtype=torch.bfloat16,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                num_q_heads=4,
                num_kv_heads=hkv,
                is_varlen=True,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "head_dim,head_dim_v,hkv",
    [
        (64, 64, 2),
        (128, 192, 4),
        (192, 128, 4),
        (256, 256, 1),
    ],
)
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_fixed_arbitrary_mask_bwd_matches_reference(
    dtype, deterministic, head_dim, head_dim_v, hkv
):
    torch.manual_seed(0)
    batch, seqlen_q, seqlen_k, hq = 2, 33, 41, 4
    q = torch.randn(batch, seqlen_q, hq, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, seqlen_k, hkv, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, seqlen_k, hkv, head_dim_v, dtype=dtype, device="cuda")
    dout = torch.randn(batch, seqlen_q, hq, head_dim_v, dtype=dtype, device="cuda")
    q_lengths = [seqlen_q] * batch
    k_lengths = [seqlen_k] * batch
    func = _causal_global_func(q_lengths, k_lengths)

    out, _, dq, dk, dv, plan = _run_arbitrary_fixed_backward(
        q,
        k,
        v,
        func,
        dout,
        deterministic=deterministic,
    )

    q_ref = q.flatten(0, 1).float().requires_grad_(True)
    k_ref = k.flatten(0, 1).float().requires_grad_(True)
    v_ref = v.flatten(0, 1).float().requires_grad_(True)
    out_ref = _reference_attention(
        q_ref, k_ref, v_ref, func, q_lengths, k_lengths
    ).view_as(out.float())
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        dout.float(),
    )
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq.float(), dq_ref.view_as(dq), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk.float(), dk_ref.view_as(dk), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dv.float(), dv_ref.view_as(dv), atol=5e-2, rtol=5e-2)
    assert plan.bwd_tensors is not None
    assert plan.bwd_tensors.cu_total_m_blocks is None
    _assert_compact_bwd_plan_invariants(plan)


@pytest.mark.parametrize(
    "dtype,head_dim,head_dim_v,hkv",
    [
        (torch.float16, 128, 128, 4),
        (torch.bfloat16, 64, 64, 1),
        (torch.bfloat16, 128, 192, 4),
        (torch.bfloat16, 192, 128, 4),
        (torch.bfloat16, 192, 192, 1),
        (torch.bfloat16, 256, 128, 4),
        (torch.bfloat16, 256, 256, 1),
    ],
)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_matches_reference(
    dtype, head_dim, head_dim_v, hkv, deterministic
):
    torch.manual_seed(0)
    q_lengths = [17, 0, 31]
    k_lengths = [23, 0, 37]
    hq = 4
    q = torch.randn(sum(q_lengths), hq, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(sum(k_lengths), hkv, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(sum(k_lengths), hkv, head_dim_v, dtype=dtype, device="cuda")
    dout = torch.randn(sum(q_lengths), hq, head_dim_v, dtype=dtype, device="cuda")
    func = _two_interval_global_func(q_lengths, k_lengths)

    out, _, dq, dk, dv, plan = _run_arbitrary_backward(
        q,
        k,
        v,
        func,
        q_lengths,
        k_lengths,
        dout,
        deterministic=deterministic,
        pack_gqa=True,
    )

    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    out_ref = _reference_attention(q_ref, k_ref, v_ref, func, q_lengths, k_lengths)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout.float()
    )
    qk_grad_atol = 6e-2 if dtype == torch.bfloat16 and head_dim < 32 else 3e-2
    torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=qk_grad_atol, rtol=3e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=qk_grad_atol, rtol=3e-2)
    torch.testing.assert_close(dv.float(), dv_ref, atol=3e-2, rtol=3e-2)
    assert plan.bwd_tensors is not None
    assert plan.bwd_tensors.pack_gqa is None
    _assert_compact_bwd_plan_invariants(plan)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_nondeterministic_row_major_dq_reuse_matches_reference():
    torch.manual_seed(0)
    q_lengths = [257]
    k_lengths = [257]
    head_dim, head_dim_v, hq, hkv = 128, 64, 4, 4
    q = torch.randn(sum(q_lengths), hq, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(sum(k_lengths), hkv, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(sum(k_lengths), hkv, head_dim_v, dtype=torch.float16, device="cuda")
    dout = torch.randn(
        sum(q_lengths), hq, head_dim_v, dtype=torch.float16, device="cuda"
    )
    func = _causal_global_func(q_lengths, k_lengths)

    _, _, dq, dk, dv, _ = _run_arbitrary_backward(
        q,
        k,
        v,
        func,
        q_lengths,
        k_lengths,
        dout,
        deterministic=False,
        pack_gqa=False,
    )

    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    out_ref = _reference_attention(q_ref, k_ref, v_ref, func, q_lengths, k_lengths)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout.float()
    )
    torch.testing.assert_close(dq.float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv.float(), dv_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_deterministic_handles_partial_and_full_k2q_blocks():
    torch.manual_seed(0)
    q_lengths = [384]
    k_lengths = [384]
    hq, hkv, head_dim, head_dim_v = 2, 2, 64, 64
    q = torch.randn(384, hq, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(384, hkv, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(384, hkv, head_dim_v, dtype=torch.bfloat16, device="cuda")
    dout = torch.randn(384, hq, head_dim_v, dtype=torch.bfloat16, device="cuda")
    q_idx = torch.arange(384, dtype=torch.int32, device="cuda")
    func = torch.zeros(1, 3, 384 + 256, dtype=torch.int32, device="cuda")
    func[0, 0, :384] = q_idx.remainder(64) + 1
    func[0, 1, :384] = 96
    func[0, 2, :384] = 320 + q_idx.remainder(32)

    out, _, dq, dk, dv, plan = _run_arbitrary_backward(
        q,
        k,
        v,
        func,
        q_lengths,
        k_lengths,
        dout,
        deterministic=True,
        pack_gqa=False,
    )
    assert int(plan.bwd_tensors.mask_block_cnt.max()) > 1
    assert int(plan.bwd_tensors.full_block_cnt.max()) > 1

    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    out_ref = _reference_attention(q_ref, k_ref, v_ref, func, q_lengths, k_lengths)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout.float()
    )
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dv.float(), dv_ref, atol=5e-2, rtol=5e-2)


@cute.jit
def _score_times_two_with_kv_bias(
    score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return score * cute.full_like(score, 2.0) + kv_idx.to(
        cutlass.Float32
    ) * cute.full_like(score, 1.0 / 256.0)


@cute.jit
def _score_times_two_with_kv_bias_bwd(
    grad, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return grad * cute.full_like(grad, 2.0)


@pytest.mark.parametrize("mode", ["softcap", "score_mod"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (256, 128)])
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_score_modifiers(
    mode, deterministic, head_dim, head_dim_v
):
    torch.manual_seed(0)
    q_lengths = [147, 131]
    k_lengths = [257, 193]
    hq = hkv = 2
    q = torch.randn(sum(q_lengths), hq, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(sum(k_lengths), hkv, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(
        sum(k_lengths), hkv, head_dim_v, dtype=torch.bfloat16, device="cuda"
    )
    dout = torch.randn(
        sum(q_lengths), hq, head_dim_v, dtype=torch.bfloat16, device="cuda"
    )
    func = _causal_global_func(q_lengths, k_lengths, hmask=hq)
    softcap = 7.0 if mode == "softcap" else 0.0
    score_mod = _score_times_two_with_kv_bias if mode == "score_mod" else None
    score_mod_bwd = _score_times_two_with_kv_bias_bwd if mode == "score_mod" else None

    out, _, dq, dk, dv, _ = _run_arbitrary_backward(
        q,
        k,
        v,
        func,
        q_lengths,
        k_lengths,
        dout,
        softcap=softcap,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        deterministic=deterministic,
        pack_gqa=False,
    )
    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    out_ref = _reference_attention(
        q_ref,
        k_ref,
        v_ref,
        func,
        q_lengths,
        k_lengths,
        softcap=softcap,
        score_multiplier=2.0 if mode == "score_mod" else 1.0,
        kv_bias_scale=1.0 / 256.0 if mode == "score_mod" else 0.0,
    )
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout.float()
    )
    torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(dv.float(), dv_ref, atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_fully_masked_is_zero(deterministic):
    torch.manual_seed(0)
    q_lengths = [9]
    k_lengths = [13]
    q = torch.randn(9, 2, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(13, 2, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    func = torch.zeros(1, 1, q.shape[0] + 256, dtype=torch.int32, device="cuda")

    out, lse, dq, dk, dv, plan = _run_arbitrary_backward(
        q,
        k,
        v,
        func,
        q_lengths,
        k_lengths,
        dout,
        deterministic=deterministic,
        pack_gqa=False,
    )
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(dk) == 0
    assert torch.count_nonzero(dv) == 0
    assert plan.bwd_tensors.mask_block_idx.numel() == 0
    assert plan.bwd_tensors.full_block_idx.numel() == 0


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_ignores_other_sample_intervals():
    torch.manual_seed(0)
    q_lengths = [3, 2]
    k_lengths = [4, 3]
    hq, hkv, head_dim, head_dim_v = 2, 2, 64, 64
    q = torch.randn(sum(q_lengths), hq, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(sum(k_lengths), hkv, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(
        sum(k_lengths), hkv, head_dim_v, dtype=torch.bfloat16, device="cuda"
    )
    dout = torch.randn(
        sum(q_lengths), hq, head_dim_v, dtype=torch.bfloat16, device="cuda"
    )
    func = torch.zeros(1, 1, sum(q_lengths) + 256, dtype=torch.int32, device="cuda")
    func[0, 0, : q_lengths[0]] = sum(k_lengths)
    func[0, 0, q_lengths[0] : sum(q_lengths)] = k_lengths[0]

    out, _, dq, dk, dv, _ = _run_arbitrary_backward(
        q,
        k,
        v,
        func,
        q_lengths,
        k_lengths,
        dout,
        deterministic=True,
        pack_gqa=False,
    )
    q_ref = q.float().requires_grad_(True)
    k_ref = k.float().requires_grad_(True)
    v_ref = v.float().requires_grad_(True)
    out_ref = _reference_attention(q_ref, k_ref, v_ref, func, q_lengths, k_lengths)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_ref, k_ref, v_ref), dout.float()
    )
    torch.testing.assert_close(out.float(), out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dq.float(), dq_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dk.float(), dk_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv.float(), dv_ref, atol=3e-2, rtol=3e-2)
    assert torch.count_nonzero(dq[q_lengths[0] :]) == 0
    assert torch.count_nonzero(dk[k_lengths[0] :]) == 0
    assert torch.count_nonzero(dv[k_lengths[0] :]) == 0


@pytest.mark.parametrize(
    "head_dim,head_dim_v",
    [
        (64, 64),
        (128, 128),
        (192, 192),
        (256, 256),
    ],
)
@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_deterministic_is_batch_invariant(
    head_dim, head_dim_v
):
    torch.manual_seed(0)
    hq, hkv = 4, 1
    target_q_len, target_k_len = 37, 53
    target_q = torch.randn(
        target_q_len, hq, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    target_k = torch.randn(
        target_k_len, hkv, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    target_v = torch.randn(
        target_k_len, hkv, head_dim_v, dtype=torch.bfloat16, device="cuda"
    )
    target_do = torch.randn(
        target_q_len, hq, head_dim_v, dtype=torch.bfloat16, device="cuda"
    )

    def run(prefix_q_len, prefix_k_len):
        if prefix_q_len == 0 and prefix_k_len == 0:
            q = target_q
            k = target_k
            v = target_v
            dout = target_do
            q_lengths = [target_q_len]
            k_lengths = [target_k_len]
            q_offset = 0
            k_offset = 0
        else:
            prefix_q = torch.randn(
                prefix_q_len, hq, head_dim, dtype=torch.bfloat16, device="cuda"
            )
            prefix_k = torch.randn(
                prefix_k_len, hkv, head_dim, dtype=torch.bfloat16, device="cuda"
            )
            prefix_v = torch.randn(
                prefix_k_len,
                hkv,
                head_dim_v,
                dtype=torch.bfloat16,
                device="cuda",
            )
            prefix_do = torch.randn(
                prefix_q_len,
                hq,
                head_dim_v,
                dtype=torch.bfloat16,
                device="cuda",
            )
            q = torch.cat((prefix_q, target_q), dim=0)
            k = torch.cat((prefix_k, target_k), dim=0)
            v = torch.cat((prefix_v, target_v), dim=0)
            dout = torch.cat((prefix_do, target_do), dim=0)
            q_lengths = [prefix_q_len, target_q_len]
            k_lengths = [prefix_k_len, target_k_len]
            q_offset = prefix_q_len
            k_offset = prefix_k_len
        func = _causal_global_func(q_lengths, k_lengths)
        _, _, dq, dk, dv, _ = _run_arbitrary_backward(
            q,
            k,
            v,
            func,
            q_lengths,
            k_lengths,
            dout,
            deterministic=True,
            pack_gqa=True,
        )
        return (
            dq[q_offset:],
            dk[k_offset:],
            dv[k_offset:],
        )

    first = run(0, 0)
    repeat = run(0, 0)
    regrouped = run(29, 41)
    for lhs, rhs in zip(first, repeat):
        assert torch.equal(lhs, rhs)
    for lhs, rhs in zip(first, regrouped):
        assert torch.equal(lhs, rhs)


@pytest.mark.skipif(COMPUTE_CAPABILITY != 9, reason="SM90-only test")
def test_varlen_arbitrary_mask_bwd_requires_backward_plan():
    q_lengths = [8]
    k_lengths = [8]
    cu_q = _cu_seqlens(q_lengths)
    cu_k = _cu_seqlens(k_lengths)
    q = torch.randn(8, 1, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    func = _causal_global_func(q_lengths, k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=8,
        max_seqlen_k=8,
        pack_gqa=False,
    )
    out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=8,
        max_seqlen_k=8,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
    )
    with pytest.raises(ValueError, match="build_backward=True"):
        out.sum().backward()
