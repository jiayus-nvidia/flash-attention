import math
import os
from dataclasses import dataclass

import pytest
import torch
from cutlass import Float32, Int32, cute
from flash_attn_cute import (
    create_arbitrary_block_sparse_tensors,
    flash_attn_func,
    flash_attn_varlen_func,
)
from flash_attn_cute.testing import is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
_FAKE_TARGET_ARCH = (
    os.getenv("FLASH_ATTENTION_ARCH") or os.getenv("CUTE_DSL_ARCH") or ""
).lower()
_FAKE_TARGET_ARCH = _FAKE_TARGET_ARCH.removeprefix("sm_").removeprefix("sm")
_FAKE_TARGET_ARCH = (
    _FAKE_TARGET_ARCH[:-1] if _FAKE_TARGET_ARCH[-1:] in "af" else _FAKE_TARGET_ARCH
)
FAKE_TARGET_IS_SM100_OR_SM103 = USE_FAKE_TENSOR and _FAKE_TARGET_ARCH in (
    "100",
    "103",
)
DEVICE_CAPABILITY = (
    torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
)
IS_SM100 = DEVICE_CAPABILITY[0] == 10 and DEVICE_CAPABILITY != (10, 1)
IS_SM100_OR_SM103 = DEVICE_CAPABILITY in ((10, 0), (10, 3))

pytestmark = [
    pytest.mark.skipif(
        not IS_SM100 and not FAKE_TARGET_IS_SM100_OR_SM103,
        reason="SM100 generic 1CTA arbitrary forward",
    ),
]


@cute.jit
def _score_to_zero(score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    """Make mask-after-score_mod ordering observable in the output."""

    return cute.full_like(score, 0.0)


@cute.jit
def _score_to_zero_bwd(
    grad, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return grad * cute.full_like(grad, 0.0)


@cute.jit
def _score_add_aux_scalar(
    score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Exercise arbitrary-plan score-mod aux plumbing without changing masking."""

    bias = aux_tensors[0]
    bias_frag = cute.make_rmem_tensor(1, bias.element_type)
    bias_frag[0] = bias[0]
    return score + bias_frag.load().to(Float32)


@cute.jit
def _score_add_aux_scalar_bwd(
    grad, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return grad


@cute.jit
def _score_add_aux_global_kv(
    score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    """Read a packed-varlen KV bias through the sample's global offset."""

    bias = aux_tensors[0]
    kv_frag = cute.make_rmem_tensor(1, Int32)
    kv_frag.store(kv_idx + seqlen_info.offset_k)
    bias_frag = cute.make_rmem_tensor(1, bias.element_type)
    bias_frag[0] = bias[kv_frag[0]]
    return score + bias_frag.load().to(Float32)


@cute.jit
def _score_add_aux_global_kv_bwd(
    grad, score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
):
    return grad


@dataclass(frozen=True)
class _ForwardCase:
    name: str
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_v: int
    dtype: torch.dtype
    pack_gqa: bool
    q_stage: int
    pattern: str = "mixed"
    per_head_mask: bool = False
    softcap: float = 0.0
    score_mode: str | None = None


_FIXED_CASES = (
    _ForwardCase(
        "mha_q1_fp16_d64_mixed",
        (64,),
        (257,),
        2,
        2,
        64,
        64,
        torch.float16,
        False,
        1,
        score_mode="aux_scalar",
    ),
    _ForwardCase(
        "mha_q2_bf16_d128_dv96_softcap",
        (129,),
        (256,),
        2,
        2,
        128,
        96,
        torch.bfloat16,
        False,
        2,
        pattern="full",
        softcap=7.0,
    ),
    _ForwardCase(
        "gqa_q1_fp16_d96_dv64_pack_score_zero",
        (32,),
        (129,),
        8,
        2,
        96,
        64,
        torch.float16,
        True,
        1,
        score_mode="zero",
    ),
    _ForwardCase(
        "mqa_q2_bf16_d192_unpack_head_mask",
        (17,),
        (256,),
        8,
        1,
        192,
        128,
        torch.bfloat16,
        False,
        2,
        per_head_mask=True,
    ),
)


_VARLEN_CASES = (
    _ForwardCase(
        "mha_q1_bf16_d96_dv128_zeros",
        (19, 0, 31, 7),
        (23, 7, 0, 129),
        2,
        2,
        96,
        128,
        torch.bfloat16,
        False,
        1,
    ),
    _ForwardCase(
        "mha_q2_fp16_d128_dv64_softcap",
        (17, 129, 0, 9),
        (23, 257, 7, 0),
        2,
        2,
        128,
        64,
        torch.float16,
        False,
        2,
        softcap=7.0,
    ),
    _ForwardCase(
        "gqa_q1_bf16_d64_dv96_unpack_head_mask",
        (13, 0, 32, 5),
        (129, 7, 0, 23),
        8,
        2,
        64,
        96,
        torch.bfloat16,
        False,
        1,
        per_head_mask=True,
    ),
    _ForwardCase(
        "mqa_q2_fp16_d128_pack",
        (9, 0, 17, 3),
        (129, 7, 0, 257),
        8,
        1,
        128,
        128,
        torch.float16,
        True,
        2,
    ),
)


def _cu_seqlens(lengths: tuple[int, ...]) -> torch.Tensor:
    prefixes = [0]
    for length in lengths:
        prefixes.append(prefixes[-1] + length)
    return torch.tensor(
        prefixes,
        dtype=torch.int32,
        device="cuda",
    )


def _make_arbitrary_func(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    *,
    hmask: int,
    pattern: str,
) -> torch.Tensor:
    """Build global interval endpoints for fixed or packed-varlen inputs.

    The mixed pattern has empty rows, partial blocks 0 and 1, and no visible
    values in block 2 when K has 257 elements.  The full pattern makes every
    valid score visible, so aligned K lengths exercise a true full-only plan.
    """

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
        q_slice = slice(q_begin, q_begin + q_len)
        if pattern == "full":
            func[:, 0, q_slice] = k_begin + k_len
            func[:, 1, q_slice] = k_begin + k_len
            func[:, 2, q_slice] = k_begin + k_len
        elif pattern == "mixed" and q_len:
            q_local = torch.arange(q_len, dtype=torch.int32, device="cuda")
            for mask_head in range(hmask):
                shifted_q = q_local + mask_head * 3
                empty = shifted_q.remainder(7) == 0
                first_end = k_begin + torch.clamp(
                    1 + shifted_q.remainder(31), max=k_len
                )
                second_begin = k_begin + min(k_len, 128)
                second_end = k_begin + torch.clamp(
                    129 + (shifted_q * 7).remainder(64), max=k_len
                )
                func[mask_head, 0, q_slice] = torch.where(empty, k_begin, first_end)
                func[mask_head, 1, q_slice] = torch.where(empty, k_begin, second_begin)
                func[mask_head, 2, q_slice] = torch.where(empty, k_begin, second_end)
        elif pattern != "mixed":
            raise ValueError(f"unknown arbitrary-mask pattern: {pattern}")
        q_begin += q_len
        k_begin += k_len
    return func


def _make_boundary_partial_func(*, q_len: int, k_len: int) -> torch.Tensor:
    """Make every K tile visible while forcing its tail tile to be partial."""

    func = torch.zeros(
        1,
        1,
        q_len + 256,
        dtype=torch.int32,
        device="cuda",
    )
    q_idx = torch.arange(q_len, dtype=torch.int32, device="cuda")
    # Alternating rows omit the final valid K element.  Consequently every
    # complete preceding K tile is full while the final K tile needs payload.
    func[0, 0, :q_len] = k_len - q_idx.remainder(2)
    return func


def _make_varlen_boundary_partial_func(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    *,
    hmask: int = 1,
) -> torch.Tensor:
    """Build sample-global endpoints with one omitted K-tail value per odd row."""

    func = torch.zeros(
        hmask,
        1,
        sum(q_lengths) + 256,
        dtype=torch.int32,
        device="cuda",
    )
    q_begin = 0
    k_begin = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        if q_len:
            q_idx = torch.arange(q_len, dtype=torch.int32, device="cuda")
            for mask_head in range(hmask):
                # Give head-specific plans different K2Q row lengths as well
                # as different payload bits.  This catches grouped-head dKdV
                # implementations that incorrectly reuse head 0's traversal.
                endpoints = (
                    k_begin + k_len - (q_idx + mask_head).remainder(2 + mask_head % 3)
                )
                func[mask_head, 0, q_begin : q_begin + q_len] = endpoints.clamp_min(
                    k_begin
                )
        q_begin += q_len
        k_begin += k_len
    return func


def _assert_rank_only_orders_are_dense(
    plan,
    *,
    k_block_prefix: tuple[int, ...],
) -> None:
    """SPT ranks each target Q block's contributors by descending local K."""

    bwd = plan.bwd_tensors
    assert bwd is not None
    assert bwd.plan_signature.dq_order_format == "rank_only"
    total_k_rows = k_block_prefix[-1]
    contributors_by_target: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
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
        assert offsets is not None
        assert indices is not None
        assert write_order is not None
        counts_cpu = counts.cpu()
        offsets_cpu = offsets.cpu()
        indices_cpu = indices.cpu()
        write_order_cpu = write_order.cpu()
        assert counts_cpu.shape[1] == total_k_rows
        assert torch.equal(counts_cpu.reshape(-1), offsets_cpu[1:] - offsets_cpu[:-1])
        for mask_head in range(counts_cpu.shape[0]):
            for k_row in range(total_k_rows):
                sample_idx = next(
                    sample_idx
                    for sample_idx in range(len(k_block_prefix) - 1)
                    if k_block_prefix[sample_idx]
                    <= k_row
                    < k_block_prefix[sample_idx + 1]
                )
                local_k_block = k_row - k_block_prefix[sample_idx]
                flat_row = mask_head * total_k_rows + k_row
                for entry_idx in range(
                    int(offsets_cpu[flat_row]), int(offsets_cpu[flat_row + 1])
                ):
                    target = (
                        sample_idx,
                        mask_head,
                        int(indices_cpu[entry_idx]),
                    )
                    contributors_by_target.setdefault(target, []).append(
                        (
                            local_k_block,
                            int(write_order_cpu[entry_idx]),
                        )
                    )
    assert contributors_by_target
    for target, contributors in contributors_by_target.items():
        local_k_blocks = [local_k_block for local_k_block, _ in contributors]
        assert len(set(local_k_blocks)) == len(local_k_blocks), target
        descending_k = sorted(
            contributors,
            key=lambda contributor: contributor[0],
            reverse=True,
        )
        assert [rank for _, rank in descending_k] == list(range(len(descending_k))), (
            target
        )


def _make_inputs(case: _ForwardCase, *, varlen: bool):
    torch.manual_seed(123)
    q_shape = (
        (sum(case.q_lengths), case.num_q_heads, case.head_dim)
        if varlen
        else (1, case.q_lengths[0], case.num_q_heads, case.head_dim)
    )
    k_shape = (
        (sum(case.k_lengths), case.num_kv_heads, case.head_dim)
        if varlen
        else (1, case.k_lengths[0], case.num_kv_heads, case.head_dim)
    )
    v_shape = (*k_shape[:-1], case.head_dim_v)
    q = torch.randn(q_shape, dtype=case.dtype, device="cuda")
    k = torch.randn(k_shape, dtype=case.dtype, device="cuda")
    v = torch.randn(v_shape, dtype=case.dtype, device="cuda")
    return q, k, v


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    func: torch.Tensor,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    *,
    softcap: float,
    score_mode: str | None,
):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    q_per_kv = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(q.shape[-1])
    func_cpu = func.cpu()
    outputs = []
    lses = []
    empty_rows = []
    q_begin = 0
    k_begin = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        if q_len == 0:
            k_begin += k_len
            continue
        if k_len == 0:
            outputs.append(
                torch.zeros(
                    q_len,
                    num_q_heads,
                    v.shape[-1],
                    dtype=torch.float32,
                    device=q.device,
                )
            )
            lses.append(
                torch.full(
                    (num_q_heads, q_len),
                    -torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                )
            )
            empty_rows.append(
                torch.ones(q_len, num_q_heads, dtype=torch.bool, device=q.device)
            )
            q_begin += q_len
            continue

        q_cur = q[q_begin : q_begin + q_len].float().transpose(0, 1)
        k_cur = (
            k[k_begin : k_begin + k_len]
            .float()
            .repeat_interleave(q_per_kv, dim=1)
            .transpose(0, 1)
        )
        v_cur = (
            v[k_begin : k_begin + k_len]
            .float()
            .repeat_interleave(q_per_kv, dim=1)
            .transpose(0, 1)
        )
        scores = torch.matmul(q_cur, k_cur.transpose(-1, -2)) * scale
        if score_mode == "zero":
            scores = torch.zeros_like(scores)
        elif score_mode == "aux_scalar":
            scores = scores + 0.375
        elif score_mode == "aux_global_kv":
            global_k_bias = (
                torch.arange(
                    k_begin,
                    k_begin + k_len,
                    dtype=torch.float32,
                    device=q.device,
                )
                * 0.003
            )
            scores = scores + global_k_bias[None, None, :]
        elif score_mode is not None:
            raise ValueError(f"unknown score mode: {score_mode}")
        if softcap:
            scores = softcap * torch.tanh(scores / softcap)

        visible = torch.zeros(
            num_q_heads,
            q_len,
            k_len,
            dtype=torch.bool,
            device=q.device,
        )
        for head_idx in range(num_q_heads):
            mask_head = 0 if func.shape[0] == 1 else head_idx
            for q_local in range(q_len):
                q_global = q_begin + q_local
                interval_begin = 0
                for endpoint_idx in range(0, func.shape[1], 2):
                    interval_end = int(func_cpu[mask_head, endpoint_idx, q_global])
                    local_begin = max(interval_begin, k_begin) - k_begin
                    local_end = min(interval_end, k_begin + k_len) - k_begin
                    if local_end > local_begin:
                        visible[head_idx, q_local, local_begin:local_end] = True
                    if endpoint_idx + 1 < func.shape[1]:
                        interval_begin = int(
                            func_cpu[mask_head, endpoint_idx + 1, q_global]
                        )

        row_has_k = visible.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~visible, -torch.inf)
        safe_scores = torch.where(
            row_has_k, masked_scores, torch.zeros_like(masked_scores)
        )
        probabilities = torch.where(
            row_has_k,
            torch.softmax(safe_scores, dim=-1),
            torch.zeros_like(safe_scores),
        )
        outputs.append(torch.matmul(probabilities, v_cur).transpose(0, 1))
        lses.append(torch.logsumexp(masked_scores, dim=-1))
        empty_rows.append(~row_has_k.squeeze(-1).transpose(0, 1))
        q_begin += q_len
        k_begin += k_len

    output = torch.cat(outputs, dim=0)
    lse = torch.cat(lses, dim=1)
    empty = torch.cat(empty_rows, dim=0)
    return output, lse, empty


def _assert_plan_contract(plan, case: _ForwardCase):
    signature = plan.plan_signature
    assert signature.arch_family == "sm100"
    assert signature.kernel_family == "sm100_generic_fwd"
    assert signature.direction == "forward"
    assert signature.tile_m == signature.tile_n == 128
    assert signature.q_stage == case.q_stage
    assert signature.cta_group_size == 1
    assert signature.pack_gqa is case.pack_gqa
    assert signature.qhead_per_kvhead == case.num_q_heads // case.num_kv_heads
    assert plan.block_size == (128 * case.q_stage, 128)
    assert plan.pack_gqa is case.pack_gqa
    assert plan.mask_block_masks.dtype == torch.uint32
    assert plan.mask_block_masks.shape[1:] == (case.q_stage, 128, 4)
    assert plan.mask_block_masks.data_ptr() % 16 == 0


def _assert_result(out, lse, ref, ref_lse, empty_rows, *, fixed: bool):
    expected_out = ref.unsqueeze(0) if fixed else ref
    expected_lse = ref_lse.unsqueeze(0) if fixed else ref_lse
    torch.testing.assert_close(out.float(), expected_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-3, rtol=3e-3)
    assert not torch.isnan(out).any()
    assert not torch.isnan(lse).any()

    if empty_rows.any():
        out_rows = out[0] if fixed else out
        lse_rows = lse[0].transpose(0, 1) if fixed else lse.transpose(0, 1)
        assert torch.count_nonzero(out_rows[empty_rows]) == 0
        assert torch.isneginf(lse_rows[empty_rows]).all()


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case.name) for case in _FIXED_CASES]
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_arbitrary_fixed_forward(case: _ForwardCase):
    q, k, v = _make_inputs(case, varlen=False)
    hmask = case.num_q_heads if case.per_head_mask else 1
    func = _make_arbitrary_func(
        case.q_lengths,
        case.k_lengths,
        hmask=hmask,
        pattern=case.pattern,
    )
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=case.pack_gqa)
    score_mod = None
    aux_tensors = None
    if case.score_mode == "zero":
        score_mod = _score_to_zero
    elif case.score_mode == "aux_scalar":
        score_mod = _score_add_aux_scalar
        aux_tensors = [torch.tensor([0.375], dtype=case.dtype, device="cuda")]
    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=case.pack_gqa,
        softcap=case.softcap,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=True,
    )
    if is_fake_mode():
        return
    q_flat, k_flat, v_flat = (tensor.flatten(0, 1) for tensor in (q, k, v))
    ref, ref_lse, empty_rows = _reference_attention(
        q_flat,
        k_flat,
        v_flat,
        func,
        case.q_lengths,
        case.k_lengths,
        softcap=case.softcap,
        score_mode=case.score_mode,
    )
    _assert_plan_contract(plan, case)
    _assert_result(out, lse, ref, ref_lse, empty_rows, fixed=True)

    if case.name == "mha_q1_fp16_d64_mixed":
        assert plan.mask_block_idx.numel() > 0
        assert plan.full_block_idx.numel() == 0
        # The third K block is entirely invisible and therefore absent from CSR.
        assert plan.mask_block_idx.numel() < math.ceil(case.k_lengths[0] / 128)
    if case.pattern == "full":
        assert plan.mask_block_idx.numel() == 0
        assert plan.full_block_idx.numel() > 0


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case.name) for case in _VARLEN_CASES]
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_arbitrary_varlen_forward(case: _ForwardCase):
    q, k, v = _make_inputs(case, varlen=True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    hmask = case.num_q_heads if case.per_head_mask else 1
    func = _make_arbitrary_func(
        case.q_lengths,
        case.k_lengths,
        hmask=hmask,
        pattern=case.pattern,
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=case.pack_gqa,
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=case.pack_gqa,
        softcap=case.softcap,
        score_mod=_score_to_zero if case.score_mode == "zero" else None,
        return_lse=True,
    )
    if is_fake_mode():
        return
    ref, ref_lse, empty_rows = _reference_attention(
        q,
        k,
        v,
        func,
        case.q_lengths,
        case.k_lengths,
        softcap=case.softcap,
        score_mode=case.score_mode,
    )
    _assert_plan_contract(plan, case)
    _assert_result(out, lse, ref, ref_lse, empty_rows, fixed=False)
    assert plan.cu_total_m_blocks.tolist()[0] == 0
    assert len(plan.cu_total_m_blocks) == len(case.q_lengths) + 1
    assert plan.topology_tensors.cu_total_q_plan_rows is plan.cu_total_m_blocks


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="requires mask payload values")
def test_sm100_arbitrary_payload_matches_tmem_consumer_coordinates():
    """Lock the payload to partition_D(tScS): row=tidx, col=value index."""

    torch.manual_seed(321)
    q_len, k_len = 129, 257
    q = torch.randn(1, q_len, 2, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, k_len, 2, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, k_len, 2, 96, dtype=torch.bfloat16, device="cuda")
    func = torch.zeros(1, 1, q_len + 256, dtype=torch.int32, device="cuda")
    func[0, 0, :q_len] = torch.arange(1, q_len + 1, dtype=torch.int32, device="cuda")

    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=False)
    assert plan.block_size == (256, 128)
    assert plan.mask_block_cnt.tolist() == [[2]]
    assert plan.full_block_cnt.tolist() == [[0]]
    assert plan.mask_block_idx.tolist() == [0, 1]
    assert plan.mask_block_masks.shape == (2, 2, 128, 4)

    expected = torch.zeros(plan.mask_block_masks.shape, dtype=torch.int64)
    for payload_idx, n_block in enumerate(plan.mask_block_idx.cpu().tolist()):
        for q_stage in range(2):
            for tidx in range(128):
                q_local = q_stage * 128 + tidx
                for word_idx in range(4):
                    mask_word = 0
                    for bit_idx in range(32):
                        # Ld32x32b.r32 gives this consumer thread one row and
                        # consecutive score values across its four words.
                        k_local = n_block * 128 + word_idx * 32 + bit_idx
                        if q_local < q_len and k_local < k_len and k_local <= q_local:
                            mask_word |= 1 << bit_idx
                    expected[payload_idx, q_stage, tidx, word_idx] = mask_word
    assert torch.equal(plan.mask_block_masks.cpu(), expected.to(torch.uint32))
    block0 = plan.mask_block_masks[0].cpu()
    block1 = plan.mask_block_masks[1].cpu()
    assert block0[1, 0].tolist() == [0xFFFFFFFF] * 4
    assert torch.count_nonzero(block0[1, 1:].to(torch.int64)) == 0
    assert torch.count_nonzero(block1[0].to(torch.int64)) == 0
    assert block1[1, 0].tolist() == [1, 0, 0, 0]


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="requires mask payload values")
def test_sm100_arbitrary_pack_gqa_qratio16_payload_coordinates():
    """Physical TMEM rows map to logical Q rows by the PackGQA ratio."""

    q_len, k_len, qratio = 8, 129, 16
    q = torch.randn(1, q_len, qratio, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, k_len, 1, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    func = torch.zeros(1, 1, q_len + 256, dtype=torch.int32, device="cuda")
    func[0, 0, :q_len] = torch.arange(1, q_len + 1, dtype=torch.int32, device="cuda")

    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=True)
    assert plan.block_size == (128, 128)
    assert plan.mask_block_idx.tolist() == [0]
    assert plan.mask_block_masks.shape == (1, 1, 128, 4)

    expected = torch.zeros((128, 4), dtype=torch.int64)
    for tidx in range(128):
        q_local = tidx // qratio
        expected[tidx, 0] = (1 << (q_local + 1)) - 1
    assert torch.equal(plan.mask_block_masks[0, 0].cpu(), expected.to(torch.uint32))


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="requires output values")
def test_sm100_arbitrary_nonzero_k_all_empty_tile():
    """An empty compact row still satisfies the SM100 mbarrier contract."""

    case = _FIXED_CASES[2]
    q, k, v = _make_inputs(case, varlen=False)
    func = torch.zeros(1, 1, case.q_lengths[0] + 256, dtype=torch.int32, device="cuda")
    plan = create_arbitrary_block_sparse_tensors(func, q, k, v, pack_gqa=case.pack_gqa)
    assert plan.mask_block_idx.numel() == 0
    assert plan.full_block_idx.numel() == 0

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=case.pack_gqa,
        score_mod=_score_to_zero,
        return_lse=True,
    )
    assert torch.count_nonzero(out) == 0
    assert torch.isneginf(lse).all()


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="requires output values")
def test_sm100_arbitrary_fixed_batch_rows_are_isolated():
    """Fixed plans map each batch to its own compact Q2K row."""

    case = _FIXED_CASES[0]
    batch_size = 2
    q_len = case.q_lengths[0]
    k_len = case.k_lengths[0]
    torch.manual_seed(456)
    q = torch.randn(
        batch_size,
        q_len,
        case.num_q_heads,
        case.head_dim,
        dtype=case.dtype,
        device="cuda",
    )
    k = torch.randn(
        batch_size,
        k_len,
        case.num_kv_heads,
        case.head_dim,
        dtype=case.dtype,
        device="cuda",
    )
    v = torch.randn_like(k)
    func = _make_arbitrary_func(
        (q_len, q_len),
        (k_len, k_len),
        hmask=1,
        pattern="mixed",
    )
    # Batch 0 sees every local K while batch 1 retains the mixed mask.  The
    # interval endpoints remain global, so using the wrong compact row would
    # make the mismatch observable without relying on identical sample masks.
    func[:, 0, :q_len] = k_len
    func[:, 1, :q_len] = k_len
    func[:, 2, :q_len] = k_len

    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
    )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    ref, ref_lse, _ = _reference_attention(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        func,
        (q_len, q_len),
        (k_len, k_len),
        softcap=0.0,
        score_mode=None,
    )
    expected_out = ref.reshape(batch_size, q_len, case.num_q_heads, -1)
    expected_lse = ref_lse.reshape(case.num_q_heads, batch_size, q_len).permute(1, 0, 2)
    torch.testing.assert_close(out.float(), expected_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-3, rtol=3e-3)
    assert plan.mask_block_cnt.shape[1] == batch_size


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_arbitrary_builder_emits_native_backward_contract():
    q = torch.empty(1, 8, 2, 64, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(1, 8, 2, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    func = torch.zeros(1, 1, 8 + 256, dtype=torch.int32, device="cuda")
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    assert plan.bwd_tensors is not None
    assert plan.bwd_tensors.block_size == (128, 128)
    assert plan.bwd_tensors.plan_signature.kernel_family == "sm100_generic_bwd"
    assert plan.bwd_tensors.plan_signature.dq_order_format == "rank_only"
    assert plan.bwd_tensors.mask_block_masks.shape[1:] == (1, 256, 4)


@pytest.mark.skipif(
    not USE_FAKE_TENSOR
    or (not IS_SM100_OR_SM103 and not FAKE_TARGET_IS_SM100_OR_SM103),
    reason="SM100/SM103 FakeTensor generic arbitrary 1CTA forward compile gate",
)
@pytest.mark.parametrize(
    "is_varlen,dtype,num_heads,num_kv_heads,head_dim,head_dim_v",
    [
        pytest.param(False, torch.bfloat16, 1, 1, 128, 128, id="fixed_bf16_d128"),
        pytest.param(
            True, torch.float16, 4, 1, 128, 96, id="varlen_mqa_fp16_d128_dv96"
        ),
        pytest.param(False, torch.bfloat16, 1, 1, 192, 128, id="fixed_bf16_d192_dv128"),
        pytest.param(True, torch.float16, 1, 1, 192, 128, id="varlen_fp16_d192_dv128"),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_arbitrary_fake_compiles_generic_forward_dispatch(
    monkeypatch,
    is_varlen,
    dtype,
    num_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
):
    """Compile generic arbitrary forward and keep every shape on 1CTA."""

    from flash_attn_cute.cache_utils import JITCache
    from flash_attn_cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    from flash_attn_cute.interface import _flash_attn_fwd

    assert is_fake_mode()
    q_lengths = (385, 0, 513) if is_varlen else (257,)
    k_lengths = (257, 129, 769) if is_varlen else (257,)
    batch_size = len(q_lengths)
    if is_varlen:
        q = torch.empty(sum(q_lengths), num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.empty(
            sum(k_lengths), num_kv_heads, head_dim, dtype=dtype, device="cuda"
        )
        cu_q = _cu_seqlens(q_lengths)
        cu_k = _cu_seqlens(k_lengths)
    else:
        q = torch.empty(
            batch_size,
            q_lengths[0],
            num_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        k = torch.empty(
            batch_size,
            k_lengths[0],
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        cu_q = cu_k = None
    v = torch.empty((*k.shape[:-1], head_dim_v), dtype=dtype, device="cuda")
    func = torch.empty(1, 1, sum(q_lengths) + 256, dtype=torch.int32, device="cuda")
    varlen_kwargs = (
        {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max(q_lengths),
            "max_seqlen_k": max(k_lengths),
        }
        if is_varlen
        else {}
    )
    plan = create_arbitrary_block_sparse_tensors(
        func, q, k, v, pack_gqa=False, **varlen_kwargs
    )

    assert plan.plan_signature.kernel_family == "sm100_generic_fwd"
    assert plan.plan_signature.cta_group_size == 1
    assert plan.plan_signature.cluster_axis == "m"
    assert plan.plan_signature.q_stage == 2
    assert plan.plan_signature.pack_gqa is False
    assert plan.plan_signature.qhead_per_kvhead == num_heads // num_kv_heads
    assert plan.block_size == (256, 128)
    assert plan.mask_block_masks.shape[1:] == (2, 128, 4)

    fwd_cache = JITCache()
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", fwd_cache)
    real_compile = cute.compile
    fwd_compile_calls = []

    def compile_spy(*args, **kwargs):
        if args and isinstance(args[0], FlashAttentionForwardSm100):
            fwd_compile_calls.append((args[0], kwargs.get("options")))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", compile_spy)
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
        **varlen_kwargs,
    )

    assert out.shape == (*q.shape[:-1], head_dim_v)
    assert lse.shape == (
        (num_heads, sum(q_lengths))
        if is_varlen
        else (batch_size, num_heads, q_lengths[0])
    )
    assert len(fwd_cache) == 1
    assert len(fwd_compile_calls) == 1
    compiled_fwd, compile_options = fwd_compile_calls[0]
    assert compiled_fwd.is_arbitrary
    assert not compiled_fwd.use_2cta_instrs
    assert compiled_fwd.cta_group_size == 1
    assert compiled_fwd.cluster_shape_mn == (1, 1)
    assert compiled_fwd.q_stage == 2
    assert compiled_fwd.pack_gqa is False
    assert compiled_fwd.qhead_per_kvhead == num_heads // num_kv_heads
    assert compiled_fwd.m_block_size == 128
    assert compiled_fwd.n_block_size == 128
    assert compiled_fwd.is_varlen_q is is_varlen
    assert compile_options is not None and "--enable-tvm-ffi" in compile_options
    assert "--opt-level" not in compile_options


@pytest.mark.skipif(
    not USE_FAKE_TENSOR
    or (not IS_SM100_OR_SM103 and not FAKE_TARGET_IS_SM100_OR_SM103),
    reason="SM100/SM103 FakeTensor arbitrary main backward compile gate",
)
@pytest.mark.parametrize(
    "head_dim,is_varlen,deterministic,dtype,num_heads,num_kv_heads",
    [
        pytest.param(128, False, False, torch.bfloat16, 1, 1, id="d128_bf16_2cta"),
        pytest.param(
            128, True, True, torch.float16, 4, 1, id="d128_varlen_mqa_fp16_2cta"
        ),
        pytest.param(192, False, False, torch.bfloat16, 1, 1, id="d192_fixed"),
        pytest.param(192, True, False, torch.float16, 1, 1, id="d192_varlen"),
        pytest.param(192, False, True, torch.bfloat16, 1, 1, id="d192_deterministic"),
        pytest.param(192, False, False, torch.bfloat16, 4, 2, id="d192_gqa"),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_arbitrary_fake_compiles_main_backward(
    monkeypatch,
    head_dim,
    is_varlen,
    deterministic,
    dtype,
    num_heads,
    num_kv_heads,
):
    """Compile the fixed D128/D192 cooperative backward topologies."""

    from flash_attn_cute.cache_utils import JITCache
    from flash_attn_cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
    from flash_attn_cute.interface import _flash_attn_bwd

    assert is_fake_mode()
    head_dim_v = 128
    q_lengths = (129, 17) if is_varlen else (129,)
    k_lengths = (257, 129) if is_varlen else (256,)
    batch_size = len(q_lengths)
    if is_varlen:
        q = torch.empty(sum(q_lengths), num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.empty(
            sum(k_lengths), num_kv_heads, head_dim, dtype=dtype, device="cuda"
        )
        cu_q = _cu_seqlens(q_lengths)
        cu_k = _cu_seqlens(k_lengths)
    else:
        q = torch.empty(
            batch_size,
            q_lengths[0],
            num_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        k = torch.empty(
            batch_size,
            k_lengths[0],
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        cu_q = cu_k = None
    v = torch.empty((*k.shape[:-1], head_dim_v), dtype=dtype, device="cuda")
    func = torch.empty(1, 1, sum(q_lengths) + 256, dtype=torch.int32, device="cuda")
    varlen_kwargs = (
        {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max(q_lengths),
            "max_seqlen_k": max(k_lengths),
        }
        if is_varlen
        else {}
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
        **varlen_kwargs,
    )

    out = torch.empty((*q.shape[:-1], head_dim_v), dtype=dtype, device="cuda")
    dout = torch.empty_like(out)
    lse_shape = (
        (num_heads, sum(q_lengths))
        if is_varlen
        else (batch_size, num_heads, q_lengths[0])
    )
    lse = torch.empty(lse_shape, dtype=torch.float32, device="cuda")

    main_bwd_cache = JITCache()
    monkeypatch.setattr(_flash_attn_bwd, "compile_cache", main_bwd_cache)
    real_compile = cute.compile
    main_bwd_compile_calls = []

    def compile_spy(*args, **kwargs):
        if args and isinstance(args[0], FlashAttentionBackwardSm100):
            main_bwd_compile_calls.append((args[0], kwargs.get("options")))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", compile_spy)
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        arbitrary=True,
        block_sparse_tensors=plan,
        deterministic=deterministic,
        **varlen_kwargs,
    )

    assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)
    assert len(main_bwd_cache) == 1
    assert len(main_bwd_compile_calls) == 1
    compiled_main, compile_options = main_bwd_compile_calls[0]
    uses_cooperative_2cta = head_dim in (128, 192)
    assert compiled_main.arch == 100
    assert compiled_main.is_arbitrary
    assert compiled_main.use_2cta_instrs is uses_cooperative_2cta
    assert compiled_main.cta_group_size == (2 if uses_cooperative_2cta else 1)
    assert compiled_main.is_varlen_q is is_varlen
    assert compiled_main.is_varlen_k is is_varlen
    assert compiled_main.deterministic is deterministic
    assert compiled_main.spt is deterministic
    qratio = num_heads // num_kv_heads
    assert compiled_main.qhead_per_kvhead == qratio
    assert compiled_main.tile_hdim == head_dim
    assert compiled_main.tile_hdimv == head_dim_v
    assert compiled_main.dV_reduce_ncol == math.gcd(32, head_dim_v // 2)
    assert compiled_main.pack_gqa is False
    assert compiled_main.dKV_postprocess is (qratio > 1)
    if head_dim == 192:
        assert compiled_main.dQ_reduce_ncol == 24
        assert compiled_main.sdQaccum_stage == 2
    elif deterministic:
        assert compiled_main.dQ_reduce_ncol == 16
        assert compiled_main.sdQaccum_stage == 2
    else:
        assert compiled_main.dQ_reduce_ncol == 8
        assert compiled_main.sdQaccum_stage == 4
    assert compile_options is not None and "--enable-tvm-ffi" in compile_options
    dispatch_arch = (
        int(_FAKE_TARGET_ARCH)
        if _FAKE_TARGET_ARCH
        else DEVICE_CAPABILITY[0] * 10 + DEVICE_CAPABILITY[1]
    )
    if dispatch_arch == 103 and uses_cooperative_2cta:
        assert "--opt-level 2" in compile_options
    else:
        assert "--opt-level" not in compile_options


@pytest.mark.skipif(
    not USE_FAKE_TENSOR
    or (not IS_SM100_OR_SM103 and not FAKE_TARGET_IS_SM100_OR_SM103),
    reason="SM100/SM103 FakeTensor hd256 arbitrary compile gate",
)
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,is_varlen",
    [
        pytest.param(1, 1, False, id="mha_fixed"),
        pytest.param(4, 1, False, id="mqa_head_broadcast_fixed"),
        pytest.param(1, 1, True, id="mha_varlen"),
        pytest.param(4, 2, True, id="gqa_head_broadcast_varlen"),
        pytest.param(4, 1, True, id="mqa_head_broadcast_varlen"),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_sm100_arbitrary_hd256_fake_compiles_forward_and_backward(
    monkeypatch,
    num_heads,
    num_kv_heads,
    is_varlen,
):
    """Compile all hd256 consumers and reject stale pre-dynamic-ABI callables."""

    from flash_attn_cute.cache_utils import JITCache
    from flash_attn_cute.interface import _flash_attn_bwd, _flash_attn_fwd
    from flash_attn_cute.sm100_hd256_2cta_fmha_backward import (
        BlackwellFusedMultiHeadAttentionBackward,
    )
    from flash_attn_cute.sm100_hd256_2cta_fmha_forward import (
        BlackwellFusedMultiHeadAttentionForward,
    )

    assert is_fake_mode()
    q_lengths = (129, 0, 17) if is_varlen else (257,)
    k_lengths = (257, 129, 0) if is_varlen else (257,)
    batch_size, head_dim = len(q_lengths), 256
    if is_varlen:
        q = torch.empty(
            sum(q_lengths),
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.empty(
            sum(k_lengths),
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        cu_q = _cu_seqlens(q_lengths)
        cu_k = _cu_seqlens(k_lengths)
        varlen_kwargs = {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max(q_lengths),
            "max_seqlen_k": max(k_lengths),
        }
    else:
        q = torch.empty(
            batch_size,
            q_lengths[0],
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k = torch.empty(
            batch_size,
            k_lengths[0],
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        varlen_kwargs = {}
    v = torch.empty_like(k)
    func = torch.empty(1, 1, sum(q_lengths) + 256, dtype=torch.int32, device="cuda")
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
        **varlen_kwargs,
    )

    assert plan.plan_signature.kernel_family == "sm100_hd256_fwd"
    assert plan.block_size == (256, 128)
    assert plan.mask_block_masks.shape[1:] == (2, 128, 4)
    assert plan.dq_tensors is not None
    assert plan.dq_tensors.plan_signature.kernel_family == "sm100_hd256_dq"
    assert plan.dq_tensors.topology_tensors is plan.topology_tensors
    assert plan.dq_tensors.mask_block_masks.shape[1:] == (2, 128, 4)
    assert plan.bwd_tensors is not None
    assert plan.bwd_tensors.plan_signature.kernel_family == "sm100_hd256_dkdv"
    assert plan.bwd_tensors.block_size == (256, 128)
    assert plan.bwd_tensors.mask_block_masks.shape[1:] == (4, 256, 1)
    if is_varlen:
        assert plan.cu_total_m_blocks.shape == (batch_size + 1,)
        assert plan.dq_tensors.cu_total_m_blocks.shape == (batch_size + 1,)
        assert plan.bwd_tensors.cu_total_m_blocks.shape == (batch_size + 1,)

    class _StaleHd256AbiCache(JITCache):
        """Seed the pre-version-marker key immediately before the first lookup."""

        def __init__(self, stale_key_from_current):
            super().__init__()
            self.stale_key_from_current = stale_key_from_current
            self.current_key = None
            self.stale_key = None
            self.stale_callable = object()

        def __contains__(self, key):
            if self.current_key is None:
                self.current_key = key
                self.stale_key = self.stale_key_from_current(key)
                assert self.stale_key != self.current_key
                self.cache[self.stale_key] = self.stale_callable
            return super().__contains__(key)

    def fwd_stale_key(current_key):
        # Forward keeps the FA log level after the dedicated hd256 ABI marker.
        assert current_key[-2] == 1
        return (*current_key[:-2], current_key[-1])

    def bwd_stale_key(current_key):
        assert current_key[-1] == 1
        return current_key[:-1]

    fwd_cache = _StaleHd256AbiCache(fwd_stale_key)
    bwd_cache = _StaleHd256AbiCache(bwd_stale_key)
    monkeypatch.setattr(_flash_attn_fwd, "compile_cache", fwd_cache)
    monkeypatch.setattr(_flash_attn_bwd, "compile_cache", bwd_cache)
    real_compile = cute.compile
    main_compile_calls = []

    def compile_spy(*args, **kwargs):
        if args and isinstance(
            args[0],
            (
                BlackwellFusedMultiHeadAttentionForward,
                BlackwellFusedMultiHeadAttentionBackward,
            ),
        ):
            main_compile_calls.append((args[0], kwargs.get("options")))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", compile_spy)
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
        **varlen_kwargs,
    )
    dout = torch.empty_like(out)
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        **varlen_kwargs,
    )

    assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)
    assert len(main_compile_calls) == 2
    for cache in (fwd_cache, bwd_cache):
        assert len(cache) == 2
        assert cache.current_key in cache.cache
        assert cache.stale_key in cache.cache
        assert cache.cache[cache.stale_key] is cache.stale_callable
        assert cache.cache[cache.current_key] is not cache.stale_callable
    compiled_fwd, fwd_options = main_compile_calls[0]
    compiled_bwd, bwd_options = main_compile_calls[1]
    assert isinstance(compiled_fwd, BlackwellFusedMultiHeadAttentionForward)
    assert compiled_fwd.is_arbitrary
    assert compiled_fwd.cluster_shape_mn == (2, 1)
    assert compiled_fwd.qhead_per_kvhead == num_heads // num_kv_heads
    assert isinstance(compiled_bwd, BlackwellFusedMultiHeadAttentionBackward)
    assert compiled_bwd.is_arbitrary
    assert compiled_bwd.qhead_per_kvhead == num_heads // num_kv_heads
    assert compiled_bwd.subtile_factor == 2
    assert compiled_bwd.dq_kernel.is_arbitrary
    assert compiled_bwd.dkdv_kernel.is_arbitrary
    assert fwd_options is not None and "--enable-tvm-ffi" in fwd_options
    assert bwd_options is not None and "--enable-tvm-ffi" in bwd_options
    dispatch_arch = (
        int(_FAKE_TARGET_ARCH)
        if _FAKE_TARGET_ARCH
        else DEVICE_CAPABILITY[0] * 10 + DEVICE_CAPABILITY[1]
    )
    if dispatch_arch in (100, 103):
        assert "--opt-level 2" in bwd_options
    else:
        assert "--opt-level" not in bwd_options


def _assert_sm100_hd256_plan_contract(
    plan,
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
) -> None:
    """Check the three independent hd256 consumer payload allocations."""

    dq_plan = plan.dq_tensors
    dkdv_plan = plan.bwd_tensors
    assert dq_plan is not None and dkdv_plan is not None

    expected_q_rows = batch_size * math.ceil(q_len / 256)
    expected_k_rows = batch_size * math.ceil(k_len / 128)
    assert tuple(plan.mask_block_cnt.shape) == (1, expected_q_rows)
    assert tuple(dq_plan.mask_block_cnt.shape) == (1, expected_q_rows)
    assert tuple(dkdv_plan.mask_block_cnt.shape) == (1, expected_k_rows)

    assert plan.block_size == dq_plan.block_size == dkdv_plan.block_size == (256, 128)
    assert plan.plan_signature.kernel_family == "sm100_hd256_fwd"
    assert dq_plan.plan_signature.kernel_family == "sm100_hd256_dq"
    assert dkdv_plan.plan_signature.kernel_family == "sm100_hd256_dkdv"
    assert dq_plan.topology_tensors is plan.topology_tensors
    assert dq_plan.mask_block_cnt is plan.mask_block_cnt
    assert dq_plan.mask_block_idx is plan.mask_block_idx
    assert (
        dkdv_plan.topology_tensors.runtime_binding
        is plan.topology_tensors.runtime_binding
    )

    # The final dimensions are the consumer-native padded word strides.  Keep
    # them explicit here: compacting any one of these allocations would make a
    # physical Q/K tail read the next payload group instead of zero padding.
    for consumer_plan, payload_tail, alignment in (
        (plan, (2, 128, 4), 16),
        (dq_plan, (2, 128, 4), 16),
        (dkdv_plan, (4, 256, 1), 4),
    ):
        payload = consumer_plan.mask_block_masks
        assert payload.dtype == torch.uint32
        assert tuple(payload.shape[1:]) == payload_tail
        assert payload.is_contiguous() and payload.stride(-1) == 1
        assert payload.shape[0] == int(consumer_plan.mask_block_cnt.sum())
        if payload.numel():
            assert payload.data_ptr() % alignment == 0

    if plan.mask_block_masks.numel():
        assert plan.mask_block_masks.data_ptr() != dq_plan.mask_block_masks.data_ptr()


def _assert_sm100_hd256_varlen_plan_contract(
    plan,
    *,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    hmask: int,
) -> None:
    """Check compact Q256/K128 prefixes and all three payload contracts."""

    dq_plan = plan.dq_tensors
    dkdv_plan = plan.bwd_tensors
    assert dq_plan is not None and dkdv_plan is not None
    q_prefix = [0]
    k_prefix = [0]
    for q_len, k_len in zip(q_lengths, k_lengths):
        q_prefix.append(q_prefix[-1] + math.ceil(q_len / 256))
        k_prefix.append(k_prefix[-1] + math.ceil(k_len / 128))

    assert plan.cu_total_m_blocks.tolist() == q_prefix
    assert dq_plan.cu_total_m_blocks.tolist() == q_prefix
    assert dkdv_plan.cu_total_m_blocks.tolist() == k_prefix
    assert tuple(plan.mask_block_cnt.shape) == (hmask, q_prefix[-1])
    assert tuple(dq_plan.mask_block_cnt.shape) == (hmask, q_prefix[-1])
    assert tuple(dkdv_plan.mask_block_cnt.shape) == (hmask, k_prefix[-1])
    assert plan.block_size == dq_plan.block_size == dkdv_plan.block_size == (256, 128)
    assert dq_plan.topology_tensors is plan.topology_tensors
    assert (
        dq_plan.topology_tensors.runtime_binding
        is plan.topology_tensors.runtime_binding
    )
    assert (
        dkdv_plan.topology_tensors.runtime_binding
        is plan.topology_tensors.runtime_binding
    )
    for consumer_plan, payload_tail in (
        (plan, (2, 128, 4)),
        (dq_plan, (2, 128, 4)),
        (dkdv_plan, (4, 256, 1)),
    ):
        assert tuple(consumer_plan.mask_block_masks.shape[1:]) == payload_tail
        assert consumer_plan.mask_block_masks.shape[0] == int(
            consumer_plan.mask_block_cnt.sum()
        )


def _assert_sm100_hd256_backward_reference(
    *,
    q,
    k,
    v,
    out,
    lse,
    dout,
    grads,
    func,
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
) -> None:
    """Compare dedicated hd256 forward and all three gradients in FP32."""

    fixed = q.dim() == 4
    q_flat = q.detach().flatten(0, 1) if fixed else q.detach()
    k_flat = k.detach().flatten(0, 1) if fixed else k.detach()
    v_flat = v.detach().flatten(0, 1) if fixed else v.detach()
    q_ref = q_flat.float().requires_grad_(True)
    k_ref = k_flat.float().requires_grad_(True)
    v_ref = v_flat.float().requires_grad_(True)
    out_ref, lse_ref, empty_rows = _reference_attention(
        q_ref,
        k_ref,
        v_ref,
        func,
        q_lengths,
        k_lengths,
        softcap=0.0,
        score_mode=None,
    )
    ref_grads = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        (dout.flatten(0, 1) if fixed else dout).float(),
        allow_unused=True,
    )
    ref_grads = tuple(
        torch.zeros_like(ref) if grad is None else grad
        for grad, ref in zip(ref_grads, (q_ref, k_ref, v_ref))
    )

    expected_out = out_ref.view_as(out)
    if fixed:
        batch_size, q_len, num_heads = q.shape[:3]
        expected_lse = lse_ref.reshape(num_heads, batch_size, q_len).permute(1, 0, 2)
    else:
        expected_lse = lse_ref
    torch.testing.assert_close(out.float(), expected_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, expected_lse, atol=4e-3, rtol=4e-3)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(
            grad.float(), ref_grad.view_as(grad), atol=1e-1, rtol=1e-1
        )

    if empty_rows.any():
        empty_output_rows = (
            empty_rows.reshape(batch_size, q_len, num_heads) if fixed else empty_rows
        )
        assert torch.count_nonzero(out[empty_output_rows]) == 0
        lse_by_row = lse.transpose(1, 2) if fixed else lse.transpose(0, 1)
        assert torch.isneginf(lse_by_row[empty_output_rows]).all()


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime hd256 arbitrary correctness",
)
def test_sm100_arbitrary_hd256_fixed_mha_mixed_backward():
    """Exercise full, partial, and empty Q256/K128 rows across two samples."""

    batch_size, q_len, k_len, num_heads, head_dim = 2, 257, 129, 2, 256
    torch.manual_seed(4001)
    q = torch.randn(
        batch_size,
        q_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        k_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)

    func = torch.zeros(
        1,
        1,
        batch_size * q_len + 256,
        dtype=torch.int32,
        device="cuda",
    )
    # Sample 0: one true full Q256xK128 block, one physical K tail,
    # and an entirely empty second Q256 row (only local q=256 is valid).
    func[0, 0, :256] = k_len
    # Sample 1: mix a partial K0 prefix with full rows, then keep its Q tail full.
    q1_begin = q_len
    k1_begin = k_len
    func[0, 0, q1_begin : q1_begin + 128] = k1_begin + 64
    func[0, 0, q1_begin + 128 : q1_begin + q_len] = k1_begin + k_len

    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    _assert_sm100_hd256_plan_contract(
        plan,
        batch_size=batch_size,
        q_len=q_len,
        k_len=k_len,
    )
    assert plan.dq_tensors is not None and plan.bwd_tensors is not None
    assert int(plan.full_block_cnt.sum()) > 0
    assert int(plan.mask_block_cnt.sum()) > 0
    assert torch.any((plan.mask_block_cnt + plan.full_block_cnt) == 0), (
        "the forward plan must retain an empty compact Q256 row"
    )

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(4002)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)

    q_ref = q.detach().flatten(0, 1).float().requires_grad_(True)
    k_ref = k.detach().flatten(0, 1).float().requires_grad_(True)
    v_ref = v.detach().flatten(0, 1).float().requires_grad_(True)
    out_ref, lse_ref, empty_rows = _reference_attention(
        q_ref,
        k_ref,
        v_ref,
        func,
        (q_len, q_len),
        (k_len, k_len),
        softcap=0.0,
        score_mode=None,
    )
    ref_grads = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        dout.flatten(0, 1).float(),
    )
    expected_out = out_ref.reshape(batch_size, q_len, num_heads, head_dim)
    expected_lse = lse_ref.reshape(num_heads, batch_size, q_len).permute(1, 0, 2)
    torch.testing.assert_close(out.float(), expected_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, expected_lse, atol=4e-3, rtol=4e-3)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(
            grad.float(), ref_grad.view_as(grad), atol=1e-1, rtol=1e-1
        )
    empty_fixed = empty_rows.reshape(batch_size, q_len, num_heads)
    assert torch.count_nonzero(out[empty_fixed]) == 0
    assert torch.isneginf(lse.transpose(1, 2)[empty_fixed]).all()


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime hd256 arbitrary GQA/MQA correctness",
)
@pytest.mark.parametrize(
    "num_kv_heads",
    [
        pytest.param(2, id="gqa_h4_kv2"),
        pytest.param(1, id="mqa_h4_kv1"),
    ],
)
def test_sm100_arbitrary_hd256_fixed_head_broadcast_gqa_mqa_backward(
    num_kv_heads,
):
    """Reuse one compact sparse row across all Q heads of each KV head."""

    q_len, k_len, num_q_heads = 257, 129, 4
    case = _ForwardCase(
        f"hd256_gqa_h{num_q_heads}_kv{num_kv_heads}",
        (q_len,),
        (k_len,),
        num_q_heads,
        num_kv_heads,
        256,
        256,
        torch.bfloat16,
        False,
        1,
        pattern="mixed",
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = _make_arbitrary_func(
        case.q_lengths,
        case.k_lengths,
        hmask=1,
        pattern=case.pattern,
    )

    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    _assert_sm100_hd256_plan_contract(
        plan,
        batch_size=1,
        q_len=q_len,
        k_len=k_len,
    )
    qratio = num_q_heads // num_kv_heads
    assert plan.plan_signature.qhead_per_kvhead == qratio
    assert plan.dq_tensors.plan_signature.qhead_per_kvhead == qratio
    assert plan.bwd_tensors.plan_signature.qhead_per_kvhead == qratio
    assert plan.mask_block_cnt.shape[0] == 1

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(4050 + num_kv_heads)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_sm100_hd256_backward_reference(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        dout=dout,
        grads=grads,
        func=func,
        q_lengths=case.q_lengths,
        k_lengths=case.k_lengths,
    )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime hd256 arbitrary boundary correctness",
)
@pytest.mark.parametrize(
    "q_len,k_len,dtype",
    [
        pytest.param(127, 257, torch.float16, id="q127_k257_fp16"),
        pytest.param(128, 256, torch.bfloat16, id="q128_k256_bf16"),
        pytest.param(129, 255, torch.float16, id="q129_k255_fp16"),
        pytest.param(255, 129, torch.bfloat16, id="q255_k129_bf16"),
        pytest.param(256, 128, torch.float16, id="q256_k128_fp16"),
        pytest.param(257, 127, torch.bfloat16, id="q257_k127_bf16"),
    ],
)
def test_sm100_arbitrary_hd256_fixed_mha_boundaries(
    q_len,
    k_len,
    dtype,
):
    """Cross every Q256/K128 boundary with physical-tail payload padding."""

    case = _ForwardCase(
        f"hd256_q{q_len}_k{k_len}_{str(dtype).removeprefix('torch.')}",
        (q_len,),
        (k_len,),
        1,
        1,
        256,
        256,
        dtype,
        False,
        1,
        pattern="mixed",
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = _make_boundary_partial_func(q_len=q_len, k_len=k_len)
    assert torch.count_nonzero(func[..., q_len:]) == 0

    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    _assert_sm100_hd256_plan_contract(
        plan,
        batch_size=1,
        q_len=q_len,
        k_len=k_len,
    )
    assert int(plan.mask_block_cnt.sum()) > 0
    assert int(plan.dq_tensors.mask_block_cnt.sum()) > 0
    assert int(plan.bwd_tensors.mask_block_cnt.sum()) > 0
    if q_len <= 128:
        # CTA rank 1 owns Q[128:256] in the forward/dQ payloads, while dKdV
        # planes 2 and 3 are the same padded Q subtile split over its K CTAs.
        assert torch.count_nonzero(plan.mask_block_masks[:, 1]) == 0
        assert torch.count_nonzero(plan.dq_tensors.mask_block_masks[:, 1]) == 0
        assert torch.count_nonzero(plan.bwd_tensors.mask_block_masks[:, 2:]) == 0

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(4100 + q_len + k_len)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_sm100_hd256_backward_reference(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        dout=dout,
        grads=grads,
        func=func,
        q_lengths=case.q_lengths,
        k_lengths=case.k_lengths,
    )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime hd256 arbitrary dummy-row correctness",
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
    ],
)
def test_sm100_arbitrary_hd256_all_empty_forward_dummy_row(dtype):
    """All compact Q rows use the synchronized zero-payload forward dummy."""

    q_len, k_len = 257, 129
    case = _ForwardCase(
        f"hd256_all_empty_{str(dtype).removeprefix('torch.')}",
        (q_len,),
        (k_len,),
        1,
        1,
        256,
        256,
        dtype,
        False,
        1,
        pattern="empty",
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = torch.zeros(
        1,
        1,
        q_len + 256,
        dtype=torch.int32,
        device="cuda",
    )

    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    _assert_sm100_hd256_plan_contract(
        plan,
        batch_size=1,
        q_len=q_len,
        k_len=k_len,
    )
    for consumer_plan in (plan, plan.dq_tensors, plan.bwd_tensors):
        assert int(consumer_plan.mask_block_cnt.sum()) == 0
        assert int(consumer_plan.full_block_cnt.sum()) == 0
        assert consumer_plan.mask_block_masks.shape[0] == 0

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(4200)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_sm100_hd256_backward_reference(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        dout=dout,
        grads=grads,
        func=func,
        q_lengths=case.q_lengths,
        k_lengths=case.k_lengths,
    )
    assert all(torch.count_nonzero(grad) == 0 for grad in grads)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime hd256 arbitrary varlen correctness",
)
@pytest.mark.parametrize(
    "dtype,hmask",
    [
        pytest.param(torch.bfloat16, 1, id="bf16_hmask1"),
        pytest.param(torch.float16, 2, id="fp16_hmask_hq"),
    ],
)
def test_sm100_arbitrary_hd256_varlen_mha_boundaries_crossed_zero(
    dtype,
    hmask,
):
    """Exercise compact prefixes, overprovisioned pairs, and crossed-zero samples."""

    q_lengths = (127, 128, 0, 129, 255, 256, 257, 17)
    k_lengths = (129, 128, 129, 127, 257, 256, 255, 0)
    case = _ForwardCase(
        f"hd256_varlen_{str(dtype).removeprefix('torch.')}_hmask{hmask}",
        q_lengths,
        k_lengths,
        2,
        2,
        256,
        256,
        dtype,
        False,
        1,
        pattern="mixed",
        per_head_mask=hmask == 2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(q_lengths)
    cu_k = _cu_seqlens(k_lengths)
    func = _make_varlen_boundary_partial_func(
        q_lengths,
        k_lengths,
        hmask=hmask,
    )
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
        build_backward=True,
    )
    _assert_sm100_hd256_varlen_plan_contract(
        plan,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        hmask=hmask,
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
    torch.manual_seed(4300 + hmask)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_sm100_hd256_backward_reference(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        dout=dout,
        grads=grads,
        func=func,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
    )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime hd256 arbitrary varlen GQA/MQA correctness",
)
@pytest.mark.parametrize("num_kv_heads", [2, 1], ids=("gqa", "mqa"))
@pytest.mark.parametrize("hmask", [1, 4], ids=("head_broadcast", "head_specific"))
def test_sm100_arbitrary_hd256_varlen_gqa_mqa_backward(
    num_kv_heads,
    hmask,
):
    """Cover grouped-head varlen with broadcast and head-specific arbitrary masks."""

    q_lengths = (129, 17, 0)
    k_lengths = (257, 129, 23)
    case = _ForwardCase(
        f"hd256_varlen_hq4_hkv{num_kv_heads}",
        q_lengths,
        k_lengths,
        4,
        num_kv_heads,
        256,
        256,
        torch.bfloat16,
        False,
        1,
        pattern="mixed",
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(q_lengths)
    cu_k = _cu_seqlens(k_lengths)
    func = _make_varlen_boundary_partial_func(q_lengths, k_lengths, hmask=hmask)
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
        build_backward=True,
    )
    _assert_sm100_hd256_varlen_plan_contract(
        plan,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
        hmask=hmask,
    )
    qratio = case.num_q_heads // case.num_kv_heads
    assert plan.plan_signature.qhead_per_kvhead == qratio
    assert plan.dq_tensors.plan_signature.qhead_per_kvhead == qratio
    assert plan.bwd_tensors.plan_signature.qhead_per_kvhead == qratio

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
    torch.manual_seed(4350 + num_kv_heads)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_sm100_hd256_backward_reference(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        dout=dout,
        grads=grads,
        func=func,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
    )


def _make_d192_backward_func(*, q_len: int, k_len: int, pattern: str):
    """Build one high-signal mask for the existing D192 K256 consumer."""

    func = torch.zeros(
        1,
        1,
        q_len + 256,
        dtype=torch.int32,
        device="cuda",
    )
    if pattern == "cta_local_zero":
        # Rank 0 owns visible K[0:128]; rank 1 owns an entirely empty slice.
        func[0, 0, :q_len] = min(k_len, 128)
    elif pattern == "mixed_union":
        # Rank 0 is full while rank 1 has a Q-dependent partial width.
        q_idx = torch.arange(q_len, dtype=torch.int32, device="cuda")
        func[0, 0, :q_len] = torch.clamp(
            129 + (q_idx * 29 + 7).remainder(64),
            max=k_len,
        )
    elif pattern == "full":
        func[0, 0, :q_len] = k_len
    else:
        raise ValueError(f"unknown D192 test pattern: {pattern}")
    return func


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime arbitrary D192 backward correctness",
)
@pytest.mark.parametrize(
    "k_len,pattern,deterministic_repeat,dtype",
    [
        pytest.param(
            129,
            "cta_local_zero",
            False,
            torch.bfloat16,
            id="bf16_k129_cta_local_zero",
        ),
        pytest.param(
            255, "mixed_union", False, torch.bfloat16, id="bf16_k255_mixed_union"
        ),
        pytest.param(256, "full", False, torch.bfloat16, id="bf16_k256_full_cluster"),
        pytest.param(257, "full", False, torch.bfloat16, id="bf16_k257_full_plus_tail"),
        pytest.param(
            129,
            "cta_local_zero",
            True,
            torch.bfloat16,
            id="bf16_k129_cta_local_zero_deterministic",
        ),
        pytest.param(
            769,
            "full",
            True,
            torch.bfloat16,
            id="bf16_k769_full_plus_tail_deterministic",
        ),
        pytest.param(
            129,
            "cta_local_zero",
            False,
            torch.float16,
            id="fp16_k129_cta_local_zero",
        ),
        pytest.param(
            257,
            "full",
            False,
            torch.float16,
            id="fp16_k257_full_plus_tail",
        ),
        pytest.param(
            769,
            "full",
            True,
            torch.float16,
            id="fp16_k769_full_plus_tail_deterministic",
        ),
    ],
)
def test_sm100_arbitrary_fixed_backward_d192(
    k_len,
    pattern,
    deterministic_repeat,
    dtype,
):
    """Validate arbitrary D192 on its pre-existing 2CTA topology."""

    q_len = 129
    case = _ForwardCase(
        f"bwd_d192_k{k_len}_{pattern}",
        (q_len,),
        (k_len,),
        1,
        1,
        192,
        128,
        dtype,
        False,
        2,
        pattern="full" if pattern == "full" else "mixed",
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = _make_d192_backward_func(
        q_len=q_len,
        k_len=k_len,
        pattern=pattern,
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )

    assert plan.bwd_tensors is not None
    bwd = plan.bwd_tensors
    signature = bwd.plan_signature
    assert signature.direction == "backward"
    assert signature.kernel_family == "sm100_generic_bwd"
    assert signature.topology.block_size == (128, 256)
    assert bwd.block_size == (128, 256)
    assert signature.cta_group_size == 2
    assert signature.cluster_axis == "n"
    assert signature.pack_gqa is False
    payload = bwd.mask_block_masks
    assert payload.dtype == torch.uint32
    assert payload.shape[1:] == (2, 256, 4)
    assert payload.data_ptr() % 16 == 0
    assert torch.count_nonzero(payload[..., 2:].to(torch.int64)) == 0
    if deterministic_repeat:
        _assert_rank_only_orders_are_dense(
            plan,
            k_block_prefix=(0, math.ceil(k_len / 256)),
        )

    partial_count = int(bwd.mask_block_cnt.sum())
    full_count = int(bwd.full_block_cnt.sum())
    if pattern == "cta_local_zero":
        assert partial_count > 0 and full_count == 0
        assert torch.count_nonzero(payload[:, 0, :, :2].to(torch.int64)) > 0
        assert torch.count_nonzero(payload[:, 1, :, :2].to(torch.int64)) == 0
    elif pattern == "mixed_union":
        assert partial_count > 0 and full_count == 0
        assert torch.count_nonzero(payload[:, 0, :, :2].to(torch.int64)) > 0
        assert torch.count_nonzero(payload[:, 1, :, :2].to(torch.int64)) > 0
    elif k_len == 256:
        assert partial_count == 0 and full_count > 0
    else:
        # K=257/769 have full K256 clusters plus one physical partial tail.
        assert partial_count > 0 and full_count > 0
        assert torch.count_nonzero(payload[:, 0, :, :2].to(torch.int64)) > 0
        assert torch.count_nonzero(payload[:, 1, :, :2].to(torch.int64)) == 0

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=deterministic_repeat,
        return_lse=True,
    )
    torch.manual_seed(900 + k_len)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(
        out,
        (q, k, v),
        dout,
        retain_graph=deterministic_repeat,
    )
    if deterministic_repeat:
        for repeat_idx in range(4):
            repeated_grads = torch.autograd.grad(
                out,
                (q, k, v),
                dout,
                retain_graph=repeat_idx < 3,
            )
            for grad, repeated_grad in zip(grads, repeated_grads):
                assert torch.equal(grad, repeated_grad)
    try:
        _assert_backward_reference(
            case,
            q,
            k,
            v,
            out,
            lse,
            dout,
            grads,
            func,
            fixed=True,
        )
    except AssertionError as error:
        raise AssertionError(f"D192 K boundary {k_len} ({pattern}): {error}") from error


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime arbitrary varlen D192 backward correctness",
)
@pytest.mark.parametrize("deterministic", [False, True], ids=["nondet", "det"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_sm100_arbitrary_varlen_backward_d192_crossed_zero(
    deterministic,
    dtype,
):
    """Validate compact K256 prefixes and cluster lockstep across zero samples."""

    case = _ForwardCase(
        "bwd_d192_varlen_crossed_zero",
        (129, 0, 17),
        (257, 129, 0),
        1,
        1,
        192,
        128,
        dtype,
        False,
        2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_varlen_boundary_partial_func(case.q_lengths, case.k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )

    bwd = plan.bwd_tensors
    assert bwd is not None
    assert bwd.block_size == (128, 256)
    assert bwd.plan_signature.cta_group_size == 2
    assert tuple(bwd.cu_total_m_blocks.tolist()) == (0, 2, 3, 3)
    topology = bwd.topology_tensors
    assert tuple(topology.cu_total_q_plan_rows.tolist()) == (0, 2, 2, 3)
    assert topology.cu_total_k_plan_rows is bwd.cu_total_m_blocks
    assert int(bwd.mask_block_cnt.sum()) > 0
    assert int(bwd.full_block_cnt.sum()) > 0
    payload = bwd.mask_block_masks
    assert payload.shape[1:] == (2, 256, 4)
    assert torch.count_nonzero(payload[..., 2:].to(torch.int64)) == 0
    assert torch.count_nonzero(payload[:, 0, :, :2].to(torch.int64)) > 0
    assert torch.count_nonzero(payload[:, 1, :, :2].to(torch.int64)) == 0

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=deterministic,
        return_lse=True,
    )
    torch.manual_seed(925)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(
        out,
        (q, k, v),
        dout,
        retain_graph=deterministic,
    )
    if deterministic:
        for repeat_idx in range(4):
            repeated_grads = torch.autograd.grad(
                out,
                (q, k, v),
                dout,
                retain_graph=repeat_idx < 3,
            )
            for grad, repeated_grad in zip(grads, repeated_grads):
                assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)

    # Sample 1 has K but no Q, while sample 2 has Q but no K.
    for grad in grads[1:]:
        assert torch.count_nonzero(grad[257:386]) == 0
    q_only = slice(129, 146)
    assert torch.count_nonzero(out[q_only]) == 0
    assert torch.count_nonzero(grads[0][q_only]) == 0
    assert torch.isneginf(lse[:, q_only]).all()


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime arbitrary varlen D192 backward boundaries",
)
@pytest.mark.parametrize("deterministic", [False, True], ids=["nondet", "det"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_sm100_arbitrary_varlen_backward_d192_boundaries(
    deterministic,
    dtype,
):
    """Cover Q128/K256 plan rows across every adjacent sequence boundary."""

    case = _ForwardCase(
        "bwd_d192_varlen_boundaries",
        (127, 128, 0, 129, 255, 256, 257, 17),
        (129, 128, 129, 127, 257, 256, 255, 0),
        1,
        1,
        192,
        128,
        dtype,
        False,
        2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_varlen_boundary_partial_func(case.q_lengths, case.k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )

    bwd = plan.bwd_tensors
    assert bwd is not None
    expected_q_prefix = (0, 1, 2, 2, 4, 6, 8, 11, 12)
    expected_k256_prefix = (0, 1, 2, 3, 4, 6, 7, 8, 8)
    assert (
        tuple(bwd.topology_tensors.cu_total_q_plan_rows.tolist()) == expected_q_prefix
    )
    assert tuple(bwd.cu_total_m_blocks.tolist()) == expected_k256_prefix
    assert bwd.topology_tensors.cu_total_k_plan_rows is bwd.cu_total_m_blocks
    assert int(bwd.mask_block_cnt.sum()) > 0
    assert int(bwd.full_block_cnt.sum()) > 0
    assert bwd.mask_block_masks.shape[1:] == (2, 256, 4)
    assert torch.count_nonzero(bwd.mask_block_masks[..., 2:].to(torch.int64)) == 0

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=deterministic,
        return_lse=True,
    )
    torch.manual_seed(926)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(
        out,
        (q, k, v),
        dout,
        retain_graph=deterministic,
    )
    if deterministic:
        for repeat_idx in range(4):
            repeated_grads = torch.autograd.grad(
                out,
                (q, k, v),
                dout,
                retain_graph=repeat_idx < 3,
            )
            for grad, repeated_grad in zip(grads, repeated_grads):
                assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime arbitrary varlen D192 grouped-head backward correctness",
)
@pytest.mark.parametrize("num_kv_heads", [2, 1], ids=("gqa", "mqa"))
def test_sm100_arbitrary_varlen_backward_d192_grouped_heads(
    num_kv_heads,
):
    """Validate D192 dK/dV head reduction on the mandatory K256 topology."""

    case = _ForwardCase(
        f"bwd_d192_varlen_hq4_hkv{num_kv_heads}",
        (129, 17),
        (257, 129),
        4,
        num_kv_heads,
        192,
        128,
        torch.bfloat16,
        False,
        2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_varlen_boundary_partial_func(case.q_lengths, case.k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )

    bwd = plan.bwd_tensors
    assert bwd is not None
    assert bwd.block_size == (128, 256)
    assert bwd.plan_signature.cta_group_size == 2
    assert bwd.plan_signature.qhead_per_kvhead == 4 // num_kv_heads
    assert bwd.mask_block_masks.shape[1:] == (2, 256, 4)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(927 + num_kv_heads)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize(
    "head_dim,dtype,num_q_heads,num_kv_heads",
    [
        pytest.param(64, torch.bfloat16, 1, 1, id="bf16_d64_mha"),
        pytest.param(128, torch.bfloat16, 1, 1, id="bf16_d128_mha_2cta"),
        pytest.param(64, torch.float16, 1, 1, id="fp16_d64_mha"),
        pytest.param(128, torch.float16, 1, 1, id="fp16_d128_mha_2cta"),
        pytest.param(128, torch.bfloat16, 4, 1, id="bf16_d128_mqa4_2cta"),
    ],
)
def test_sm100_arbitrary_fixed_backward_topology_matrix(
    head_dim,
    dtype,
    num_q_heads,
    num_kv_heads,
):
    """Cover every compact K2Q row shape with one JIT per head dimension."""

    for pattern in ("partial_only", "full_only", "empty", "mixed"):
        _run_sm100_arbitrary_fixed_backward(
            head_dim=head_dim,
            dtype=dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            pattern=pattern,
        )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64_1cta", "d128_2cta"])
def test_sm100_arbitrary_fixed_backward_deterministic_repeat(head_dim):
    """The rank-only dQ order must make repeated deterministic bwd bitwise stable."""

    _run_sm100_arbitrary_fixed_backward(
        head_dim=head_dim,
        pattern="mixed",
        deterministic_repeat=True,
    )


def _run_sm100_arbitrary_fixed_backward(
    *,
    head_dim: int,
    pattern: str,
    dtype: torch.dtype = torch.bfloat16,
    num_q_heads: int = 1,
    num_kv_heads: int = 1,
    deterministic_repeat: bool = False,
):
    dtype_name = str(dtype).removeprefix("torch.")
    case = _ForwardCase(
        f"bwd_mha_{dtype_name}_d{head_dim}_{pattern}",
        (129,),
        (256,),
        num_q_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        dtype,
        False,
        2,
        pattern=pattern,
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = torch.zeros(
        1,
        1,
        case.q_lengths[0] + 256,
        dtype=torch.int32,
        device="cuda",
    )
    endpoint = {
        "partial_only": 64,
        "full_only": case.k_lengths[0],
        "empty": 0,
        # This is full+partial for K128, but one partial cluster tile for K256.
        "mixed": 192,
    }[pattern]
    func[0, 0, : case.q_lengths[0]] = endpoint
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    assert plan.bwd_tensors is not None
    expected_cta_group_size = 2 if head_dim == 128 else 1
    assert plan.bwd_tensors.block_size == (
        128,
        128 * expected_cta_group_size,
    )
    assert plan.bwd_tensors.plan_signature.cta_group_size == expected_cta_group_size
    partial_count = int(plan.bwd_tensors.mask_block_cnt.sum())
    full_count = int(plan.bwd_tensors.full_block_cnt.sum())
    expected_nonempty = {
        "partial_only": (True, False),
        "full_only": (False, True),
        "empty": (False, False),
        "mixed": (True, expected_cta_group_size == 1),
    }[pattern]
    assert (partial_count > 0, full_count > 0) == expected_nonempty

    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=deterministic_repeat,
        return_lse=True,
    )
    torch.manual_seed(321 + head_dim)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(
        out,
        (q, k, v),
        dout,
        retain_graph=deterministic_repeat,
    )
    if deterministic_repeat:
        # Keep the unit test bounded; the longer 50--100 launch stress belongs
        # in the dedicated SM100/SM103 soak run described by the migration plan.
        for repeat_idx in range(4):
            repeated_grads = torch.autograd.grad(
                out,
                (q, k, v),
                dout,
                retain_graph=repeat_idx < 3,
            )
            for grad, repeated_grad in zip(grads, repeated_grads):
                assert torch.equal(grad, repeated_grad)
    dq, dk, dv = grads

    q_ref = q.detach().flatten(0, 1).float().requires_grad_(True)
    k_ref = k.detach().flatten(0, 1).float().requires_grad_(True)
    v_ref = v.detach().flatten(0, 1).float().requires_grad_(True)
    out_ref, lse_ref, empty_rows = _reference_attention(
        q_ref,
        k_ref,
        v_ref,
        func,
        case.q_lengths,
        case.k_lengths,
        softcap=case.softcap,
        score_mode=case.score_mode,
    )
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        dout.flatten(0, 1).float(),
    )
    try:
        _assert_result(
            out,
            lse,
            out_ref,
            lse_ref,
            empty_rows,
            fixed=True,
        )
        torch.testing.assert_close(dq.float(), dq_ref.view_as(dq), atol=6e-2, rtol=6e-2)
        torch.testing.assert_close(dk.float(), dk_ref.view_as(dk), atol=6e-2, rtol=6e-2)
        torch.testing.assert_close(dv.float(), dv_ref.view_as(dv), atol=6e-2, rtol=6e-2)
    except AssertionError as error:
        raise AssertionError(f"{case.name}: {error}") from error


def _assert_backward_reference(
    case: _ForwardCase,
    q,
    k,
    v,
    out,
    lse,
    dout,
    grads,
    func,
    *,
    fixed: bool,
):
    q_ref = q.detach().float()
    k_ref = k.detach().float()
    v_ref = v.detach().float()
    if fixed:
        q_ref = q_ref.flatten(0, 1)
        k_ref = k_ref.flatten(0, 1)
        v_ref = v_ref.flatten(0, 1)
    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)
    out_ref, lse_ref, empty_rows = _reference_attention(
        q_ref,
        k_ref,
        v_ref,
        func,
        case.q_lengths,
        case.k_lengths,
        softcap=case.softcap,
        score_mode=case.score_mode,
    )
    dout_ref = dout.flatten(0, 1).float() if fixed else dout.float()
    ref_grads = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        dout_ref,
        allow_unused=True,
    )
    ref_grads = tuple(
        torch.zeros_like(ref) if grad is None else grad
        for grad, ref in zip(ref_grads, (q_ref, k_ref, v_ref))
    )
    _assert_result(
        out,
        lse,
        out_ref,
        lse_ref,
        empty_rows,
        fixed=fixed,
    )
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(
            grad.float(), ref_grad.view_as(grad), atol=8e-2, rtol=8e-2
        )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize(
    "deterministic", [False, True], ids=("nondeterministic", "deterministic")
)
def test_sm100_arbitrary_varlen_backward_internal_zero_d64(
    deterministic,
):
    case = _ForwardCase(
        "bwd_varlen_internal_zero_d64",
        (17, 0, 31),
        (23, 0, 37),
        1,
        1,
        64,
        64,
        torch.bfloat16,
        False,
        1,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=1, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=deterministic,
        return_lse=True,
    )
    torch.manual_seed(654)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=deterministic)
    if deterministic:
        for repeat_idx in range(2):
            repeated_grads = torch.autograd.grad(
                out,
                (q, k, v),
                dout,
                retain_graph=repeat_idx == 0,
            )
            for grad, repeated_grad in zip(grads, repeated_grads):
                assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)
    assert plan.bwd_tensors is not None
    assert plan.bwd_tensors.cu_total_m_blocks is not None
    assert plan.bwd_tensors.cu_total_m_blocks.tolist() == [0, 1, 1, 2]


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_varlen_backward_d96_dv64_partial_full():
    """Cover independent QK/V dimensions with both K2Q row kinds."""

    case = _ForwardCase(
        "bwd_varlen_bf16_d96_dv64_partial_full",
        (129, 17),
        (257, 193),
        1,
        1,
        96,
        64,
        torch.bfloat16,
        False,
        2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_varlen_boundary_partial_func(case.q_lengths, case.k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    assert plan.bwd_tensors is not None
    assert int(plan.bwd_tensors.mask_block_cnt.sum()) > 0
    assert int(plan.bwd_tensors.full_block_cnt.sum()) > 0

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(664)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_varlen_backward_boundaries_and_crossed_zero():
    """Cover every Q/K 128-boundary plus Q-only and K-only samples."""

    case = _ForwardCase(
        "bwd_varlen_boundaries_crossed_zero",
        (127, 128, 0, 129, 255, 256, 257, 17),
        (129, 128, 129, 127, 257, 256, 255, 0),
        1,
        1,
        64,
        64,
        torch.bfloat16,
        False,
        2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_varlen_boundary_partial_func(case.q_lengths, case.k_lengths)
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    assert plan.bwd_tensors is not None
    bwd_topology = plan.bwd_tensors.topology_tensors
    expected_q_prefix = (0, 1, 2, 2, 4, 6, 8, 11, 12)
    expected_k_prefix = (0, 2, 3, 5, 6, 9, 11, 13, 13)
    assert bwd_topology.cu_total_q_plan_rows is not None
    assert tuple(bwd_topology.cu_total_q_plan_rows.tolist()) == expected_q_prefix
    assert tuple(plan.bwd_tensors.cu_total_m_blocks.tolist()) == expected_k_prefix
    _assert_rank_only_orders_are_dense(plan, k_block_prefix=expected_k_prefix)

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=True,
        return_lse=True,
    )
    torch.manual_seed(665)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    for repeat_idx in range(2):
        repeated = torch.autograd.grad(
            out, (q, k, v), dout, retain_graph=repeat_idx == 0
        )
        for grad, repeated_grad in zip(grads, repeated):
            assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)

    q_offsets = [0]
    k_offsets = [0]
    for q_len, k_len in zip(case.q_lengths, case.k_lengths):
        q_offsets.append(q_offsets[-1] + q_len)
        k_offsets.append(k_offsets[-1] + k_len)
    # Sample 2 has K but no Q, so it must not contribute dK/dV.
    for grad in grads[1:]:
        assert torch.count_nonzero(grad[k_offsets[2] : k_offsets[3]]) == 0
    # Sample 7 has Q but no K, so its output and dQ must be exactly zero.
    q_only = slice(q_offsets[7], q_offsets[8])
    assert torch.count_nonzero(out[q_only]) == 0
    assert torch.count_nonzero(grads[0][q_only]) == 0
    assert torch.isneginf(lse[:, q_only]).all()


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize("num_kv_heads", [2, 1], ids=("gqa", "mqa"))
def test_sm100_arbitrary_varlen_backward_grouped_heads(num_kv_heads):
    case = _ForwardCase(
        f"bwd_varlen_hq4_hkv{num_kv_heads}_d64",
        (17, 0, 31),
        (23, 0, 37),
        4,
        num_kv_heads,
        64,
        64,
        torch.bfloat16,
        False,
        1,
        per_head_mask=True,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_arbitrary_func(
        case.q_lengths,
        case.k_lengths,
        hmask=case.num_q_heads,
        pattern="mixed",
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(654 + num_kv_heads)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize("num_kv_heads", [2, 1], ids=("gqa", "mqa"))
def test_sm100_arbitrary_varlen_deterministic_empty_head_tokens(
    num_kv_heads,
):
    """Empty Q heads and crossed-zero samples must advance dK/dV tokens."""

    case = _ForwardCase(
        f"bwd_varlen_deterministic_hq4_hkv{num_kv_heads}_empty_heads",
        (129, 0, 17),
        (257, 129, 0),
        4,
        num_kv_heads,
        64,
        64,
        torch.bfloat16,
        False,
        2,
        per_head_mask=True,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_varlen_boundary_partial_func(
        case.q_lengths, case.k_lengths, hmask=case.num_q_heads
    )
    func[0::2].zero_()
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=True,
        return_lse=True,
    )
    torch.manual_seed(666 + num_kv_heads)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    for repeat_idx in range(2):
        repeated = torch.autograd.grad(
            out, (q, k, v), dout, retain_graph=repeat_idx == 0
        )
        for grad, repeated_grad in zip(grads, repeated):
            assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)
    assert torch.count_nonzero(out[:, 0::2]) == 0
    assert torch.count_nonzero(grads[0][:, 0::2]) == 0
    # The middle sample has no Q, and therefore cannot contribute dK/dV.
    k_middle = slice(case.k_lengths[0], sum(case.k_lengths[:2]))
    for grad in grads[1:]:
        assert torch.count_nonzero(grad[k_middle]) == 0
    # The last sample has Q but no K.
    q_last = slice(case.q_lengths[0], sum(case.q_lengths))
    assert torch.count_nonzero(out[q_last]) == 0
    assert torch.count_nonzero(grads[0][q_last]) == 0


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_varlen_pack_gqa_backward_plan_transition():
    """Autograd preserves a Q2K outer plan with a K2Q consumer."""

    case = _ForwardCase(
        "bwd_varlen_pack_gqa_qratio8",
        (17, 0, 31),
        (23, 0, 37),
        8,
        1,
        64,
        64,
        torch.bfloat16,
        True,
        2,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=1, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=True,
        build_backward=True,
    )
    assert plan.bwd_tensors is not None
    assert plan.plan_signature.pack_gqa is True
    assert plan.plan_signature.qhead_per_kvhead == 8
    assert plan.bwd_tensors.plan_signature.pack_gqa is False
    assert plan.bwd_tensors.plan_signature.qhead_per_kvhead == 8
    assert (
        plan.bwd_tensors.topology_tensors.runtime_binding
        is plan.topology_tensors.runtime_binding
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=True,
        deterministic=True,
        return_lse=True,
    )
    torch.manual_seed(669)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    repeated = torch.autograd.grad(out, (q, k, v), dout)
    for grad, repeated_grad in zip(grads, repeated):
        assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize("mode", ["softcap", "score_mod"])
def test_sm100_arbitrary_varlen_backward_score_modifiers(mode):
    case = _ForwardCase(
        f"bwd_varlen_{mode}_d64",
        (147, 131),
        (257, 193),
        1,
        1,
        64,
        64,
        torch.bfloat16,
        False,
        2,
        softcap=7.0 if mode == "softcap" else 0.0,
        score_mode="zero" if mode == "score_mod" else None,
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=1, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    score_mod = _score_to_zero if mode == "score_mod" else None
    score_mod_bwd = _score_to_zero_bwd if mode == "score_mod" else None
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=True,
        softcap=case.softcap,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        return_lse=True,
    )
    torch.manual_seed(670 + len(mode))
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    repeated = torch.autograd.grad(out, (q, k, v), dout)
    for grad, repeated_grad in zip(grads, repeated):
        assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime arbitrary varlen score_mod aux correctness",
)
def test_sm100_arbitrary_varlen_backward_score_mod_aux_global_kv():
    """Use aux data and varlen offsets in score recomputation and backward."""

    case = _ForwardCase(
        "bwd_varlen_score_mod_aux_global_kv_d64",
        (147, 131),
        (257, 193),
        1,
        1,
        64,
        64,
        torch.bfloat16,
        False,
        2,
        score_mode="aux_global_kv",
    )
    q, k, v = _make_inputs(case, varlen=True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=1, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=False,
        build_backward=True,
    )
    kv_bias = (
        torch.arange(sum(case.k_lengths), dtype=torch.float32, device="cuda") * 0.003
    )
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        deterministic=True,
        score_mod=_score_add_aux_global_kv,
        score_mod_bwd=_score_add_aux_global_kv_bwd,
        aux_tensors=[kv_bias],
        return_lse=True,
    )
    torch.manual_seed(675)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
    repeated = torch.autograd.grad(out, (q, k, v), dout)
    for grad, repeated_grad in zip(grads, repeated):
        assert torch.equal(grad, repeated_grad)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=False)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_fixed_backward_aux_metadata_specializes_cache():
    """Aux dtype and stride-0 layout must not reuse one compiled callable."""

    case = _ForwardCase(
        "bwd_fixed_aux_cache_metadata",
        (129,),
        (257,),
        1,
        1,
        64,
        64,
        torch.bfloat16,
        False,
        2,
        score_mode="aux_scalar",
    )
    q_base, k_base, v_base = _make_inputs(case, varlen=False)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=1, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func, q_base, k_base, v_base, pack_gqa=False, build_backward=True
    )
    torch.manual_seed(681)
    dout = torch.randn_like(q_base)
    aux_variants = (
        torch.tensor([0.375], dtype=torch.float32, device="cuda"),
        torch.tensor([0.375], dtype=torch.bfloat16, device="cuda").expand(2),
    )
    for aux in aux_variants:
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        out, lse = flash_attn_func(
            q,
            k,
            v,
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=False,
            deterministic=True,
            score_mod=_score_add_aux_scalar,
            score_mod_bwd=_score_add_aux_scalar_bwd,
            aux_tensors=[aux],
            return_lse=True,
        )
        grads = torch.autograd.grad(out, (q, k, v), dout)
        _assert_backward_reference(
            case, q, k, v, out, lse, dout, grads, func, fixed=True
        )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime API validation",
)
def test_sm100_arbitrary_score_mod_api_guards_preserve_no_grad():
    case = _ForwardCase(
        "score_mod_api_guards",
        (64,),
        (129,),
        1,
        1,
        64,
        64,
        torch.bfloat16,
        False,
        1,
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=1, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func, q, k, v, pack_gqa=False, build_backward=True
    )
    call_kwargs = {
        "arbitrary": True,
        "block_sparse_tensors": plan,
        "pack_gqa": False,
        "return_lse": True,
    }
    with pytest.raises(ValueError, match="must be provided together"):
        flash_attn_func(q, k, v, score_mod=_score_to_zero, **call_kwargs)
    with pytest.raises(ValueError, match="cannot be provided without"):
        flash_attn_func(q, k, v, score_mod_bwd=_score_to_zero_bwd, **call_kwargs)
    with torch.no_grad():
        out, lse = flash_attn_func(q, k, v, score_mod=_score_to_zero, **call_kwargs)
    assert torch.isfinite(out).all()
    assert not out.requires_grad
    assert not lse.requires_grad
    with pytest.raises(ValueError, match="device cpu"):
        flash_attn_func(
            q,
            k,
            v,
            score_mod=_score_add_aux_scalar,
            score_mod_bwd=_score_add_aux_scalar_bwd,
            aux_tensors=[torch.tensor([0.375])],
            **call_kwargs,
        )


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
@pytest.mark.parametrize("num_kv_heads", [2, 1], ids=("gqa", "mqa"))
def test_sm100_arbitrary_fixed_backward_grouped_heads(num_kv_heads):
    case = _ForwardCase(
        f"bwd_hq4_hkv{num_kv_heads}_d64",
        (129,),
        (257,),
        4,
        num_kv_heads,
        64,
        64,
        torch.bfloat16,
        False,
        2,
        per_head_mask=True,
    )
    q, k, v = _make_inputs(case, varlen=False)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    func = _make_arbitrary_func(
        case.q_lengths, case.k_lengths, hmask=case.num_q_heads, pattern="mixed"
    )
    plan = create_arbitrary_block_sparse_tensors(
        func, q, k, v, pack_gqa=False, build_backward=True
    )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(655 + num_kv_heads)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)
    _assert_backward_reference(case, q, k, v, out, lse, dout, grads, func, fixed=True)


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_fixed_backward_batch_isolation():
    batch_size, q_len, k_len, head_dim = 2, 129, 257, 64
    torch.manual_seed(658)
    q = torch.randn(
        batch_size,
        q_len,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        k_len,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    q_lengths = (q_len,) * batch_size
    k_lengths = (k_len,) * batch_size
    func = _make_arbitrary_func(q_lengths, k_lengths, hmask=1, pattern="mixed")
    # Sample 0 is full while sample 1 retains its global-offset mixed mask.
    # A fixed-row stride bug therefore cannot hide behind identical samples.
    func[:, :, :q_len] = k_len
    plan = create_arbitrary_block_sparse_tensors(
        func, q, k, v, pack_gqa=False, build_backward=True
    )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        arbitrary=True,
        block_sparse_tensors=plan,
        pack_gqa=False,
        return_lse=True,
    )
    torch.manual_seed(659)
    dout = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dout)

    q_ref = q.detach().flatten(0, 1).float().requires_grad_(True)
    k_ref = k.detach().flatten(0, 1).float().requires_grad_(True)
    v_ref = v.detach().flatten(0, 1).float().requires_grad_(True)
    out_ref, lse_ref, _ = _reference_attention(
        q_ref,
        k_ref,
        v_ref,
        func,
        q_lengths,
        k_lengths,
        softcap=0.0,
        score_mode=None,
    )
    ref_grads = torch.autograd.grad(
        out_ref,
        (q_ref, k_ref, v_ref),
        dout.flatten(0, 1).float(),
    )
    torch.testing.assert_close(out.float(), out_ref.view_as(out), atol=3e-2, rtol=3e-2)
    expected_lse = lse_ref.reshape(1, batch_size, q_len).permute(1, 0, 2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-3, rtol=3e-3)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(
            grad.float(), ref_grad.view_as(grad), atol=8e-2, rtol=8e-2
        )
    assert plan.bwd_tensors is not None
    assert plan.bwd_tensors.mask_block_cnt.shape[1] == batch_size * 3


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_fixed_backward_batch_invariance():
    """A deterministic target sample is bitwise invariant to a batch prefix."""

    q_len, k_len, head_dim = 129, 257, 64
    torch.manual_seed(660)
    q_target = torch.randn(1, q_len, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    k_target = torch.randn(1, k_len, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    v_target = torch.randn_like(k_target)
    q_prefix = torch.randn_like(q_target)
    k_prefix = torch.randn_like(k_target)
    v_prefix = torch.randn_like(v_target)
    dout_target = torch.randn_like(q_target)
    dout_prefix = torch.randn_like(q_prefix)

    def run(q_value, k_value, v_value, func, dout):
        q = q_value.detach().clone().requires_grad_(True)
        k = k_value.detach().clone().requires_grad_(True)
        v = v_value.detach().clone().requires_grad_(True)
        plan = create_arbitrary_block_sparse_tensors(
            func, q, k, v, pack_gqa=False, build_backward=True
        )
        out, lse = flash_attn_func(
            q,
            k,
            v,
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=False,
            deterministic=True,
            return_lse=True,
        )
        grads = torch.autograd.grad(out, (q, k, v), dout)
        return out, lse, grads

    single = run(
        q_target,
        k_target,
        v_target,
        _make_arbitrary_func((q_len,), (k_len,), hmask=1, pattern="mixed"),
        dout_target,
    )
    prefixed = run(
        torch.cat((q_prefix, q_target), dim=0),
        torch.cat((k_prefix, k_target), dim=0),
        torch.cat((v_prefix, v_target), dim=0),
        _make_arbitrary_func((q_len, q_len), (k_len, k_len), hmask=1, pattern="mixed"),
        torch.cat((dout_prefix, dout_target), dim=0),
    )
    assert torch.equal(single[0][0], prefixed[0][1])
    assert torch.equal(single[1][0], prefixed[1][1])
    for single_grad, prefixed_grad in zip(single[2], prefixed[2]):
        assert torch.equal(single_grad[0], prefixed_grad[1])


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_fixed_backward_k_boundaries():
    """Exercise all-visible and explicitly partial tails at each K boundary."""

    for k_len in (127, 128, 129, 255, 256, 257):
        for tail_kind in ("full", "partial"):
            case = _ForwardCase(
                f"bwd_k{k_len}_{tail_kind}",
                (129,),
                (k_len,),
                1,
                1,
                64,
                64,
                torch.bfloat16,
                False,
                2,
                pattern="full" if tail_kind == "full" else "mixed",
            )
            q, k, v = _make_inputs(case, varlen=False)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            func = (
                _make_arbitrary_func(
                    case.q_lengths, case.k_lengths, hmask=1, pattern="full"
                )
                if tail_kind == "full"
                else _make_boundary_partial_func(q_len=129, k_len=k_len)
            )
            plan = create_arbitrary_block_sparse_tensors(
                func, q, k, v, pack_gqa=False, build_backward=True
            )
            assert plan.bwd_tensors is not None
            partial_count = int(plan.bwd_tensors.mask_block_cnt.sum())
            full_count = int(plan.bwd_tensors.full_block_cnt.sum())
            if tail_kind == "partial":
                assert partial_count > 0
            else:
                # An all-visible logical tail can still need payload because
                # its physical Q/K tile contains out-of-sequence coordinates.
                assert partial_count + full_count > 0
            out, lse = flash_attn_func(
                q,
                k,
                v,
                arbitrary=True,
                block_sparse_tensors=plan,
                pack_gqa=False,
                return_lse=True,
            )
            torch.manual_seed(700 + k_len)
            dout = torch.randn_like(out)
            grads = torch.autograd.grad(out, (q, k, v), dout)
            try:
                _assert_backward_reference(
                    case, q, k, v, out, lse, dout, grads, func, fixed=True
                )
            except AssertionError as error:
                raise AssertionError(
                    f"K boundary {k_len} ({tail_kind}): {error}"
                ) from error


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_fixed_backward_q_boundaries():
    """Exercise Q tails while every Q block consumes a partial K tile payload."""

    for q_len in (127, 128, 129, 255, 256, 257):
        case = _ForwardCase(
            f"bwd_q{q_len}",
            (q_len,),
            (129,),
            1,
            1,
            64,
            64,
            torch.bfloat16,
            False,
            1 if q_len <= 128 else 2,
            pattern="mixed",
        )
        q, k, v = _make_inputs(case, varlen=False)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        func = _make_boundary_partial_func(q_len=q_len, k_len=129)
        plan = create_arbitrary_block_sparse_tensors(
            func, q, k, v, pack_gqa=False, build_backward=True
        )
        assert plan.bwd_tensors is not None
        assert int(plan.bwd_tensors.mask_block_cnt.sum()) > 0
        out, lse = flash_attn_func(
            q,
            k,
            v,
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=False,
            return_lse=True,
        )
        torch.manual_seed(800 + q_len)
        dout = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), dout)
        try:
            _assert_backward_reference(
                case, q, k, v, out, lse, dout, grads, func, fixed=True
            )
        except AssertionError as error:
            raise AssertionError(f"Q boundary {q_len}: {error}") from error


@pytest.mark.skipif(
    USE_FAKE_TENSOR or not IS_SM100_OR_SM103,
    reason="SM100/SM103 runtime backward correctness",
)
def test_sm100_arbitrary_deterministic_gqa_empty_head_tokens():
    """An empty earlier Q head must advance both dK/dV accumulation tokens."""

    case = _ForwardCase(
        "bwd_deterministic_gqa_empty_head",
        (129,),
        (257,),
        4,
        2,
        64,
        64,
        torch.bfloat16,
        False,
        2,
        per_head_mask=True,
    )
    q_base, k_base, v_base = _make_inputs(case, varlen=False)
    func = torch.zeros(
        case.num_q_heads,
        3,
        case.q_lengths[0] + 256,
        dtype=torch.int32,
        device="cuda",
    )
    func[1::2, :, : case.q_lengths[0]] = case.k_lengths[0]
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q_base,
        k_base,
        v_base,
        pack_gqa=False,
        build_backward=True,
    )
    torch.manual_seed(657)
    dout = torch.randn_like(q_base)
    repeated = []
    outputs = []
    lses = []
    for _ in range(3):
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        out, lse = flash_attn_func(
            q,
            k,
            v,
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=False,
            deterministic=True,
            return_lse=True,
        )
        repeated.append(torch.autograd.grad(out, (q, k, v), dout))
        outputs.append(out)
        lses.append(lse)
    for later in repeated[1:]:
        for first_grad, later_grad in zip(repeated[0], later):
            assert torch.equal(first_grad, later_grad)
    _assert_backward_reference(
        case,
        q_base,
        k_base,
        v_base,
        outputs[0],
        lses[0],
        dout,
        repeated[0],
        func,
        fixed=True,
    )


@pytest.mark.skipif(USE_FAKE_TENSOR, reason="requires runtime tensor provenance")
def test_sm100_arbitrary_rejects_stale_varlen_prefixes_before_launch():
    """Equal aggregate geometry cannot make a plan valid for a new partition."""

    case = _VARLEN_CASES[0]
    q, k, v = _make_inputs(case, varlen=True)
    cu_q = _cu_seqlens(case.q_lengths)
    cu_k = _cu_seqlens(case.k_lengths)
    func = _make_arbitrary_func(
        case.q_lengths,
        case.k_lengths,
        hmask=1,
        pattern=case.pattern,
    )
    plan = create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(case.q_lengths),
        max_seqlen_k=max(case.k_lengths),
        pack_gqa=case.pack_gqa,
    )

    replacement_cu_q = _cu_seqlens((18, 1, 31, 7))
    replacement_cu_k = _cu_seqlens((22, 8, 0, 129))
    with pytest.raises(ValueError, match="cu_seqlens_q provenance mismatch"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=replacement_cu_q,
            cu_seqlens_k=replacement_cu_k,
            max_seqlen_q=max(case.q_lengths),
            max_seqlen_k=max(case.k_lengths),
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=case.pack_gqa,
        )

    cu_q[1] -= 1
    with pytest.raises(ValueError, match="cu_seqlens_q was modified in-place"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(case.q_lengths),
            max_seqlen_k=max(case.k_lengths),
            arbitrary=True,
            block_sparse_tensors=plan,
            pack_gqa=case.pack_gqa,
        )
