from __future__ import annotations

import math

import pytest
import torch

from flex_attn import create_mask_plan, flex_attn_func
from tests.datas.mask_func_cases import make_mask_func
from tests.datas.sequence_cases import smoke_cases


pytestmark = pytest.mark.gpu

_FWD_TOPOLOGY_CASE_IDS = {0, 7, 14, 28, 43, 57, 339, 1027}
assert _FWD_TOPOLOGY_CASE_IDS <= {
    case.case_id for case in smoke_cases("fixed")
}


def _selected_rows(seqlen: int) -> tuple[int, ...]:
    candidates = (0, 1, 127, 128, 255, seqlen // 2, seqlen - 2, seqlen - 1)
    return tuple(sorted({index for index in candidates if 0 <= index < seqlen}))


def _assert_default_sm100_forward_topology(plan, case) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        return
    packed_plan, _, _ = plan._runtime_args
    expected_family = (
        "sm100_hd256_fwd"
        if (case.head_dim, case.head_dim_v) == (256, 256)
        else "sm100_qstage1_2cta_fwd"
    )
    assert packed_plan.plan_signature.kernel_family == expected_family


def _visible_columns(endpoints: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    columns = torch.arange(seqlen_k, device=endpoints.device)
    visible = columns < endpoints[0]
    for index in range(1, endpoints.numel(), 2):
        visible |= (columns >= endpoints[index]) & (columns < endpoints[index + 1])
    return visible


def _assert_fa_error(
    actual: torch.Tensor,
    reference: torch.Tensor,
    pytorch: torch.Tensor,
    *,
    name: str = "tensor",
) -> None:
    if actual.numel() == 0:
        return
    actual = actual.detach().float()
    reference = reference.detach().float()
    pytorch = pytorch.detach().float()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(reference))
    finite = torch.isfinite(reference)
    actual = actual[finite]
    reference = reference[finite]
    pytorch = pytorch[finite]
    if actual.numel() == 0:
        return
    absolute_error = (actual - reference).abs()
    max_error_idx = int(absolute_error.argmax().item())
    test_error = absolute_error[max_error_idx].item()
    pytorch_error = (pytorch - reference).abs().max().item()
    arithmetic_atol = 2.0 * (
        reference + 0.3 - 0.3 - reference
    ).abs().max().item()
    error_limit = 2.0 * pytorch_error + arithmetic_atol
    assert test_error <= error_limit, (
        f"{name} max error {test_error:.8g} exceeds {error_limit:.8g} "
        f"(reordered PyTorch error {pytorch_error:.8g}); flat index "
        f"{max_error_idx}, actual {actual[max_error_idx].item():.8g}, "
        f"reference {reference[max_error_idx].item():.8g}"
    )


def _attention_reference_rows(q_rows, k_rows, v_rows, visible, *, reorder_ops=False):
    softmax_scale = 1.0 / math.sqrt(q_rows.shape[-1])
    score = (
        q_rows @ (k_rows * softmax_scale).transpose(0, 1)
        if reorder_ops
        else (q_rows @ k_rows.transpose(0, 1)) * softmax_scale
    )
    has_keys = visible.any(dim=-1)
    score = score.masked_fill(~visible, -torch.inf)
    probability = torch.zeros_like(score)
    probability[has_keys] = torch.softmax(score[has_keys], dim=-1)
    out_rows = probability @ v_rows
    lse_rows = torch.full(
        (q_rows.shape[0],),
        -torch.inf,
        dtype=score.dtype,
        device=score.device,
    )
    lse_rows[has_keys] = torch.logsumexp(score[has_keys], dim=-1)
    return out_rows, lse_rows


def _reference_and_sparse_dout(q, k, v, mask_func, case):
    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    q_pt = q.detach().requires_grad_(True)
    k_pt = k.detach().requires_grad_(True)
    v_pt = v.detach().requires_grad_(True)
    dout = torch.zeros(
        (*q.shape[:-1], case.head_dim_v), dtype=q.dtype, device=q.device
    )
    generator = torch.Generator(device=q.device).manual_seed(case.seed ^ 0x5A5A5A5A)
    sampled_out = []
    sampled_lse = []
    pytorch_out = []
    pytorch_lse = []
    reference_nodes = []
    pytorch_nodes = []
    dout_nodes = []
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    qratio = case.num_q_heads // case.num_kv_heads
    rows = tuple(range(seqlen_q)) if case.full_reference else _selected_rows(seqlen_q)
    for batch_idx in range(case.batch_size):
        flat_rows = torch.tensor(
            [batch_idx * seqlen_q + row for row in rows], device=q.device
        )
        for head_idx in range(case.num_q_heads):
            mask_head = 0 if case.hmask == 1 else head_idx
            endpoints = mask_func[mask_head, :, flat_rows].transpose(0, 1)
            kv_head = head_idx // qratio
            visible = torch.stack(
                [_visible_columns(row_endpoints, seqlen_k) for row_endpoints in endpoints]
            )
            out_rows, lse_rows = _attention_reference_rows(
                q_ref[batch_idx, rows, head_idx],
                k_ref[batch_idx, :, kv_head],
                v_ref[batch_idx, :, kv_head],
                visible,
            )
            out_rows_pt, lse_rows_pt = _attention_reference_rows(
                q_pt[batch_idx, rows, head_idx],
                k_pt[batch_idx, :, kv_head],
                v_pt[batch_idx, :, kv_head],
                visible,
                reorder_ops=True,
            )
            dout_rows = torch.randn(
                (len(rows), case.head_dim_v),
                dtype=q.dtype,
                device=q.device,
                generator=generator,
            )
            dout[batch_idx, rows, head_idx] = dout_rows
            sampled_out.append(out_rows.detach())
            sampled_lse.append(lse_rows.detach())
            pytorch_out.append(out_rows_pt.detach())
            pytorch_lse.append(lse_rows_pt.detach())
            reference_nodes.append(out_rows)
            pytorch_nodes.append(out_rows_pt)
            dout_nodes.append(dout_rows)
    gradients = torch.autograd.grad(
        tuple(reference_nodes),
        (q_ref, k_ref, v_ref),
        tuple(node.float() for node in dout_nodes),
    )
    pytorch_gradients = torch.autograd.grad(
        tuple(pytorch_nodes),
        (q_pt, k_pt, v_pt),
        tuple(dout_nodes),
    )
    return (
        dout,
        torch.cat(sampled_out),
        torch.cat(sampled_lse),
        gradients,
        torch.cat(pytorch_out),
        torch.cat(pytorch_lse),
        pytorch_gradients,
        rows,
    )


def test_flex_attn(case):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[case.dtype]
    torch.manual_seed(case.seed)
    seqlen_q, seqlen_k = case.q_lengths[0], case.k_lengths[0]
    q = torch.randn(
        case.batch_size,
        seqlen_q,
        case.num_q_heads,
        case.head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        case.batch_size,
        seqlen_k,
        case.num_kv_heads,
        case.head_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn(
        case.batch_size,
        seqlen_k,
        case.num_kv_heads,
        case.head_dim_v,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    mask_func = make_mask_func(case)
    plan = create_mask_plan(mask_func, q, k, v)
    _assert_default_sm100_forward_topology(plan, case)
    run_fwd_topology_variants = (
        torch.cuda.get_device_capability()[0] == 10
        and case.case_id in _FWD_TOPOLOGY_CASE_IDS
    )
    qstage1_2cta_plan = (
        create_mask_plan(
            mask_func,
            q,
            k,
            v,
            build_backward=False,
            _fwd_variant="qstage1_2cta",
        )
        if run_fwd_topology_variants
        else None
    )
    qstage2_1cta_plan = (
        create_mask_plan(
            mask_func,
            q,
            k,
            v,
            build_backward=False,
            _fwd_variant="qstage2_1cta",
        )
        if run_fwd_topology_variants and case.head_dim != 256
        else None
    )
    qstage1_1cta_plan = (
        create_mask_plan(
            mask_func,
            q,
            k,
            v,
            build_backward=False,
            _fwd_variant="qstage1_1cta",
        )
        if run_fwd_topology_variants
        else None
    )
    (
        dout,
        ref_out,
        ref_lse,
        ref_grads,
        pytorch_out,
        pytorch_lse,
        pytorch_grads,
        rows,
    ) = _reference_and_sparse_dout(q, k, v, mask_func, case)
    out, lse = flex_attn_func(
        q,
        k,
        v,
        mask_plan=plan,
        deterministic=case.deterministic,
        return_lse=True,
    )
    sampled_out = torch.cat(
        [out[batch_idx, rows].transpose(0, 1).reshape(-1, case.head_dim_v) for batch_idx in range(case.batch_size)]
    )
    sampled_lse = torch.cat(
        [lse[batch_idx, :, rows].reshape(-1) for batch_idx in range(case.batch_size)]
    )
    _assert_fa_error(sampled_out, ref_out, pytorch_out)
    _assert_fa_error(sampled_lse, ref_lse, pytorch_lse)
    for candidate_plan in (
        qstage1_1cta_plan,
        qstage1_2cta_plan,
        qstage2_1cta_plan,
    ):
        if candidate_plan is None:
            continue
        with torch.no_grad():
            candidate_out, candidate_lse = flex_attn_func(
                q,
                k,
                v,
                mask_plan=candidate_plan,
                return_lse=True,
            )
        candidate_sampled_out = torch.cat(
            [
                candidate_out[batch_idx, rows]
                .transpose(0, 1)
                .reshape(-1, case.head_dim_v)
                for batch_idx in range(case.batch_size)
            ]
        )
        candidate_sampled_lse = torch.cat(
            [
                candidate_lse[batch_idx, :, rows].reshape(-1)
                for batch_idx in range(case.batch_size)
            ]
        )
        _assert_fa_error(candidate_sampled_out, ref_out, pytorch_out)
        _assert_fa_error(candidate_sampled_lse, ref_lse, pytorch_lse)
    if run_fwd_topology_variants and case.case_id == 0:
        empty_mask_func = torch.zeros(
            (1, 1, case.batch_size * seqlen_q),
            dtype=torch.int32,
            device=q.device,
        )
        empty_plan = create_mask_plan(
            empty_mask_func,
            q,
            k,
            v,
            build_backward=False,
            _fwd_variant="qstage1_1cta",
        )
        with torch.no_grad():
            empty_out, empty_lse = flex_attn_func(
                q,
                k,
                v,
                mask_plan=empty_plan,
                return_lse=True,
            )
        assert torch.count_nonzero(empty_out).item() == 0
        assert torch.isneginf(empty_lse).all().item()
    out.backward(dout)
    for name, actual, reference, pytorch in zip(
        ("dQ", "dK", "dV"),
        (q.grad, k.grad, v.grad),
        ref_grads,
        pytorch_grads,
    ):
        _assert_fa_error(actual, reference, pytorch, name=name)
