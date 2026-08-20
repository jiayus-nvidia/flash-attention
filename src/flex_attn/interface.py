"""Public FlexAttention functions."""

from __future__ import annotations

import torch

from flex_attn.autograd import FlexAttnFunc, FlexAttnVarlenFunc
from flex_attn.plan.mask_plan import MaskPlan
from flex_attn.plan.validation import validate_call_options


def _validate_plan(mask_plan: MaskPlan) -> None:
    if not isinstance(mask_plan, MaskPlan):
        raise TypeError("mask_plan must be returned by create_mask_plan")


def flex_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask_plan: MaskPlan,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """Run fixed-length BSHD attention with a reusable packed mask plan."""

    _validate_plan(mask_plan)
    validate_call_options(
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        return_lse=return_lse,
    )
    mask_plan._validate_runtime(q, k, v, mode="fixed")
    result = FlexAttnFunc.apply(
        q,
        k,
        v,
        mask_plan,
        softmax_scale,
        deterministic,
        return_lse,
    )
    return result if return_lse else result[0]


def flex_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask_plan: MaskPlan,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """Run true-varlen THD attention using geometry owned by ``mask_plan``."""

    _validate_plan(mask_plan)
    validate_call_options(
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        return_lse=return_lse,
    )
    mask_plan._validate_runtime(q, k, v, mode="varlen")
    result = FlexAttnVarlenFunc.apply(
        q,
        k,
        v,
        mask_plan,
        softmax_scale,
        deterministic,
        return_lse,
    )
    return result if return_lse else result[0]


__all__ = ["flex_attn_func", "flex_attn_varlen_func"]
