"""Eager custom autograd bindings for packed-mask attention."""

from __future__ import annotations

import torch

from flex_attn.dispatch import _flex_attn_bwd, _flex_attn_fwd
from flex_attn.plan.mask_plan import MaskPlan


class FlexAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_plan: MaskPlan,
        softmax_scale: float | None,
        deterministic: bool,
        return_lse: bool,
        return_max_logit: bool,
    ):
        packed_plan, _, _ = mask_plan._runtime_args
        out, lse, max_logit = _flex_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            pack_gqa=mask_plan.metadata.pack_gqa,
            block_sparse_tensors=packed_plan,
            return_lse=return_lse,
            return_max_logit=return_max_logit,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.mask_plan = mask_plan
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        if max_logit is not None:
            ctx.mark_non_differentiable(max_logit)
        return out, lse, max_logit

    @staticmethod
    def backward(ctx, dout, dlse, _dmax_logit):
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        packed_plan, _, _ = ctx.mask_plan._runtime_args
        dq, dk, dv = _flex_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
            block_sparse_tensors=packed_plan,
            dlse=dlse if ctx.return_lse else None,
        )
        return dq, dk, dv, None, None, None, None, None


class FlexAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_plan: MaskPlan,
        softmax_scale: float | None,
        deterministic: bool,
        return_lse: bool,
        return_max_logit: bool,
    ):
        packed_plan, cu_seqlens_q, cu_seqlens_k = mask_plan._runtime_args
        metadata = mask_plan.metadata
        out, lse, max_logit = _flex_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            softmax_scale=softmax_scale,
            pack_gqa=metadata.pack_gqa,
            block_sparse_tensors=packed_plan,
            return_lse=return_lse,
            return_max_logit=return_max_logit,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.mask_plan = mask_plan
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        if max_logit is not None:
            ctx.mark_non_differentiable(max_logit)
        return out, lse, max_logit

    @staticmethod
    def backward(ctx, dout, dlse, _dmax_logit):
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        packed_plan, cu_seqlens_q, cu_seqlens_k = ctx.mask_plan._runtime_args
        metadata = ctx.mask_plan.metadata
        dq, dk, dv = _flex_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=ctx.softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_k=metadata.max_seqlen_k,
            deterministic=ctx.deterministic,
            block_sparse_tensors=packed_plan,
            dlse=dlse if ctx.return_lse else None,
        )
        return dq, dk, dv, None, None, None, None, None


__all__ = ["FlexAttnFunc", "FlexAttnVarlenFunc"]
