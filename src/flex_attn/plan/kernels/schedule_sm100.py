"""Plan-side work descriptor materialization for SM100 forward CLC scheduling."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

import cuda.bindings.driver as cuda

_L2_SECTION_BYTES = 50 * 1024 * 1024


class Sm100ForwardSchedulePlan:
    """Materialize exact forward work and locality keys once per mask plan."""

    def __init__(
        self,
        *,
        plan_tile_m: int,
        tile_n: int,
        qhead_per_kvhead: int,
        pack_gqa: bool,
        is_varlen: bool,
    ) -> None:
        self.plan_tile_m = plan_tile_m
        self.tile_n = tile_n
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        self.is_varlen = is_varlen

    @cute.jit
    def __call__(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mSequenceDesc: Optional[cute.Tensor],
        mWorkDesc: cute.Tensor,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        batch_size: Int32,
        num_scheduled_heads: Int32,
        num_kv_heads: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        max_m_blocks: Int32,
        head_dim: Int32,
        head_dim_v: Int32,
        element_size: Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mPartialCount,
            mFullCount,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            mSequenceDesc,
            mWorkDesc,
            mTaskCost,
            mSectionId,
            batch_size,
            num_scheduled_heads,
            num_kv_heads,
            seqlen_q_fixed,
            seqlen_k_fixed,
            max_m_blocks,
            head_dim,
            head_dim_v,
            element_size,
        ).launch(
            grid=(cutlass.max(max_m_blocks, Int32(1)), num_scheduled_heads, batch_size),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mSequenceDesc: Optional[cute.Tensor],
        mWorkDesc: cute.Tensor,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        batch_size: Int32,
        num_scheduled_heads: Int32,
        num_kv_heads: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        max_m_blocks: Int32,
        head_dim: Int32,
        head_dim_v: Int32,
        element_size: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        q_offset = batch_idx * seqlen_q_fixed
        k_offset = batch_idx * seqlen_k_fixed
        q_len = seqlen_q_fixed
        k_len = seqlen_k_fixed
        q_plan_row_begin = batch_idx * max_m_blocks
        if const_expr(self.is_varlen):
            assert mCuSeqlensQ is not None
            assert mCuSeqlensK is not None
            assert mCuTotalMBlocks is not None
            assert mSequenceDesc is not None
            q_offset = mCuSeqlensQ[batch_idx]
            k_offset = mCuSeqlensK[batch_idx]
            q_len = mCuSeqlensQ[batch_idx + Int32(1)] - q_offset
            k_len = mCuSeqlensK[batch_idx + Int32(1)] - k_offset
            q_plan_row_begin = mCuTotalMBlocks[batch_idx]

        physical_q_len = q_len
        if const_expr(self.pack_gqa):
            physical_q_len *= Int32(self.qhead_per_kvhead)
        q_plan_row_count = cute.ceil_div(physical_q_len, self.plan_tile_m)
        num_k_blocks = cute.ceil_div(k_len, self.tile_n)

        if const_expr(mSequenceDesc is not None):
            if tidx == Int32(0) and m_block == Int32(0) and head_idx == Int32(0):
                mSequenceDesc[batch_idx, Int32(0)] = q_offset
                mSequenceDesc[batch_idx, Int32(1)] = k_offset
                mSequenceDesc[batch_idx, Int32(2)] = q_len
                mSequenceDesc[batch_idx, Int32(3)] = k_len
                mSequenceDesc[batch_idx, Int32(4)] = q_plan_row_begin
                mSequenceDesc[batch_idx, Int32(5)] = q_plan_row_count
                mSequenceDesc[batch_idx, Int32(6)] = num_k_blocks
                mSequenceDesc[batch_idx, Int32(7)] = Int32(0)

        if tidx == Int32(0) and m_block < q_plan_row_count:
            outer_row = q_plan_row_begin + m_block
            task_idx = outer_row * num_scheduled_heads + head_idx
            q_valid_rows = cutlass.min(
                Int32(self.plan_tile_m),
                physical_q_len - m_block * Int32(self.plan_tile_m),
            )
            plan_head = Int32(0)
            if mPartialCount.shape[0] != 1:
                plan_head = head_idx
            task_cost = (
                mPartialCount[plan_head, outer_row]
                + mFullCount[plan_head, outer_row]
            )
            kv_head_idx = head_idx
            if const_expr(not self.pack_gqa):
                kv_head_idx = head_idx // Int32(self.qhead_per_kvhead)

            kv_head_bytes = (
                Int64(k_len)
                * Int64(head_dim + head_dim_v)
                * Int64(element_size)
            )
            heads_per_section = Int32(1)
            while (
                heads_per_section * Int32(2) <= num_kv_heads
                and kv_head_bytes * Int64(heads_per_section * Int32(2))
                <= Int64(_L2_SECTION_BYTES)
            ):
                heads_per_section *= Int32(2)
            section_id = (
                batch_idx * num_kv_heads
                + kv_head_idx // heads_per_section
            )

            mWorkDesc[task_idx, Int32(0)] = m_block
            mWorkDesc[task_idx, Int32(1)] = head_idx
            mWorkDesc[task_idx, Int32(2)] = batch_idx
            mWorkDesc[task_idx, Int32(3)] = q_valid_rows
            mTaskCost[task_idx] = task_cost
            mSectionId[task_idx] = section_id


__all__ = ["Sm100ForwardSchedulePlan"]
