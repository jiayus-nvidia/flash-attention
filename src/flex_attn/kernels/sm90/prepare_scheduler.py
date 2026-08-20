# Copyright (c) 2026, MagiAttention contributors.

"""SM90 varlen scheduler metadata preparation."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Int32, const_expr


_PREPARE_THREADS = 256
_PREPARE_LOG_THREADS = 8
_INVALID_KEY = -2147483647
_INVALID_BATCH = 2147483647
_L2_BUDGET_BYTES = 8 * 1024 * 1024


class FlexAttentionVarlenPrepareSchedulerSm90:
    """Build deterministic, compact metadata for the SM90 persistent scheduler."""

    def __init__(
        self,
        *,
        tile_m: int,
        tile_n: int,
        head_dim: int,
        head_dim_v: int,
        element_size: int,
        qhead_per_kvhead: int,
        pack_gqa: bool,
        sort_by_remaining: bool,
    ):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        self.sort_by_remaining = sort_by_remaining
        size_one_kv_block = tile_n * (head_dim + head_dim_v) * element_size
        self.max_kv_blocks_in_l2 = max(_L2_BUDGET_BYTES // size_one_kv_block, 1)

    @cute.jit
    def __call__(
        self,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mMetadata: cute.Tensor,
        mTileCounter: cute.Tensor,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        total_q: Int32,
        total_k: Int32,
        num_head: Int32,
        stream: cuda.CUstream,
    ):
        num_batch = mMetadata.shape[1]
        self.kernel(
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mCuTotalMBlocks,
            mMetadata,
            mTileCounter,
            seqlen_q_static,
            seqlen_k_static,
            total_q,
            total_k,
            num_head,
        ).launch(
            grid=(cute.ceil_div(num_batch, _PREPARE_THREADS), 1, 1),
            block=(_PREPARE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _get_seqlen(
        self,
        batch_idx: Int32,
        mCuSeqlens: Optional[cute.Tensor],
        mSeqUsed: Optional[cute.Tensor],
        seqlen_static: Int32,
        total: Int32,
    ) -> Int32:
        seqlen = seqlen_static
        if const_expr(mSeqUsed is not None):
            seqlen = mSeqUsed[batch_idx]
        elif const_expr(mCuSeqlens is not None):
            begin = cutlass.max(Int32(0), cutlass.min(mCuSeqlens[batch_idx], total))
            end = cutlass.max(begin, cutlass.min(mCuSeqlens[batch_idx + 1], total))
            seqlen = end - begin
        return cutlass.max(seqlen, Int32(0))

    @cute.jit
    def _get_nheads_in_l2(self, num_n_blocks: Int32, num_head: Int32) -> Int32:
        nheads = Int32(1)
        if num_n_blocks * Int32(16) <= Int32(self.max_kv_blocks_in_l2):
            nheads = Int32(16)
        elif num_n_blocks * Int32(8) <= Int32(self.max_kv_blocks_in_l2):
            nheads = Int32(8)
        elif num_n_blocks * Int32(4) <= Int32(self.max_kv_blocks_in_l2):
            nheads = Int32(4)
        elif num_n_blocks * Int32(2) <= Int32(self.max_kv_blocks_in_l2):
            nheads = Int32(2)
        if const_expr(not self.pack_gqa):
            nheads *= Int32(self.qhead_per_kvhead)
        return cutlass.min(nheads, num_head)

    @cute.jit
    def _comes_before(
        self,
        lhs_key: Int32,
        lhs_batch: Int32,
        rhs_key: Int32,
        rhs_batch: Int32,
    ):
        return (lhs_key > rhs_key) | ((lhs_key == rhs_key) & (lhs_batch < rhs_batch))

    @cute.kernel
    def kernel(
        self,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mMetadata: cute.Tensor,
        mTileCounter: cute.Tensor,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        total_q: Int32,
        total_k: Int32,
        num_head: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, _, _ = cute.arch.block_idx()
        num_batch = mMetadata.shape[1]
        batch_idx = chunk_idx * Int32(_PREPARE_THREADS) + tidx

        if chunk_idx == Int32(0) and tidx == Int32(0):
            mTileCounter[0] = Int32(0)

        key = Int32(_INVALID_KEY)
        num_m_blocks = Int32(0)
        nheads_in_l2 = Int32(1)
        actual_batch = Int32(_INVALID_BATCH)
        if batch_idx < num_batch:
            q_len = self._get_seqlen(batch_idx, mCuSeqlensQ, mSeqUsedQ, seqlen_q_static, total_q)
            k_len = self._get_seqlen(batch_idx, mCuSeqlensK, mSeqUsedK, seqlen_k_static, total_k)
            if const_expr(mCuTotalMBlocks is not None):
                num_m_blocks = mCuTotalMBlocks[batch_idx + 1] - mCuTotalMBlocks[batch_idx]
            else:
                physical_q_len = q_len
                if const_expr(self.pack_gqa):
                    physical_q_len *= Int32(self.qhead_per_kvhead)
                num_m_blocks = cute.ceil_div(physical_q_len, self.tile_m)
            num_n_blocks = cute.ceil_div(k_len, self.tile_n)
            key = num_n_blocks
            if const_expr(self.sort_by_remaining):
                key = num_n_blocks * Int32(self.tile_n) - num_m_blocks * Int32(self.tile_m)
            nheads_in_l2 = self._get_nheads_in_l2(cutlass.max(num_n_blocks, Int32(1)), num_head)
            actual_batch = batch_idx

        smem = cutlass_utils.SmemAllocator()
        sMeta = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((4, _PREPARE_THREADS), stride=(_PREPARE_THREADS, 1)),
            byte_alignment=16,
        )
        sMeta[0, tidx] = key
        sMeta[1, tidx] = num_m_blocks
        sMeta[2, tidx] = nheads_in_l2
        sMeta[3, tidx] = actual_batch
        cute.arch.sync_threads()

        # Deterministic descending bitonic sort with an ascending batch tie-break.
        for log_size in cutlass.range_constexpr(_PREPARE_LOG_THREADS):
            size = 1 << (log_size + 1)
            for log_stride in cutlass.range_constexpr(log_size + 1):
                stride = 1 << (log_size - log_stride)
                partner = tidx ^ Int32(stride)
                if partner > tidx:
                    lhs_key = sMeta[0, tidx]
                    lhs_batch = sMeta[3, tidx]
                    rhs_key = sMeta[0, partner]
                    rhs_batch = sMeta[3, partner]
                    descending = (tidx & Int32(size)) == Int32(0)
                    swap = self._comes_before(rhs_key, rhs_batch, lhs_key, lhs_batch)
                    if not descending:
                        swap = self._comes_before(lhs_key, lhs_batch, rhs_key, rhs_batch)
                    if swap:
                        for row in cutlass.range_constexpr(4):
                            lhs = sMeta[row, tidx]
                            rhs = sMeta[row, partner]
                            sMeta[row, tidx] = rhs
                            sMeta[row, partner] = lhs
                cute.arch.sync_threads()

        virtual_batch = chunk_idx * Int32(_PREPARE_THREADS) + tidx
        if virtual_batch < num_batch:
            mMetadata[0, virtual_batch] = sMeta[1, tidx]
            mMetadata[1, virtual_batch] = sMeta[2, tidx]
            mMetadata[2, virtual_batch] = sMeta[3, tidx]
