"""Compile arbitrary interval masks into compact SM90 forward/backward plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Boolean, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flash_attn_cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn_cute.cache_utils import get_jit_cache
from flash_attn_cute.cute_dsl_utils import to_cute_tensor
from flash_attn_cute.sm90_fwd_config import (
    _ResolvedSm90FwdConsumerConfig,
    make_sm90_fwd_tiled_mma_qk,
    resolve_sm90_fwd_consumer_config,
)
from flash_attn_cute.sm90_bwd_config import (
    _ResolvedSm90BwdConsumerConfig,
    make_sm90_bwd_tiled_mma_sdp,
    resolve_sm90_bwd_consumer_config,
)
from flash_attn_cute.testing import is_fake_mode

_PLAN_THREADS = 256
_ERROR_INVALID_INTERVAL = 1
_ERROR_INVALID_SEQLENS = 2
_DQ_ORDER_COMPONENT_BITS = 16
_DQ_ORDER_COMPONENT_LIMIT = 1 << _DQ_ORDER_COMPONENT_BITS

_CLASSIFY_COMPILE_CACHE = get_jit_cache("arbitrary_plan_classify")
_MATERIALIZE_COMPILE_CACHE = get_jit_cache("arbitrary_plan_materialize")
_K2Q_COUNT_COMPILE_CACHE = get_jit_cache("arbitrary_plan_k2q_count")
_K2Q_MATERIALIZE_COMPILE_CACHE = get_jit_cache("arbitrary_plan_k2q_materialize")


@dataclass(frozen=True)
class _ResolvedSm90BwdTopologyConfig:
    """Adapt the backward sparse Q tile to the shared topology classifier."""

    consumer: _ResolvedSm90BwdConsumerConfig

    @property
    def dtype(self) -> torch.dtype:
        return self.consumer.dtype

    @property
    def tile_m(self) -> int:
        return self.consumer.sparse_tile_m

    @property
    def tile_n(self) -> int:
        return self.consumer.tile_n

    @property
    def pack_gqa(self) -> bool:
        return False

    @property
    def qhead_per_kvhead(self) -> int:
        return self.consumer.qhead_per_kvhead

    @property
    def num_mma_threads(self) -> int:
        return self.consumer.num_mma_threads

    @property
    def payload_values_per_thread(self) -> int:
        return self.consumer.payload_values_per_thread

    @property
    def payload_valid_words(self) -> int:
        return self.consumer.payload_valid_words

    @property
    def payload_padded_words(self) -> int:
        return self.consumer.payload_padded_words

    @property
    def is_varlen(self) -> bool:
        return self.consumer.is_varlen

    @property
    def topology_planner_compile_key(self) -> tuple:
        return (
            self.consumer.arch,
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            False,
            1,
        )


@dsl_user_op
def _shr_u32(val: Uint32, shift: Uint32, *, loc=None, ip=None) -> Uint32:
    """Perform a defined PTX unsigned shift, including a shift by 32."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [val.ir_value(loc=loc, ip=ip), shift.ir_value(loc=loc, ip=ip)],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _load_endpoint(
    mArbitraryFunc: cute.Tensor,
    mask_head: Int32,
    endpoint_idx: Int32,
    q_global: Int32,
) -> Int32:
    return mArbitraryFunc[(mask_head, endpoint_idx, q_global)]


class _ArbitraryPlanCommonSm90:
    def __init__(
        self,
        config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm90BwdTopologyConfig,
    ):
        self.dtype = cutlass.BFloat16 if config.dtype == torch.bfloat16 else cutlass.Float16
        self.tile_m = config.tile_m
        self.tile_n = config.tile_n
        self.pack_gqa = config.pack_gqa
        self.qhead_per_kvhead = config.qhead_per_kvhead
        self.num_mma_threads = config.num_mma_threads
        self.payload_values_per_thread = config.payload_values_per_thread
        self.payload_valid_words = config.payload_valid_words
        self.payload_padded_words = config.payload_padded_words
        self.is_varlen = config.is_varlen

    @cute.jit
    def _sample_info(
        self,
        upper_outer_row: Int32,
        max_m_blocks: Int32,
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
    ):
        batch_idx = upper_outer_row // max_m_blocks
        local_m_block = upper_outer_row - batch_idx * max_m_blocks
        q_begin = batch_idx * seqlen_q_fixed
        q_end = q_begin + seqlen_q_fixed
        k_begin = batch_idx * seqlen_k_fixed
        k_end = k_begin + seqlen_k_fixed
        compact_outer_row = upper_outer_row
        if const_expr(self.is_varlen):
            q_begin = mCuSeqlensQ[batch_idx]
            q_end = mCuSeqlensQ[batch_idx + 1]
            k_begin = mCuSeqlensK[batch_idx]
            k_end = mCuSeqlensK[batch_idx + 1]
            q_begin = cutlass.max(Int32(0), cutlass.min(q_begin, total_q))
            q_end = cutlass.max(q_begin, cutlass.min(q_end, total_q))
            k_begin = cutlass.max(Int32(0), cutlass.min(k_begin, total_k))
            k_end = cutlass.max(k_begin, cutlass.min(k_end, total_k))
            compact_outer_row = mCuTotalMBlocks[batch_idx] + local_m_block
        q_len = q_end - q_begin
        k_len = k_end - k_begin
        physical_q_len = (
            q_len * Int32(self.qhead_per_kvhead) if const_expr(self.pack_gqa) else q_len
        )
        num_m_blocks = cute.ceil_div(physical_q_len, self.tile_m)
        valid_m_block = (batch_idx < batch_size) & (local_m_block < num_m_blocks)
        return (
            batch_idx,
            local_m_block,
            compact_outer_row,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_m_block,
        )

    @cute.jit
    def _physical_q_info(
        self,
        local_m_block: Int32,
        row_in_tile: Int32,
        q_begin: Int32,
        q_len: Int32,
    ):
        physical_q = local_m_block * Int32(self.tile_m) + row_in_tile
        if const_expr(self.pack_gqa):
            q_local = physical_q // Int32(self.qhead_per_kvhead)
        else:
            q_local = physical_q
        q_valid = q_local < q_len
        return q_begin + q_local, q_local, q_valid

    @cute.jit
    def _safe_interval(
        self,
        mArbitraryFunc: cute.Tensor,
        mask_head: Int32,
        interval_idx: Int32,
        q_global: Int32,
        k_begin: Int32,
        k_len: Int32,
        total_k: Int32,
    ):
        global_begin = Int32(0)
        if interval_idx > Int32(0):
            global_begin = _load_endpoint(
                mArbitraryFunc, mask_head, interval_idx * Int32(2) - Int32(1), q_global
            )
        global_end = _load_endpoint(mArbitraryFunc, mask_head, interval_idx * Int32(2), q_global)
        safe_begin = cutlass.max(Int32(0), cutlass.min(global_begin, total_k))
        safe_end = cutlass.max(Int32(0), cutlass.min(global_end, total_k))
        if safe_end < safe_begin:
            safe_end = safe_begin
        k_end = k_begin + k_len
        local_begin = cutlass.max(safe_begin, k_begin) - k_begin
        local_end = cutlass.min(safe_end, k_end) - k_begin
        if local_end < local_begin:
            local_end = local_begin
        return global_begin, global_end, local_begin, local_end

    @cute.jit
    def _row_block_state(
        self,
        mArbitraryFunc: cute.Tensor,
        mask_head: Int32,
        q_global: Int32,
        block_id: Int32,
        nfunc: Int32,
        k_begin: Int32,
        k_len: Int32,
        total_k: Int32,
    ):
        block_begin = block_id * Int32(self.tile_n)
        block_end = block_begin + Int32(self.tile_n)
        covered_end = block_begin
        visible = Boolean(False)
        num_intervals = (nfunc + Int32(1)) // Int32(2)
        for interval_idx in cutlass.range(num_intervals, unroll=1):
            _, _, local_begin, local_end = self._safe_interval(
                mArbitraryFunc,
                mask_head,
                interval_idx,
                q_global,
                k_begin,
                k_len,
                total_k,
            )
            lo = cutlass.max(local_begin, block_begin)
            hi = cutlass.min(local_end, block_end)
            if hi > lo:
                visible = Boolean(True)
                if lo <= covered_end and hi > covered_end:
                    covered_end = hi
        return visible, covered_end >= block_end

    @cute.jit
    def _is_visible(
        self,
        mArbitraryFunc: cute.Tensor,
        mask_head: Int32,
        q_global: Int32,
        k_global: Int32,
        nfunc: Int32,
    ) -> Boolean:
        keep = Boolean(False)
        num_intervals = (nfunc + Int32(1)) // Int32(2)
        for interval_idx in cutlass.range(num_intervals, unroll=1):
            global_begin = Int32(0)
            if interval_idx > Int32(0):
                global_begin = _load_endpoint(
                    mArbitraryFunc,
                    mask_head,
                    interval_idx * Int32(2) - Int32(1),
                    q_global,
                )
            global_end = _load_endpoint(
                mArbitraryFunc, mask_head, interval_idx * Int32(2), q_global
            )
            if k_global >= global_begin and k_global < global_end:
                keep = Boolean(True)
        return keep


class _ArbitraryPlanClassifySm90(_ArbitraryPlanCommonSm90):
    """Validate intervals and classify candidate QK tiles."""

    @cute.jit
    def __call__(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mError: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
        stream: cuda.CUstream = None,
    ):
        upper_total_m_blocks = mPartialCounts.shape[1]
        hmask = mArbitraryFunc.shape[0]
        self.kernel(
            mArbitraryFunc,
            mVisibleBits,
            mFullBits,
            mPartialCounts,
            mFullCounts,
            mError,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_m_blocks,
            max_n_blocks,
            nfunc,
        ).launch(
            grid=(upper_total_m_blocks, hmask, 1),
            block=(_PLAN_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _mark_error(self, mError: cute.Tensor, value: Uint32) -> None:
        cute.arch.atomic_or(
            mError.iterator.llvm_ptr,
            value,
            sem="relaxed",
            scope="gpu",
        )

    @cute.jit
    def _set_candidate_range(
        self,
        mVisibleBits: cute.Tensor,
        mask_head: Int32,
        compact_outer_row: Int32,
        block_begin: Int32,
        block_end: Int32,
    ) -> None:
        first_word = block_begin // Int32(32)
        last_word = (block_end - Int32(1)) // Int32(32)
        word_idx = first_word
        while word_idx <= last_word:
            lo = Int32(0)
            if word_idx == first_word:
                lo = block_begin - first_word * Int32(32)
            hi = Int32(32)
            if word_idx == last_word:
                hi = block_end - last_word * Int32(32)
            upper = _shr_u32(Uint32(0xFFFF_FFFF), Uint32(Int32(32) - hi))
            lower = _shr_u32(Uint32(0xFFFF_FFFF), Uint32(Int32(32) - lo))
            offset = cute.crd2idx((mask_head, compact_outer_row, word_idx), mVisibleBits.layout)
            cute.arch.atomic_or(
                (mVisibleBits.iterator + offset).llvm_ptr,
                upper ^ lower,
                sem="relaxed",
                scope="gpu",
            )
            word_idx += Int32(1)

    @cute.kernel
    def kernel(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mError: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        upper_outer_row, mask_head, _ = cute.arch.block_idx()
        (
            _,
            local_m_block,
            compact_outer_row,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_m_block,
        ) = self._sample_info(
            upper_outer_row,
            max_m_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
        )
        q_global, _, q_valid = self._physical_q_info(local_m_block, tidx, q_begin, q_len)
        q_valid = q_valid & valid_m_block & (tidx < Int32(self.tile_m))

        if q_valid:
            previous_begin = Int32(-1)
            num_intervals = (nfunc + Int32(1)) // Int32(2)
            for interval_idx in cutlass.range(num_intervals, unroll=1):
                global_begin, global_end, local_begin, local_end = self._safe_interval(
                    mArbitraryFunc,
                    mask_head,
                    interval_idx,
                    q_global,
                    k_begin,
                    k_len,
                    total_k,
                )
                invalid = (
                    (global_begin < Int32(0))
                    | (global_end < Int32(0))
                    | (global_begin > total_k)
                    | (global_end > total_k)
                    | (global_end < global_begin)
                )
                if global_end > global_begin and global_begin < previous_begin:
                    invalid = Boolean(True)
                if invalid:
                    self._mark_error(mError, Uint32(_ERROR_INVALID_INTERVAL))
                if global_end > global_begin:
                    previous_begin = global_begin
                if local_end > local_begin:
                    block_begin = local_begin // Int32(self.tile_n)
                    block_end = cute.ceil_div(local_end, self.tile_n)
                    self._set_candidate_range(
                        mVisibleBits,
                        mask_head,
                        compact_outer_row,
                        block_begin,
                        block_end,
                    )

        smem = cutlass_utils.SmemAllocator()
        sWarpVisible = smem.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout((8,)),
            byte_alignment=16,
        )
        sWarpFull = smem.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout((8,)),
            byte_alignment=16,
        )
        cute.arch.sync_threads()

        partial_count = Int32(0)
        full_count = Int32(0)
        anchor_full_word = Int32(0)
        anchor_full_mask = Uint32(0)
        word_idx = Int32(0)
        num_words = cute.ceil_div(max_n_blocks, 32)
        while word_idx < num_words:
            candidate_word = Uint32(0)
            if valid_m_block:
                candidate_word = mVisibleBits[mask_head, compact_outer_row, word_idx]
            for bit_idx in cutlass.range_constexpr(32):
                block_id = word_idx * Int32(32) + Int32(bit_idx)
                candidate = (block_id < max_n_blocks) & (
                    (candidate_word & Uint32(1 << bit_idx)) != Uint32(0)
                )
                if candidate:
                    row_visible = Boolean(False)
                    row_full = Boolean(True)
                    if q_valid:
                        row_visible, row_full = self._row_block_state(
                            mArbitraryFunc,
                            mask_head,
                            q_global,
                            block_id,
                            nfunc,
                            k_begin,
                            k_len,
                            total_k,
                        )
                    warp_idx = cute.arch.warp_idx()
                    lane_idx = cute.arch.lane_idx()
                    warp_visible = cute.arch.vote_ballot_sync(row_visible)
                    warp_full = cute.arch.vote_ballot_sync(row_full)
                    if lane_idx == Int32(0):
                        sWarpVisible[warp_idx] = Uint32(warp_visible)
                        sWarpFull[warp_idx] = Uint32(warp_full)
                    cute.arch.sync_threads()
                    if tidx == Int32(0):
                        any_visible = Uint32(0)
                        all_full = Uint32(0xFFFF_FFFF)
                        for warp in cutlass.range_constexpr(8):
                            any_visible |= sWarpVisible[warp]
                            all_full &= sWarpFull[warp]
                        is_full = all_full == Uint32(0xFFFF_FFFF)
                        if (block_id + Int32(1)) * Int32(self.tile_n) > k_len:
                            is_full = Boolean(False)
                        if any_visible != Uint32(0):
                            if is_full:
                                mFullBits[mask_head, compact_outer_row, word_idx] |= Uint32(
                                    1 << bit_idx
                                )
                                full_count += Int32(1)
                                anchor_full_word = word_idx
                                anchor_full_mask = Uint32(1 << bit_idx)
                            else:
                                partial_count += Int32(1)
                    cute.arch.sync_threads()
            word_idx += Int32(1)

        if tidx == Int32(0) and valid_m_block:
            if partial_count == Int32(0) and full_count > Int32(0):
                mFullBits[mask_head, compact_outer_row, anchor_full_word] &= (
                    Uint32(0xFFFF_FFFF) ^ anchor_full_mask
                )
                partial_count += Int32(1)
                full_count -= Int32(1)
            mPartialCounts[mask_head, compact_outer_row] = partial_count
            mFullCounts[mask_head, compact_outer_row] = full_count


class _ArbitraryPlanMaterializeSm90(_ArbitraryPlanCommonSm90):
    """Materialize stable CSR lists and MMA-thread-native payloads."""

    @cute.jit
    def __call__(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
        stream: cuda.CUstream = None,
    ):
        upper_total_m_blocks = mVisibleBits.shape[1]
        hmask = mArbitraryFunc.shape[0]
        tiled_mma_qk = make_sm90_fwd_tiled_mma_qk(
            self.dtype,
            self.tile_m,
            self.tile_n,
        )
        self.kernel(
            tiled_mma_qk,
            mArbitraryFunc,
            mVisibleBits,
            mFullBits,
            mPartialOffsets,
            mPartialIndices,
            mPartialMasks,
            mFullOffsets,
            mFullIndices,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_m_blocks,
            max_n_blocks,
            nfunc,
        ).launch(
            grid=(upper_total_m_blocks, hmask, 1),
            block=(_PLAN_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _store_payload(
        self,
        tiled_mma_qk: cute.TiledMma,
        mArbitraryFunc: cute.Tensor,
        mPartialMasks: cute.Tensor,
        payload_idx: Int32,
        planner_tidx: Int32,
        mask_head: Int32,
        local_m_block: Int32,
        block_id: Int32,
        q_begin: Int32,
        q_len: Int32,
        k_begin: Int32,
        k_len: Int32,
        nfunc: Int32,
    ) -> None:
        consumer_tidx = planner_tidx
        while consumer_tidx < Int32(self.num_mma_threads):
            thr_mma_qk = tiled_mma_qk.get_slice(consumer_tidx)
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)
            rMask = cute.make_rmem_tensor((self.payload_padded_words,), Uint32)
            rMask.fill(Uint32(0))
            for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                packed = Uint32(0)
                for bit_idx in cutlass.range(32, unroll=1):
                    value_idx = word_idx * 32 + bit_idx
                    keep = Boolean(False)
                    if value_idx < self.payload_values_per_thread:
                        coord = tScS[value_idx]
                        row_in_tile = Int32(coord[0])
                        col_in_tile = Int32(coord[1])
                        q_global, _, q_valid = self._physical_q_info(
                            local_m_block, row_in_tile, q_begin, q_len
                        )
                        k_local = block_id * Int32(self.tile_n) + col_in_tile
                        k_valid = k_local < k_len
                        if q_valid and k_valid:
                            keep = self._is_visible(
                                mArbitraryFunc,
                                mask_head,
                                q_global,
                                k_begin + k_local,
                                nfunc,
                            )
                    if keep:
                        packed |= Uint32(1) << Uint32(bit_idx)
                rMask[word_idx] = packed
            mask_iter = mPartialMasks.iterator + cute.crd2idx(
                (payload_idx, Int32(0), consumer_tidx, Int32(0)),
                mPartialMasks.layout,
            )
            mask_ptr = cute.make_ptr(
                Uint32,
                mask_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=min(
                    16,
                    4 * (self.payload_padded_words & -self.payload_padded_words),
                ),
            )
            gMask = cute.make_tensor(mask_ptr, (self.payload_padded_words,))
            cute.autovec_copy(rMask, gMask)
            consumer_tidx += Int32(_PLAN_THREADS)

    @cute.kernel
    def kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
    ):
        planner_tidx, _, _ = cute.arch.thread_idx()
        upper_outer_row, mask_head, _ = cute.arch.block_idx()
        (
            _,
            local_m_block,
            compact_outer_row,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_m_block,
        ) = self._sample_info(
            upper_outer_row,
            max_m_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
        )
        if valid_m_block:
            total_m_blocks = mPartialOffsets.shape[0] - Int32(1)
            total_m_blocks = total_m_blocks // mArbitraryFunc.shape[0]
            plan_row = mask_head * total_m_blocks + compact_outer_row
            partial_ordinal = Int32(0)
            full_ordinal = Int32(0)
            num_words = cute.ceil_div(max_n_blocks, 32)
            for word_idx in cutlass.range(num_words, unroll=1):
                visible_word = mVisibleBits[mask_head, compact_outer_row, word_idx]
                full_word = mFullBits[mask_head, compact_outer_row, word_idx]
                for bit_idx in cutlass.range_constexpr(32):
                    block_id = word_idx * Int32(32) + Int32(bit_idx)
                    visible = (block_id < max_n_blocks) & (
                        (visible_word & Uint32(1 << bit_idx)) != Uint32(0)
                    )
                    full = (full_word & Uint32(1 << bit_idx)) != Uint32(0)
                    if visible:
                        if full:
                            if planner_tidx == Int32(0):
                                mFullIndices[mFullOffsets[plan_row] + full_ordinal] = block_id
                            full_ordinal += Int32(1)
                        else:
                            payload_idx = mPartialOffsets[plan_row] + partial_ordinal
                            if planner_tidx == Int32(0):
                                mPartialIndices[payload_idx] = block_id
                            self._store_payload(
                                tiled_mma_qk,
                                mArbitraryFunc,
                                mPartialMasks,
                                payload_idx,
                                planner_tidx,
                                mask_head,
                                local_m_block,
                                block_id,
                                q_begin,
                                q_len,
                                k_begin,
                                k_len,
                                nfunc,
                            )
                            partial_ordinal += Int32(1)


class _ArbitraryPlanK2QCommonSm90(_ArbitraryPlanCommonSm90):
    def __init__(self, config: _ResolvedSm90BwdConsumerConfig):
        super().__init__(_ResolvedSm90BwdTopologyConfig(config))
        self.consumer_tile_m = config.tile_m
        self.subtile_factor = config.subtile_factor
        self.sdp_swap_ab = config.sdp_swap_ab
        self.atom_layout_m_sdp = config.atom_layout_m_sdp
        self.num_wg_mma = config.num_wg
        self.spt = config.spt

    @cute.jit
    def _sample_info_k(
        self,
        upper_n_row: Int32,
        max_n_blocks: Int32,
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
    ):
        batch_idx = upper_n_row // max_n_blocks
        local_n_block = upper_n_row - batch_idx * max_n_blocks
        q_begin = batch_idx * seqlen_q_fixed
        q_end = q_begin + seqlen_q_fixed
        k_begin = batch_idx * seqlen_k_fixed
        k_end = k_begin + seqlen_k_fixed
        compact_q_begin = batch_idx * cute.ceil_div(seqlen_q_fixed, self.tile_m)
        compact_n_row = upper_n_row
        if const_expr(self.is_varlen):
            q_begin = cutlass.max(Int32(0), cutlass.min(mCuSeqlensQ[batch_idx], total_q))
            q_end = cutlass.max(q_begin, cutlass.min(mCuSeqlensQ[batch_idx + 1], total_q))
            k_begin = cutlass.max(Int32(0), cutlass.min(mCuSeqlensK[batch_idx], total_k))
            k_end = cutlass.max(k_begin, cutlass.min(mCuSeqlensK[batch_idx + 1], total_k))
            compact_q_begin = mCuTotalQBlocks[batch_idx]
            compact_n_row = mCuTotalKBlocks[batch_idx] + local_n_block
        q_len = q_end - q_begin
        k_len = k_end - k_begin
        num_q_blocks = cute.ceil_div(q_len, self.tile_m)
        num_k_blocks = cute.ceil_div(k_len, self.tile_n)
        valid_n_block = (batch_idx < batch_size) & (local_n_block < num_k_blocks)
        return (
            batch_idx,
            local_n_block,
            compact_n_row,
            compact_q_begin,
            num_q_blocks,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_n_block,
        )

    @cute.jit
    def _dq_write_rank(
        self,
        mVisibleBits: cute.Tensor,
        mask_head: Int32,
        compact_q_row: Int32,
        local_n_block: Int32,
        contributor_count: Int32,
    ) -> Int32:
        rank = Int32(0)
        word_limit = local_n_block // Int32(32)
        word_idx = Int32(0)
        while word_idx < word_limit:
            rank += Int32(cute.arch.popc(mVisibleBits[mask_head, compact_q_row, word_idx]))
            word_idx += Int32(1)
        bit_idx = local_n_block - word_limit * Int32(32)
        low_bits = _shr_u32(Uint32(0xFFFF_FFFF), Uint32(Int32(32) - bit_idx))
        rank += Int32(cute.arch.popc(mVisibleBits[mask_head, compact_q_row, word_limit] & low_bits))
        if const_expr(self.spt):
            rank = contributor_count - Int32(1) - rank
        return rank


class _ArbitraryPlanK2QCountSm90(_ArbitraryPlanK2QCommonSm90):
    """Count K-major partial/full Q lists from Q-major packed topology."""

    @cute.jit
    def __call__(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
        stream: cuda.CUstream = None,
    ):
        upper_total_n_blocks = mPartialCounts.shape[1]
        hmask = mPartialCounts.shape[0]
        self.kernel(
            mVisibleBits,
            mFullBits,
            mPartialCounts,
            mFullCounts,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_n_blocks,
        ).launch(
            grid=(upper_total_n_blocks, hmask, 1),
            block=(_PLAN_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        upper_n_row, mask_head, _ = cute.arch.block_idx()
        (
            _,
            local_n_block,
            compact_n_row,
            compact_q_begin,
            num_q_blocks,
            _,
            _,
            _,
            _,
            valid_n_block,
        ) = self._sample_info_k(
            upper_n_row,
            max_n_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
        )

        if tidx < Int32(32):
            lane_idx = cute.arch.lane_idx()
            partial_count = Int32(0)
            full_count = Int32(0)
            q_group = Int32(0)
            word_idx = local_n_block // Int32(32)
            bit_idx = local_n_block - word_idx * Int32(32)
            bit = Uint32(1) << Uint32(bit_idx)
            while q_group * Int32(32) < num_q_blocks:
                local_q_block = q_group * Int32(32) + lane_idx
                partial = Boolean(False)
                full = Boolean(False)
                if valid_n_block and local_q_block < num_q_blocks:
                    compact_q_row = compact_q_begin + local_q_block
                    visible = (mVisibleBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                    full = (mFullBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                    partial = visible & ~full
                partial_ballot = cute.arch.vote_ballot_sync(partial)
                full_ballot = cute.arch.vote_ballot_sync(full)
                if lane_idx == Int32(0):
                    partial_count += Int32(cute.arch.popc(partial_ballot))
                    full_count += Int32(cute.arch.popc(full_ballot))
                q_group += Int32(1)
            if lane_idx == Int32(0) and valid_n_block:
                mPartialCounts[mask_head, compact_n_row] = partial_count
                mFullCounts[mask_head, compact_n_row] = full_count


class _ArbitraryPlanK2QMaterializeSm90(_ArbitraryPlanK2QCommonSm90):
    """Materialize stable K2Q CSR, native backward payload, and dQ ranks."""

    @cute.jit
    def __call__(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mQPartialCounts: cute.Tensor,
        mQFullCounts: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mPartialDQOrder: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mFullDQOrder: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
        stream: cuda.CUstream = None,
    ):
        # The kernel decodes an upper-bound row as batch_idx * max_n_blocks +
        # local_n_block. Launching only the compact K-block count would skip
        # samples after an interior zero-length sample.
        upper_total_n_blocks = batch_size * max_n_blocks
        hmask = mArbitraryFunc.shape[0]
        tiled_mma_sdp = make_sm90_bwd_tiled_mma_sdp(
            self.dtype,
            self.consumer_tile_m,
            self.tile_n,
            self.num_wg_mma,
            self.atom_layout_m_sdp,
            self.sdp_swap_ab,
        )
        self.kernel(
            tiled_mma_sdp,
            mArbitraryFunc,
            mVisibleBits,
            mFullBits,
            mQPartialCounts,
            mQFullCounts,
            mPartialOffsets,
            mPartialIndices,
            mPartialMasks,
            mPartialDQOrder,
            mFullOffsets,
            mFullIndices,
            mFullDQOrder,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_n_blocks,
            nfunc,
        ).launch(
            grid=(upper_total_n_blocks, hmask, 1),
            block=(_PLAN_THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _store_payload(
        self,
        tiled_mma_sdp: cute.TiledMma,
        mArbitraryFunc: cute.Tensor,
        mPartialMasks: cute.Tensor,
        payload_idx: Int32,
        planner_tidx: Int32,
        mask_head: Int32,
        local_q_block: Int32,
        local_n_block: Int32,
        q_begin: Int32,
        q_len: Int32,
        k_begin: Int32,
        k_len: Int32,
        nfunc: Int32,
    ) -> None:
        consumer_tidx = planner_tidx
        while consumer_tidx < Int32(self.num_mma_threads):
            thr_mma_sdp = tiled_mma_sdp.get_slice(consumer_tidx)
            acc_shape = (self.consumer_tile_m, self.tile_n)
            cS = cute.make_identity_tensor(
                acc_shape if const_expr(not self.sdp_swap_ab) else acc_shape[::-1]
            )
            tScS = thr_mma_sdp.partition_C(cS)
            row_coord = 0 if const_expr(not self.sdp_swap_ab) else 1
            col_coord = 1 if const_expr(not self.sdp_swap_ab) else 0
            for subtile_idx in cutlass.range_constexpr(self.subtile_factor):
                rMask = cute.make_rmem_tensor((self.payload_padded_words,), Uint32)
                rMask.fill(Uint32(0))
                for word_idx in cutlass.range_constexpr(self.payload_valid_words):
                    packed = Uint32(0)
                    for bit_idx in cutlass.range(32, unroll=1):
                        value_idx = word_idx * 32 + bit_idx
                        keep = Boolean(False)
                        if value_idx < self.payload_values_per_thread:
                            coord = tScS[value_idx]
                            q_local = (
                                local_q_block * Int32(self.tile_m)
                                + Int32(subtile_idx * self.consumer_tile_m)
                                + Int32(coord[row_coord])
                            )
                            k_local = local_n_block * Int32(self.tile_n) + Int32(coord[col_coord])
                            if q_local < q_len and k_local < k_len:
                                keep = self._is_visible(
                                    mArbitraryFunc,
                                    mask_head,
                                    q_begin + q_local,
                                    k_begin + k_local,
                                    nfunc,
                                )
                        if keep:
                            packed |= Uint32(1) << Uint32(bit_idx)
                    rMask[word_idx] = packed
                mask_iter = mPartialMasks.iterator + cute.crd2idx(
                    (payload_idx, Int32(subtile_idx), consumer_tidx, Int32(0)),
                    mPartialMasks.layout,
                )
                mask_ptr = cute.make_ptr(
                    Uint32,
                    mask_iter.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=min(
                        16,
                        4 * (self.payload_padded_words & -self.payload_padded_words),
                    ),
                )
                gMask = cute.make_tensor(mask_ptr, (self.payload_padded_words,))
                cute.autovec_copy(rMask, gMask)
            consumer_tidx += Int32(_PLAN_THREADS)

    @cute.kernel
    def kernel(
        self,
        tiled_mma_sdp: cute.TiledMma,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mQPartialCounts: cute.Tensor,
        mQFullCounts: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialMasks: cute.Tensor,
        mPartialDQOrder: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mFullDQOrder: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
        nfunc: Int32,
    ):
        planner_tidx, _, _ = cute.arch.thread_idx()
        upper_n_row, mask_head, _ = cute.arch.block_idx()
        (
            _,
            local_n_block,
            compact_n_row,
            compact_q_begin,
            num_q_blocks,
            q_begin,
            q_len,
            k_begin,
            k_len,
            valid_n_block,
        ) = self._sample_info_k(
            upper_n_row,
            max_n_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
        )
        if valid_n_block:
            total_n_blocks = mPartialOffsets.shape[0] - Int32(1)
            total_n_blocks = total_n_blocks // mArbitraryFunc.shape[0]
            plan_row = mask_head * total_n_blocks + compact_n_row
            partial_ordinal = Int32(0)
            full_ordinal = Int32(0)
            word_idx = local_n_block // Int32(32)
            bit_idx = local_n_block - word_idx * Int32(32)
            bit = Uint32(1) << Uint32(bit_idx)
            for local_q_block in cutlass.range(num_q_blocks, unroll=1):
                compact_q_row = compact_q_begin + local_q_block
                visible = (mVisibleBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                full = (mFullBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                if visible:
                    rank = Int32(0)
                    if planner_tidx == Int32(0):
                        rank = self._dq_write_rank(
                            mVisibleBits,
                            mask_head,
                            compact_q_row,
                            local_n_block,
                            mQPartialCounts[mask_head, compact_q_row]
                            + mQFullCounts[mask_head, compact_q_row],
                        )
                    if full:
                        if planner_tidx == Int32(0):
                            output_idx = mFullOffsets[plan_row] + full_ordinal
                            mFullIndices[output_idx] = local_q_block
                            mFullDQOrder[output_idx] = Int32(
                                (Uint32(rank) << Uint32(_DQ_ORDER_COMPONENT_BITS))
                                | Uint32(local_q_block)
                            )
                        full_ordinal += Int32(1)
                    else:
                        output_idx = mPartialOffsets[plan_row] + partial_ordinal
                        if planner_tidx == Int32(0):
                            mPartialIndices[output_idx] = local_q_block
                            mPartialDQOrder[output_idx] = Int32(
                                (Uint32(rank) << Uint32(_DQ_ORDER_COMPONENT_BITS))
                                | Uint32(local_q_block)
                            )
                        self._store_payload(
                            tiled_mma_sdp,
                            mArbitraryFunc,
                            mPartialMasks,
                            output_idx,
                            planner_tidx,
                            mask_head,
                            local_q_block,
                            local_n_block,
                            q_begin,
                            q_len,
                            k_begin,
                            k_len,
                            nfunc,
                        )
                        partial_ordinal += Int32(1)


def _exclusive_offsets(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty((counts.numel() + 1,), dtype=torch.int32, device=counts.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts.reshape(-1), dim=0, dtype=torch.int32)
    return offsets


def _to_cute_optional(tensor: torch.Tensor | None):
    return to_cute_tensor(tensor, assumed_align=4, leading_dim=0) if tensor is not None else None


def _compile_classify(
    config: _ResolvedSm90FwdConsumerConfig | _ResolvedSm90BwdTopologyConfig,
    arbitrary_func: torch.Tensor,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    error: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor | None,
):
    key = ("arbitrary_plan_classify_sm90", config.topology_planner_compile_key)
    if key not in _CLASSIFY_COMPILE_CACHE:
        kernel = _ArbitraryPlanClassifySm90(config)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _CLASSIFY_COMPILE_CACHE[key] = cute.compile(
            kernel,
            to_cute_tensor(arbitrary_func, assumed_align=4, leading_dim=2),
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(error, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_m_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    return _CLASSIFY_COMPILE_CACHE[key]


def _compile_materialize(
    config: _ResolvedSm90FwdConsumerConfig,
    arbitrary_func: torch.Tensor,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_offsets: torch.Tensor,
    partial_indices: torch.Tensor,
    partial_masks: torch.Tensor,
    full_offsets: torch.Tensor,
    full_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_m_blocks: torch.Tensor | None,
):
    key = ("arbitrary_plan_materialize_sm90", config.payload_planner_compile_key)
    if key not in _MATERIALIZE_COMPILE_CACHE:
        kernel = _ArbitraryPlanMaterializeSm90(config)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _MATERIALIZE_COMPILE_CACHE[key] = cute.compile(
            kernel,
            to_cute_tensor(arbitrary_func, assumed_align=4, leading_dim=2),
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_indices, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_masks, assumed_align=16, leading_dim=3),
            to_cute_tensor(full_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_indices, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_m_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    return _MATERIALIZE_COMPILE_CACHE[key]


def _compile_k2q_count(
    config: _ResolvedSm90BwdConsumerConfig,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_q_blocks: torch.Tensor | None,
    cu_total_k_blocks: torch.Tensor | None,
):
    key = ("arbitrary_plan_k2q_count_sm90", config.planner_compile_key)
    if key not in _K2Q_COUNT_COMPILE_CACHE:
        kernel = _ArbitraryPlanK2QCountSm90(config)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _K2Q_COUNT_COMPILE_CACHE[key] = cute.compile(
            kernel,
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(full_counts, assumed_align=4, leading_dim=1),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_q_blocks),
            _to_cute_optional(cu_total_k_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    return _K2Q_COUNT_COMPILE_CACHE[key]


def _compile_k2q_materialize(
    config: _ResolvedSm90BwdConsumerConfig,
    arbitrary_func: torch.Tensor,
    visible_bits: torch.Tensor,
    full_bits: torch.Tensor,
    q_partial_counts: torch.Tensor,
    q_full_counts: torch.Tensor,
    partial_offsets: torch.Tensor,
    partial_indices: torch.Tensor,
    partial_masks: torch.Tensor,
    partial_dq_order: torch.Tensor,
    full_offsets: torch.Tensor,
    full_indices: torch.Tensor,
    full_dq_order: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    cu_total_q_blocks: torch.Tensor | None,
    cu_total_k_blocks: torch.Tensor | None,
):
    key = ("arbitrary_plan_k2q_materialize_sm90", config.planner_compile_key)
    if key not in _K2Q_MATERIALIZE_COMPILE_CACHE:
        kernel = _ArbitraryPlanK2QMaterializeSm90(config)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _K2Q_MATERIALIZE_COMPILE_CACHE[key] = cute.compile(
            kernel,
            to_cute_tensor(arbitrary_func, assumed_align=4, leading_dim=2),
            to_cute_tensor(visible_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(full_bits, assumed_align=16, leading_dim=2),
            to_cute_tensor(q_partial_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(q_full_counts, assumed_align=4, leading_dim=1),
            to_cute_tensor(partial_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_indices, assumed_align=4, leading_dim=0),
            to_cute_tensor(partial_masks, assumed_align=16, leading_dim=3),
            to_cute_tensor(partial_dq_order, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_offsets, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_indices, assumed_align=4, leading_dim=0),
            to_cute_tensor(full_dq_order, assumed_align=4, leading_dim=0),
            _to_cute_optional(cu_seqlens_q),
            _to_cute_optional(cu_seqlens_k),
            _to_cute_optional(cu_total_q_blocks),
            _to_cute_optional(cu_total_k_blocks),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
    return _K2Q_MATERIALIZE_COMPILE_CACHE[key]


def _validate_builder_inputs(
    arbitrary_func: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
):
    varlen_values = (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    is_varlen = all(value is not None for value in varlen_values)
    if not is_varlen and any(value is not None for value in varlen_values):
        raise ValueError(
            "cu_seqlens_q, cu_seqlens_k, max_seqlen_q, and max_seqlen_k must be provided together"
        )
    expected_rank = 3 if is_varlen else 4
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must have rank {expected_rank} in this mode")
        if tensor.device != q.device:
            raise ValueError("q, k, and v must be on the same device")
        if tensor.dtype != q.dtype:
            raise TypeError("q, k, and v must have the same dtype")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")
    if not q.is_cuda and not is_fake_mode():
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-3] != v.shape[-3]:
        raise ValueError("k and v must have the same sequence extent")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same number of heads")
    if q.shape[-2] % k.shape[-2] != 0:
        raise ValueError("Hq must be divisible by Hkv")

    if arbitrary_func.ndim != 3:
        raise ValueError("arbitrary_func must have shape [Hmask, nfunc, total_q + padding]")
    if arbitrary_func.dtype != torch.int32:
        raise TypeError("arbitrary_func must have dtype torch.int32")
    if arbitrary_func.device != q.device:
        raise ValueError("arbitrary_func must be on the same device as q")
    if not arbitrary_func.is_contiguous():
        raise ValueError("arbitrary_func must be contiguous")
    hmask, nfunc, func_q_extent = arbitrary_func.shape
    if hmask not in (1, q.shape[-2]):
        raise ValueError(f"Hmask must be 1 or Hq ({q.shape[-2]}); got {hmask}")
    if nfunc <= 0 or nfunc % 2 == 0:
        raise ValueError("nfunc must be a positive odd runtime value")

    if is_varlen:
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_q/k must be rank-1")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
        for name, tensor in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if tensor.dtype != torch.int32 or tensor.device != q.device:
                raise ValueError(f"{name} must be int32 on the same device as q")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        batch_size = cu_seqlens_q.numel() - 1
        total_q = q.shape[0]
        total_k = k.shape[0]
        if not isinstance(max_seqlen_q, int) or not isinstance(max_seqlen_k, int):
            raise TypeError("max_seqlen_q/k must be Python ints")
        if max_seqlen_q < 0 or max_seqlen_k < 0:
            raise ValueError("max_seqlen_q/k must be non-negative")
        seqlen_q_fixed = 0
        seqlen_k_fixed = 0
    else:
        batch_size, seqlen_q_fixed = q.shape[:2]
        if k.shape[0] != batch_size or v.shape[0] != batch_size:
            raise ValueError("fixed q, k, and v must have the same batch size")
        seqlen_k_fixed = k.shape[1]
        total_q = batch_size * seqlen_q_fixed
        total_k = batch_size * seqlen_k_fixed
        max_seqlen_q = seqlen_q_fixed
        max_seqlen_k = seqlen_k_fixed
    if func_q_extent < total_q + 256:
        raise ValueError(
            f"arbitrary_func last extent must be at least total_q + 256 ({total_q + 256})"
        )
    return {
        "is_varlen": is_varlen,
        "batch_size": batch_size,
        "seqlen_q_fixed": seqlen_q_fixed,
        "seqlen_k_fixed": seqlen_k_fixed,
        "total_q": total_q,
        "total_k": total_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "hmask": hmask,
        "nfunc": nfunc,
    }


def create_arbitrary_block_sparse_tensors(
    arbitrary_func: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    pack_gqa: bool | None = None,
    build_backward: bool = False,
) -> BlockSparseTensorsTorch:
    """Build a compact sample-local arbitrary-mask plan for SM90 attention."""

    metadata = _validate_builder_inputs(
        arbitrary_func,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    device = q.device
    arch = (
        90
        if is_fake_mode()
        else sum(
            value * multiplier
            for value, multiplier in zip(torch.cuda.get_device_capability(device), (10, 1))
        )
    )
    fwd_config = resolve_sm90_fwd_consumer_config(
        arch=arch,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        num_q_heads=q.shape[-2],
        num_kv_heads=k.shape[-2],
        is_varlen=metadata["is_varlen"],
        hmask=metadata["hmask"],
        pack_gqa=pack_gqa,
    )
    bwd_config = (
        resolve_sm90_bwd_consumer_config(
            arch=arch,
            dtype=q.dtype,
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            num_q_heads=q.shape[-2],
            num_kv_heads=k.shape[-2],
            is_varlen=metadata["is_varlen"],
        )
        if build_backward
        else None
    )

    qratio = fwd_config.qhead_per_kvhead if fwd_config.pack_gqa else 1
    fwd_max_m_blocks = math.ceil(metadata["max_seqlen_q"] * qratio / fwd_config.tile_m)
    fwd_max_n_blocks = math.ceil(metadata["max_seqlen_k"] / fwd_config.tile_n)
    fwd_upper_total_m_blocks = metadata["batch_size"] * fwd_max_m_blocks
    fwd_num_words = max(1, math.ceil(fwd_max_n_blocks / 32))

    bwd_max_m_blocks = (
        math.ceil(metadata["max_seqlen_q"] / bwd_config.sparse_tile_m)
        if bwd_config is not None
        else 0
    )
    bwd_max_n_blocks = (
        math.ceil(metadata["max_seqlen_k"] / bwd_config.tile_n) if bwd_config is not None else 0
    )
    bwd_upper_total_m_blocks = metadata["batch_size"] * bwd_max_m_blocks
    bwd_upper_total_n_blocks = metadata["batch_size"] * bwd_max_n_blocks
    bwd_num_words = max(1, math.ceil(bwd_max_n_blocks / 32))
    if bwd_config is not None and (
        bwd_max_m_blocks > _DQ_ORDER_COMPONENT_LIMIT or bwd_max_n_blocks > _DQ_ORDER_COMPONENT_LIMIT
    ):
        raise ValueError(
            "SM90 arbitrary backward supports at most 65536 local Q/K blocks per sample"
        )

    cu_total_m_blocks = None
    cu_total_bwd_m_blocks = None
    cu_total_bwd_n_blocks = None
    metadata_invalid = torch.zeros((), dtype=torch.bool, device=device)
    if metadata["is_varlen"]:
        q_lengths_raw = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        k_lengths_raw = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        q_lengths = q_lengths_raw.clamp(min=0, max=metadata["max_seqlen_q"])
        physical_q_lengths = q_lengths * qratio
        m_counts = torch.div(
            physical_q_lengths + fwd_config.tile_m - 1,
            fwd_config.tile_m,
            rounding_mode="floor",
        ).to(torch.int32)
        cu_total_m_blocks = torch.empty(
            (metadata["batch_size"] + 1,), dtype=torch.int32, device=device
        )
        cu_total_m_blocks[0] = 0
        cu_total_m_blocks[1:] = torch.cumsum(m_counts, dim=0, dtype=torch.int32)
        if bwd_config is not None:
            bwd_m_counts = torch.div(
                q_lengths + bwd_config.sparse_tile_m - 1,
                bwd_config.sparse_tile_m,
                rounding_mode="floor",
            ).to(torch.int32)
            bwd_n_counts = torch.div(
                k_lengths_raw.clamp(min=0, max=metadata["max_seqlen_k"]) + bwd_config.tile_n - 1,
                bwd_config.tile_n,
                rounding_mode="floor",
            ).to(torch.int32)
            cu_total_bwd_m_blocks = torch.empty(
                (metadata["batch_size"] + 1,), dtype=torch.int32, device=device
            )
            cu_total_bwd_n_blocks = torch.empty_like(cu_total_bwd_m_blocks)
            cu_total_bwd_m_blocks[0] = 0
            cu_total_bwd_n_blocks[0] = 0
            cu_total_bwd_m_blocks[1:] = torch.cumsum(bwd_m_counts, dim=0, dtype=torch.int32)
            cu_total_bwd_n_blocks[1:] = torch.cumsum(bwd_n_counts, dim=0, dtype=torch.int32)
        metadata_invalid = (
            (cu_seqlens_q[0] != 0)
            | (cu_seqlens_k[0] != 0)
            | (cu_seqlens_q[-1] != metadata["total_q"])
            | (cu_seqlens_k[-1] != metadata["total_k"])
            | torch.any(q_lengths_raw < 0)
            | torch.any(k_lengths_raw < 0)
            | torch.any(q_lengths_raw > metadata["max_seqlen_q"])
            | torch.any(k_lengths_raw > metadata["max_seqlen_k"])
        )

    visible_bits = torch.zeros(
        (metadata["hmask"], fwd_upper_total_m_blocks, fwd_num_words),
        dtype=torch.uint32,
        device=device,
    )
    full_bits = torch.zeros_like(visible_bits)
    partial_counts_tmp = torch.zeros(
        (metadata["hmask"], fwd_upper_total_m_blocks),
        dtype=torch.int32,
        device=device,
    )
    full_counts_tmp = torch.zeros_like(partial_counts_tmp)
    error = torch.zeros((1,), dtype=torch.uint32, device=device)

    bwd_visible_bits = None
    bwd_full_bits = None
    bwd_q_partial_counts_tmp = None
    bwd_q_full_counts_tmp = None
    bwd_partial_counts_tmp = None
    bwd_full_counts_tmp = None
    if bwd_config is not None:
        bwd_visible_bits = torch.zeros(
            (metadata["hmask"], bwd_upper_total_m_blocks, bwd_num_words),
            dtype=torch.uint32,
            device=device,
        )
        bwd_full_bits = torch.zeros_like(bwd_visible_bits)
        bwd_q_partial_counts_tmp = torch.zeros(
            (metadata["hmask"], bwd_upper_total_m_blocks),
            dtype=torch.int32,
            device=device,
        )
        bwd_q_full_counts_tmp = torch.zeros_like(bwd_q_partial_counts_tmp)
        bwd_partial_counts_tmp = torch.zeros(
            (metadata["hmask"], bwd_upper_total_n_blocks),
            dtype=torch.int32,
            device=device,
        )
        bwd_full_counts_tmp = torch.zeros_like(bwd_partial_counts_tmp)

    if fwd_upper_total_m_blocks > 0 and fwd_max_n_blocks > 0:
        classify = _compile_classify(
            fwd_config,
            arbitrary_func,
            visible_bits,
            full_bits,
            partial_counts_tmp,
            full_counts_tmp,
            error,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_m_blocks,
        )
        if not is_fake_mode():
            classify(
                arbitrary_func,
                visible_bits,
                full_bits,
                partial_counts_tmp,
                full_counts_tmp,
                error,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(fwd_max_m_blocks),
                Int32(fwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
    if bwd_config is not None and bwd_upper_total_m_blocks > 0 and bwd_max_n_blocks > 0:
        bwd_classify = _compile_classify(
            _ResolvedSm90BwdTopologyConfig(bwd_config),
            arbitrary_func,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_q_partial_counts_tmp,
            bwd_q_full_counts_tmp,
            error,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
        )
        if not is_fake_mode():
            bwd_classify(
                arbitrary_func,
                bwd_visible_bits,
                bwd_full_bits,
                bwd_q_partial_counts_tmp,
                bwd_q_full_counts_tmp,
                error,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_m_blocks),
                Int32(bwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
    if bwd_config is not None and bwd_upper_total_n_blocks > 0 and bwd_max_m_blocks > 0:
        k2q_count = _compile_k2q_count(
            bwd_config,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_partial_counts_tmp,
            bwd_full_counts_tmp,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
            cu_total_bwd_n_blocks,
        )
        if not is_fake_mode():
            k2q_count(
                bwd_visible_bits,
                bwd_full_bits,
                bwd_partial_counts_tmp,
                bwd_full_counts_tmp,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                cu_total_bwd_n_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_n_blocks),
            )
    if is_fake_mode():
        total_m_blocks = fwd_upper_total_m_blocks
        partial_nnz = metadata["hmask"] * fwd_upper_total_m_blocks * fwd_max_n_blocks
        full_nnz = 0
        bwd_total_m_blocks = bwd_upper_total_m_blocks
        bwd_total_n_blocks = bwd_upper_total_n_blocks
        bwd_partial_nnz = metadata["hmask"] * bwd_upper_total_n_blocks * bwd_max_m_blocks
        bwd_full_nnz = 0
    else:
        partial_scan = torch.cumsum(partial_counts_tmp.reshape(-1), dim=0, dtype=torch.int64)
        full_scan = torch.cumsum(full_counts_tmp.reshape(-1), dim=0, dtype=torch.int64)
        total_m_tensor = (
            cu_total_m_blocks[-1].to(torch.int64)
            if cu_total_m_blocks is not None
            else torch.tensor(fwd_upper_total_m_blocks, dtype=torch.int64, device=device)
        )
        partial_total = (
            partial_scan[-1]
            if partial_scan.numel()
            else torch.zeros((), dtype=torch.int64, device=device)
        )
        full_total = (
            full_scan[-1]
            if full_scan.numel()
            else torch.zeros((), dtype=torch.int64, device=device)
        )
        zero = torch.zeros((), dtype=torch.int64, device=device)
        bwd_partial_scan = (
            torch.cumsum(bwd_partial_counts_tmp.reshape(-1), dim=0, dtype=torch.int64)
            if bwd_config is not None
            else None
        )
        bwd_full_scan = (
            torch.cumsum(bwd_full_counts_tmp.reshape(-1), dim=0, dtype=torch.int64)
            if bwd_config is not None
            else None
        )
        bwd_total_m_tensor = (
            cu_total_bwd_m_blocks[-1].to(torch.int64)
            if cu_total_bwd_m_blocks is not None
            else torch.tensor(bwd_upper_total_m_blocks, dtype=torch.int64, device=device)
            if bwd_config is not None
            else zero
        )
        bwd_total_n_tensor = (
            cu_total_bwd_n_blocks[-1].to(torch.int64)
            if cu_total_bwd_n_blocks is not None
            else torch.tensor(bwd_upper_total_n_blocks, dtype=torch.int64, device=device)
            if bwd_config is not None
            else zero
        )
        bwd_partial_total = (
            bwd_partial_scan[-1]
            if bwd_partial_scan is not None and bwd_partial_scan.numel()
            else zero
        )
        bwd_full_total = (
            bwd_full_scan[-1] if bwd_full_scan is not None and bwd_full_scan.numel() else zero
        )
        combined_error = error[0].to(torch.int64) | (
            metadata_invalid.to(torch.int64) * _ERROR_INVALID_SEQLENS
        )
        header = torch.stack(
            (
                total_m_tensor,
                partial_total,
                full_total,
                bwd_total_m_tensor,
                bwd_total_n_tensor,
                bwd_partial_total,
                bwd_full_total,
                combined_error,
            )
        )
        (
            total_m_blocks,
            partial_nnz,
            full_nnz,
            bwd_total_m_blocks,
            bwd_total_n_blocks,
            bwd_partial_nnz,
            bwd_full_nnz,
            error_value,
        ) = (int(value) for value in header.cpu().tolist())
        if error_value & _ERROR_INVALID_SEQLENS:
            raise ValueError(
                "cu_seqlens_q/k must start at zero, be nondecreasing, end at "
                "total_q/total_k, and respect max_seqlen_q/k"
            )
        if error_value & _ERROR_INVALID_INTERVAL:
            raise ValueError(
                "arbitrary_func endpoints must be in [0, total_k], each begin must "
                "not exceed its end, and non-empty interval begins must be ordered"
            )

    partial_counts = partial_counts_tmp[:, :total_m_blocks].clone()
    full_counts = full_counts_tmp[:, :total_m_blocks].clone()
    partial_offsets = _exclusive_offsets(partial_counts)
    full_offsets = _exclusive_offsets(full_counts)
    partial_indices = torch.empty((partial_nnz,), dtype=torch.int32, device=device)
    full_indices = torch.empty((full_nnz,), dtype=torch.int32, device=device)
    partial_masks = torch.empty(
        (
            partial_nnz,
            fwd_config.physical_subtiles,
            fwd_config.num_mma_threads,
            fwd_config.payload_padded_words,
        ),
        dtype=torch.uint32,
        device=device,
    )
    bwd_plan = None
    if bwd_config is not None:
        bwd_partial_counts = bwd_partial_counts_tmp[:, :bwd_total_n_blocks].clone()
        bwd_full_counts = bwd_full_counts_tmp[:, :bwd_total_n_blocks].clone()
        bwd_partial_offsets = _exclusive_offsets(bwd_partial_counts)
        bwd_full_offsets = _exclusive_offsets(bwd_full_counts)
        bwd_partial_indices = torch.empty((bwd_partial_nnz,), dtype=torch.int32, device=device)
        bwd_full_indices = torch.empty((bwd_full_nnz,), dtype=torch.int32, device=device)
        bwd_partial_dq_order = torch.empty_like(bwd_partial_indices)
        bwd_full_dq_order = torch.empty_like(bwd_full_indices)
        bwd_partial_masks = torch.empty(
            (
                bwd_partial_nnz,
                bwd_config.physical_subtiles,
                bwd_config.num_mma_threads,
                bwd_config.payload_padded_words,
            ),
            dtype=torch.uint32,
            device=device,
        )
    if fwd_upper_total_m_blocks > 0 and partial_nnz + full_nnz > 0:
        materialize = _compile_materialize(
            fwd_config,
            arbitrary_func,
            visible_bits,
            full_bits,
            partial_offsets,
            partial_indices,
            partial_masks,
            full_offsets,
            full_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_m_blocks,
        )
        if not is_fake_mode():
            materialize(
                arbitrary_func,
                visible_bits,
                full_bits,
                partial_offsets,
                partial_indices,
                partial_masks,
                full_offsets,
                full_indices,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_m_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(fwd_max_m_blocks),
                Int32(fwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
    if (
        bwd_config is not None
        and bwd_upper_total_n_blocks > 0
        and bwd_partial_nnz + bwd_full_nnz > 0
    ):
        bwd_materialize = _compile_k2q_materialize(
            bwd_config,
            arbitrary_func,
            bwd_visible_bits,
            bwd_full_bits,
            bwd_q_partial_counts_tmp,
            bwd_q_full_counts_tmp,
            bwd_partial_offsets,
            bwd_partial_indices,
            bwd_partial_masks,
            bwd_partial_dq_order,
            bwd_full_offsets,
            bwd_full_indices,
            bwd_full_dq_order,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_total_bwd_m_blocks,
            cu_total_bwd_n_blocks,
        )
        if not is_fake_mode():
            bwd_materialize(
                arbitrary_func,
                bwd_visible_bits,
                bwd_full_bits,
                bwd_q_partial_counts_tmp,
                bwd_q_full_counts_tmp,
                bwd_partial_offsets,
                bwd_partial_indices,
                bwd_partial_masks,
                bwd_partial_dq_order,
                bwd_full_offsets,
                bwd_full_indices,
                bwd_full_dq_order,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_total_bwd_m_blocks,
                cu_total_bwd_n_blocks,
                Int32(metadata["batch_size"]),
                Int32(metadata["seqlen_q_fixed"]),
                Int32(metadata["seqlen_k_fixed"]),
                Int32(metadata["total_q"]),
                Int32(metadata["total_k"]),
                Int32(bwd_max_n_blocks),
                Int32(metadata["nfunc"]),
            )
    if bwd_config is not None:
        bwd_plan = BlockSparseTensorsTorch(
            mask_block_cnt=bwd_partial_counts,
            mask_block_idx=bwd_partial_indices,
            full_block_cnt=bwd_full_counts,
            full_block_idx=bwd_full_indices,
            cu_total_m_blocks=(cu_total_bwd_n_blocks if metadata["is_varlen"] else None),
            cu_block_idx_offsets=None,
            block_size=bwd_config.block_size,
            dq_write_order=bwd_partial_dq_order,
            dq_write_order_full=bwd_full_dq_order,
            spt=bwd_config.spt,
            mask_block_offset=bwd_partial_offsets,
            full_block_offset=bwd_full_offsets,
            mask_block_masks=bwd_partial_masks,
            pack_gqa=None,
            bwd_tensors=None,
        )

    plan = BlockSparseTensorsTorch(
        mask_block_cnt=partial_counts,
        mask_block_idx=partial_indices,
        full_block_cnt=full_counts,
        full_block_idx=full_indices,
        cu_total_m_blocks=cu_total_m_blocks,
        cu_block_idx_offsets=None,
        block_size=fwd_config.block_size,
        dq_write_order=None,
        dq_write_order_full=None,
        spt=None,
        mask_block_offset=partial_offsets,
        full_block_offset=full_offsets,
        mask_block_masks=partial_masks,
        pack_gqa=fwd_config.pack_gqa,
        bwd_tensors=bwd_plan,
    )

    return plan


__all__ = ["create_arbitrary_block_sparse_tensors"]
