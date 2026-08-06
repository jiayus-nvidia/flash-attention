"""
Block-sparse runtime utilities for CUTE DSL kernels.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from typing import Callable, Optional, Tuple
from functools import partial
import math
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from quack import copy_utils

# Import data structures from block_sparsity
from flash_attn_cute import barrier
from flash_attn_cute.block_sparsity import BlockSparseTensors
from flash_attn_cute.named_barrier import NamedBarrierBwd
from flash_attn_cute.seqlen_info import SeqlenInfoQK
from flash_attn_cute.sm90_fwd_config import _sm90_fwd_mask_payload_group_idx


@dsl_user_op
def _prefetch_global_l1(ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Prefetch one global cache line into the local L1 data cache."""
    ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64],
        "prefetch.global.L1 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _get_curr_blocksparse_tensors_varlen(
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Varlen path: tensors are 2D [nheads, total_m_blocks] / [nheads, total_n_blocks]."""
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, *_ = blocksparse_tensors
    curr_m_block = seqlen_info.m_block_offset + m_block
    curr_block_idx_offset = seqlen_info.block_idx_offset + m_block * seqlen_info.num_n_blocks
    curr_mask_block_cnt = mask_block_cnt[head_idx, curr_m_block]
    curr_mask_block_idx = cute.domain_offset(curr_block_idx_offset, mask_block_idx[head_idx, None])
    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[head_idx, curr_m_block]
        curr_full_block_idx = cute.domain_offset(
            curr_block_idx_offset, full_block_idx[head_idx, None]
        )
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None
    return (curr_mask_block_cnt, curr_mask_block_idx, curr_full_block_cnt, curr_full_block_idx)


@cute.jit
def _get_curr_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Fixed-length path: tensors are 4D [batch, nheads, m_block, n_block]."""
    mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, *_ = blocksparse_tensors
    curr_mask_block_cnt = mask_block_cnt[batch_idx, head_idx, m_block]
    curr_mask_block_idx = mask_block_idx[batch_idx, head_idx, m_block, None]
    if const_expr(full_block_cnt is not None):
        curr_full_block_cnt = full_block_cnt[batch_idx, head_idx, m_block]
        curr_full_block_idx = full_block_idx[batch_idx, head_idx, m_block, None]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None
    return (curr_mask_block_cnt, curr_mask_block_idx, curr_full_block_cnt, curr_full_block_idx)


@cute.jit
def _get_curr_blocksparse_tensors_linear(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Fixed-length CSR path: counts are [B, H, row], indices are compact 1D."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    assert mask_block_offset is not None

    batch, nheads, n_blocks = mask_block_cnt.shape
    sparse_batch_idx = 0 if batch == 1 else batch_idx
    sparse_head_idx = 0 if nheads == 1 else head_idx
    offset_idx = (
        (0 if batch == 1 else batch_idx * nheads * n_blocks)
        + (0 if nheads == 1 else head_idx * n_blocks)
        + m_block
    )
    curr_mask_block_cnt = mask_block_cnt[sparse_batch_idx, sparse_head_idx, m_block]
    curr_mask_block_idx = cute.domain_offset(mask_block_offset[offset_idx], mask_block_idx)
    if const_expr(full_block_cnt is not None):
        assert full_block_offset is not None
        assert full_block_idx is not None
        curr_full_block_cnt = full_block_cnt[sparse_batch_idx, sparse_head_idx, m_block]
        curr_full_block_idx = cute.domain_offset(full_block_offset[offset_idx], full_block_idx)
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_idx = None
    return (curr_mask_block_cnt, curr_mask_block_idx, curr_full_block_cnt, curr_full_block_idx)


@cute.jit
def get_curr_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Extract head, m_block, and batch-local blocksparsity data from blocksparse_tensors"""
    if const_expr(len(blocksparse_tensors.mask_block_cnt.shape) == 2):
        return _get_curr_blocksparse_tensors_varlen(
            head_idx, m_block, blocksparse_tensors, seqlen_info
        )
    return get_curr_blocksparse_tensors_fixed(batch_idx, head_idx, m_block, blocksparse_tensors)


@cute.jit
def get_curr_blocksparse_tensors_fixed(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
) -> Tuple[cutlass.Int32, cute.Tensor, cutlass.Int32, Optional[cute.Tensor]]:
    """Extract fixed-length 4D or CSR block-sparse data."""
    if const_expr(blocksparse_tensors.mask_block_offset is not None):
        return _get_curr_blocksparse_tensors_linear(
            batch_idx, head_idx, m_block, blocksparse_tensors
        )
    return _get_curr_blocksparse_tensors(batch_idx, head_idx, m_block, blocksparse_tensors)


@cute.jit
def get_curr_arbitrary_blocksparse_tensors(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Extract one compact arbitrary-plan row and its packed-mask base."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    assert mask_block_offset is not None
    assert full_block_cnt is not None
    assert full_block_idx is not None
    assert full_block_offset is not None

    total_m_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    if const_expr(seqlen_info.has_cu_seqlens_q):
        outer_row = seqlen_info.m_block_offset + m_block
    else:
        physical_q_len = seqlen_info.seqlen_q * qhead_per_kvhead
        m_blocks_per_sample = cute.ceil_div(physical_q_len, tile_m)
        outer_row = batch_idx * m_blocks_per_sample + m_block
    plan_row = plan_head * total_m_blocks + outer_row

    partial_base = mask_block_offset[plan_row]
    full_base = full_block_offset[plan_row]
    curr_mask_block_cnt = mask_block_cnt[plan_head, outer_row]
    curr_mask_block_idx = cute.domain_offset(partial_base, mask_block_idx)
    curr_full_block_cnt = full_block_cnt[plan_head, outer_row]
    curr_full_block_idx = cute.domain_offset(full_base, full_block_idx)
    return (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
        partial_base,
    )


@cute.jit
def get_curr_arbitrary_block_counts_fwd_sm90(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    seqlen_info: SeqlenInfoQK,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Load only the arbitrary row metadata needed by SM90 MMA consumers."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    full_block_cnt = blocksparse_tensors.full_block_cnt
    mask_block_offset = blocksparse_tensors.mask_block_offset
    assert full_block_cnt is not None
    assert mask_block_offset is not None

    total_m_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    if const_expr(seqlen_info.has_cu_seqlens_q):
        outer_row = seqlen_info.m_block_offset + m_block
    else:
        physical_q_len = seqlen_info.seqlen_q * qhead_per_kvhead
        m_blocks_per_sample = cute.ceil_div(physical_q_len, tile_m)
        outer_row = batch_idx * m_blocks_per_sample + m_block
    return (
        mask_block_cnt[plan_head, outer_row],
        full_block_cnt[plan_head, outer_row],
        mask_block_offset[plan_head * total_m_blocks + outer_row],
    )


@cute.jit
def get_curr_arbitrary_blocksparse_tensors_bwd(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    n_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    n_blocks_per_sample: cutlass.Int32,
):
    """Extract one compact arbitrary K2Q row and its packed-mask base."""

    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    cu_total_k_blocks = blocksparse_tensors.cu_total_m_blocks
    assert mask_block_offset is not None
    assert full_block_cnt is not None
    assert full_block_idx is not None
    assert full_block_offset is not None

    total_n_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    outer_row = batch_idx * n_blocks_per_sample + n_block
    if const_expr(cu_total_k_blocks is not None):
        outer_row = cu_total_k_blocks[batch_idx] + n_block
    plan_row = plan_head * total_n_blocks + outer_row

    partial_base = mask_block_offset[plan_row]
    full_base = full_block_offset[plan_row]
    curr_mask_block_cnt = mask_block_cnt[plan_head, outer_row]
    curr_mask_block_idx = cute.domain_offset(partial_base, mask_block_idx)
    curr_full_block_cnt = full_block_cnt[plan_head, outer_row]
    curr_full_block_idx = cute.domain_offset(full_base, full_block_idx)
    return (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
        partial_base,
        full_base,
    )


@cute.jit
def get_curr_arbitrary_block_counts_bwd_sm90(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    n_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
    n_blocks_per_sample: cutlass.Int32,
):
    """Load the K2Q counts and partial-payload base needed by the MMA consumer."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    full_block_cnt = blocksparse_tensors.full_block_cnt
    mask_block_offset = blocksparse_tensors.mask_block_offset
    cu_total_k_blocks = blocksparse_tensors.cu_total_m_blocks
    assert full_block_cnt is not None
    assert mask_block_offset is not None

    total_n_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    outer_row = batch_idx * n_blocks_per_sample + n_block
    if const_expr(cu_total_k_blocks is not None):
        outer_row = cu_total_k_blocks[batch_idx] + n_block
    return (
        mask_block_cnt[plan_head, outer_row],
        full_block_cnt[plan_head, outer_row],
        mask_block_offset[plan_head * total_n_blocks + outer_row],
    )


@cute.jit
def load_packed_mask_payload(
    mask_payloads: Optional[cute.Tensor],
    payload_idx: Int32,
    payload_group_idx: Int32,
    subtile_idx: Int32 = Int32(0),
    payload_words: cutlass.Constexpr[int] = 4,
):
    """Load one compact consumer-native arbitrary-mask payload."""
    if const_expr(mask_payloads is None):
        return None
    payload_alignment = min(16, 4 * (payload_words & -payload_words))
    mask_iter = mask_payloads.iterator + cute.crd2idx(
        (payload_idx, subtile_idx, payload_group_idx, Int32(0)),
        mask_payloads.layout,
    )
    mask_ptr = cute.make_ptr(
        Uint32,
        mask_iter.toint(),
        cute.AddressSpace.gmem,
        assumed_align=payload_alignment,
    )
    g_mask = cute.make_tensor(mask_ptr, (payload_words,))
    r_mask = cute.make_rmem_tensor_like(g_mask, Uint32)
    cute.autovec_copy(g_mask, r_mask)
    return r_mask


# NOTE [SM100 block-sparse empty tiles: mbarrier contract]
#
# For block-sparse SM100 forward, a given (m_block, stage) Q tile can have zero active
# KV blocks (total_block_cnt == 0). In that case there is no seqlen_kv iteration, so
# the softmax warp-group has no row stats to publish.
#
# The correction warp-group seeds fully-masked-row stats and runs the usual correction
# epilogue so output/LSE have well-defined values. Both warp-groups must still perform
# the softmax<->correction mbarrier handshake so phases advance correctly across
# empty->empty and empty->non-empty tile sequences.
#
# In the no-sink case, this corresponds to the usual fully-masked-row convention:
# output is zero and LSE is -inf.
#
# Barrier contract (each is `mbar_ptr + <offset> + stage`):
#
# Producer/consumer pairs:
# - `mbar_softmax_corr_full`    : softmax arrive        -> correction wait
# - `mbar_softmax_corr_empty`   : correction arrive     -> softmax wait
# - `mbar_P_full_O_rescaled`    : softmax arrive (+ correction arrive) -> MMA wait
# - `mbar_P_full_2`             : softmax arrive        -> MMA wait
# - `mbar_corr_epi_full_/empty` : correction <-> epilogue (only when epilogue is separate)
#
# Empty tile (`total_block_cnt == 0`):
# - Softmax: skips the seqlen_kv softmax path entirely (no P stores, no `mbar_P_full_*`).
#   It only arrives `mbar_softmax_corr_full` once per stage as a synthetic "no work" signal.
#   At the `softmax_loop` level, softmax unconditionally waits `mbar_softmax_corr_empty`
#   before each tile (when block-sparse) to drain a prior correction arrival and keep
#   phases aligned across non-empty -> empty transitions.
# - Correction: waits `mbar_softmax_corr_full`, seeds stats + runs `correction_epilogue(scale=0)`,
#   and arrives `mbar_softmax_corr_empty` (and `mbar_corr_epi_full_/empty` when applicable).
# - No `mbar_P_full_*` barriers are arrived (no P, no MMA O); only the softmax<->correction
#   (and correction<->epilogue) handshakes advance phases.
#
# Non-empty tile:
# - Softmax: runs `softmax_step` (produces P) and uses `mbar_softmax_corr_full/empty` to
#   publish row_max (during seqlen_kv) and final row stats (once per tile), and to advance phases;
#   arrives `mbar_P_full_*` when P is stored.
# - Correction: waits `mbar_softmax_corr_full`, may rescale/release O, arrives `mbar_softmax_corr_empty`
#   to ack/advance, and arrives `mbar_P_full_O_rescaled` when MMA can proceed.
#
# Backward (SM100):
# - Empty KV tile: for a given `n_block`, `total_m_block_cnt == 0` means no Q tiles contribute.
# - Both the load and compute loops guard all pipeline work on `process_tile`, so empty tiles
#   skip producer/consumer operations entirely (no per-tile mbarrier phase handshake like forward).
# - In the `not dKV_postprocess` path, dK/dV for empty KV tiles are explicitly written as zeros
#   even when `process_tile == False` (see `flash_bwd_sm100.py` `should_zero_dKV`).


@cute.jit
def load_block_list(
    block_indices: cute.Tensor,
    block_count,
    first_block_preloaded: cutlass.Constexpr,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
):
    """Iterate over the sparse blocks and load K, V into the pipeline.
    For the intra_wg_overlap case, we overlap the loads of K and V. And this
    means we need to pipeline the last V load from the partial block case,
    with the loads for the full blocks. Set first_block_preloaded when the
    caller has already issued the first K load for the list.

    Q is loaded separately on its own mbarrier before this function is called.

    Note:
        we iterate along the n_block indices in reverse.

    Returns:
        Updated kv_producer_state after processing the block list.

    """
    if block_count > 0:
        if const_expr(not intra_wg_overlap):
            for offset in cutlass.range(block_count):
                n_block = block_indices[block_count - 1 - offset]
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
        else:
            n_block_first = block_indices[block_count - 1]
            if const_expr(not first_block_preloaded):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_first, producer_state=kv_producer_state)

            for idx in cutlass.range(block_count - 1, unroll=1):
                n_block_prev = block_indices[block_count - 1 - idx]
                n_block = block_indices[block_count - 2 - idx]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)

    return kv_producer_state


@cute.jit
def finish_overlap_v_load(
    block_indices: cute.Tensor,
    block_count,
    load_V,
    pipeline_v,
    kv_producer_state,
):
    """Load the final V block after overlapped K/V loads."""
    if block_count > 0:
        n_block_last = block_indices[0]
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_last, producer_state=kv_producer_state)
        kv_producer_state.advance()

    return kv_producer_state


@cute.jit
def sparse_tensor_m_block(
    m_block,
    qhead_per_kvhead: cutlass.Constexpr[int],
    q_subtile_factor: cutlass.Constexpr[int],
):
    """Map packed m_block indices to block-sparse tensor indices."""
    block = m_block
    if const_expr(qhead_per_kvhead != 1):
        block = block // qhead_per_kvhead
    if const_expr(q_subtile_factor != 1):
        block = block // q_subtile_factor
    return block


@cute.jit
def prefetch_arbitrary_forward_block_index(
    block_idx: cute.Tensor,
    list_idx: Int32,
    valid,
):
    """Prefetch one indirect block index without selecting a CSR list."""
    if valid:
        block_iter = block_idx.iterator + cute.crd2idx((list_idx,), block_idx.layout)
        with cute.arch.elect_one():
            block_ptr = cute.make_ptr(
                Int32,
                block_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            _prefetch_global_l1(block_ptr)


@cute.jit
def prefetch_arbitrary_forward_mask(
    mask_payloads: cute.Tensor,
    payload_idx: Int32,
    is_partial,
    payload_words: cutlass.Constexpr[int],
):
    """Use the TMA producer warp to stage a partial payload in L1."""
    if is_partial:
        lane_idx = cute.arch.lane_idx()
        cache_line_words = 128 // 4
        payload_total_words = mask_payloads.shape[2] * payload_words
        word_offset = lane_idx * cache_line_words
        if word_offset < payload_total_words:
            mask_iter = mask_payloads.iterator + cute.crd2idx(
                (payload_idx, Int32(0), Int32(0), Int32(0)),
                mask_payloads.layout,
            )
            mask_ptr = cute.make_ptr(
                Uint32,
                mask_iter.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            _prefetch_global_l1(mask_ptr + word_offset)


@cute.jit
def apply_arbitrary_forward_mask(
    acc_S: cute.Tensor,
    n_block: Int32,
    base_mask_fn: Callable,
    mask_payloads: cute.Tensor,
    payload_idx: Int32,
    payload_group_idx: Int32,
    payload_words: cutlass.Constexpr[int],
    mask_seqlen: cutlass.Constexpr[bool] = True,
    r_bitmask: Optional[cute.Tensor] = None,
):
    """Apply one packed partial-block payload."""
    # The FA mask callback passes mask_seqlen; the payload already encodes it.
    if const_expr(r_bitmask is None):
        r_bitmask = load_packed_mask_payload(
            mask_payloads,
            payload_idx,
            payload_group_idx,
            payload_words=payload_words,
        )
    base_mask_fn(
        acc_S=acc_S,
        n_block=n_block,
        mask_mod=None,
        mask_seqlen=False,
        rBitmask=r_bitmask,
    )


@cute.jit
def produce_arbitrary_forward_nonoverlap(
    partial_block_cnt,
    partial_block_idx: cute.Tensor,
    full_block_cnt,
    full_block_idx: cute.Tensor,
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    payload_words: cutlass.Constexpr[int],
    kv_producer_state,
    load_K: Callable,
    load_V: Callable,
    pipeline_k,
    pipeline_v,
    o_empty_mbar_ptr: Optional[cute.Pointer] = None,
    o_empty_phase: Optional[Int32] = None,
):
    """Produce anchored partial/full CSR loops without K/V overlap."""
    if const_expr(o_empty_mbar_ptr is not None):
        assert o_empty_phase is not None
        cute.arch.mbarrier_wait(o_empty_mbar_ptr, phase=o_empty_phase)

    for iteration in cutlass.range(partial_block_cnt, unroll=1):
        partial_list_idx = partial_block_cnt - Int32(1) - iteration
        n_block = partial_block_idx[partial_list_idx]
        payload_idx = partial_payload_base + partial_list_idx
        prefetch_arbitrary_forward_mask(mask_payloads, payload_idx, True, payload_words)
        prefetch_arbitrary_forward_block_index(
            partial_block_idx,
            partial_list_idx - Int32(1),
            iteration + Int32(1) < partial_block_cnt,
        )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(src_idx=n_block, producer_state=kv_producer_state)
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block, producer_state=kv_producer_state)
        kv_producer_state.advance()

    for iteration in cutlass.range(full_block_cnt, unroll=1):
        full_list_idx = full_block_cnt - Int32(1) - iteration
        n_block = full_block_idx[full_list_idx]
        prefetch_arbitrary_forward_block_index(
            full_block_idx,
            full_list_idx - Int32(1),
            iteration + Int32(1) < full_block_cnt,
        )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(src_idx=n_block, producer_state=kv_producer_state)
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block, producer_state=kv_producer_state)
        kv_producer_state.advance()
    return kv_producer_state


@cute.jit
def produce_arbitrary_forward_overlap(
    partial_block_cnt,
    partial_block_idx: cute.Tensor,
    full_block_cnt,
    full_block_idx: cute.Tensor,
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    payload_words: cutlass.Constexpr[int],
    kv_producer_state,
    load_K: Callable,
    load_V: Callable,
    pipeline_k,
    pipeline_v,
    o_empty_mbar_ptr: Optional[cute.Pointer] = None,
    o_empty_phase: Optional[Int32] = None,
):
    """Produce anchored partial/full CSR loops with overlapped K/V loads."""
    total_block_cnt = partial_block_cnt + full_block_cnt
    n_block_prev = Int32(0)
    if total_block_cnt > Int32(0):
        # The planner guarantees at least one partial anchor for every nonempty row.
        partial_list_idx = partial_block_cnt - Int32(1)
        n_block_prev = partial_block_idx[partial_list_idx]
        payload_idx = partial_payload_base + partial_list_idx
        prefetch_arbitrary_forward_mask(mask_payloads, payload_idx, True, payload_words)
        prefetch_arbitrary_forward_mask(
            mask_payloads,
            payload_idx - Int32(1),
            partial_block_cnt > Int32(1),
            payload_words,
        )
        prefetch_arbitrary_forward_block_index(
            partial_block_idx,
            partial_list_idx - Int32(1),
            partial_block_cnt > Int32(1),
        )
        prefetch_arbitrary_forward_block_index(
            full_block_idx,
            full_block_cnt - Int32(1),
            full_block_cnt > Int32(0),
        )
        pipeline_k.producer_acquire(kv_producer_state)
        load_K(src_idx=n_block_prev, producer_state=kv_producer_state)

    # K uses independent shared storage. Issue the first K TMA before waiting
    # for the previous O epilogue to release the aliased V/O buffer.
    if const_expr(o_empty_mbar_ptr is not None):
        assert o_empty_phase is not None
        cute.arch.mbarrier_wait(o_empty_mbar_ptr, phase=o_empty_phase)

    if total_block_cnt > Int32(0):
        for iteration in cutlass.range(1, partial_block_cnt, unroll=1):
            partial_list_idx = partial_block_cnt - Int32(1) - iteration
            n_block = partial_block_idx[partial_list_idx]
            payload_idx = partial_payload_base + partial_list_idx
            prefetch_arbitrary_forward_mask(
                mask_payloads,
                payload_idx - Int32(1),
                iteration + Int32(1) < partial_block_cnt,
                payload_words,
            )
            prefetch_arbitrary_forward_block_index(
                partial_block_idx,
                partial_list_idx - Int32(1),
                iteration + Int32(1) < partial_block_cnt,
            )
            kv_producer_state_prev = kv_producer_state.clone()
            kv_producer_state.advance()
            pipeline_k.producer_acquire(kv_producer_state)
            load_K(src_idx=n_block, producer_state=kv_producer_state)
            pipeline_v.producer_acquire(kv_producer_state_prev)
            load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)
            n_block_prev = n_block

        for iteration in cutlass.range(full_block_cnt, unroll=1):
            full_list_idx = full_block_cnt - Int32(1) - iteration
            n_block = full_block_idx[full_list_idx]
            prefetch_arbitrary_forward_block_index(
                full_block_idx,
                full_list_idx - Int32(1),
                iteration + Int32(1) < full_block_cnt,
            )
            kv_producer_state_prev = kv_producer_state.clone()
            kv_producer_state.advance()
            pipeline_k.producer_acquire(kv_producer_state)
            load_K(src_idx=n_block, producer_state=kv_producer_state)
            pipeline_v.producer_acquire(kv_producer_state_prev)
            load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)
            n_block_prev = n_block

    if total_block_cnt > Int32(0):
        pipeline_v.producer_acquire(kv_producer_state)
        load_V(src_idx=n_block_prev, producer_state=kv_producer_state)
        kv_producer_state.advance()
    return kv_producer_state


@cute.jit
def consume_arbitrary_forward_nonoverlap(
    partial_block_cnt,
    partial_block_idx: Optional[cute.Tensor],
    full_block_cnt,
    full_block_idx: Optional[cute.Tensor],
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    kv_consumer_state,
    mma_pv_fn: Callable,
    mma_one_n_block: Callable,
    base_mask_fn: Callable,
    payload_group_idx: Int32,
    payload_words: cutlass.Constexpr[int],
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
):
    """Consume anchored partial/full CSR loops without K/V overlap."""
    total_block_cnt = partial_block_cnt + full_block_cnt
    processed_any = total_block_cnt > Int32(0)
    if processed_any:
        warp_scheduler_barrier_sync()
        partial_list_idx = partial_block_cnt - Int32(1)
        payload_idx = partial_payload_base + partial_list_idx
        n_block = Int32(0)
        if const_expr(partial_block_idx is not None):
            n_block = partial_block_idx[partial_list_idx]
        kv_consumer_state = mma_one_n_block(
            kv_consumer_state,
            n_block=n_block,
            mma_pv_fn=partial(mma_pv_fn, zero_init=True),
            mask_fn=partial(
                apply_arbitrary_forward_mask,
                base_mask_fn=base_mask_fn,
                mask_payloads=mask_payloads,
                payload_idx=payload_idx,
                payload_group_idx=payload_group_idx,
                payload_words=payload_words,
            ),
            is_first_n_block=True,
        )
        for iteration in cutlass.range(1, partial_block_cnt, unroll=1):
            partial_list_idx = partial_block_cnt - Int32(1) - iteration
            payload_idx = partial_payload_base + partial_list_idx
            n_block = Int32(0)
            if const_expr(partial_block_idx is not None):
                n_block = partial_block_idx[partial_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                mask_fn=partial(
                    apply_arbitrary_forward_mask,
                    base_mask_fn=base_mask_fn,
                    mask_payloads=mask_payloads,
                    payload_idx=payload_idx,
                    payload_group_idx=payload_group_idx,
                    payload_words=payload_words,
                ),
                is_first_n_block=False,
            )
        for iteration in cutlass.range(full_block_cnt, unroll=1):
            full_list_idx = full_block_cnt - Int32(1) - iteration
            n_block = Int32(0)
            if const_expr(full_block_idx is not None):
                n_block = full_block_idx[full_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=False),
                mask_fn=None,
                is_first_n_block=False,
            )
        warp_scheduler_barrier_arrive()
    return kv_consumer_state, processed_any


@cute.jit
def consume_arbitrary_forward_overlap(
    partial_block_cnt,
    partial_block_idx: Optional[cute.Tensor],
    full_block_cnt,
    full_block_idx: Optional[cute.Tensor],
    partial_payload_base: Int32,
    mask_payloads: cute.Tensor,
    seqlen_info,
    kv_consumer_state,
    mma_pv_fn: Callable,
    mma_one_n_block: Callable,
    process_first_half_block: Callable,
    process_last_half_block: Callable,
    base_mask_fn: Callable,
    score_mod_fn: Optional[Callable],
    payload_group_idx: Int32,
    payload_words: cutlass.Constexpr[int],
):
    """Consume anchored partial/full CSR loops with K/V overlap."""
    processed_any = partial_block_cnt + full_block_cnt > Int32(0)
    O_should_accumulate = False
    if processed_any:
        partial_list_idx = partial_block_cnt - Int32(1)
        payload_idx = partial_payload_base + partial_list_idx
        n_block = Int32(0)
        if const_expr(partial_block_idx is not None):
            n_block = partial_block_idx[partial_list_idx]
        kv_consumer_state = process_first_half_block(
            n_block=n_block,
            seqlen=seqlen_info,
            kv_consumer_state=kv_consumer_state,
            mask_fn=partial(
                apply_arbitrary_forward_mask,
                base_mask_fn=base_mask_fn,
                mask_payloads=mask_payloads,
                payload_idx=payload_idx,
                payload_group_idx=payload_group_idx,
                payload_words=payload_words,
            ),
            mask_prefetch_fn=partial(
                load_packed_mask_payload,
                mask_payloads,
                payload_idx,
                payload_group_idx,
                payload_words=payload_words,
            ),
            score_mod_fn=score_mod_fn,
            is_first_block=True,
        )
        for iteration in cutlass.range(1, partial_block_cnt, unroll=1):
            partial_list_idx = partial_block_cnt - Int32(1) - iteration
            payload_idx = partial_payload_base + partial_list_idx
            n_block = Int32(0)
            if const_expr(partial_block_idx is not None):
                n_block = partial_block_idx[partial_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                seqlen=seqlen_info,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    apply_arbitrary_forward_mask,
                    base_mask_fn=base_mask_fn,
                    mask_payloads=mask_payloads,
                    payload_idx=payload_idx,
                    payload_group_idx=payload_group_idx,
                    payload_words=payload_words,
                ),
                mask_prefetch_fn=partial(
                    load_packed_mask_payload,
                    mask_payloads,
                    payload_idx,
                    payload_group_idx,
                    payload_words=payload_words,
                ),
            )
            O_should_accumulate = True

        for iteration in cutlass.range(full_block_cnt, unroll=1):
            full_list_idx = full_block_cnt - Int32(1) - iteration
            n_block = Int32(0)
            if const_expr(full_block_idx is not None):
                n_block = full_block_idx[full_list_idx]
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=n_block,
                seqlen=seqlen_info,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=None,
                mask_prefetch_fn=None,
            )
            O_should_accumulate = True

        kv_consumer_state = process_last_half_block(
            kv_consumer_state=kv_consumer_state,
            zero_init=not O_should_accumulate,
        )
        O_should_accumulate = True
    return kv_consumer_state, O_should_accumulate, processed_any


@cute.jit
def produce_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info: SeqlenInfoQK,
    kv_producer_state,
    load_K,
    load_V,
    pipeline_k,
    pipeline_v,
    intra_wg_overlap: cutlass.Constexpr,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
    o_empty_mbar_ptr: Optional[cute.Pointer] = None,
    o_empty_phase: Optional[Int32] = None,
):
    """Iterate over the mask and full block lists for a single tile.

    Q is loaded separately on its own mbarrier before this function is called.

    The masked (partial) list may leave the last V load pending when intra-warp-group
    overlap is enabled. The first full block must consume that pending V while
    issuing its own K load on the next pipeline stage.

    In the intra-wg-overlap path, the last masked block leaves its V copy in flight
    while we advance the producer state to start the next full K. Either the full list
    overlaps that pending V load, or, if no full blocks exist, we explicitly drain it.

    Args:
        qhead_per_kvhead: Pack-GQA factor. When > 1, m_block is in packed space and
            must be converted to unpacked for sparse tensor indexing.
    """
    if const_expr(blocksparse_tensors.mask_block_masks is not None):
        (
            curr_mask_block_cnt,
            curr_mask_block_idx,
            curr_full_block_cnt,
            curr_full_block_idx,
            mask_payload_base,
        ) = get_curr_arbitrary_blocksparse_tensors(
            batch_idx,
            head_idx,
            m_block,
            blocksparse_tensors,
            seqlen_info,
            tile_m,
            qhead_per_kvhead,
        )
    else:
        m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
        (
            curr_mask_block_cnt,
            curr_mask_block_idx,
            curr_full_block_cnt,
            curr_full_block_idx,
        ) = get_curr_blocksparse_tensors(
            batch_idx,
            head_idx,
            m_block_sparse,
            blocksparse_tensors,
            seqlen_info,
        )
        mask_payload_base = Int32(0)

    if const_expr(blocksparse_tensors.mask_block_masks is not None):
        if const_expr(not intra_wg_overlap):
            return produce_arbitrary_forward_nonoverlap(
                curr_mask_block_cnt,
                curr_mask_block_idx,
                curr_full_block_cnt,
                curr_full_block_idx,
                mask_payload_base,
                blocksparse_tensors.mask_block_masks,
                blocksparse_tensors.mask_block_masks.shape[3],
                kv_producer_state,
                load_K,
                load_V,
                pipeline_k,
                pipeline_v,
                o_empty_mbar_ptr,
                o_empty_phase,
            )
        else:
            return produce_arbitrary_forward_overlap(
                curr_mask_block_cnt,
                curr_mask_block_idx,
                curr_full_block_cnt,
                curr_full_block_idx,
                mask_payload_base,
                blocksparse_tensors.mask_block_masks,
                blocksparse_tensors.mask_block_masks.shape[3],
                kv_producer_state,
                load_K,
                load_V,
                pipeline_k,
                pipeline_v,
                o_empty_mbar_ptr,
                o_empty_phase,
            )

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0

    first_block_preloaded = cutlass.const_expr(intra_wg_overlap and o_empty_mbar_ptr is not None)
    if const_expr(first_block_preloaded):
        # The sparse row metadata is available before V/O shared storage is
        # reusable. Start the first K TMA so its latency overlaps the epilogue.
        if mask_empty:
            if curr_full_block_cnt > 0:
                n_block_first = curr_full_block_idx[curr_full_block_cnt - 1]
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_first, producer_state=kv_producer_state)
        else:
            n_block_first = curr_mask_block_idx[curr_mask_block_cnt - 1]
            pipeline_k.producer_acquire(kv_producer_state)
            load_K(src_idx=n_block_first, producer_state=kv_producer_state)

    if const_expr(o_empty_mbar_ptr is not None):
        assert o_empty_phase is not None
        cute.arch.mbarrier_wait(o_empty_mbar_ptr, phase=o_empty_phase)

    if mask_empty:
        # No masked blocks: the full list owns the initial K load.
        kv_producer_state = load_block_list(
            curr_full_block_idx,
            curr_full_block_cnt,
            first_block_preloaded=first_block_preloaded,
            kv_producer_state=kv_producer_state,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            intra_wg_overlap=intra_wg_overlap,
        )

        if const_expr(intra_wg_overlap) and curr_full_block_cnt > 0:
            kv_producer_state = finish_overlap_v_load(
                curr_full_block_idx,
                curr_full_block_cnt,
                load_V,
                pipeline_v,
                kv_producer_state,
            )
    else:
        # Masked blocks present. When overlap is disabled this fully drains the list.
        kv_producer_state = load_block_list(
            curr_mask_block_idx,
            curr_mask_block_cnt,
            first_block_preloaded=first_block_preloaded,
            kv_producer_state=kv_producer_state,
            load_K=load_K,
            load_V=load_V,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            intra_wg_overlap=intra_wg_overlap,
        )

        if full_empty:
            if const_expr(intra_wg_overlap):
                kv_producer_state = finish_overlap_v_load(
                    curr_mask_block_idx,
                    curr_mask_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
        else:
            if const_expr(intra_wg_overlap):
                # Bridge the masked list to the full list by overlapping the pending masked V
                # with the first full K load.
                n_block_mask_last = curr_mask_block_idx[0]
                n_block_full_first = curr_full_block_idx[curr_full_block_cnt - 1]
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_full_first, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_mask_last, producer_state=kv_producer_state_prev)

                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    first_block_preloaded=True,
                    kv_producer_state=kv_producer_state,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    intra_wg_overlap=intra_wg_overlap,
                )

                kv_producer_state = finish_overlap_v_load(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    load_V,
                    pipeline_v,
                    kv_producer_state,
                )
            else:
                # Non-overlap path with both lists: run the full list normally.
                kv_producer_state = load_block_list(
                    curr_full_block_idx,
                    curr_full_block_cnt,
                    first_block_preloaded=False,
                    kv_producer_state=kv_producer_state,
                    load_K=load_K,
                    load_V=load_V,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    intra_wg_overlap=intra_wg_overlap,
                )

    return kv_producer_state


@cute.jit
def consume_block_sparse_loads(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info,
    kv_consumer_state,
    mma_pv_fn,
    mma_one_n_block,
    process_first_half_block,
    process_last_half_block,
    mask_fn,
    score_mod_fn,
    O_should_accumulate,
    mask_mod,
    fastdiv_mods,
    intra_wg_overlap: cutlass.Constexpr,
    warp_scheduler_barrier_sync: Callable,
    warp_scheduler_barrier_arrive: Callable,
    tile_m: cutlass.Constexpr[int],
    consumer_tidx: Int32,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
    payload_words: cutlass.Constexpr[int] = 4,
):
    """Consume the mask and full block lists for a single tile on the consumer side.

    Mirrors `produce_block_sparse_loads` so that the consumer pipeline uses
    the same sparse tensor indexing.

    Args:
        qhead_per_kvhead: Pack-GQA factor. When > 1, m_block is in packed space and
            must be converted to unpacked for sparse tensor indexing.
    """
    mask_payloads = blocksparse_tensors.mask_block_masks
    if const_expr(mask_payloads is not None):
        # Packed payloads are self-contained for arbitrary-only attention.
        # Only score_mod consumers need CSR indices to reconstruct logical K coordinates.
        if const_expr(score_mod_fn is not None):
            (
                curr_mask_block_cnt,
                curr_mask_block_idx,
                curr_full_block_cnt,
                curr_full_block_idx,
                mask_payload_base,
            ) = get_curr_arbitrary_blocksparse_tensors(
                batch_idx,
                head_idx,
                m_block,
                blocksparse_tensors,
                seqlen_info,
                tile_m,
                qhead_per_kvhead,
            )
        else:
            (
                curr_mask_block_cnt,
                curr_full_block_cnt,
                mask_payload_base,
            ) = get_curr_arbitrary_block_counts_fwd_sm90(
                batch_idx,
                head_idx,
                m_block,
                blocksparse_tensors,
                seqlen_info,
                tile_m,
                qhead_per_kvhead,
            )
            curr_mask_block_idx = None
            curr_full_block_idx = None
    else:
        m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
        (
            curr_mask_block_cnt,
            curr_mask_block_idx,
            curr_full_block_cnt,
            curr_full_block_idx,
        ) = get_curr_blocksparse_tensors(
            batch_idx,
            head_idx,
            m_block_sparse,
            blocksparse_tensors,
            seqlen_info,
        )
        mask_payload_base = Int32(0)
    processed_any = curr_mask_block_cnt + curr_full_block_cnt > 0

    if const_expr(mask_payloads is not None):
        payload_group_idx = _sm90_fwd_mask_payload_group_idx(
            consumer_tidx,
            qhead_per_kvhead,
        )
        if const_expr(not intra_wg_overlap):
            kv_consumer_state, processed_any = consume_arbitrary_forward_nonoverlap(
                curr_mask_block_cnt,
                curr_mask_block_idx,
                curr_full_block_cnt,
                curr_full_block_idx,
                mask_payload_base,
                mask_payloads,
                kv_consumer_state,
                mma_pv_fn,
                mma_one_n_block,
                mask_fn,
                payload_group_idx,
                payload_words,
                warp_scheduler_barrier_sync,
                warp_scheduler_barrier_arrive,
            )
            return kv_consumer_state, processed_any, processed_any
        return consume_arbitrary_forward_overlap(
            curr_mask_block_cnt,
            curr_mask_block_idx,
            curr_full_block_cnt,
            curr_full_block_idx,
            mask_payload_base,
            mask_payloads,
            seqlen_info,
            kv_consumer_state,
            mma_pv_fn,
            mma_one_n_block,
            process_first_half_block,
            process_last_half_block,
            mask_fn,
            score_mod_fn,
            payload_group_idx,
            payload_words,
        )

    if const_expr(not intra_wg_overlap):
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            warp_scheduler_barrier_sync()
            kv_consumer_state = mma_one_n_block(
                kv_consumer_state,
                n_block=mask_n_block,
                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=(
                        fastdiv_mods if cutlass.const_expr(mask_mod is not None) else None
                    ),
                ),
                is_first_n_block=True,
            )
            O_should_accumulate = True
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
            if curr_full_block_cnt == 0:
                warp_scheduler_barrier_arrive()

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                warp_scheduler_barrier_sync()
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first_n_block=True,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    is_first_n_block=False,
                )
                O_should_accumulate = True
                for i in cutlass.range(1, curr_full_block_cnt):
                    full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=full_n_block,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                        is_first_n_block=False,
                    )
                    O_should_accumulate = True
            warp_scheduler_barrier_arrive()
    else:
        if curr_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1]
            kv_consumer_state = process_first_half_block(
                n_block=mask_n_block,
                seqlen=seqlen_info,
                kv_consumer_state=kv_consumer_state,
                mask_fn=partial(
                    mask_fn,
                    mask_mod=mask_mod,
                    mask_seqlen=True,
                    fastdiv_mods=(
                        fastdiv_mods if cutlass.const_expr(mask_mod is not None) else None
                    ),
                ),
                score_mod_fn=score_mod_fn,
                is_first_block=True,
            )
            for i in cutlass.range(1, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[curr_mask_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=mask_n_block,
                    seqlen=seqlen_info,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=mask_mod, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[curr_full_block_cnt - 1]
            if curr_mask_block_cnt == 0:
                kv_consumer_state = process_first_half_block(
                    n_block=full_n_block,
                    seqlen=seqlen_info,
                    kv_consumer_state=kv_consumer_state,
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                    score_mod_fn=score_mod_fn,
                    is_first_block=True,
                )
            else:
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen_info,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),
                )
                O_should_accumulate = True
            for i in cutlass.range(1, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[curr_full_block_cnt - 1 - i]
                kv_consumer_state = mma_one_n_block(
                    kv_consumer_state,
                    n_block=full_n_block,
                    seqlen=seqlen_info,
                    mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                    mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),
                )
                O_should_accumulate = True

        if curr_mask_block_cnt + curr_full_block_cnt > 0:
            kv_consumer_state = process_last_half_block(
                kv_consumer_state=kv_consumer_state,
                zero_init=not O_should_accumulate,
            )
            O_should_accumulate = True

    return kv_consumer_state, O_should_accumulate, processed_any


@cute.jit
def split_block_range(block_count, split_idx: Int32, num_splits: Int32):
    """Return the half-open block-list range assigned to one SplitKV partition."""
    blocks_per_split = cute.ceil_div(block_count, num_splits)
    block_begin = cutlass.min(split_idx * blocks_per_split, block_count)
    block_end = cutlass.min(block_begin + blocks_per_split, block_count)
    return block_begin, block_end


@cute.jit
def load_block_list_sm100(
    block_indices: cute.Tensor,
    block_begin,
    block_end,
    load_q_with_first: cutlass.Constexpr,
    q_stage: cutlass.Constexpr,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
):
    """SM100 version of load_block_list (no intra_wg_overlap, no extra_tx_count)."""
    block_count = block_end - block_begin
    if block_count > 0:
        # First iteration: load Q alongside K if requested
        n_block_first = block_indices[block_end - 1]

        if const_expr(load_q_with_first):
            # SM100 loads Q0 and optionally Q1
            load_Q(block=0, stage=0)
            if const_expr(q_stage == 2):
                load_Q(block=1, stage=1)

        # SM100 doesn't use producer_acquire for pipeline_kv in load path
        # The pipeline barriers are handled inside load_KV
        load_K(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()
        load_V(block=n_block_first, producer_state=kv_producer_state, page_idx=None)
        kv_producer_state.advance()

        # Remaining blocks
        for offset in cutlass.range(1, block_count):
            n_block = block_indices[block_end - 1 - offset]
            load_K(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()
            load_V(block=n_block, producer_state=kv_producer_state, page_idx=None)
            kv_producer_state.advance()

    return kv_producer_state


# SM100-specific tile processor using SM100 helpers
@cute.jit
def produce_block_sparse_loads_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info: SeqlenInfoQK,
    split_idx: Int32,
    num_splits: Int32,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    """SM100 entry point for sparse block iteration.

    SM100 uses PipelineTmaUmma which doesn't support extra_tx_count, so we use
    simplified block processing that just calls producer_acquire without extras.

    Args:
        m_block: which tile of m we are processing
        qhead_per_kvhead: Constexpr pack factor
    """
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    mask_begin, mask_end = split_block_range(curr_mask_block_cnt, split_idx, num_splits)
    full_begin, full_end = split_block_range(curr_full_block_cnt, split_idx, num_splits)
    mask_empty = mask_begin == mask_end
    full_empty = full_begin == full_end

    q_phase_flipped = False

    if mask_empty:
        # No masked blocks: process full list with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            full_begin,
            full_end,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        # Process masked blocks with Q loading
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            mask_begin,
            mask_end,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            # Process full blocks without Q loading
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                full_begin,
                full_end,
                load_q_with_first=False,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def _get_curr_blocksparse_tensors_linear_raw(
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    m_block: cutlass.Int32,
    blocksparse_tensors: BlockSparseTensors,
):
    """Fixed-length CSR path with raw compact offsets."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    assert mask_block_offset is not None

    batch, nheads, n_blocks = mask_block_cnt.shape
    sparse_batch_idx = 0 if batch == 1 else batch_idx
    sparse_head_idx = 0 if nheads == 1 else head_idx
    offset_idx = (
        (0 if batch == 1 else batch_idx * nheads * n_blocks)
        + (0 if nheads == 1 else head_idx * n_blocks)
        + m_block
    )

    curr_mask_block_cnt = mask_block_cnt[sparse_batch_idx, sparse_head_idx, m_block]
    curr_mask_block_offset = mask_block_offset[offset_idx]

    if const_expr(full_block_cnt is not None):
        assert full_block_offset is not None
        assert full_block_idx is not None
        curr_full_block_cnt = full_block_cnt[sparse_batch_idx, sparse_head_idx, m_block]
        curr_full_block_offset = full_block_offset[offset_idx]
    else:
        curr_full_block_cnt = Int32(0)
        curr_full_block_offset = Int32(0)
        full_block_idx = None

    return (
        curr_mask_block_cnt,
        curr_mask_block_offset,
        mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_offset,
        full_block_idx,
    )


@cute.jit
def produce_block_sparse_loads_sm100_linear(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    kv_producer_state,
    load_Q,
    load_K,
    load_V,
    pipeline_kv,
    q_stage: cutlass.Constexpr,
    q_producer_phase: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    """SM100 non-SplitKV load path for fixed-length linear CSR block sparsity."""
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_offset,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_offset,
        curr_full_block_idx,
    ) = _get_curr_blocksparse_tensors_linear_raw(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
    )

    mask_empty = curr_mask_block_cnt == 0
    full_empty = curr_full_block_cnt == 0
    q_phase_flipped = False

    if mask_empty:
        kv_producer_state = load_block_list_sm100(
            curr_full_block_idx,
            curr_full_block_offset,
            curr_full_block_offset + curr_full_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = not full_empty
    else:
        kv_producer_state = load_block_list_sm100(
            curr_mask_block_idx,
            curr_mask_block_offset,
            curr_mask_block_offset + curr_mask_block_cnt,
            load_q_with_first=True,
            q_stage=q_stage,
            kv_producer_state=kv_producer_state,
            load_Q=load_Q,
            load_K=load_K,
            load_V=load_V,
            pipeline_kv=pipeline_kv,
        )
        q_phase_flipped = True

        if not full_empty:
            kv_producer_state = load_block_list_sm100(
                curr_full_block_idx,
                curr_full_block_offset,
                curr_full_block_offset + curr_full_block_cnt,
                load_q_with_first=False,
                q_stage=q_stage,
                kv_producer_state=kv_producer_state,
                load_Q=load_Q,
                load_K=load_K,
                load_V=load_V,
                pipeline_kv=pipeline_kv,
            )

    if q_phase_flipped:
        q_producer_phase ^= 1

    return kv_producer_state, q_producer_phase


@cute.jit
def get_total_block_count_linear_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    full_block_cnt = blocksparse_tensors.full_block_cnt
    batch, nheads, _ = mask_block_cnt.shape
    sparse_batch_idx = 0 if batch == 1 else batch_idx
    sparse_head_idx = 0 if nheads == 1 else head_idx

    total = mask_block_cnt[sparse_batch_idx, sparse_head_idx, m_block_sparse]
    if const_expr(full_block_cnt is not None):
        total = total + full_block_cnt[sparse_batch_idx, sparse_head_idx, m_block_sparse]
    return total


@cute.jit
def get_total_block_count(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    split_idx: Int32,
    num_splits: Int32,
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr,
    seqlen_info: SeqlenInfoQK,
):
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)
    (
        curr_mask_block_cnt,
        _,
        curr_full_block_cnt,
        _,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    mask_begin, mask_end = split_block_range(curr_mask_block_cnt, split_idx, num_splits)
    full_begin, full_end = split_block_range(curr_full_block_cnt, split_idx, num_splits)
    return mask_end - mask_begin + full_end - full_begin


@cute.jit
def handle_block_sparse_empty_tile_correction_sm100(
    tidx: Int32,
    q_stage: cutlass.Constexpr,
    m_block_size: cutlass.Constexpr,
    qhead_per_kvhead,
    pack_gqa: cutlass.Constexpr,
    is_split_kv: cutlass.Constexpr,
    learnable_sink,
    mLSE,
    seqlen_info,
    m_block: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    split_idx: Int32,
    sScale: cute.Tensor,
    stats: list,
    correction_epilogue: Callable,
    thr_mma_pv: cute.ThrMma,
    tOtO: cute.Tensor,
    sO: cute.Tensor,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    pipeline_o_epi: cutlass.pipeline.PipelineAsync,
    sm_stats_consumer_phase: Int32,
    o_corr_consumer_phase: Int32,
    corr_epi_producer_phase: Int32,
    softmax_scale_log2: Float32,
    max_offset: Float32,
    max_offset_scale: Float32,
    mO_cur: Optional[cute.Tensor] = None,
    gO: Optional[cute.Tensor] = None,
    gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
):
    """Handle SM100 forward block-sparse tiles with no active KV blocks.

    This path is taken when `total_block_cnt == 0`. The softmax warp-group still
    arrives `mbar_softmax_corr_full` (synthetic "no work") so the correction
    warp-group can:

    - seed fully-masked-row stats (row_sum=1; row_max=-inf when tracked) for LSE
    - run `correction_epilogue` with `scale=0` so the output tile is written as zeros
      (independent of any prior tmem contents)
    - wait on `mbar_softmax_corr_full` and arrive `mbar_softmax_corr_empty`
      (and `mbar_corr_epi_*` when applicable) so phases stay aligned across tiles

    This helper intentionally does not touch `mbar_P_full_*` since no P is produced.
    See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
    """
    LOG2_E = Float32(math.log2(math.e))
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

    for stage in cutlass.range_constexpr(q_stage):
        row_sum_value = Float32(1.0)
        row_max_value = (
            -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None
        )
        if const_expr(learnable_sink is not None):
            sink_val = -Float32.inf
            if const_expr(not pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            elif tidx < m_block_size:
                q_head_idx = (
                    (q_stage * m_block + stage) * m_block_size + tidx
                ) % qhead_per_kvhead + head_idx * qhead_per_kvhead
                sink_val = Float32(learnable_sink[q_head_idx])
            if sink_val != -Float32.inf and (const_expr(not is_split_kv) or split_idx == 0):
                if row_max_value == -Float32.inf:
                    row_max_value = sink_val * (LOG2_E / softmax_scale_log2)
                    row_sum_value = max_offset_scale
                else:
                    row_sum_value = row_sum_value + cute.math.exp2(
                        sink_val * LOG2_E - row_max_value * softmax_scale_log2 + max_offset,
                        fastmath=True,
                    )
        if tidx < m_block_size:
            scale_row_idx = tidx + stage * m_block_size
            sScale[scale_row_idx] = row_sum_value
            if const_expr(mLSE is not None or learnable_sink is not None):
                sScale[scale_row_idx + q_stage * m_block_size] = row_max_value
        acc_flag = row_sum_value == Float32(0.0) or row_sum_value != row_sum_value
        stats[stage] = (row_sum_value, row_max_value, acc_flag)

        # See NOTE [SM100 block-sparse empty tiles: mbarrier contract].
        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
        pipeline_sm_stats.consumer_release_w_index(stage)

        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)

        gO_stage = gO[None, None, stage] if const_expr(gO is not None) else None
        correction_epilogue(
            thr_mma_pv,
            tOtO[None, None, None, stage],
            tidx,
            stage,
            m_block,
            seqlen_info.seqlen_q,
            Float32(0.0),  # zero scale ensures empty tile writes zeros into staged outputs
            sO[None, None, stage],
            mO_cur,
            gO_stage,
            gmem_tiled_copy_O,
        )
        if const_expr(gmem_tiled_copy_O is None):
            pipeline_o_epi.producer_commit_w_index(stage)

    sm_stats_consumer_phase ^= 1
    corr_epi_producer_phase ^= 1

    return (
        sm_stats_consumer_phase,
        o_corr_consumer_phase,
        corr_epi_producer_phase,
    )


@cute.jit
def softmax_block_sparse_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    seqlen_info: SeqlenInfoQK,
    split_idx: Int32,
    num_splits: Int32,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    is_arbitrary: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_idx,
    ) = get_curr_blocksparse_tensors(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
        seqlen_info,
    )

    mask_begin, mask_end = split_block_range(curr_mask_block_cnt, split_idx, num_splits)
    full_begin, full_end = split_block_range(curr_full_block_cnt, split_idx, num_splits)
    split_mask_block_cnt = mask_end - mask_begin
    split_full_block_cnt = full_end - full_begin
    total_block_cnt = split_mask_block_cnt + split_full_block_cnt

    if total_block_cnt == 0:
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)
    else:
        if split_mask_block_cnt > 0:
            mask_n_block = curr_mask_block_idx[mask_end - 1]
            (
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
            ) = softmax_step(
                mma_si_consumer_phase,
                si_corr_producer_phase,
                s0_s1_sequence_phase,
                mask_n_block,
                is_first=True,
                mask_fn=partial(
                    mask_fn,
                    mask_seqlen=not is_arbitrary,
                    check_q_boundary=check_m_boundary,
                ),
            )
            for i in cutlass.range(1, split_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[mask_end - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False, check_q_boundary=check_m_boundary),
                )

        if split_full_block_cnt > 0:
            full_n_block = curr_full_block_idx[full_end - 1]
            if split_mask_block_cnt == 0:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=True,
                    mask_fn=None,
                )
            else:
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    is_first=False,
                    mask_fn=None,
                )
            for i in cutlass.range(1, split_full_block_cnt):
                full_n_block = curr_full_block_idx[full_end - 1 - i]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=None,
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )


@cute.jit
def softmax_block_sparse_sm100_linear(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    m_block,
    softmax_step: Callable,
    mask_fn: Callable,
    mask_fn_none: Callable,
    mma_si_consumer_phase: Int32,
    si_corr_producer_phase: Int32,
    s0_s1_sequence_phase: Int32,
    pipeline_sm_stats: cutlass.pipeline.PipelineAsync,
    sm_stats_barrier: cutlass.pipeline.NamedBarrier,
    q_stage: cutlass.Constexpr,
    stage_idx: Int32,
    check_m_boundary: bool,
    is_arbitrary: cutlass.Constexpr[bool],
    qhead_per_kvhead: cutlass.Constexpr,
    q_subtile_factor: cutlass.Constexpr[int] = 1,
):
    """SM100 non-SplitKV softmax path for fixed-length linear CSR block sparsity."""
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
    m_block_sparse = sparse_tensor_m_block(m_block, qhead_per_kvhead, q_subtile_factor)

    (
        curr_mask_block_cnt,
        curr_mask_block_offset,
        curr_mask_block_idx,
        curr_full_block_cnt,
        curr_full_block_offset,
        curr_full_block_idx,
    ) = _get_curr_blocksparse_tensors_linear_raw(
        batch_idx,
        head_idx,
        m_block_sparse,
        blocksparse_tensors,
    )
    total_block_cnt = curr_mask_block_cnt + curr_full_block_cnt

    if total_block_cnt == 0:
        sm_stats_barrier.arrive_w_index(index=stage_idx * 4 + warp_idx)
    else:
        if curr_mask_block_cnt > 0:
            for i in cutlass.range(0, curr_mask_block_cnt):
                mask_n_block = curr_mask_block_idx[
                    curr_mask_block_offset + curr_mask_block_cnt - 1 - i
                ]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    mask_n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=False),
                )

        if curr_full_block_cnt > 0:
            for i in cutlass.range(0, curr_full_block_cnt):
                full_n_block = curr_full_block_idx[
                    curr_full_block_offset + curr_full_block_cnt - 1 - i
                ]
                (
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                ) = softmax_step(
                    mma_si_consumer_phase,
                    si_corr_producer_phase,
                    s0_s1_sequence_phase,
                    full_n_block,
                    mask_fn=None,
                )

    return (
        mma_si_consumer_phase,
        si_corr_producer_phase,
        s0_s1_sequence_phase,
        total_block_cnt == 0,
    )


# =============================================================================
# Backward-specific block-sparse helpers (SM100)
# =============================================================================
#
# In backward, iteration is transposed compared to forward:
# - Forward: outer loop over m_blocks (Q tiles), inner loop over n_blocks (KV tiles)
# - Backward: outer loop over n_blocks (KV tiles), inner loop over m_blocks (Q tiles)
#
# The backward block-sparse tensors use "Q direction" indexing:
# - q_block_cnt[batch, head, n_block] → count of m_blocks to process for this KV tile
# - q_block_idx[batch, head, n_block, :] → indices of m_blocks to process
#


@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    n_blocks_per_sample: int = 0,
):
    """Count total tile iterations for given n_block (KV tile) in backward."""
    if const_expr(blocksparse_tensors.mask_block_masks is not None):
        curr_q_cnt, _, curr_full_cnt, _, _, _ = get_curr_arbitrary_blocksparse_tensors_bwd(
            batch_idx,
            head_idx,
            n_block,
            blocksparse_tensors,
            n_blocks_per_sample,
        )
    else:
        curr_q_cnt, _, curr_full_cnt, _ = get_curr_blocksparse_tensors_fixed(
            batch_idx, head_idx, n_block, blocksparse_tensors
        )
    total = curr_q_cnt + curr_full_cnt
    return total * subtile_factor


@cute.jit
def produce_block_sparse_q_loads_bwd_sm100(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    # Pipeline states (will be returned after advancing)
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    # Pipelines
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    # Load functions
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    # Global tensors for LSE/dPsum
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    # TMA copy bytes for extra_tx_count
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    # Flags for which loads to perform
    should_load_Q: cutlass.Constexpr,
    should_load_dO: cutlass.Constexpr,
    # Subtiling factor and bounds
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """SM100 backward block sparse loading with subtiling.

    Returns updated (producer_state_Q_LSE, producer_state_dO_dPsum).
    First iteration loads K/V alongside Q/dO; subsequent iterations load only Q/dO.
    """
    (
        curr_q_cnt,
        curr_q_idx,
        curr_full_cnt,
        curr_full_idx,
        loop_count,
    ) = get_block_sparse_iteration_info_bwd(
        blocksparse_tensors, batch_idx, head_idx, n_block, subtile_factor, m_block_max
    )

    split_sparse_blocks = const_expr(curr_full_idx is not None)
    block_group_count: cutlass.Constexpr[int] = 2 if const_expr(split_sparse_blocks) else 1
    for block_group in cutlass.range_constexpr(block_group_count):
        group_loop_count = loop_count
        iter_offset = Int32(0)
        if const_expr(split_sparse_blocks):
            group_loop_count = curr_q_cnt * subtile_factor
            if const_expr(block_group == 1):
                group_loop_count = curr_full_cnt * subtile_factor
                iter_offset = curr_q_cnt * subtile_factor

        for group_iter_idx in cutlass.range(group_loop_count, unroll=1):
            iter_idx = group_iter_idx + iter_offset
            if const_expr(split_sparse_blocks):
                sparse_iter_idx = group_iter_idx // subtile_factor
                subtile_offset = group_iter_idx % subtile_factor
                if const_expr(block_group == 0):
                    m_block = curr_q_idx[sparse_iter_idx] * subtile_factor + subtile_offset
                else:
                    assert curr_full_idx is not None
                    m_block = curr_full_idx[sparse_iter_idx] * subtile_factor + subtile_offset
            else:
                m_block, _ = get_m_block_from_iter_bwd(
                    iter_idx,
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    subtile_factor,
                    m_block_max,
                )
            m_block_safe = m_block
            if m_block_max > 0:
                m_block_safe = cutlass.min(m_block, m_block_max - 1)

            if iter_idx == 0:
                # First block: load K/V alongside Q/dO
                if const_expr(should_load_Q):
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=tma_copy_bytes_K
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block_safe],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum, extra_tx_count=tma_copy_bytes_V
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                    load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block_safe],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()
            else:
                # Subsequent blocks: just load Q/dO (K/V already loaded)
                if const_expr(should_load_Q):
                    pipeline_Q.producer_acquire(producer_state_Q_LSE)
                    load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block_safe],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    pipeline_dO.producer_acquire(producer_state_dO_dPsum)
                    load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block_safe],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()

    return producer_state_Q_LSE, producer_state_dO_dPsum


@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Extract block-sparse iteration info for backward pass.

    Returns (curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count).
    """
    curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx = get_curr_blocksparse_tensors_fixed(
        batch_idx, head_idx, n_block, blocksparse_tensors
    )

    sparse_block_count = curr_q_cnt
    if const_expr(curr_full_idx is not None):
        sparse_block_count = sparse_block_count + curr_full_cnt
    total_count = sparse_block_count * subtile_factor

    return curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count


@cute.jit
def get_curr_dq_write_order_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
):
    curr_dq_write_order = None
    curr_dq_write_order_full = None
    if const_expr(blocksparse_tensors.dq_write_order is not None):
        assert blocksparse_tensors.dq_write_order is not None
        if const_expr(len(blocksparse_tensors.mask_block_cnt.shape) == 2):
            mask_block_cnt = blocksparse_tensors.mask_block_cnt
            mask_block_offset = blocksparse_tensors.mask_block_offset
            cu_total_k_blocks = blocksparse_tensors.cu_total_m_blocks
            assert mask_block_offset is not None
            assert cu_total_k_blocks is not None
            plan_head = Int32(0) if mask_block_cnt.shape[0] == 1 else head_idx
            outer_row = cu_total_k_blocks[batch_idx] + n_block
            offset_idx = plan_head * mask_block_cnt.shape[1] + outer_row
            curr_dq_write_order = cute.domain_offset(
                mask_block_offset[offset_idx], blocksparse_tensors.dq_write_order
            )
            if const_expr(blocksparse_tensors.dq_write_order_full is not None):
                assert blocksparse_tensors.dq_write_order_full is not None
                full_block_offset = blocksparse_tensors.full_block_offset
                assert full_block_offset is not None
                curr_dq_write_order_full = cute.domain_offset(
                    full_block_offset[offset_idx], blocksparse_tensors.dq_write_order_full
                )
        elif const_expr(blocksparse_tensors.mask_block_offset is not None):
            mask_block_cnt = blocksparse_tensors.mask_block_cnt
            mask_block_offset = blocksparse_tensors.mask_block_offset
            assert mask_block_offset is not None
            batch, nheads, n_blocks = mask_block_cnt.shape
            offset_idx = (
                (0 if batch == 1 else batch_idx * nheads * n_blocks)
                + (0 if nheads == 1 else head_idx * n_blocks)
                + n_block
            )
            curr_dq_write_order = cute.domain_offset(
                mask_block_offset[offset_idx], blocksparse_tensors.dq_write_order
            )
            if const_expr(blocksparse_tensors.dq_write_order_full is not None):
                assert blocksparse_tensors.dq_write_order_full is not None
                full_block_offset = blocksparse_tensors.full_block_offset
                assert full_block_offset is not None
                curr_dq_write_order_full = cute.domain_offset(
                    full_block_offset[offset_idx], blocksparse_tensors.dq_write_order_full
                )
        else:
            curr_dq_write_order = blocksparse_tensors.dq_write_order[
                batch_idx, head_idx, n_block, None
            ]
            if const_expr(blocksparse_tensors.dq_write_order_full is not None):
                assert blocksparse_tensors.dq_write_order_full is not None
                curr_dq_write_order_full = blocksparse_tensors.dq_write_order_full[
                    batch_idx, head_idx, n_block, None
                ]
    return curr_dq_write_order, curr_dq_write_order_full


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: Optional[cute.Tensor],
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    full_first: cutlass.Constexpr[bool] = False,
):
    """Derive m_block index and is_full_block flag from iteration index.

    Returns (m_block, is_full_block):
        - m_block: The actual Q-tile block index
        - is_full_block: True if this is a full block (no mask_mod needed)
    """
    sparse_iter_idx = iter_idx // subtile_factor
    subtile_offset = iter_idx % subtile_factor

    sparse_m_block = Int32(0)
    is_full_block = False
    if const_expr(curr_full_idx is not None):
        if const_expr(full_first):
            if sparse_iter_idx < curr_full_cnt:
                sparse_m_block = curr_full_idx[sparse_iter_idx]
                is_full_block = True
            else:
                sparse_m_block = curr_q_idx[sparse_iter_idx - curr_full_cnt]
        else:
            if sparse_iter_idx < curr_q_cnt:
                sparse_m_block = curr_q_idx[sparse_iter_idx]
            else:
                sparse_m_block = curr_full_idx[sparse_iter_idx - curr_q_cnt]
                is_full_block = True
    else:
        sparse_m_block = curr_q_idx[sparse_iter_idx]

    return sparse_m_block * subtile_factor + subtile_offset, is_full_block


@cute.jit
def get_physical_subtile_count_bwd_sm90(
    sparse_block_count,
    sparse_block_indices: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
):
    """Return the exact physical-Q iteration count for one sorted K2Q list."""
    if const_expr(subtile_factor == 1):
        return sparse_block_count
    physical_count = sparse_block_count * subtile_factor
    if sparse_block_count > Int32(0):
        last_sparse_m_block = sparse_block_indices[sparse_block_count - Int32(1)]
        tail_excess = (last_sparse_m_block + Int32(1)) * subtile_factor - m_block_max
        if tail_excess > Int32(0):
            physical_count -= tail_excess
    return physical_count


@cute.jit
def _load_q_do_block_sm90(
    m_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    sQ_block_metadata: Optional[cute.Tensor],
    producer_tidx: Int32,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    load_kv: bool,
):
    """Load one Q/dO block, optionally loading K/V on first iteration."""
    if load_kv:
        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=tma_copy_bytes_K)
    else:
        pipeline_Q.producer_acquire(producer_state_Q)
    if const_expr(sQ_block_metadata is not None):
        if producer_tidx == Int32(0):
            stage = producer_state_Q.index
            sQ_block_metadata[0, stage] = m_block
            cute.arch.fence_view_async_shared()
    if load_kv:
        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
    load_Q(m_block, producer_state=producer_state_Q)
    load_LSE(m_block, producer_state=producer_state_Q)

    producer_state_dO_cur = (
        producer_state_dO if const_expr(not Q_stage_eq_dO_stage) else producer_state_Q
    )
    if load_kv:
        pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=tma_copy_bytes_V)
        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
    else:
        pipeline_dO.producer_acquire(producer_state_dO_cur)
    load_dO(m_block, producer_state=producer_state_dO_cur)
    load_dPsum(m_block, producer_state=producer_state_dO_cur)

    producer_state_Q.advance()
    producer_state_dO.advance()
    return producer_state_Q, producer_state_dO


@cute.jit
def produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    sQ_block_metadata: Optional[cute.Tensor],
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    n_blocks_per_sample: int,
    producer_tidx: Int32,
):
    """SM90 backward block sparse loading with separate partial/full loops.

    K/V are loaded with the first valid block. Iterates partial blocks first,
    then full blocks, matching consumer order.

    Returns updated (producer_state_Q, producer_state_dO).
    """
    if const_expr(blocksparse_tensors.mask_block_masks is not None):
        curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, _, _ = (
            get_curr_arbitrary_blocksparse_tensors_bwd(
                batch_idx,
                head_idx,
                n_block,
                blocksparse_tensors,
                n_blocks_per_sample,
            )
        )
    else:
        curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx = get_curr_blocksparse_tensors_fixed(
            batch_idx, head_idx, n_block, blocksparse_tensors
        )

    kv_loaded = False

    for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
        sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
        for subtile_offset in cutlass.range(subtile_factor, unroll=1):
            m_block = sparse_m_block + subtile_offset

            if m_block < m_block_max:
                producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                    m_block,
                    producer_state_Q,
                    producer_state_dO,
                    pipeline_Q,
                    pipeline_dO,
                    load_K,
                    load_V,
                    load_Q,
                    load_dO,
                    load_LSE,
                    load_dPsum,
                    sQ_block_metadata,
                    producer_tidx,
                    tma_copy_bytes_K,
                    tma_copy_bytes_V,
                    Q_stage_eq_dO_stage,
                    load_kv=not kv_loaded,
                )
                kv_loaded = True

    if const_expr(curr_full_idx is not None):
        for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
            sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
            for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                m_block = sparse_m_block + subtile_offset

                if m_block < m_block_max:
                    producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                        m_block,
                        producer_state_Q,
                        producer_state_dO,
                        pipeline_Q,
                        pipeline_dO,
                        load_K,
                        load_V,
                        load_Q,
                        load_dO,
                        load_LSE,
                        load_dPsum,
                        sQ_block_metadata,
                        producer_tidx,
                        tma_copy_bytes_K,
                        tma_copy_bytes_V,
                        Q_stage_eq_dO_stage,
                        load_kv=not kv_loaded,
                    )
                    kv_loaded = True

    return producer_state_Q, producer_state_dO


@cute.jit
def consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    mask,
    mask_mod,
    is_causal: cutlass.Constexpr,
    is_local: cutlass.Constexpr,
    thr_mma_SdP,
    consumer_tidx: Int32,
    sQ_block_metadata: Optional[cute.Tensor],
    pipeline_Q,
    n_blocks_per_sample: int,
    payload_words: cutlass.Constexpr[int] = 4,
    score_mod_fn=None,
    score_mod_bwd_fn=None,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    aux_tensors=None,
    fastdiv_mods=(None, None),
):
    """SM90 backward block sparse MMA consumption with separate partial/full loops.

    Partial blocks are processed first (with mask_mod applied), then full blocks
    (without mask_mod). This ensures mask_mod is only applied where needed.

    Returns updated (consumer_state_Q, consumer_state_dO) and whether any Q block was consumed.
    """
    mask_payloads = blocksparse_tensors.mask_block_masks
    mask_payload_base = Int32(0)
    if const_expr(mask_payloads is not None):
        if const_expr(subtile_factor == 1 and sQ_block_metadata is not None):
            curr_q_cnt, curr_full_cnt, mask_payload_base = get_curr_arbitrary_block_counts_bwd_sm90(
                batch_idx,
                head_idx,
                n_block,
                blocksparse_tensors,
                n_blocks_per_sample,
            )
            curr_q_idx = None
            curr_full_idx = None
        else:
            (
                curr_q_cnt,
                curr_q_idx,
                curr_full_cnt,
                curr_full_idx,
                mask_payload_base,
                _,
            ) = get_curr_arbitrary_blocksparse_tensors_bwd(
                batch_idx,
                head_idx,
                n_block,
                blocksparse_tensors,
                n_blocks_per_sample,
            )
    else:
        curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx = get_curr_blocksparse_tensors_fixed(
            batch_idx, head_idx, n_block, blocksparse_tensors
        )

    dKV_accumulate = False

    mask_fn_partial = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
    )

    mask_fn_full = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
    )
    if const_expr(sQ_block_metadata is not None):
        partial_iter_count = get_physical_subtile_count_bwd_sm90(
            curr_q_cnt, curr_q_idx, subtile_factor, m_block_max
        )
        full_iter_count = get_physical_subtile_count_bwd_sm90(
            curr_full_cnt, curr_full_idx, subtile_factor, m_block_max
        )
        for iter_idx in cutlass.range(partial_iter_count, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            packed_mask_fn = partial(
                load_packed_mask_payload,
                mask_payloads,
                mask_payload_base + sparse_idx,
                consumer_tidx,
                subtile_idx=subtile_offset,
                payload_words=payload_words,
            )
            pipeline_Q.consumer_wait(
                consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q)
            )
            stage = consumer_state_Q.index
            m_block = sQ_block_metadata[0, stage]
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                packed_mask_fn=packed_mask_fn,
                score_mod_fn=score_mod_fn,
                score_mod_bwd_fn=score_mod_bwd_fn,
                dKV_accumulate=dKV_accumulate,
                q_pipeline_already_waited=True,
            )
            dKV_accumulate = True

        for _ in cutlass.range(full_iter_count, unroll=1):
            pipeline_Q.consumer_wait(
                consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q)
            )
            stage = consumer_state_Q.index
            m_block = sQ_block_metadata[0, stage]
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                mask_fn=None,
                score_mod_fn=score_mod_fn,
                score_mod_bwd_fn=score_mod_bwd_fn,
                dKV_accumulate=dKV_accumulate,
                q_pipeline_already_waited=True,
            )
            dKV_accumulate = True
        return consumer_state_Q, consumer_state_dO, dKV_accumulate
    else:
        if const_expr(mask_payloads is not None):
            for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
                sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
                for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                    m_block = sparse_m_block + subtile_offset
                    if m_block < m_block_max:
                        packed_mask_fn = partial(
                            load_packed_mask_payload,
                            mask_payloads,
                            mask_payload_base + sparse_idx,
                            consumer_tidx,
                            subtile_idx=subtile_offset,
                            payload_words=payload_words,
                        )
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            packed_mask_fn=packed_mask_fn,
                            score_mod_fn=score_mod_fn,
                            score_mod_bwd_fn=score_mod_bwd_fn,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True

            for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
                sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
                for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                    m_block = sparse_m_block + subtile_offset
                    if m_block < m_block_max:
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=None,
                            score_mod_fn=score_mod_fn,
                            score_mod_bwd_fn=score_mod_bwd_fn,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True
        else:
            for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
                sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
                for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                    m_block = sparse_m_block + subtile_offset

                    if m_block < m_block_max:
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn_partial,
                            score_mod_fn=score_mod_fn,
                            score_mod_bwd_fn=score_mod_bwd_fn,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True

            if const_expr(curr_full_idx is not None):
                for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
                    sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
                    for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                        m_block = sparse_m_block + subtile_offset

                        if m_block < m_block_max:
                            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=mask_fn_full,
                                score_mod_fn=score_mod_fn,
                                score_mod_bwd_fn=score_mod_bwd_fn,
                                dKV_accumulate=dKV_accumulate,
                            )
                            dKV_accumulate = True

    return consumer_state_Q, consumer_state_dO, dKV_accumulate


@cute.jit
def _store_one_dQaccum_sm90(
    m_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
    accum_row_major: cutlass.Constexpr[bool] = False,
    deterministic: cutlass.Constexpr[bool] = False,
    mdQ_semaphore_cur: Optional[cute.Tensor] = None,
    warp_local_tidx: Int32 = Int32(0),
    lock_value: Int32 = Int32(0),
    prefetch_dq_order: Optional[Callable] = None,
    release_dq_empty: cutlass.Constexpr[bool] = True,
):
    """Store dQaccum for a single m_block."""
    if const_expr(accum_row_major and not deterministic):
        # A row-major copy chunk contains columns produced by every warp
        # group. Do not release any writer for the next iteration until all
        # previous chunks have finished reading shared memory.
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    if const_expr(release_dq_empty):
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            if const_expr(not deterministic and not accum_row_major):
                cute.arch.cp_async_bulk_wait_group(
                    num_dQ_warp_groups - 1 - warp_group_idx, read=True
                )
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )

    next_dq_order = Uint32(0)
    if const_expr(prefetch_dq_order is not None):
        next_dq_order = Uint32(prefetch_dq_order())

    if const_expr(deterministic):
        assert mdQ_semaphore_cur is not None
        barrier.wait_eq(
            mdQ_semaphore_cur[(m_block, None)].iterator,
            warp_local_tidx,
            0,  # flag_offset
            lock_value,
        )

    if const_expr(accum_row_major):
        # A contiguous store chunk spans rows and therefore contains columns
        # produced by every dQ warp group. Wait for all writers before reading
        # any chunk from the row-major shared-memory matrix.
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            with cute.arch.elect_one():
                copy_utils.cpasync_reduce_bulk_add_f32(
                    sdQaccum[None, warp_group_idx].iterator,
                    gdQaccum[(None, warp_group_idx), m_block].iterator,
                    tma_copy_bytes_dQ,
                )
            cute.arch.cp_async_bulk_commit_group()
    else:
        for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            with cute.arch.elect_one():
                copy_utils.cpasync_reduce_bulk_add_f32(
                    sdQaccum[None, warp_group_idx].iterator,
                    gdQaccum[(None, warp_group_idx), m_block].iterator,
                    tma_copy_bytes_dQ,
                )
            cute.arch.cp_async_bulk_commit_group()

    if const_expr(deterministic):
        assert mdQ_semaphore_cur is not None
        # The next contributor must not start until every dQ chunk from this
        # CTA has completed its global-memory reduction.
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        barrier.arrive_inc(
            mdQ_semaphore_cur[(m_block, None)].iterator,
            warp_local_tidx,
            0,  # flag_offset
            1,
        )
    return next_dq_order


@cute.jit
def _load_dq_order_entry(dq_write_order: cute.Tensor, index: Int32):
    """Load one packed deterministic dQ entry for software prefetching."""
    return Uint32(dq_write_order[index])


@cute.jit
def _store_one_dQaccum_from_packed_order_sm90(
    packed_order: Uint32,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
    mdQ_semaphore_cur: cute.Tensor,
    warp_local_tidx: Int32,
    accum_row_major: cutlass.Constexpr[bool],
    prefetch_dq_order: Optional[Callable] = None,
    release_dq_empty: cutlass.Constexpr[bool] = True,
):
    """Store one deterministic dQ tile from a packed (m_block, rank) entry."""
    m_block = Int32(packed_order & Uint32(0xFFFF))
    lock_value = Int32(packed_order >> Uint32(16))
    return _store_one_dQaccum_sm90(
        m_block,
        sdQaccum,
        gdQaccum,
        num_dQ_warp_groups,
        num_threads_per_warp_group,
        tma_copy_bytes_dQ,
        accum_row_major=accum_row_major,
        deterministic=True,
        mdQ_semaphore_cur=mdQ_semaphore_cur,
        warp_local_tidx=warp_local_tidx,
        lock_value=lock_value,
        prefetch_dq_order=prefetch_dq_order,
        release_dq_empty=release_dq_empty,
    )


@cute.jit
def _store_dQaccum_packed_order_sequence_sm90(
    packed_order: Uint32,
    dq_write_order: cute.Tensor,
    count: Int32,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
    mdQ_semaphore_cur: cute.Tensor,
    warp_local_tidx: Int32,
    accum_row_major: cutlass.Constexpr[bool],
    start_idx: Int32 = Int32(0),
):
    """Store a non-empty packed write-order sequence with pipelined loads."""
    for sparse_idx in cutlass.range(start_idx, count - Int32(1), unroll=1):
        packed_order = _store_one_dQaccum_from_packed_order_sm90(
            packed_order,
            sdQaccum,
            gdQaccum,
            num_dQ_warp_groups,
            num_threads_per_warp_group,
            tma_copy_bytes_dQ,
            mdQ_semaphore_cur,
            warp_local_tidx,
            accum_row_major,
            prefetch_dq_order=partial(
                _load_dq_order_entry,
                dq_write_order,
                sparse_idx + Int32(1),
            ),
        )
    _store_one_dQaccum_from_packed_order_sm90(
        packed_order,
        sdQaccum,
        gdQaccum,
        num_dQ_warp_groups,
        num_threads_per_warp_group,
        tma_copy_bytes_dQ,
        mdQ_semaphore_cur,
        warp_local_tidx,
        accum_row_major,
    )


@cute.jit
def dQaccum_store_block_sparse_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    n_blocks_per_sample: int,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
    deterministic: cutlass.Constexpr[bool] = False,
    accum_row_major: cutlass.Constexpr[bool] = False,
    mdQ_semaphore_cur: Optional[cute.Tensor] = None,
    warp_local_tidx: Int32 = Int32(0),
):
    """SM90 backward block sparse dQaccum store with separate partial/full loops.

    Iterates partial blocks first, then full blocks, matching producer/consumer order.
    """
    if const_expr(blocksparse_tensors.mask_block_masks is not None):
        (
            curr_q_cnt,
            curr_q_idx,
            curr_full_cnt,
            curr_full_idx,
            partial_base,
            full_base,
        ) = get_curr_arbitrary_blocksparse_tensors_bwd(
            batch_idx,
            head_idx,
            n_block,
            blocksparse_tensors,
            n_blocks_per_sample,
        )
    else:
        curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx = get_curr_blocksparse_tensors_fixed(
            batch_idx, head_idx, n_block, blocksparse_tensors
        )
    curr_dq_write_order = None
    curr_dq_write_order_full = None
    if const_expr(deterministic):
        if const_expr(blocksparse_tensors.mask_block_masks is not None):
            assert blocksparse_tensors.dq_write_order is not None
            assert blocksparse_tensors.dq_write_order_full is not None
            curr_dq_write_order = cute.domain_offset(
                partial_base, blocksparse_tensors.dq_write_order
            )
            curr_dq_write_order_full = cute.domain_offset(
                full_base, blocksparse_tensors.dq_write_order_full
            )
        else:
            curr_dq_write_order, curr_dq_write_order_full = get_curr_dq_write_order_bwd(
                blocksparse_tensors, batch_idx, head_idx, n_block
            )
        assert curr_dq_write_order is not None

    if const_expr(
        deterministic and blocksparse_tensors.mask_block_masks is not None and subtile_factor == 1
    ):
        # Pipeline the packed write-order load behind dQEmpty release.  This
        # keeps metadata latency off the MMA warp group's critical path while
        # preserving the stable K2Q rank order required by deterministic mode.
        assert mdQ_semaphore_cur is not None
        if curr_q_cnt + curr_full_cnt > Int32(0):
            for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                    number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
                )
        if curr_q_cnt > Int32(0):
            packed_order = _load_dq_order_entry(curr_dq_write_order, Int32(0))
            if curr_q_cnt > Int32(1):
                packed_order = _store_one_dQaccum_from_packed_order_sm90(
                    packed_order,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                    mdQ_semaphore_cur,
                    warp_local_tidx,
                    accum_row_major,
                    prefetch_dq_order=partial(
                        _load_dq_order_entry,
                        curr_dq_write_order,
                        Int32(1),
                    ),
                    release_dq_empty=False,
                )
                for sparse_idx in cutlass.range(Int32(1), curr_q_cnt - Int32(1), unroll=1):
                    packed_order = _store_one_dQaccum_from_packed_order_sm90(
                        packed_order,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                        prefetch_dq_order=partial(
                            _load_dq_order_entry,
                            curr_dq_write_order,
                            sparse_idx + Int32(1),
                        ),
                    )
                if curr_full_cnt > Int32(0):
                    assert curr_dq_write_order_full is not None
                    packed_order_full = _store_one_dQaccum_from_packed_order_sm90(
                        packed_order,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                        prefetch_dq_order=partial(
                            _load_dq_order_entry,
                            curr_dq_write_order_full,
                            Int32(0),
                        ),
                    )
                    _store_dQaccum_packed_order_sequence_sm90(
                        packed_order_full,
                        curr_dq_write_order_full,
                        curr_full_cnt,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                    )
                else:
                    _store_one_dQaccum_from_packed_order_sm90(
                        packed_order,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                    )
            else:
                if curr_full_cnt > Int32(0):
                    assert curr_dq_write_order_full is not None
                    packed_order_full = _store_one_dQaccum_from_packed_order_sm90(
                        packed_order,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                        prefetch_dq_order=partial(
                            _load_dq_order_entry,
                            curr_dq_write_order_full,
                            Int32(0),
                        ),
                        release_dq_empty=False,
                    )
                    _store_dQaccum_packed_order_sequence_sm90(
                        packed_order_full,
                        curr_dq_write_order_full,
                        curr_full_cnt,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                    )
                else:
                    _store_one_dQaccum_from_packed_order_sm90(
                        packed_order,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        mdQ_semaphore_cur,
                        warp_local_tidx,
                        accum_row_major,
                        release_dq_empty=False,
                    )
        elif curr_full_cnt > Int32(0):
            assert curr_dq_write_order_full is not None
            packed_order_full = _load_dq_order_entry(curr_dq_write_order_full, Int32(0))
            if curr_full_cnt > Int32(1):
                packed_order_full = _store_one_dQaccum_from_packed_order_sm90(
                    packed_order_full,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                    mdQ_semaphore_cur,
                    warp_local_tidx,
                    accum_row_major,
                    prefetch_dq_order=partial(
                        _load_dq_order_entry,
                        curr_dq_write_order_full,
                        Int32(1),
                    ),
                    release_dq_empty=False,
                )
                _store_dQaccum_packed_order_sequence_sm90(
                    packed_order_full,
                    curr_dq_write_order_full,
                    curr_full_cnt,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                    mdQ_semaphore_cur,
                    warp_local_tidx,
                    accum_row_major,
                    start_idx=Int32(1),
                )
            else:
                _store_one_dQaccum_from_packed_order_sm90(
                    packed_order_full,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                    mdQ_semaphore_cur,
                    warp_local_tidx,
                    accum_row_major,
                    release_dq_empty=False,
                )
        return

    for sparse_idx in cutlass.range(curr_q_cnt, unroll=1):
        if const_expr(deterministic and blocksparse_tensors.mask_block_masks is not None):
            packed_order = Uint32(curr_dq_write_order[sparse_idx])
            sparse_m_block = Int32(packed_order & Uint32(0xFFFF)) * subtile_factor
            lock_value = Int32(packed_order >> Uint32(16))
        else:
            sparse_m_block = curr_q_idx[sparse_idx] * subtile_factor
            lock_value = curr_dq_write_order[sparse_idx] if const_expr(deterministic) else Int32(0)
        for subtile_offset in cutlass.range(subtile_factor, unroll=1):
            m_block = sparse_m_block + subtile_offset

            if m_block < m_block_max:
                _store_one_dQaccum_sm90(
                    m_block,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                    accum_row_major=accum_row_major,
                    deterministic=deterministic,
                    mdQ_semaphore_cur=mdQ_semaphore_cur,
                    warp_local_tidx=warp_local_tidx,
                    lock_value=lock_value,
                )

    if const_expr(curr_full_idx is not None):
        if const_expr(deterministic):
            assert curr_dq_write_order_full is not None
        for sparse_idx in cutlass.range(curr_full_cnt, unroll=1):
            if const_expr(deterministic and blocksparse_tensors.mask_block_masks is not None):
                packed_order = Uint32(curr_dq_write_order_full[sparse_idx])
                sparse_m_block = Int32(packed_order & Uint32(0xFFFF)) * subtile_factor
                lock_value = Int32(packed_order >> Uint32(16))
            else:
                sparse_m_block = curr_full_idx[sparse_idx] * subtile_factor
                lock_value = (
                    curr_dq_write_order_full[sparse_idx] if const_expr(deterministic) else Int32(0)
                )
            for subtile_offset in cutlass.range(subtile_factor, unroll=1):
                m_block = sparse_m_block + subtile_offset

                if m_block < m_block_max:
                    _store_one_dQaccum_sm90(
                        m_block,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups,
                        num_threads_per_warp_group,
                        tma_copy_bytes_dQ,
                        accum_row_major=accum_row_major,
                        deterministic=deterministic,
                        mdQ_semaphore_cur=mdQ_semaphore_cur,
                        warp_local_tidx=warp_local_tidx,
                        lock_value=lock_value,
                    )
