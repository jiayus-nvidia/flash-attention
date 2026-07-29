#pragma once

#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward declarations
struct BlockSparsityArguments;
struct BlockSparsityInfo;
struct BlockSparsityIterator;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Block sparsity tensors for forward pass (Q2K direction)
// For each m_block, we have lists of n_blocks that need to be processed:
// - mask_block: blocks that require masking (partial blocks)
// - full_block: blocks that don't require masking (full blocks)
//
// Data layout:
// - cnt: [B * H * num_blocks] flattened, where flat_idx = bidb * H * num_blocks + bidh * num_blocks + m_block
// - offset: [B * H * num_blocks + 1], CSR-style exclusive prefix sum
// - idx: [total_blocks], compact n_block indices
struct BlockSparsityArguments {
    // mask_block_cnt[flat_idx]: number of masked (partial) blocks for this m_block
    int const* mask_block_cnt = nullptr;
    // mask_block_offset[flat_idx]: cumulative offset into mask_block_idx
    int const* mask_block_offset = nullptr;
    // mask_block_idx[offset]: the n_block index
    int const* mask_block_idx = nullptr;

    // full_block_cnt[flat_idx]: number of full blocks for this m_block
    int const* full_block_cnt = nullptr;
    // full_block_offset[flat_idx]: cumulative offset into full_block_idx
    int const* full_block_offset = nullptr;
    // full_block_idx[offset]: the n_block index
    int const* full_block_idx = nullptr;

    // Number of m_blocks (needed for computing flat index)
    int num_blocks = 0;
    // Number of heads (needed for computing flat index, can be 1 for broadcasting)
    int num_heads = 0;
    // Number of batches (needed for computing flat index, can be 1 for broadcasting)
    int num_batches = 0;

    // Deterministic backward write ranks, parallel to mask/full_block_idx.
    // These are only consumed by the K2Q backward dQ store.
    int const* dq_write_order = nullptr;
    int const* dq_write_order_full = nullptr;
};

// Device-side params (same as Arguments for now, but can be optimized separately)
using BlockSparsityParams = BlockSparsityArguments;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper struct to hold block sparsity info for a single m_block during kernel execution
//
// - Mask blocks are processed first, in REVERSE order (highest index first)
// - Full blocks are processed second, in REVERSE order (highest index first)
// - This matches the producer/consumer pipeline synchronization
//
// Masking logic (all blocks need basic mask, difference is mask_mod):
//   - First block (iter 0):        mask_seqlen=true,  mask_mod=true,  is_first=true
//   - Other mask blocks:           mask_seqlen=false, mask_mod=true,  is_first=false
//   - First full block:            mask_seqlen=true,  mask_mod=false, is_first=false
//   - Other full blocks:           mask_seqlen=false, mask_mod=false, is_first=false
//
// Note: full_blocks still need basic masking (seqlen check), just not arbitrary mask (mask_mod)
//
// Example with mask_block_cnt=3, full_block_cnt=2:
//   iteration 0: mask_block[2] (last mask block)   -> mask_seqlen=true,  mask_mod=true
//   iteration 1: mask_block[1]                     -> mask_seqlen=false, mask_mod=true
//   iteration 2: mask_block[0] (first mask block)  -> mask_seqlen=false, mask_mod=true
//   iteration 3: full_block[1] (last full block)   -> mask_seqlen=false, mask_mod=false
//   iteration 4: full_block[0] (first full block)  -> mask_seqlen=false, mask_mod=false
struct BlockSparsityInfo {
    int mask_block_cnt = 0;
    int mask_block_offset = 0;
    int full_block_cnt = 0;
    int full_block_offset = 0;
    int const* mask_block_idx = nullptr;
    int const* full_block_idx = nullptr;

    CUTLASS_DEVICE
    BlockSparsityInfo() = default;

    // Compute flat index for accessing block sparsity arrays
    CUTLASS_DEVICE
    static int compute_flat_idx(int bidb, int bidh, int m_block, int num_heads, int num_blocks) {
        return bidb * num_heads * num_blocks + bidh * num_blocks + m_block;
    }

    // Initialize from params for a specific (bidb, bidh, m_block)
    // Supports broadcasting: when num_heads=1 or num_batches=1, all heads/batches share the same sparsity pattern
    CUTLASS_DEVICE
    void init(BlockSparsityParams const& params, int bidb, int bidh, int m_block) {
        // Support broadcasting: use 0 when dimension is 1
        int const sparse_batch_idx = params.num_batches == 1 ? 0 : bidb;
        int const sparse_head_idx = params.num_heads == 1 ? 0 : bidh;
        int flat_idx = compute_flat_idx(sparse_batch_idx, sparse_head_idx, m_block, params.num_heads, params.num_blocks);
        mask_block_cnt = params.mask_block_cnt[flat_idx];
        mask_block_offset = params.mask_block_offset[flat_idx];
        mask_block_idx = params.mask_block_idx;
        full_block_cnt = params.full_block_cnt[flat_idx];
        full_block_offset = params.full_block_offset[flat_idx];
        full_block_idx = params.full_block_idx;
    }

    CUTLASS_DEVICE
    int get_total_blocks() const {
        return mask_block_cnt + full_block_cnt;
    }

    CUTLASS_DEVICE
    bool is_empty() const {
        return get_total_blocks() == 0;
    }

    // Get n_block at iteration position idx (REVERSE order within each block type)
    // idx=0 returns the LAST mask block, idx=mask_block_cnt-1 returns the FIRST mask block
    // idx=mask_block_cnt returns the LAST full block, etc.
    CUTLASS_DEVICE
    int get_n_block(int idx) const {
        if (idx < mask_block_cnt) {
            // Reverse order: idx=0 -> last mask block, idx=mask_block_cnt-1 -> first mask block
            int reverse_idx = mask_block_cnt - 1 - idx;
            return mask_block_idx[mask_block_offset + reverse_idx];
        } else {
            // Reverse order: idx=mask_block_cnt -> last full block
            int full_idx = idx - mask_block_cnt;
            int reverse_idx = full_block_cnt - 1 - full_idx;
            return full_block_idx[full_block_offset + reverse_idx];
        }
    }

    // Get n_block at raw position (no reverse, for direct index access)
    CUTLASS_DEVICE
    int get_n_block_raw(int idx) const {
        if (idx < mask_block_cnt) {
            return mask_block_idx[mask_block_offset + idx];
        } else {
            int full_idx = idx - mask_block_cnt;
            return full_block_idx[full_block_offset + full_idx];
        }
    }

    // Check if iteration position idx is a mask block (requires masking with mask_mod/arbitrary mask)
    CUTLASS_DEVICE
    bool is_mask_block(int idx) const {
        return idx < mask_block_cnt;
    }

    // Check if this is the first block being processed (for softmax is_first initialization)
    CUTLASS_DEVICE
    bool is_first_block(int idx) const {
        return idx == 0;
    }

    // Check if this is the first full block (transition from mask to full blocks)
    CUTLASS_DEVICE
    bool is_first_full_block(int idx) const {
        return idx == mask_block_cnt && full_block_cnt > 0;
    }

    // Check if mask_seqlen should be true for this iteration
    // mask_seqlen is true for:
    //   1. The very first block (idx == 0)
    //   2. The first full block (idx == mask_block_cnt) if there are full blocks
    CUTLASS_DEVICE
    bool needs_seqlen_mask(int idx) const {
        return is_first_block(idx) || is_first_full_block(idx);
    }

    // Check if mask_mod (arbitrary mask) should be applied for this iteration
    // mask_mod is only applied for mask blocks, not full blocks
    CUTLASS_DEVICE
    bool needs_mask_mod(int idx) const {
        return is_mask_block(idx);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Iterator for block sparsity - iterates through mask blocks first, then full blocks
// In REVERSE order (Mask Seqlen for first block)
struct BlockSparsityIterator {
    BlockSparsityInfo const& info;
    int current_idx = 0;  // iteration index (0 = first to process = last mask block)

    CUTLASS_DEVICE
    BlockSparsityIterator(BlockSparsityInfo const& info_, int start_idx = 0)
        : info(info_), current_idx(start_idx) {}

    // Returns the n_block index for the current position
    CUTLASS_DEVICE
    int get_n_block() const {
        return info.get_n_block(current_idx);
    }

    // Returns whether current block is a mask (partial) block that needs mask_mod
    CUTLASS_DEVICE
    bool is_mask_block() const {
        return info.is_mask_block(current_idx);
    }

    // Returns whether current block is a full block (no mask_mod needed)
    CUTLASS_DEVICE
    bool is_full_block() const {
        return !is_mask_block();
    }

    // Check if this is the first block being processed (for softmax is_first)
    CUTLASS_DEVICE
    bool is_first() const {
        return info.is_first_block(current_idx);
    }

    // Check if this is the first full block (boundary between mask and full)
    CUTLASS_DEVICE
    bool is_first_full() const {
        return info.is_first_full_block(current_idx);
    }

    // Check if mask_seqlen should be true for current iteration
    // True for: first block overall OR first full block
    CUTLASS_DEVICE
    bool needs_seqlen_mask() const {
        return info.needs_seqlen_mask(current_idx);
    }

    // Check if mask_mod (arbitrary mask) should be applied for current iteration
    // Only true for mask blocks, not full blocks
    CUTLASS_DEVICE
    bool needs_mask_mod() const {
        return info.needs_mask_mod(current_idx);
    }

    // Check if iterator is valid (more blocks to process)
    CUTLASS_DEVICE
    bool is_valid() const {
        return current_idx < info.get_total_blocks();
    }

    // Advance to next block
    CUTLASS_DEVICE
    void advance() {
        ++current_idx;
    }

    // Get number of remaining blocks (including current)
    CUTLASS_DEVICE
    int remaining() const {
        return info.get_total_blocks() - current_idx;
    }

    // Get current iteration index
    CUTLASS_DEVICE
    int get_idx() const {
        return current_idx;
    }

    // Get total number of blocks
    CUTLASS_DEVICE
    int get_total() const {
        return info.get_total_blocks();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Unified BlockSparsity class for consistent usage pattern (similar to TileScheduler)
// Usage:
//   using BlockSparsity = flash::BlockSparsity;
//   using BlockSparsityArguments = typename BlockSparsity::Arguments;
//   using BlockSparsityParams = typename BlockSparsity::Params;
struct BlockSparsity {
    using Arguments = BlockSparsityArguments;
    using Params = BlockSparsityParams;
    using Info = BlockSparsityInfo;
    using Iterator = BlockSparsityIterator;

    // Convert Arguments to Params (identity for now, can be optimized later)
    static Params
    to_underlying_arguments(Arguments const& args) {
        return args;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Block Sparse Load Functions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Load the final V block after overlapped K/V loads.
 *
 * @tparam Transpose_V Whether V needs transposition
 * @param block_idx Block indices array
 * @param block_offset Offset into block_idx
 * @param block_count Number of blocks
 * @param load_V Lambda to load V
 * @param smem_pipe_write Pipeline state
 * @param should_load_KV Whether to actually perform the load
 */
template <bool Transpose_V, class LoadV, class PipelineState>
CUTLASS_DEVICE
void finish_overlap_v_load(
    int const* block_idx,
    int block_offset,
    int block_count,
    LoadV&& load_V,
    PipelineState& smem_pipe_write,
    bool should_load_KV
) {
    if (block_count > 0) {
        // First index = last block in reverse order
        int n_block_last = block_idx[block_offset];
        if constexpr (!Transpose_V) {
            if (should_load_KV) {
                load_V(n_block_last, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
            }
        }
        ++smem_pipe_write;
    }
}

/**
 * Load a block list (mask or full blocks).
 *
 * @tparam IntraWGOverlap Whether to use intra-warpgroup overlap
 * @tparam Transpose_V Whether V needs transposition
 * @tparam PagedKVNonTMA Whether using paged KV without TMA
 * @param block_idx Block indices array
 * @param block_offset Offset into block_idx
 * @param block_count Number of blocks
 * @param load_q_with_first Whether to load Q with the first K
 * @param first_block_preloaded Whether first K was already loaded (for bridging)
 * @param load_K Lambda to load K
 * @param load_V Lambda to load V
 * @param load_Q_fn Lambda to load Q (and wait for barrier_O)
 * @param copy_Vt_to_V Lambda to copy transposed V (only used if Transpose_V)
 * @param paged_kv_manager PagedKV manager
 * @param smem_pipe_write Pipeline state
 * @param should_load_KV Whether to actually perform the loads
 */
template <bool IntraWGOverlap, bool Transpose_V, bool PagedKVNonTMA,
          class LoadK, class LoadV, class LoadQ, class CopyVtToV,
          class PagedKVManager, class PipelineState>
CUTLASS_DEVICE
void load_block_list(
    int const* block_idx,
    int block_offset,
    int block_count,
    bool load_q_with_first,
    bool first_block_preloaded,
    LoadK&& load_K,
    LoadV&& load_V,
    LoadQ&& load_Q_fn,
    CopyVtToV&& copy_Vt_to_V,
    PagedKVManager& paged_kv_manager,
    PipelineState& smem_pipe_write,
    bool should_load_KV
) {
    if (block_count <= 0) return;

    // Reverse iteration: from block_count-1 to 0
    int n_block_first = block_idx[block_offset + block_count - 1];

    if constexpr (!IntraWGOverlap) {
        // ============================================================
        // Non-overlap path: sequential K, V loads
        // ============================================================
        if (!first_block_preloaded) {
            if (should_load_KV) {
                if constexpr (PagedKVNonTMA) {
                    paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/, true /*First_iter*/>(n_block_first);
                } else {
                    paged_kv_manager.template load_page_table_TMA<true /*First_iter*/>(n_block_first);
                }
                if constexpr (Transpose_V) {
                    load_V(n_block_first, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
                }
                load_K(n_block_first, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
            }
            if (load_q_with_first) {
                load_Q_fn();
            }
        }
        if constexpr (!Transpose_V) {
            if (should_load_KV) {
                load_V(n_block_first, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
            }
        }
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
        ++smem_pipe_write;

        // Remaining iterations
        for (int i = 1; i < block_count; ++i) {
            int n_block = block_idx[block_offset + block_count - 1 - i];
            if (should_load_KV) {
                if constexpr (PagedKVNonTMA) {
                    paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block);
                } else {
                    paged_kv_manager.load_page_table_TMA(n_block);
                }
                if constexpr (Transpose_V) {
                    load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                }
                load_K(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                if constexpr (!Transpose_V) {
                    load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                }
            }
            if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
            ++smem_pipe_write;
        }
    } else {
        // ============================================================
        // Overlap path: interleaved K/V loads
        // ============================================================
        if (!first_block_preloaded) {
            if (should_load_KV) {
                if constexpr (PagedKVNonTMA) {
                    paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/, true /*First_iter*/>(n_block_first);
                } else {
                    paged_kv_manager.template load_page_table_TMA<true /*First_iter*/>(n_block_first);
                }
                if constexpr (Transpose_V) {
                    load_V(n_block_first, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
                }
                load_K(n_block_first, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
            }
            if (load_q_with_first) {
                load_Q_fn();
            }
        }
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }

        // Interleaved K[i+1] and V[i] loads
        for (int i = 0; i < block_count - 1; ++i) {
            int n_block_prev = block_idx[block_offset + block_count - 1 - i];
            int n_block = block_idx[block_offset + block_count - 2 - i];

            PipelineState smem_pipe_write_prev = smem_pipe_write;
            ++smem_pipe_write;

            if (should_load_KV) {
                if constexpr (PagedKVNonTMA) {
                    paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block);
                } else {
                    paged_kv_manager.load_page_table_TMA(n_block);
                }
                if constexpr (Transpose_V) {
                    load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                }
                load_K(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                if constexpr (!Transpose_V) {
                    load_V(n_block_prev, smem_pipe_write_prev, cute::true_type{} /*Seqlenk_mask*/);
                }
            }
            if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
        }
        // Note: last V is loaded by finish_overlap_v_load
    }
}

/**
 * Bridge mask list to full list (overlap mode only).
 * Overlaps the pending masked V with the first full K load.
 *
 * @tparam Transpose_V Whether V needs transposition
 * @tparam PagedKVNonTMA Whether using paged KV without TMA
 */
template <bool Transpose_V, bool PagedKVNonTMA,
          class LoadK, class LoadV, class CopyVtToV,
          class PagedKVManager, class PipelineState>
CUTLASS_DEVICE
void bridge_mask_to_full(
    BlockSparsityInfo const& info,
    LoadK&& load_K,
    LoadV&& load_V,
    CopyVtToV&& copy_Vt_to_V,
    PagedKVManager& paged_kv_manager,
    PipelineState& smem_pipe_write,
    bool should_load_KV
) {
    // Last mask block (first in stored order due to reverse iteration)
    int n_block_mask_last = info.mask_block_idx[info.mask_block_offset];
    // First full block (last in stored order due to reverse iteration)
    int n_block_full_first = info.full_block_idx[info.full_block_offset + info.full_block_cnt - 1];

    PipelineState smem_pipe_write_prev = smem_pipe_write;
    ++smem_pipe_write;

    if (should_load_KV) {
        if constexpr (PagedKVNonTMA) {
            paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block_full_first);
        } else {
            paged_kv_manager.load_page_table_TMA(n_block_full_first);
        }
        if constexpr (Transpose_V) {
            load_V(n_block_full_first, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
        }
        load_K(n_block_full_first, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
        if constexpr (!Transpose_V) {
            load_V(n_block_mask_last, smem_pipe_write_prev, cute::true_type{} /*Seqlenk_mask*/);
        }
    }
    if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
}

/**
 * Main entry point for block sparse loads.
 * Iterates over the mask and full block lists for a single tile.
 *
 * @tparam IntraWGOverlap Whether to use intra-warpgroup overlap
 * @tparam Transpose_V Whether V needs transposition
 * @tparam PagedKVNonTMA Whether using paged KV without TMA
 * @param info Block sparsity info for the current m_block
 * @param load_K Lambda to load K
 * @param load_V Lambda to load V
 * @param load_Q_fn Lambda to load Q (and wait for barrier_O)
 * @param copy_Vt_to_V Lambda to copy transposed V
 * @param paged_kv_manager PagedKV manager
 * @param smem_pipe_write Pipeline state
 * @param should_load_KV Whether to actually perform the loads
 */
template <bool IntraWGOverlap, bool Transpose_V, bool PagedKVNonTMA,
          class LoadK, class LoadV, class LoadQ, class CopyVtToV,
          class PagedKVManager, class PipelineState>
CUTLASS_DEVICE
void produce_block_sparse_loads(
    BlockSparsityInfo const& info,
    LoadK&& load_K,
    LoadV&& load_V,
    LoadQ&& load_Q_fn,
    CopyVtToV&& copy_Vt_to_V,
    PagedKVManager& paged_kv_manager,
    PipelineState& smem_pipe_write,
    bool should_load_KV
) {
    bool const mask_empty = (info.mask_block_cnt == 0);
    bool const full_empty = (info.full_block_cnt == 0);

    if (mask_empty) {
        // ============================================================
        // Case 1: No masked blocks - full list owns initial Q+K load
        // ============================================================
        load_block_list<IntraWGOverlap, Transpose_V, PagedKVNonTMA>(
            info.full_block_idx,
            info.full_block_offset,
            info.full_block_cnt,
            /*load_q_with_first=*/true,
            /*first_block_preloaded=*/false,
            load_K, load_V, load_Q_fn, copy_Vt_to_V,
            paged_kv_manager, smem_pipe_write, should_load_KV
        );

        if constexpr (IntraWGOverlap) {
            finish_overlap_v_load<Transpose_V>(
                info.full_block_idx,
                info.full_block_offset,
                info.full_block_cnt,
                load_V, smem_pipe_write, should_load_KV
            );
        }
    } else {
        // ============================================================
        // Case 2: Masked blocks present - load mask list first with Q
        // ============================================================
        load_block_list<IntraWGOverlap, Transpose_V, PagedKVNonTMA>(
            info.mask_block_idx,
            info.mask_block_offset,
            info.mask_block_cnt,
            /*load_q_with_first=*/true,
            /*first_block_preloaded=*/false,
            load_K, load_V, load_Q_fn, copy_Vt_to_V,
            paged_kv_manager, smem_pipe_write, should_load_KV
        );

        if (full_empty) {
            // No full blocks: just finish the mask list
            if constexpr (IntraWGOverlap) {
                finish_overlap_v_load<Transpose_V>(
                    info.mask_block_idx,
                    info.mask_block_offset,
                    info.mask_block_cnt,
                    load_V, smem_pipe_write, should_load_KV
                );
            }
        } else {
            // Both mask and full blocks present
            if constexpr (IntraWGOverlap) {
                // Bridge: overlap pending masked V with first full K
                bridge_mask_to_full<Transpose_V, PagedKVNonTMA>(
                    info, load_K, load_V, copy_Vt_to_V,
                    paged_kv_manager, smem_pipe_write, should_load_KV
                );

                // Continue with full list (first K already loaded)
                load_block_list<IntraWGOverlap, Transpose_V, PagedKVNonTMA>(
                    info.full_block_idx,
                    info.full_block_offset,
                    info.full_block_cnt,
                    /*load_q_with_first=*/false,
                    /*first_block_preloaded=*/true,
                    load_K, load_V, load_Q_fn, copy_Vt_to_V,
                    paged_kv_manager, smem_pipe_write, should_load_KV
                );

                // Finish last V of full list
                finish_overlap_v_load<Transpose_V>(
                    info.full_block_idx,
                    info.full_block_offset,
                    info.full_block_cnt,
                    load_V, smem_pipe_write, should_load_KV
                );
            } else {
                // Non-overlap: just run full list normally (skip Q reload)
                load_block_list<IntraWGOverlap, Transpose_V, PagedKVNonTMA>(
                    info.full_block_idx,
                    info.full_block_offset,
                    info.full_block_cnt,
                    /*load_q_with_first=*/false,
                    /*first_block_preloaded=*/false,
                    load_K, load_V, load_Q_fn, copy_Vt_to_V,
                    paged_kv_manager, smem_pipe_write, should_load_KV
                );
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Block Sparse Consume Functions (Consumer-side MMA)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Consume the mask and full block lists for a single tile on the consumer side.
 * Mirrors `produce_block_sparse_loads` so that the consumer pipeline matches the producer.
 *
 * masking logic for block sparsity:
 *   - mask_blocks: apply arbitrary mask (MaskFunc already contains Seqlenk info), check_inf=true
 *   - full_blocks: no mask needed (Seqlenk info is in MaskFunc, Seqlenq doesn't need mask
 *                  as out-of-bound won't be written back in epilogue), check_inf=false
 *
 * Iteration order (REVERSE within each block type to match producer):
 *   - mask_block[mask_cnt-1] -> ... -> mask_block[0]  (with arbitrary mask)
 *   - full_block[full_cnt-1] -> ... -> full_block[0]  (no mask)
 *
 * @tparam IntraWGOverlap Whether to use intra-warpgroup overlap
 * @param info Block sparsity info for the current m_block
 * @param fwd_step Lambda for one forward step
 * @param process_first_half_block Lambda for first half block (overlap mode): (n_block, mask_fn) -> void
 * @param process_last_half_block Lambda for last half block (overlap mode): () -> void
 * @param arbitrary_mask_fn Mask function with arbitrary mask (for mask_blocks)
 * @param no_mask_fn No mask function (for full_blocks)
 * @param warp_scheduler_barrier_sync Warp scheduler barrier sync function (non-overlap mode)
 * @param warp_scheduler_barrier_arrive Warp scheduler barrier arrive function (non-overlap mode)
 */
template <bool IntraWGOverlap,
          class FwdStep, class ProcessFirstHalfBlock, class ProcessLastHalfBlock,
          class ArbitraryMaskFn, class NoMaskFn,
          class WarpSchedulerBarrierSync, class WarpSchedulerBarrierArrive>
CUTLASS_DEVICE
void consume_block_sparse_loads(
    BlockSparsityInfo const& info,
    FwdStep&& fwd_step,
    ProcessFirstHalfBlock&& process_first_half_block,
    ProcessLastHalfBlock&& process_last_half_block,
    ArbitraryMaskFn&& arbitrary_mask_fn,
    NoMaskFn&& no_mask_fn,
    WarpSchedulerBarrierSync&& warp_scheduler_barrier_sync,
    WarpSchedulerBarrierArrive&& warp_scheduler_barrier_arrive
) {
    // Note: Empty case (no blocks to process) is handled at mma() function entry.
    // When this function is called, there must be at least one block to process.
    int const curr_mask_cnt = info.mask_block_cnt;
    int const curr_mask_offset = info.mask_block_offset;
    int const curr_full_cnt = info.full_block_cnt;
    int const curr_full_offset = info.full_block_offset;

    if constexpr (!IntraWGOverlap) {
        // ============================================================
        // Non-overlap path
        // ============================================================

        // Process mask_blocks (need arbitrary mask, check_inf=true)
        if (curr_mask_cnt > 0) {
            // First mask_block: is_first=true, check_inf=true
            int n_block = info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 1];
            warp_scheduler_barrier_sync();
            fwd_step(n_block, arbitrary_mask_fn, cute::true_type{} /*is_first*/, cute::true_type{} /*check_inf*/);

            // Subsequent mask_blocks: is_first=false, check_inf=true
            CUTLASS_PRAGMA_NO_UNROLL
            for (int i = 1; i < curr_mask_cnt; ++i) {
                n_block = info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 1 - i];
                fwd_step(n_block, arbitrary_mask_fn, cute::false_type{} /*is_first*/, cute::true_type{} /*check_inf*/);
            }

            if (curr_full_cnt == 0) {
                warp_scheduler_barrier_arrive();
            }
        }

        // Process full_blocks (no mask, check_inf=false)
        if (curr_full_cnt > 0) {
            int n_block = info.full_block_idx[curr_full_offset + curr_full_cnt - 1];

            if (curr_mask_cnt == 0) {
                // No mask_blocks: first full_block is_first=true
                warp_scheduler_barrier_sync();
                fwd_step(n_block, no_mask_fn, cute::true_type{} /*is_first*/, cute::false_type{} /*check_inf*/);
            } else {
                // Has mask_blocks: is_first=false
                fwd_step(n_block, no_mask_fn, cute::false_type{} /*is_first*/, cute::false_type{} /*check_inf*/);
            }

            // Subsequent full_blocks: is_first=false, check_inf=false
            CUTLASS_PRAGMA_NO_UNROLL
            for (int i = 1; i < curr_full_cnt; ++i) {
                n_block = info.full_block_idx[curr_full_offset + curr_full_cnt - 1 - i];
                fwd_step(n_block, no_mask_fn, cute::false_type{} /*is_first*/, cute::false_type{} /*check_inf*/);
            }
            warp_scheduler_barrier_arrive();
        }

    } else {
        // ============================================================
        // Overlap path (IntraWGOverlap)
        // ============================================================

        // Process mask_blocks (need arbitrary mask, check_inf=true)
        if (curr_mask_cnt > 0) {
            // First mask_block: process_first_half_block
            int n_block = info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 1];
            process_first_half_block(n_block, arbitrary_mask_fn);

            // Subsequent mask_blocks: fwd_step with check_inf=true
            CUTLASS_PRAGMA_NO_UNROLL
            for (int i = 1; i < curr_mask_cnt; ++i) {
                n_block = info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 1 - i];
                fwd_step(n_block, arbitrary_mask_fn, cute::true_type{} /*check_inf*/);
            }
        }

        // Process full_blocks (no mask, check_inf=false)
        if (curr_full_cnt > 0) {
            int n_block = info.full_block_idx[curr_full_offset + curr_full_cnt - 1];

            if (curr_mask_cnt == 0) {
                // No mask_blocks: first full_block needs process_first_half_block
                process_first_half_block(n_block, no_mask_fn);
            } else {
                // Has mask_blocks: use fwd_step
                fwd_step(n_block, no_mask_fn, cute::false_type{} /*check_inf*/);
            }

            // Subsequent full_blocks: check_inf=false
            CUTLASS_PRAGMA_NO_UNROLL
            for (int i = 1; i < curr_full_cnt; ++i) {
                n_block = info.full_block_idx[curr_full_offset + curr_full_cnt - 1 - i];
                fwd_step(n_block, no_mask_fn, cute::false_type{} /*check_inf*/);
            }
        }

        // Last half block - always called since empty case is handled at function entry
        process_last_half_block();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Block Sparse Backward Support (K2Q direction)
// For backward pass, we iterate over m_blocks for a fixed n_block
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Block sparsity info for backward pass (K2Q direction)
 * For each n_block, we have lists of m_blocks that need to be processed:
 * - mask_block: blocks that require masking (partial blocks, need arbitrary mask)
 * - full_block: blocks that don't require masking (full blocks)
 *
 * Unlike forward (which uses REVERSE order), backward uses FORWARD order:
 * - Process mask blocks in order: m_block[0], m_block[1], ...
 * - Then full blocks in order: m_block[0], m_block[1], ...
 */
struct BlockSparsityInfoBwd {
    int mask_block_cnt = 0;
    int mask_block_offset = 0;
    int full_block_cnt = 0;
    int full_block_offset = 0;
    int const* mask_block_idx = nullptr;
    int const* full_block_idx = nullptr;
    int const* dq_write_order = nullptr;
    int const* dq_write_order_full = nullptr;

    CUTLASS_DEVICE
    BlockSparsityInfoBwd() = default;

    // Initialize from params for a specific (bidb, bidh, n_block)
    // Note: For backward, the block index is n_block (K block), not m_block
    CUTLASS_DEVICE
    void init(BlockSparsityParams const& params, int bidb, int bidh, int n_block) {
        // Support broadcasting: use 0 when dimension is 1
        int const sparse_batch_idx = params.num_batches == 1 ? 0 : bidb;
        int const sparse_head_idx = params.num_heads == 1 ? 0 : bidh;
        int flat_idx = BlockSparsityInfo::compute_flat_idx(sparse_batch_idx, sparse_head_idx, n_block, params.num_heads, params.num_blocks);
        mask_block_cnt = params.mask_block_cnt[flat_idx];
        mask_block_offset = params.mask_block_offset[flat_idx];
        mask_block_idx = params.mask_block_idx;
        full_block_cnt = params.full_block_cnt[flat_idx];
        full_block_offset = params.full_block_offset[flat_idx];
        full_block_idx = params.full_block_idx;
        dq_write_order = params.dq_write_order;
        dq_write_order_full = params.dq_write_order_full;
    }

    CUTLASS_DEVICE
    int get_total_blocks() const {
        return mask_block_cnt + full_block_cnt;
    }

    CUTLASS_DEVICE
    bool is_empty() const {
        return get_total_blocks() == 0;
    }

    // Get m_block at iteration position idx (FORWARD order)
    // For mask blocks: idx=0 -> first mask block, idx=mask_block_cnt-1 -> last mask block
    // For full blocks: idx=0 -> first full block after mask blocks
    CUTLASS_DEVICE
    int get_mask_m_block(int idx) const {
        return mask_block_idx[mask_block_offset + idx];
    }

    CUTLASS_DEVICE
    int get_full_m_block(int idx) const {
        return full_block_idx[full_block_offset + idx];
    }

    CUTLASS_DEVICE
    int get_mask_dq_write_order(int idx) const {
        return dq_write_order[mask_block_offset + idx];
    }

    CUTLASS_DEVICE
    int get_full_dq_write_order(int idx) const {
        return dq_write_order_full[full_block_offset + idx];
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Block Sparse Backward Load Functions (Producer-side)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Load a list of m_blocks for backward pass.
 * First iteration loads K, V together with Q, LSE, dO, dPsum.
 * Subsequent iterations only load Q, LSE, dO, dPsum.
 *
 * @param block_indices Pointer to m_block indices
 * @param block_offset Starting offset in block_indices
 * @param block_count Number of blocks to load
 * @param load_kv_with_first Whether to load K,V with the first block
 * @param load_Q_LSE Lambda to load Q and LSE: (m_block, smem_pipe_write) -> void
 * @param load_dO_dPsum Lambda to load dO and dPsum: (m_block, smem_pipe_write_do) -> void
 * @param load_KV Lambda to load K and V (only called once): () -> void
 * @param smem_pipe_write Pipeline state for Q
 * @param smem_pipe_write_do Pipeline state for dO
 */
template <bool Q_dO_same_stages,
          typename LoadQLSE, typename LoadDODPsum, typename LoadKV,
          typename PipelineState, typename PipelineStateDO>
CUTLASS_DEVICE
void load_block_list_bwd(
    int const* block_indices,
    int block_offset,
    int block_count,
    bool load_kv_with_first,
    LoadQLSE&& load_Q_LSE,
    LoadDODPsum&& load_dO_dPsum,
    LoadKV&& load_KV,
    PipelineState& smem_pipe_write,
    PipelineStateDO& smem_pipe_write_do
) {
    if (block_count == 0) return;

    // First m_block
    int m_block = block_indices[block_offset];

    if (load_kv_with_first) {
        // Load K, V together with first Q, LSE
        load_KV();
    }
    load_Q_LSE(m_block, smem_pipe_write);

    // Load dO, dPsum for first block
    PipelineStateDO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
    load_dO_dPsum(m_block, smem_pipe_write_do_cur);

    if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
    ++smem_pipe_write;

    // Subsequent m_blocks
    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 1; i < block_count; ++i) {
        m_block = block_indices[block_offset + i];
        load_Q_LSE(m_block, smem_pipe_write);

        smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
        load_dO_dPsum(m_block, smem_pipe_write_do_cur);

        if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
        ++smem_pipe_write;
    }
}

/**
 * Main entry point for block sparse backward loading.
 *
 * Processing order:
 * 1. If mask blocks exist, process them first (with K,V loading on first block)
 * 2. Then process full blocks (K,V already loaded)
 */
template <bool Q_dO_same_stages,
          typename LoadQLSE, typename LoadDODPsum, typename LoadKV,
          typename PipelineState, typename PipelineStateDO>
CUTLASS_DEVICE
void produce_block_sparse_loads_bwd(
    BlockSparsityInfoBwd const& info,
    LoadQLSE&& load_Q_LSE,
    LoadDODPsum&& load_dO_dPsum,
    LoadKV&& load_KV,
    PipelineState& smem_pipe_write,
    PipelineStateDO& smem_pipe_write_do
) {
    bool const mask_empty = info.mask_block_cnt == 0;
    bool const full_empty = info.full_block_cnt == 0;

    if (mask_empty) {
        // No mask blocks: process full blocks only (with K,V loading)
        load_block_list_bwd<Q_dO_same_stages>(
            info.full_block_idx,
            info.full_block_offset,
            info.full_block_cnt,
            /*load_kv_with_first=*/true,
            std::forward<LoadQLSE>(load_Q_LSE),
            std::forward<LoadDODPsum>(load_dO_dPsum),
            std::forward<LoadKV>(load_KV),
            smem_pipe_write,
            smem_pipe_write_do
        );
    } else {
        // Process mask blocks first (with K,V loading)
        load_block_list_bwd<Q_dO_same_stages>(
            info.mask_block_idx,
            info.mask_block_offset,
            info.mask_block_cnt,
            /*load_kv_with_first=*/true,
            std::forward<LoadQLSE>(load_Q_LSE),
            std::forward<LoadDODPsum>(load_dO_dPsum),
            std::forward<LoadKV>(load_KV),
            smem_pipe_write,
            smem_pipe_write_do
        );

        if (!full_empty) {
            // Process full blocks (K,V already loaded)
            load_block_list_bwd<Q_dO_same_stages>(
                info.full_block_idx,
                info.full_block_offset,
                info.full_block_cnt,
                /*load_kv_with_first=*/false,
                std::forward<LoadQLSE>(load_Q_LSE),
                std::forward<LoadDODPsum>(load_dO_dPsum),
                std::forward<LoadKV>(load_KV),
                smem_pipe_write,
                smem_pipe_write_do
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Block Sparse Backward MMA Functions (Consumer-side)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Consume block sparse MMA for backward pass.
 *
 * Processing order (matches producer):
 * 1. Process mask blocks with arbitrary mask (mask_fn)
 * 2. Process full blocks without arbitrary mask (non_mask_fn)
 *
 * @param info Block sparsity info for the current n_block
 * @param bwd_step Lambda for one backward step: (m_block, mask_fn) -> void
 * @param mask_fn Mask function with arbitrary mask (for mask_blocks)
 * @param non_mask_fn Mask function without arbitrary mask (for full_blocks)
 */
template <typename BwdStep, typename MaskFn, typename NonMaskFn>
CUTLASS_DEVICE
void consume_block_sparse_mma_bwd(
    BlockSparsityInfoBwd const& info,
    BwdStep&& bwd_step,
    MaskFn&& mask_fn,
    NonMaskFn&& non_mask_fn
) {
    // Process mask_blocks (need arbitrary mask)
    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 0; i < info.mask_block_cnt; ++i) {
        int m_block = info.get_mask_m_block(i);
        bwd_step(m_block, mask_fn);
    }

    // Process full_blocks (no arbitrary mask)
    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 0; i < info.full_block_cnt; ++i) {
        int m_block = info.get_full_m_block(i);
        bwd_step(m_block, non_mask_fn);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Block Sparse Backward Store dQ Functions
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Iterate over all m_blocks for store_dq with block sparsity.
 * Simply iterates through mask blocks and full blocks.
 *
 * @param info Block sparsity info for the current n_block
 * @param store_step Lambda to store dQ for one m_block:
 *                   (m_block, compact_write_rank) -> void
 */
template <bool Deterministic, typename StoreStep>
CUTLASS_DEVICE
void store_dq_block_sparse(
    BlockSparsityInfoBwd const& info,
    StoreStep&& store_step
) {
    // Process mask_blocks
    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 0; i < info.mask_block_cnt; ++i) {
        int m_block = info.get_mask_m_block(i);
        int write_rank = 0;
        if constexpr (Deterministic) {
            write_rank = info.get_mask_dq_write_order(i);
        }
        store_step(m_block, write_rank);
    }

    // Process full_blocks
    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 0; i < info.full_block_cnt; ++i) {
        int m_block = info.get_full_m_block(i);
        int write_rank = 0;
        if constexpr (Deterministic) {
            write_rank = info.get_full_dq_write_order(i);
        }
        store_step(m_block, write_rank);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// SM80 Block Sparse Functions (uses cp.async, simpler pipeline)
// SM80 does not use IntraWGOverlap, so the pipeline is simpler.
// Pipeline state is just an integer (smem_pipe_write) instead of PipelineState struct.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Load block list for SM80 (non-overlap, cp.async based).
 * SM80 uses cp.async instead of TMA and has simpler pipeline management.
 *
 * @tparam PagedKV Whether using paged KV
 * @param block_idx Block indices array
 * @param block_offset Offset into block_idx
 * @param block_count Number of blocks
 * @param load_first_seqlenk_mask Whether first block needs seqlenk masking
 * @param load_K Lambda to load K: (n_block, smem_pipe_write, seqlenk_mask_type) -> void
 * @param load_V Lambda to load V: (n_block, smem_pipe_write, seqlenk_mask_type) -> void
 * @param paged_kv_manager PagedKV manager
 * @param smem_pipe_write Pipeline write index (integer)
 * @param kStages Number of pipeline stages
 */
template <bool PagedKV,
          class LoadK, class LoadV,
          class PagedKVManager>
CUTLASS_DEVICE
void load_block_list_sm80(
    int const* block_idx,
    int block_offset,
    int block_count,
    bool load_first_seqlenk_mask,
    LoadK&& load_K,
    LoadV&& load_V,
    PagedKVManager& paged_kv_manager,
    int& smem_pipe_write,
    int kStages
) {
    if (block_count <= 0) return;

    // First block (reverse order: block_count-1 is first to process)
    int n_block = block_idx[block_offset + block_count - 1];

    if constexpr (PagedKV) {
        paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/>(n_block);
    }

    if (load_first_seqlenk_mask) {
        load_K(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
        cute::cp_async_fence();
        // V must also use Seqlenk_mask=true for the boundary block to zero-fill
        // out-of-bounds rows via cp.async ZFILL (IEEE 754: 0 * NaN = NaN).
        load_V(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
        cute::cp_async_fence();
    } else {
        load_K(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
        cute::cp_async_fence();
        load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
        cute::cp_async_fence();
    }
    smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;

    // Remaining blocks (reverse order)
    for (int i = 1; i < block_count; ++i) {
        n_block = block_idx[block_offset + block_count - 1 - i];

        if constexpr (PagedKV) {
            paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block);
        }

        load_K(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
        cute::cp_async_fence();
        load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
        cute::cp_async_fence();
        smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
    }
}

/**
 * Main entry point for SM80 block sparse KV loading.
 * Uses cp.async for async memory copy and simple integer pipeline counters.
 *
 * @tparam PagedKV Whether using paged KV
 * @param info Block sparsity info for the current m_block
 * @param load_K Lambda to load K
 * @param load_V Lambda to load V
 * @param paged_kv_manager PagedKV manager
 * @param smem_pipe_write Pipeline write index
 * @param kStages Number of pipeline stages
 */
template <bool PagedKV,
          class LoadK, class LoadV,
          class PagedKVManager>
CUTLASS_DEVICE
void produce_block_sparse_loads_sm80(
    BlockSparsityInfo const& info,
    LoadK&& load_K,
    LoadV&& load_V,
    PagedKVManager& paged_kv_manager,
    int& smem_pipe_write,
    int kStages
) {
    bool const mask_empty = (info.mask_block_cnt == 0);
    bool const full_empty = (info.full_block_cnt == 0);

    if (mask_empty) {
        // No mask blocks: load full blocks only
        load_block_list_sm80<PagedKV>(
            info.full_block_idx,
            info.full_block_offset,
            info.full_block_cnt,
            /*load_first_seqlenk_mask=*/true,
            load_K, load_V,
            paged_kv_manager, smem_pipe_write, kStages
        );
    } else {
        // Load mask blocks first
        load_block_list_sm80<PagedKV>(
            info.mask_block_idx,
            info.mask_block_offset,
            info.mask_block_cnt,
            /*load_first_seqlenk_mask=*/true,
            load_K, load_V,
            paged_kv_manager, smem_pipe_write, kStages
        );

        // Then load full blocks (no need for first seqlenk mask)
        if (!full_empty) {
            load_block_list_sm80<PagedKV>(
                info.full_block_idx,
                info.full_block_offset,
                info.full_block_cnt,
                /*load_first_seqlenk_mask=*/false,
                load_K, load_V,
                paged_kv_manager, smem_pipe_write, kStages
            );
        }
    }
}

/**
 * Consume block sparse MMA for SM80 forward pass (no IntraWGOverlap).
 * This version includes support for prefetching next K block.
 *
 * @param info Block sparsity info
 * @param fwd_step Lambda for one forward step: (n_block, next_n_block, mask_fn, is_first_type, check_inf_type) -> void
 *                 where next_n_block is -1 if there is no next block
 * @param arbitrary_mask_fn Mask function with arbitrary mask
 * @param no_mask_fn No mask function
 */
template <class FwdStep, class ArbitraryMaskFn, class NoMaskFn>
CUTLASS_DEVICE
void consume_block_sparse_loads_sm80(
    BlockSparsityInfo const& info,
    FwdStep&& fwd_step,
    ArbitraryMaskFn&& arbitrary_mask_fn,
    NoMaskFn&& no_mask_fn
) {
    int const curr_mask_cnt = info.mask_block_cnt;
    int const curr_mask_offset = info.mask_block_offset;
    int const curr_full_cnt = info.full_block_cnt;
    int const curr_full_offset = info.full_block_offset;

    // Helper to get next block index
    auto get_next_n_block = [&](int curr_idx_in_mask, int curr_idx_in_full, bool in_mask_blocks) -> int {
        if (in_mask_blocks) {
            // Currently in mask blocks
            if (curr_idx_in_mask < curr_mask_cnt - 1) {
                // Next mask block exists
                return info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 2 - curr_idx_in_mask];
            } else if (curr_full_cnt > 0) {
                // Transition to first full block
                return info.full_block_idx[curr_full_offset + curr_full_cnt - 1];
            }
        } else {
            // Currently in full blocks
            if (curr_idx_in_full < curr_full_cnt - 1) {
                // Next full block exists
                return info.full_block_idx[curr_full_offset + curr_full_cnt - 2 - curr_idx_in_full];
            }
        }
        return -1;  // No next block
    };

    // Process mask_blocks (need arbitrary mask, check_inf=true)
    if (curr_mask_cnt > 0) {
        // First mask_block: is_first=true, check_inf=true
        int n_block = info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 1];
        int next_n_block = get_next_n_block(0, 0, true);
        fwd_step(n_block, next_n_block, arbitrary_mask_fn, cute::true_type{} /*is_first*/, cute::true_type{} /*check_inf*/);

        // Subsequent mask_blocks: is_first=false, check_inf=true
        CUTLASS_PRAGMA_NO_UNROLL
        for (int i = 1; i < curr_mask_cnt; ++i) {
            n_block = info.mask_block_idx[curr_mask_offset + curr_mask_cnt - 1 - i];
            next_n_block = get_next_n_block(i, 0, true);
            fwd_step(n_block, next_n_block, arbitrary_mask_fn, cute::false_type{} /*is_first*/, cute::true_type{} /*check_inf*/);
        }
    }

    // Process full_blocks (no mask, check_inf=false)
    if (curr_full_cnt > 0) {
        int n_block = info.full_block_idx[curr_full_offset + curr_full_cnt - 1];
        int next_n_block = get_next_n_block(0, 0, false);

        if (curr_mask_cnt == 0) {
            // No mask_blocks: first full_block is_first=true
            fwd_step(n_block, next_n_block, no_mask_fn, cute::true_type{} /*is_first*/, cute::false_type{} /*check_inf*/);
        } else {
            // Has mask_blocks: is_first=false
            fwd_step(n_block, next_n_block, no_mask_fn, cute::false_type{} /*is_first*/, cute::false_type{} /*check_inf*/);
        }

        // Subsequent full_blocks: is_first=false, check_inf=false
        CUTLASS_PRAGMA_NO_UNROLL
        for (int i = 1; i < curr_full_cnt; ++i) {
            n_block = info.full_block_idx[curr_full_offset + curr_full_cnt - 1 - i];
            next_n_block = get_next_n_block(0, i, false);
            fwd_step(n_block, next_n_block, no_mask_fn, cute::false_type{} /*is_first*/, cute::false_type{} /*check_inf*/);
        }
    }
}

}  // namespace flash
