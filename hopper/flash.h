/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t v_dim_stride;

    // The number of heads.
    int h, h_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
    using index_t = int64_t;

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // For FP8 scaling
    float * __restrict__ q_descale_ptr;
    float * __restrict__ k_descale_ptr;
    float * __restrict__ v_descale_ptr;
    index_t q_descale_batch_stride;
    index_t q_descale_head_stride;
    index_t k_descale_batch_stride;
    index_t k_descale_head_stride;
    index_t v_descale_batch_stride;
    index_t v_descale_head_stride;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
    int total_q, total_k, total_knew;
    int b_k;  // When having KV cache and with cache_batch_idx, K & V might have larger batch size than Q
    int dv, dv_rounded;  // For the case where V headdim is different from Q/K headdim

    // The scaling factors for the kernel.
    float scale_softmax;
    float softcap;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ cu_seqlens_knew;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each q/k sequence.
    int *__restrict__ seqused_q;
    int *__restrict__ seqused_k;

    // The stride between rows of Oaccum.
    index_t oaccum_split_stride;
    index_t oaccum_batch_stride;
    index_t oaccum_row_stride;
    index_t oaccum_head_stride;

    // The stride between rows of LSEaccum.
    index_t lseaccum_split_stride;
    index_t lseaccum_batch_stride;
    index_t lseaccum_head_stride;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    void *__restrict__ qv_ptr;
    index_t qv_batch_stride;
    index_t qv_row_stride;
    index_t qv_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;
    int *__restrict__ seqlens_rotary;

    // The indices to index into the KV cache.
    int * __restrict__ kv_batch_idx;

    // Paged KV cache
    int * __restrict__ page_table;
    index_t page_table_batch_stride;
    int page_size;
    int num_pages;
    bool pagedkv_tma;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    int attention_chunk;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_fp32;
    bool is_e4m3;
    bool is_causal;
    bool is_local;
    bool is_arbitrary;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version
    bool pack_gqa;

    int * __restrict__ tile_count_semaphore;
    int * __restrict__ num_m_blocks_ptr;
    // int * __restrict__ num_n_blocks_ptr;
    int * __restrict__ num_splits_dynamic_ptr;
    int * __restrict__ varlen_batch_idx_ptr; // virtual -> actual
    int * __restrict__ num_nheads_in_l2_ptr;
    bool skip_scheduler_metadata_computation;
    bool varlen_sort_batches;
    int tile_count_semaphore_offset;
    bool head_swizzle;
    bool prepare_varlen_pdl;

    // Block sparsity parameters (Q2K direction)
    // For each m_block, we have lists of n_blocks to process:
    // mask_block: blocks requiring masking, full_block: blocks without masking
    // batch and head_q can be 1 for broadcasting
    bool use_block_sparsity;
    int * __restrict__ block_sparse_mask_cnt;      // [batch, head_q, num_m_blocks]: count of mask blocks per m_block
    int * __restrict__ block_sparse_mask_offset;   // [batch * head_q * num_m_blocks+1]: cumulative offset into mask_idx
    int * __restrict__ block_sparse_mask_idx;      // [total_mask_blocks]: n_block indices for mask blocks
    int * __restrict__ block_sparse_full_cnt;      // [batch, head_q, num_m_blocks]: count of full blocks per m_block
    int * __restrict__ block_sparse_full_offset;   // [batch * head_q * num_m_blocks+1]: cumulative offset into full_idx
    int * __restrict__ block_sparse_full_idx;      // [total_full_blocks]: n_block indices for full blocks
    int block_sparse_num_blocks;                   // num_m_blocks (for computing flat index)
    int block_sparse_num_heads;                    // num_heads (for computing flat index, can be 1 for broadcasting)
    int block_sparse_num_batches;                  // num_batches (for computing flat index, can be 1 for broadcasting)
    // Deterministic backward write ranks, parallel to the compact K2Q idx arrays.
    int * __restrict__ block_sparse_dq_write_order = nullptr;
    int * __restrict__ block_sparse_dq_write_order_full = nullptr;

    // Arbitrary mask function tensor for element-level masking within mask_blocks
    // Shape: [batch or 1, head_q or 1, func_num, seqlen_q + 256] where func_num is odd, 1 for broadcasting if head_q is 1 or batch is 1
    // For each row, stores valid column ranges:
    //   col_max[0] = arbitrary_func[batch, 0, 0, row] - first valid range upper bound
    //   col_min[i] = arbitrary_func[batch, 0, 2*i+1, row] - (i+1)th valid range lower bound
    //   col_max[i+1] = arbitrary_func[batch, 0, 2*i+2, row] - (i+1)th valid range upper bound
    // Arbitrary mask function tensor for element-level masking (controlled by Is_arbitrary template param)
    int * __restrict__ mask_func_ptr;              // [batch or 1, head_q or 1, func_num, seqlen_q + 256], contiguous (seq_stride = 1)
    // Shape of mask_func tensor (for CuTe tensor construction)
    int func_seqlen;                               // arbitrary_func.size(3), >= seqlen_q + 256
    int arbitrary_func_num;                        // arbitrary_func.size(2), func_num (must be odd), passed as template parameter
    int func_head;                                 // arbitrary_func.size(1), should be head_q (num heads)
    int func_batch;                                // arbitrary_func.size(0), should be batch (batch size), 1 for broadcasting if head_q is 1 or batch is 1
    // Strides of mask_func tensor
    index_t func_batch_stride;                         // arbitrary_func.stride(0)
    index_t func_head_stride;                          // arbitrary_func.stride(1)
    index_t func_nfunc_stride;                         // arbitrary_func.stride(2)
    // Note: seq_stride is 1 (contiguous)

    int arch;
    int num_sm;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {
    using index_t = int64_t;

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;
    void *__restrict__ softmax_lse_log2_ptr;

    int *__restrict__ dq_semaphore;
    int *__restrict__ dk_semaphore;
    int *__restrict__ dv_semaphore;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA, int kNFunc>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa, int blockM, int blockN, bool enable_pdl);
template <int Arch, typename T, int kHeadDim, bool Has_softcap, int kNFunc>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
template <typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);
