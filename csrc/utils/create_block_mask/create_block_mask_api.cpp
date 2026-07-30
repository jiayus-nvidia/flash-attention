#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "create_block_mask.h"
#include "../../../hopper/tile_size.h"

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// ============================================================================
// Tile size computation helpers
// Hopper C++ kernels use hopper/tile_size.h as the single source of truth.
// ============================================================================

/**
 * Get GPU architecture as int (80, 86, 89, 90, 100, etc.)
 */
inline int get_gpu_arch() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

/**
 * Match hopper/flash_api{,_stable}.cpp for builds with disabled hdim buckets.
 */
inline int round_up_hopper_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

/**
 * Get forward pass tile sizes based on architecture.
 * Uses Hopper C++ tile_size.h for SM8x/SM90 and the FA4 selector contract for
 * Blackwell.
 *
 * @param arch GPU architecture (80, 86, 89, 90, 100). If -1, auto-detect.
 * @param headdim Head dimension
 * @param is_causal Whether using causal attention
 * @param is_local Whether using local attention
 * @param is_arbitrary Whether using arbitrary mask
 *
 * @return Pair of (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for forward pass
 */
inline std::pair<int, int> get_fwd_tile_sizes(
    int arch,
    int headdim,
    bool is_causal = false,
    bool is_local = false,
    bool is_arbitrary = false,
    bool paged_kv = false,
    bool varlen_and_split = false
) {
    if (arch < 0) {
        arch = get_gpu_arch();
    }
    int const hopper_rounded_headdim = round_up_hopper_headdim(headdim);

    if (arch >= 100) {
        return {128, 128};
    } else if (arch >= 90) {
        auto const tile = tile_size_fwd_sm90(
            hopper_rounded_headdim,
            hopper_rounded_headdim,
            is_causal,
            is_local,
            is_arbitrary,
            /*element_size=*/2,
            /*v_colmajor=*/false,
            /*paged_kv_non_TMA=*/paged_kv);
        return {std::get<0>(tile), std::get<1>(tile)};
    } else {
        auto const tile = tile_size_fwd_sm8x(
            arch == 86 || arch == 89,
            hopper_rounded_headdim,
            hopper_rounded_headdim,
            is_causal,
            is_local,
            is_arbitrary,
            /*element_size=*/2,
            paged_kv,
            varlen_and_split);
        return {std::get<0>(tile), std::get<1>(tile)};
    }
}

/**
 * Get backward pass tile sizes based on architecture.
 * Uses Hopper C++ tile_size.h for SM8x/SM90 and the FA4 selector contract for
 * Blackwell.
 *
 * @param arch GPU architecture (80, 86, 89, 90, 100). If -1, auto-detect.
 * @param headdim Head dimension
 * @param is_causal Whether using causal attention
 * @param is_local Whether using local attention
 * @param is_arbitrary Whether using arbitrary mask
 * @param has_softcap Whether using softcap
 * @param headdim_v Value head dimension; negative means the same as headdim
 *
 * @return Pair of (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for backward pass
 */
inline std::pair<int, int> get_bwd_tile_sizes(
    int arch,
    int headdim,
    bool is_causal = false,
    bool is_local = false,
    bool is_arbitrary = false,
    bool has_softcap = false,
    int headdim_v = -1
) {
    if (arch < 0) {
        arch = get_gpu_arch();
    }

    if (headdim_v < 0) {
        headdim_v = headdim;
    }
    int const hopper_rounded_headdim =
        round_up_hopper_headdim(std::max(headdim, headdim_v));

    if (arch >= 100 && arch < 120) {
        if (headdim == 256 && headdim_v == 256) {
            return {256, 128};
        }
        const bool use_2cta_linear_csr =
            is_arbitrary && !is_causal && !is_local && !has_softcap &&
            ((headdim == 128 && headdim_v == 128) ||
             (headdim == 192 && headdim_v == 128));
        if (use_2cta_linear_csr) {
            return {128, 256};
        }
        return {128, 128};
    } else if (arch >= 120) {
        return {128, 128};
    } else if (arch >= 90) {
        // Hopper C++ kernels round and specialize on max(Q/K dim, V dim).
        // Keep this selector identical to hopper/tile_size.h, including the
        // arbitrary/softcap cases that use a smaller M tile.
        auto const tile = tile_size_bwd_sm90(
            hopper_rounded_headdim,
            is_causal, is_local, is_arbitrary, has_softcap);
        return {std::get<0>(tile), std::get<1>(tile)};
    } else {
        // SM86/SM89 share one tuned SM8x configuration, while SM80 uses a
        // different tile for several head dimensions.
        auto const tile = tile_size_bwd_sm8x(
            arch == 86 || arch == 89,
            hopper_rounded_headdim,
            is_causal, is_local, is_arbitrary, has_softcap);
        return {std::get<0>(tile), std::get<1>(tile)};
    }
}

/**
 * Q2K (Forward): Convert function encoding tensor to block sparse tensors.
 *
 * Forward pass: fix q_block, loop kv_blocks.
 *
 * Function encoding format (same as magi_to_hstu output):
 *   interval 0: [0, F0)       - valid range (no mask needed)
 *   interval 1: [F1, F2)      - valid range (no mask needed)
 *   interval 2: [F3, F4)      - valid range (no mask needed)
 *   ...
 *
 * These intervals represent positions where attention scores are VALID (not masked).
 *
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 *                     func_q_len must be >= Q_LEN + 256 (to avoid bounds checking in kernel)
 *                     n_func must be odd (e.g., 1, 3, 5, 7, ...)
 *                     Supports non-contiguous tensors (uses strides for access)
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 128)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * @param check_q_boundary: if true (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks;
 *                          if false, partial q_blocks can have FULL kv_blocks (default: false)
 * @param debug: if true, initialize output tensors to -1 for debugging; if false, use uninitialized memory (default: false)
 *
 * @return tuple of (mask_block_cnt, full_block_cnt, block_idx)
 *         mask_block_cnt: [B, H, num_q_blocks], int32 - number of partial kv_blocks per q_block
 *         full_block_cnt: [B, H, num_q_blocks], int32 - number of full kv_blocks per q_block
 *         block_idx: [B, H, num_q_blocks, num_kv_blocks], int32 - combined block indices
 *                    full blocks stored left-to-right (0, 1, 2, ...)
 *                    mask blocks stored right-to-left (num_kv_blocks-1, num_kv_blocks-2, ...)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> create_q2k_block_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 128,
    int KV_BLOCK_SIZE = 128,
    bool check_q_boundary = false,
    bool debug = false
) {
    // Input validation
    TORCH_CHECK(func_tensor.is_cuda(), "func_tensor must be a CUDA tensor");
    TORCH_CHECK(func_tensor.dtype() == torch::kInt32, "func_tensor must be int32");
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D [B, H, n_func, func_q_len]");
    TORCH_CHECK(Q_LEN > 0, "Q_LEN must be positive");
    TORCH_CHECK(KV_LEN > 0, "KV_LEN must be positive");
    TORCH_CHECK(Q_BLOCK_SIZE > 0, "Q_BLOCK_SIZE must be positive");
    TORCH_CHECK(KV_BLOCK_SIZE > 0, "KV_BLOCK_SIZE must be positive");

    int B = func_tensor.size(0);
    int H = func_tensor.size(1);
    int n_func = func_tensor.size(2);
    int func_q_len = func_tensor.size(3);

    // Get strides for non-contiguous tensor support
    int stride_b = func_tensor.stride(0);
    int stride_h = func_tensor.stride(1);
    int stride_f = func_tensor.stride(2);
    int stride_q = func_tensor.stride(3);

    // Check that func_tensor has enough padding to avoid bounds checking in kernel
    TORCH_CHECK(func_q_len >= Q_LEN + 256,
                "func_tensor.size(3) must be >= Q_LEN + 256 to avoid bounds checking in kernel. "
                "Got func_tensor.size(3)=", func_q_len, ", Q_LEN=", Q_LEN);
    TORCH_CHECK(n_func % 2 == 1, "n_func must be odd (e.g., 1, 3, 5, 7, ...)");

    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);

    // Ensure tensors are on the same device
    at::cuda::CUDAGuard device_guard(func_tensor.device());

    // Allocate output tensors (always contiguous)
    auto opts = func_tensor.options().dtype(torch::kInt32);
    at::Tensor mask_block_cnt, full_block_cnt, block_idx;
    if (debug) {
        // Initialize to -1 for debugging (helps detect unexecuted kernels)
        mask_block_cnt = torch::full({B, H, num_q_blocks}, -1, opts);
        full_block_cnt = torch::full({B, H, num_q_blocks}, -1, opts);
        block_idx = torch::full({B, H, num_q_blocks, num_kv_blocks}, -1, opts);
    } else {
        // Use uninitialized memory for performance
        mask_block_cnt = torch::empty({B, H, num_q_blocks}, opts);
        full_block_cnt = torch::empty({B, H, num_q_blocks}, opts);
        block_idx = torch::empty({B, H, num_q_blocks, num_kv_blocks}, opts);
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    launch_create_q2k_block_sparse_from_func(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        check_q_boundary,
        mask_block_cnt.data_ptr<int>(),
        full_block_cnt.data_ptr<int>(),
        block_idx.data_ptr<int>(),
        stream
    );

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx);
}

/**
 * K2Q (Backward): Convert function encoding tensor to block sparse tensors.
 *
 * Backward pass: fix kv_block, loop q_blocks.
 * Always performs boundary checking (partial q_blocks cannot have FULL status).
 *
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 128)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * @param debug: if true, initialize output tensors to -1 for debugging; if false, use uninitialized memory (default: false)
 *
 * @return tuple of (mask_block_cnt, full_block_cnt, block_idx)
 *         mask_block_cnt: [B, H, num_kv_blocks], int32 - number of partial q_blocks per kv_block
 *         full_block_cnt: [B, H, num_kv_blocks], int32 - number of full q_blocks per kv_block
 *         block_idx: [B, H, num_kv_blocks, num_q_blocks], int32 - combined block indices
 *                    full blocks stored left-to-right (0, 1, 2, ...)
 *                    mask blocks stored right-to-left (num_q_blocks-1, num_q_blocks-2, ...)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> create_k2q_block_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 128,
    int KV_BLOCK_SIZE = 128,
    bool debug = false
) {
    // Input validation
    TORCH_CHECK(func_tensor.is_cuda(), "func_tensor must be a CUDA tensor");
    TORCH_CHECK(func_tensor.dtype() == torch::kInt32, "func_tensor must be int32");
    TORCH_CHECK(func_tensor.dim() == 4, "func_tensor must be 4D [B, H, n_func, func_q_len]");
    TORCH_CHECK(Q_LEN > 0, "Q_LEN must be positive");
    TORCH_CHECK(KV_LEN > 0, "KV_LEN must be positive");
    TORCH_CHECK(Q_BLOCK_SIZE > 0, "Q_BLOCK_SIZE must be positive");
    TORCH_CHECK(KV_BLOCK_SIZE > 0, "KV_BLOCK_SIZE must be positive");

    int B = func_tensor.size(0);
    int H = func_tensor.size(1);
    int n_func = func_tensor.size(2);
    int func_q_len = func_tensor.size(3);

    // Get strides for non-contiguous tensor support
    int stride_b = func_tensor.stride(0);
    int stride_h = func_tensor.stride(1);
    int stride_f = func_tensor.stride(2);
    int stride_q = func_tensor.stride(3);

    // Check that func_tensor has enough padding to avoid bounds checking in kernel
    TORCH_CHECK(func_q_len >= Q_LEN + 256,
                "func_tensor.size(3) must be >= Q_LEN + 256 to avoid bounds checking in kernel. "
                "Got func_tensor.size(3)=", func_q_len, ", Q_LEN=", Q_LEN);
    TORCH_CHECK(n_func % 2 == 1, "n_func must be odd (e.g., 1, 3, 5, 7, ...)");

    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);

    // Ensure tensors are on the same device
    at::cuda::CUDAGuard device_guard(func_tensor.device());

    // Allocate output tensors (always contiguous)
    // Note: For K2Q, output shape is [B, H, num_kv_blocks, num_q_blocks]
    auto opts = func_tensor.options().dtype(torch::kInt32);
    at::Tensor mask_block_cnt, full_block_cnt, block_idx;
    if (debug) {
        // Initialize to -1 for debugging (helps detect unexecuted kernels)
        mask_block_cnt = torch::full({B, H, num_kv_blocks}, -1, opts);
        full_block_cnt = torch::full({B, H, num_kv_blocks}, -1, opts);
        block_idx = torch::full({B, H, num_kv_blocks, num_q_blocks}, -1, opts);
    } else {
        // Use uninitialized memory for performance
        mask_block_cnt = torch::empty({B, H, num_kv_blocks}, opts);
        full_block_cnt = torch::empty({B, H, num_kv_blocks}, opts);
        block_idx = torch::empty({B, H, num_kv_blocks, num_q_blocks}, opts);
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int* d_q_block_kv_min = nullptr;
    int* d_q_block_kv_max = nullptr;
    at::Tensor q_block_kv_min, q_block_kv_max;

#ifndef DISABLE_KV_RANGE_OPT
    // Default (optimization enabled):
    // - Q2K kernel uses inline kv_range computation
    // - K2Q kernel uses precompute optimization:
    //   First launch a kernel to compute q_block kv ranges,
    //   then K2Q kernel can skip q_blocks that don't overlap with the current kv_block
    q_block_kv_min = torch::empty({B, H, num_q_blocks}, opts);
    q_block_kv_max = torch::empty({B, H, num_q_blocks}, opts);

    launch_compute_q_block_kv_range(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, n_func,
        Q_BLOCK_SIZE,
        q_block_kv_min.data_ptr<int>(),
        q_block_kv_max.data_ptr<int>(),
        stream
    );

    d_q_block_kv_min = q_block_kv_min.data_ptr<int>();
    d_q_block_kv_max = q_block_kv_max.data_ptr<int>();
#endif  // DISABLE_KV_RANGE_OPT

    // Step 2: Launch K2Q kernel (with or without precomputed q_block kv ranges)
    launch_create_k2q_block_sparse_from_func(
        func_tensor.data_ptr<int>(),
        stride_b, stride_h, stride_f, stride_q,
        B, H, Q_LEN, KV_LEN, n_func,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        d_q_block_kv_min,
        d_q_block_kv_max,
        mask_block_cnt.data_ptr<int>(),
        full_block_cnt.data_ptr<int>(),
        block_idx.data_ptr<int>(),
        stream
    );

    return std::make_tuple(mask_block_cnt, full_block_cnt, block_idx);
}

/**
 * Compact block indices from BHQK format to linear sparse format (CSR-like).
 *
 * Converts the combined block_idx tensor to compact mask_block_idx and full_block_idx tensors
 * using prefix sum (cumulative sum) to compute offsets.
 *
 * Input format (from create_q2k/k2q_block_sparse_from_func):
 *   - mask_block_cnt: [B, H, num_blocks], count of mask blocks per row
 *   - full_block_cnt: [B, H, num_blocks], count of full blocks per row
 *   - block_idx: [B, H, num_blocks, max_blocks], combined indices
 *       - full blocks stored left-to-right: block_idx[..., 0:full_cnt]
 *       - mask blocks stored right-to-left: block_idx[..., max_blocks-mask_cnt:max_blocks]
 *
 * Output format (LinearBlockSparseTensorsTorch compatible):
 *   - mask_block_cnt: [B, H, num_blocks] (unchanged)
 *   - mask_block_offset: [B * H * num_blocks + 1], exclusive prefix sum (flattened, starts with 0)
 *   - mask_block_idx: [total_mask_blocks], compact indices
 *   - full_block_cnt: [B, H, num_blocks] (unchanged)
 *   - full_block_offset: [B * H * num_blocks + 1], exclusive prefix sum (flattened, starts with 0)
 *   - full_block_idx: [total_full_blocks], compact indices
 *
 * Memory savings: O(num_blocks * max_blocks) -> O(total_mask_blocks + total_full_blocks)
 *
 * @param mask_block_cnt: [B, H, num_blocks], int32
 * @param full_block_cnt: [B, H, num_blocks], int32
 * @param block_idx: [B, H, num_blocks, max_blocks], int32
 *
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx_compact,
 *                   full_block_cnt, full_block_offset, full_block_idx_compact)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compact_block_idx(
    const at::Tensor& mask_block_cnt,
    const at::Tensor& full_block_cnt,
    const at::Tensor& block_idx
) {
    // Input validation
    TORCH_CHECK(mask_block_cnt.is_cuda(), "mask_block_cnt must be a CUDA tensor");
    TORCH_CHECK(full_block_cnt.is_cuda(), "full_block_cnt must be a CUDA tensor");
    TORCH_CHECK(block_idx.is_cuda(), "block_idx must be a CUDA tensor");
    TORCH_CHECK(mask_block_cnt.dtype() == torch::kInt32, "mask_block_cnt must be int32");
    TORCH_CHECK(full_block_cnt.dtype() == torch::kInt32, "full_block_cnt must be int32");
    TORCH_CHECK(block_idx.dtype() == torch::kInt32, "block_idx must be int32");
    TORCH_CHECK(mask_block_cnt.dim() == 3, "mask_block_cnt must be 3D [B, H, num_blocks]");
    TORCH_CHECK(full_block_cnt.dim() == 3, "full_block_cnt must be 3D [B, H, num_blocks]");
    TORCH_CHECK(block_idx.dim() == 4, "block_idx must be 4D [B, H, num_blocks, max_blocks]");

    // Ensure same device
    at::cuda::CUDAGuard device_guard(mask_block_cnt.device());

    int64_t B = mask_block_cnt.size(0);
    int64_t H = mask_block_cnt.size(1);
    int64_t num_blocks = mask_block_cnt.size(2);
    int64_t max_blocks = block_idx.size(3);

    auto opts = mask_block_cnt.options();

    // Ensure contiguous
    at::Tensor mask_cnt = mask_block_cnt.contiguous();
    at::Tensor full_cnt = full_block_cnt.contiguous();
    at::Tensor block_idx_cont = block_idx.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Create offset arrays with leading 0 for exclusive prefix sum
    // offset shape: [B * H * num_blocks + 1], format: [0, c0, c0+c1, ..., total]
    int64_t n_blocks_flat = B * H * num_blocks;
    int64_t offset_size = n_blocks_flat + 1;

    // Allocate offset tensors (uninitialized, kernel sets offset[0]=0)
    at::Tensor mask_block_offset = torch::empty({offset_size}, opts);
    at::Tensor full_block_offset = torch::empty({offset_size}, opts);

    // Compute exclusive prefix sum: offset[0]=0, offset[i+1]=sum(cnt[0..i])
    launch_dual_inclusive_sum(
        mask_cnt.data_ptr<int>(),
        full_cnt.data_ptr<int>(),
        mask_block_offset.data_ptr<int>(),
        full_block_offset.data_ptr<int>(),
        n_blocks_flat,  // number of count elements
        stream
    );

    // Synchronize to read total counts from the last element of offset arrays
    cudaStreamSynchronize(stream);

    // Read total counts from the last element of offset arrays
    int total_mask = mask_block_offset[-1].item<int>();
    int total_full = full_block_offset[-1].item<int>();

    // Allocate compact idx tensors (empty if total is 0)
    at::Tensor mask_block_idx_compact = torch::empty({total_mask}, opts);
    at::Tensor full_block_idx_compact = torch::empty({total_full}, opts);

    // Step 2: Extract indices using CUDA kernel
    if (total_mask > 0 || total_full > 0) {
        launch_extract_compact_indices(
            block_idx_cont.data_ptr<int>(),
            mask_cnt.data_ptr<int>(),
            full_cnt.data_ptr<int>(),
            mask_block_offset.data_ptr<int>(),
            full_block_offset.data_ptr<int>(),
            n_blocks_flat,
            max_blocks,
            mask_block_idx_compact.data_ptr<int>(),
            full_block_idx_compact.data_ptr<int>(),
            stream
        );
    }

    // Return tensors:
    // - mask_cnt, full_cnt: [B, H, num_blocks] (unchanged)
    // - mask_block_offset, full_block_offset: [B * H * num_blocks + 1] (flattened, starts with 0)
    // - mask_block_idx_compact, full_block_idx_compact: [total_mask], [total_full]
    return std::make_tuple(
        mask_cnt, mask_block_offset, mask_block_idx_compact,
        full_cnt, full_block_offset, full_block_idx_compact
    );
}

/**
 * Q2K CSR (Forward): Convert function encoding tensor directly to CSR sparse format.
 *
 * This is a convenience function that combines create_q2k_block_sparse_from_func and compact_block_idx.
 *
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 256)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 * @param check_q_boundary: if true (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks
 *
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx,
 *                   full_block_cnt, full_block_offset, full_block_idx)
 *         - mask_block_cnt: [B, H, num_q_blocks + 1], int32
 *         - mask_block_offset: [B * H * (num_q_blocks + 1)], int32 (flattened prefix sum)
 *         - mask_block_idx: [total_mask_blocks], int32
 *         - full_block_cnt: [B, H, num_q_blocks + 1], int32
 *         - full_block_offset: [B * H * (num_q_blocks + 1)], int32 (flattened prefix sum)
 *         - full_block_idx: [total_full_blocks], int32
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
create_q2k_csr_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 256,
    int KV_BLOCK_SIZE = 128,
    bool check_q_boundary = false
) {
    // Step 1: Create block sparse tensors
    auto [mask_block_cnt, full_block_cnt, block_idx] =
        create_q2k_block_sparse_from_func(
            func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary, false  // debug=false for performance
        );

    // Step 2: Compact to CSR format
    return compact_block_idx(mask_block_cnt, full_block_cnt, block_idx);
}

/**
 * K2Q CSR (Backward): Convert function encoding tensor directly to CSR sparse format.
 *
 * This is a convenience function that combines create_k2q_block_sparse_from_func and compact_block_idx.
 *
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param Q_BLOCK_SIZE: block size for query dimension (default: 128)
 * @param KV_BLOCK_SIZE: block size for key/value dimension (default: 128)
 *
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx,
 *                   full_block_cnt, full_block_offset, full_block_idx)
 *         - mask_block_cnt: [B, H, num_kv_blocks + 1], int32
 *         - mask_block_offset: [B * H * (num_kv_blocks + 1)], int32 (flattened prefix sum)
 *         - mask_block_idx: [total_mask_blocks], int32
 *         - full_block_cnt: [B, H, num_kv_blocks + 1], int32
 *         - full_block_offset: [B * H * (num_kv_blocks + 1)], int32 (flattened prefix sum)
 *         - full_block_idx: [total_full_blocks], int32
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
create_k2q_csr_sparse_from_func(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int Q_BLOCK_SIZE = 128,
    int KV_BLOCK_SIZE = 128
) {
    // Step 1: Create block sparse tensors
    auto [mask_block_cnt, full_block_cnt, block_idx] =
        create_k2q_block_sparse_from_func(
            func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            false  // debug=false for performance
        );

    // Step 2: Compact to CSR format
    return compact_block_idx(mask_block_cnt, full_block_cnt, block_idx);
}

// ============================================================================
// Auto tile size functions - automatically compute tile sizes from architecture
// ============================================================================

/**
 * Q2K CSR Auto (Forward): Convert with automatic tile size detection.
 *
 * Automatically detects GPU architecture and computes the correct tile sizes
 * based on headdim and mask configuration. Users don't need to specify tile sizes.
 *
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param headdim: head dimension (required for tile size computation)
 * @param is_causal: whether using causal attention
 * @param is_local: whether using local attention
 * @param is_arbitrary: whether using arbitrary mask (default: true since using func_tensor)
 * @param check_q_boundary: if true, partial q_blocks cannot have FULL kv_blocks
 *
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx,
 *                   full_block_cnt, full_block_offset, full_block_idx,
 *                   Q_BLOCK_SIZE, KV_BLOCK_SIZE)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int>
create_q2k_csr_sparse_auto(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int headdim,
    bool is_causal = false,
    bool is_local = false,
    bool is_arbitrary = true,
    bool check_q_boundary = false
) {
    // Auto-detect tile sizes
    auto [Q_BLOCK_SIZE, KV_BLOCK_SIZE] = get_fwd_tile_sizes(
        -1,  // auto-detect arch
        headdim,
        is_causal,
        is_local,
        is_arbitrary
    );

    // Create CSR sparse tensors
    auto [mask_block_cnt, mask_block_offset, mask_block_idx,
          full_block_cnt, full_block_offset, full_block_idx] =
        create_q2k_csr_sparse_from_func(
            func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary);

    return std::make_tuple(
        mask_block_cnt, mask_block_offset, mask_block_idx,
        full_block_cnt, full_block_offset, full_block_idx,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE
    );
}

/**
 * K2Q CSR Auto (Backward): Convert with automatic tile size detection.
 *
 * Automatically detects GPU architecture and computes the correct tile sizes
 * based on headdim and mask configuration. Users don't need to specify tile sizes.
 *
 * @param func_tensor: [B, H, n_func, func_q_len], int32, function encoding tensor
 * @param Q_LEN: query sequence length
 * @param KV_LEN: key/value sequence length
 * @param headdim: head dimension (required for tile size computation)
 * @param is_causal: whether using causal attention
 * @param is_local: whether using local attention
 * @param is_arbitrary: whether using arbitrary mask (default: true since using func_tensor)
 * @param has_softcap: whether using softcap
 *
 * @return tuple of (mask_block_cnt, mask_block_offset, mask_block_idx,
 *                   full_block_cnt, full_block_offset, full_block_idx,
 *                   Q_BLOCK_SIZE, KV_BLOCK_SIZE)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int>
create_k2q_csr_sparse_auto(
    const at::Tensor& func_tensor,
    int Q_LEN,
    int KV_LEN,
    int headdim,
    bool is_causal = false,
    bool is_local = false,
    bool is_arbitrary = true,
    bool has_softcap = false,
    int headdim_v = -1
) {
    TORCH_CHECK(func_tensor.is_cuda(), "func_tensor must be a CUDA tensor");
    at::cuda::CUDAGuard device_guard(func_tensor.device());

    // Auto-detect tile sizes
    auto [Q_BLOCK_SIZE, KV_BLOCK_SIZE] = get_bwd_tile_sizes(
        -1,  // auto-detect arch
        headdim,
        is_causal,
        is_local,
        is_arbitrary,
        has_softcap,
        headdim_v
    );

    // Create CSR sparse tensors
    auto [mask_block_cnt, mask_block_offset, mask_block_idx,
          full_block_cnt, full_block_offset, full_block_idx] =
        create_k2q_csr_sparse_from_func(
            func_tensor, Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE);

    return std::make_tuple(
        mask_block_cnt, mask_block_offset, mask_block_idx,
        full_block_cnt, full_block_offset, full_block_idx,
        Q_BLOCK_SIZE, KV_BLOCK_SIZE
    );
}

/**
 * Python-accessible function to get forward tile sizes.
 */
std::tuple<int, int> py_get_fwd_tile_sizes(
    int headdim,
    bool is_causal = false,
    bool is_local = false,
    bool is_arbitrary = true,
    int arch = -1
) {
    auto [Q_BLOCK_SIZE, KV_BLOCK_SIZE] = get_fwd_tile_sizes(
        arch, headdim, is_causal, is_local, is_arbitrary);
    return std::make_tuple(Q_BLOCK_SIZE, KV_BLOCK_SIZE);
}

/**
 * Python-accessible function to get backward tile sizes.
 */
std::tuple<int, int> py_get_bwd_tile_sizes(
    int headdim,
    bool is_causal = false,
    bool is_local = false,
    bool is_arbitrary = true,
    bool has_softcap = false,
    int arch = -1,
    int headdim_v = -1
) {
    auto [Q_BLOCK_SIZE, KV_BLOCK_SIZE] = get_bwd_tile_sizes(
        arch, headdim, is_causal, is_local, is_arbitrary, has_softcap, headdim_v);
    return std::make_tuple(Q_BLOCK_SIZE, KV_BLOCK_SIZE);
}

/**
 * Python-accessible function to get GPU architecture.
 */
int py_get_gpu_arch() {
    return get_gpu_arch();
}

int py_round_up_hopper_headdim(int headdim) {
    return round_up_hopper_headdim(headdim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA kernels for creating block sparse tensors (FlexAttention/BlockSparseTensorsTorch format)";

    // ========================================================================
    // Original functions with explicit tile sizes (for advanced users)
    // ========================================================================

    m.def("create_q2k_block_sparse_from_func", &create_q2k_block_sparse_from_func,
          "Q2K (Forward): Convert function encoding tensor to block sparse tensors. "
          "Fix q_block, loop kv_blocks. "
          "Returns (mask_block_cnt, full_block_cnt, block_idx) "
          "where block_idx contains full blocks left-to-right and mask blocks right-to-left.\n\n"
          "WARNING: Q_BLOCK_SIZE and KV_BLOCK_SIZE must match kernel tile sizes! "
          "Use create_q2k_csr_sparse_auto() instead for automatic tile size detection.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE")  = 256,  // default: 256x128 for forward pass
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("check_q_boundary") = false,
          py::arg("debug") = false);

    m.def("create_k2q_block_sparse_from_func", &create_k2q_block_sparse_from_func,
          "K2Q (Backward): Convert function encoding tensor to block sparse tensors. "
          "Fix kv_block, loop q_blocks. Always checks boundary. "
          "Returns (mask_block_cnt, full_block_cnt, block_idx) "
          "where block_idx contains full blocks left-to-right and mask blocks right-to-left.\n\n"
          "WARNING: Q_BLOCK_SIZE and KV_BLOCK_SIZE must match kernel tile sizes! "
          "Use create_k2q_csr_sparse_auto() instead for automatic tile size detection.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 128,  // default: 128x128 for backward pass
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("debug") = false);

    m.def("compact_block_idx", &compact_block_idx,
          "Compact block indices from BHQK format to linear sparse format (CSR-like). "
          "Converts combined block_idx tensor to compact mask_block_idx and full_block_idx. "
          "Total counts are obtained from the last element of offset arrays after scan. "
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx).",
          py::arg("mask_block_cnt"),
          py::arg("full_block_cnt"),
          py::arg("block_idx"));

    m.def("create_q2k_csr_sparse_from_func", &create_q2k_csr_sparse_from_func,
          "Q2K CSR (Forward): Convert function encoding tensor directly to CSR sparse format. "
          "Combines create_q2k_block_sparse_from_func and compact_block_idx in one call. "
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx).\n\n"
          "WARNING: Q_BLOCK_SIZE and KV_BLOCK_SIZE must match kernel tile sizes! "
          "Use create_q2k_csr_sparse_auto() instead for automatic tile size detection.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 256,
          py::arg("KV_BLOCK_SIZE") = 128,
          py::arg("check_q_boundary") = false);

    m.def("create_k2q_csr_sparse_from_func", &create_k2q_csr_sparse_from_func,
          "K2Q CSR (Backward): Convert function encoding tensor directly to CSR sparse format. "
          "Combines create_k2q_block_sparse_from_func and compact_block_idx in one call. "
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx).\n\n"
          "WARNING: Q_BLOCK_SIZE and KV_BLOCK_SIZE must match kernel tile sizes! "
          "Use create_k2q_csr_sparse_auto() instead for automatic tile size detection.",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("Q_BLOCK_SIZE") = 128,
          py::arg("KV_BLOCK_SIZE") = 128);

    // ========================================================================
    // Auto tile size functions (RECOMMENDED)
    // These automatically detect GPU architecture and compute correct tile sizes
    // ========================================================================

    m.def("create_q2k_csr_sparse_auto", &create_q2k_csr_sparse_auto,
          "Q2K CSR Auto (Forward): Convert with AUTOMATIC tile size detection.\n\n"
          "RECOMMENDED: This function automatically detects GPU architecture and "
          "computes the correct tile sizes based on headdim and mask configuration.\n\n"
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx, Q_BLOCK_SIZE, KV_BLOCK_SIZE).",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("headdim"),
          py::arg("is_causal") = false,
          py::arg("is_local") = false,
          py::arg("is_arbitrary") = true,
          py::arg("check_q_boundary") = false);

    m.def("create_k2q_csr_sparse_auto", &create_k2q_csr_sparse_auto,
          "K2Q CSR Auto (Backward): Convert with AUTOMATIC tile size detection.\n\n"
          "RECOMMENDED: This function automatically detects GPU architecture and "
          "computes the correct tile sizes based on headdim and mask configuration.\n\n"
          "Returns (mask_block_cnt, mask_block_offset, mask_block_idx, "
          "full_block_cnt, full_block_offset, full_block_idx, Q_BLOCK_SIZE, KV_BLOCK_SIZE).",
          py::arg("func_tensor"),
          py::arg("Q_LEN"),
          py::arg("KV_LEN"),
          py::arg("headdim"),
          py::arg("is_causal") = false,
          py::arg("is_local") = false,
          py::arg("is_arbitrary") = true,
          py::arg("has_softcap") = false,
          py::arg("headdim_v") = -1);

    // ========================================================================
    // Helper functions for tile size queries
    // ========================================================================

    m.def("get_fwd_tile_sizes", &py_get_fwd_tile_sizes,
          "Get forward pass tile sizes (Q_BLOCK_SIZE, KV_BLOCK_SIZE) based on configuration.\n\n"
          "Use this result when constructing Q2K CSR metadata for the selected backend.",
          py::arg("headdim"),
          py::arg("is_causal") = false,
          py::arg("is_local") = false,
          py::arg("is_arbitrary") = true,
          py::arg("arch") = -1);

    m.def("get_bwd_tile_sizes", &py_get_bwd_tile_sizes,
          "Get backward pass tile sizes (Q_BLOCK_SIZE, KV_BLOCK_SIZE) based on configuration.\n\n"
          "Use this result when constructing K2Q CSR metadata for the selected backend.",
          py::arg("headdim"),
          py::arg("is_causal") = false,
          py::arg("is_local") = false,
          py::arg("is_arbitrary") = true,
          py::arg("has_softcap") = false,
          py::arg("arch") = -1,
          py::arg("headdim_v") = -1);

    m.def("get_gpu_arch", &py_get_gpu_arch,
          "Get GPU architecture as int (e.g., 80, 86, 89, 90, 100).");

    m.def("round_up_hopper_headdim", &py_round_up_hopper_headdim,
          "Round a head dimension to the buckets compiled into the Hopper extension.",
          py::arg("headdim"));
}
