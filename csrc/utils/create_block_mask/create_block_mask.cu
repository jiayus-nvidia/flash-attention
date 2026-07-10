#include "create_block_mask.h"
#include <cassert>
#include <cub/cub.cuh>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// ============================================================================
// Optimization Control Macros (for benchmarking)
// ============================================================================
// Define these macros at compile time to enable/disable specific optimizations:
//
// 1. DISABLE_REG_CACHE - Disable register caching for func_tensor values
//    Without this optimization: func_tensor is read from global memory
//    multiple times (once for get_token_kv_range, and once per kv_block loop)
//    With this optimization: func_tensor is loaded once into registers
//    Expected speedup: ~4% on dense patterns (full, causal, random)
//
// 2. DISABLE_KV_RANGE_OPT - Disable kv_range optimization for both Q2K and K2Q
//
//    Default (not defined): Optimization ENABLED for both kernels
//      - Q2K kernel: computes min/max kv range via block reduce, then iterates
//        only over kv_blocks that could potentially overlap
//      - K2Q kernel: first launches a separate kernel (precompute) to compute
//        q_block_kv_min/max, then uses these values to skip q_blocks that don't
//        overlap with the current kv_block
//      - Expected speedup: significant for sparse patterns (e.g., diagonal ~7x)
//
//    With DISABLE_KV_RANGE_OPT defined: Optimization DISABLED
//      - Q2K kernel: iterates over all kv_blocks (0 to num_kv_blocks)
//      - K2Q kernel: iterates over all q_blocks (no precompute kernel launched)
//
// 3. DISABLE_BLOCK_SIZE_TEMPLATE - Disable block size template specialization
//    Without this optimization: Q_BLOCK_SIZE and KV_BLOCK_SIZE are runtime values
//    With this optimization: common block sizes (256x128, 128x128, 64x64) use
//    compile-time constants, enabling loop unrolling and constant propagation
//    Expected speedup: varies based on pattern and compiler optimizations
//
// 4. ENABLE_WARP_LEVEL_OPT - Enable warp-level optimization (EXPERIMENTAL)
//    With this optimization: uses only 32 threads (one warp) per block
//    Each thread processes Q_BLOCK_SIZE/32 q_tokens (e.g., 8 for Q_BLOCK_SIZE=256)
//    Performs local reduce within thread, then warp reduce (no block reduce)
//    Benefits:
//      - Avoids __syncthreads() for state reduction
//      - No shared memory needed for state reduction
//      - Better instruction-level parallelism within each thread
//    Requirements:
//      - Q_BLOCK_SIZE must be multiple of 32 (64, 128, 256)
//      - n_func must be 1-33 (compile-time constant for unrolling)
//    Note: This is experimental and may not always be faster than baseline
//
// Usage: Add -DDISABLE_XXX or -DENABLE_XXX to nvcc flags
// ============================================================================

// Block state enumeration
// EMPTY=0, MASK=1, FULL=2
constexpr int STATE_EMPTY = 0;
constexpr int STATE_MASK = 1;
constexpr int STATE_FULL = 2;

// Warp size
constexpr int WARP_SIZE_CONST = 32;

// Maximum Q_BLOCK_SIZE supported (for shared memory allocation)
#ifndef MAX_Q_BLOCK_SIZE
#define MAX_Q_BLOCK_SIZE 256
#endif

// Maximum number of warps per block
#define MAX_NUM_WARPS (MAX_Q_BLOCK_SIZE / WARP_SIZE_CONST)

/**
 * Device function to load func values from global memory into a register array.
 *
 * @tparam N_FUNC: number of function values (must be > 0 for register caching)
 * @param func_ptr: pointer to func_tensor at position [b, h, 0, q_idx]
 * @param stride_f: stride for the n_func dimension
 * @param func_regs: output register array of size N_FUNC
 */
template <int N_FUNC>
__device__ __forceinline__ void load_func_to_registers(
    const int* func_ptr,
    int stride_f,
    int* func_regs  // Must be a compile-time sized array
) {
    #pragma unroll
    for (int i = 0; i < N_FUNC; i++) {
        func_regs[i] = func_ptr[i * stride_f];
    }
}

/**
 * Device function to compute the min and max kv positions using register-cached func values.
 *
 * @tparam N_FUNC: number of function values (compile-time constant)
 * @param func_regs: register array containing cached func values
 * @param out_min_kv: output minimum kv position (INT_MAX if no valid interval)
 * @param out_max_kv: output maximum kv position (0 if no valid interval)
 */
template <int N_FUNC>
__device__ __forceinline__ void get_token_kv_range_from_regs(
    const int* func_regs,
    int& out_min_kv,
    int& out_max_kv
) {
    constexpr int num_intervals = (N_FUNC + 1) / 2;

    int min_kv = INT_MAX;
    int max_kv = 0;

    // Check first interval: [0, F0)
    {
        int interval_start = 0;
        int interval_end = func_regs[0];  // F0

        if (interval_end > interval_start) {
            min_kv = min(min_kv, interval_start);
            max_kv = max(max_kv, interval_end);
        }
    }

    // Check remaining intervals: [F_{2i-1}, F_{2i}) for i = 1, 2, ...
    #pragma unroll
    for (int i = 1; i < num_intervals; i++) {
        int interval_start = func_regs[2 * i - 1];
        int interval_end = func_regs[2 * i];

        if (interval_end > interval_start) {
            min_kv = min(min_kv, interval_start);
            max_kv = max(max_kv, interval_end);
        }
    }

    out_min_kv = min_kv;
    out_max_kv = max_kv;
}

/**
 * Device function to compute the min and max kv positions for a q_token's valid intervals.
 *
 * For a given q_token, compute the minimum and maximum kv positions across all valid intervals:
 *   interval 0: [0, F0)
 *   interval 1: [F1, F2)
 *   interval 2: [F3, F4)
 *   ...
 *
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @param func_ptr: pointer to func_tensor at position [b, h, 0, q_idx]
 * @param n_func: number of function values (only used when N_FUNC == 0)
 * @param stride_f: stride for the n_func dimension
 * @param out_min_kv: output minimum kv position (INT_MAX if no valid interval)
 * @param out_max_kv: output maximum kv position (0 if no valid interval)
 */
template <int N_FUNC>
__device__ __forceinline__ void get_token_kv_range(
    const int* func_ptr,
    int n_func,
    int stride_f,
    int& out_min_kv,
    int& out_max_kv
) {
    const int num_intervals = (N_FUNC > 0) ? ((N_FUNC + 1) / 2) : ((n_func + 1) / 2);

    int min_kv = INT_MAX;
    int max_kv = 0;

    // Check first interval: [0, F0)
    {
        int interval_start = 0;
        int interval_end = func_ptr[0 * stride_f];  // F0

        if (interval_end > interval_start) {
            min_kv = min(min_kv, interval_start);
            max_kv = max(max_kv, interval_end);
        }
    }

    // Check remaining intervals: [F_{2i-1}, F_{2i}) for i = 1, 2, ...
    #pragma unroll
    for (int i = 1; i < num_intervals; i++) {
        int interval_start = func_ptr[(2 * i - 1) * stride_f];
        int interval_end = func_ptr[(2 * i) * stride_f];

        if (interval_end > interval_start) {
            min_kv = min(min_kv, interval_start);
            max_kv = max(max_kv, interval_end);
        }
    }

    out_min_kv = min_kv;
    out_max_kv = max_kv;
}

/**
 * Device function to determine the state of a (q_token, kv_block) pair.
 *
 * For a given q_token and kv_block [kv_start, kv_end), check against
 * the valid intervals defined by func_vals:
 *   interval 0: [0, F0)
 *   interval 1: [F1, F2)
 *   interval 2: [F3, F4)
 *   ...
 *
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @param func_ptr: pointer to func_tensor at position [b, h, 0, q_idx]
 * @param n_func: number of function values (only used when N_FUNC == 0)
 * @param stride_f: stride for the n_func dimension
 * @param kv_start: start of kv_block (inclusive)
 * @param kv_end: end of kv_block (exclusive)
 *
 * Returns:
 *   STATE_FULL: kv_block is completely within some valid interval
 *   STATE_MASK: kv_block partially overlaps with valid intervals
 *   STATE_EMPTY: kv_block has no overlap with any valid interval
 */
template <int N_FUNC>
__device__ __forceinline__ int get_token_kv_block_state(
    const int* func_ptr,
    int n_func,
    int stride_f,
    int kv_start,
    int kv_end
) {
    // Use compile-time constant if N_FUNC > 0, otherwise use runtime value
    const int num_intervals = (N_FUNC > 0) ? ((N_FUNC + 1) / 2) : ((n_func + 1) / 2);

    bool has_any_overlap = false;
    bool is_fully_covered = false;

    // Check first interval: [0, F0)
    {
        int interval_start = 0;
        int interval_end = func_ptr[0 * stride_f];  // F0

        if (interval_end > interval_start) {
            // Check if kv_block is fully covered by this interval
            if (interval_start <= kv_start && interval_end >= kv_end) {
                is_fully_covered = true;
            }
            // Check if there's any overlap
            else if (interval_start < kv_end && interval_end > kv_start) {
                has_any_overlap = true;
            }
        }
    }

    // Check remaining intervals: [F_{2i-1}, F_{2i}) for i = 1, 2, ...
    // When N_FUNC is a compile-time constant, the loop can be fully unrolled
    #pragma unroll
    for (int i = 1; i < num_intervals; i++) {
        int interval_start = func_ptr[(2 * i - 1) * stride_f];  // F_{2i-1}
        int interval_end = func_ptr[(2 * i) * stride_f];        // F_{2i}

        if (interval_end <= interval_start) {
            continue;
        }

        // Check if kv_block is fully covered by this interval
        if (interval_start <= kv_start && interval_end >= kv_end) {
            is_fully_covered = true;
        }
        // Check if there's any overlap
        else if (interval_start < kv_end && interval_end > kv_start) {
            has_any_overlap = true;
        }
    }

    if (is_fully_covered) {
        return STATE_FULL;
    } else if (has_any_overlap) {
        return STATE_MASK;
    } else {
        return STATE_EMPTY;
    }
}

/**
 * Device function to determine kv_block state using register-cached func values.
 *
 * @tparam N_FUNC: number of function values (compile-time constant)
 * @param func_regs: register array containing cached func values
 * @param kv_start: start of kv_block (inclusive)
 * @param kv_end: end of kv_block (exclusive)
 */
template <int N_FUNC>
__device__ __forceinline__ int get_token_kv_block_state_from_regs(
    const int* func_regs,
    int kv_start,
    int kv_end
) {
    constexpr int num_intervals = (N_FUNC + 1) / 2;

    bool has_any_overlap = false;
    bool is_fully_covered = false;

    // Check first interval: [0, F0)
    {
        int interval_start = 0;
        int interval_end = func_regs[0];  // F0

        if (interval_end > interval_start) {
            if (interval_start <= kv_start && interval_end >= kv_end) {
                is_fully_covered = true;
            } else if (interval_start < kv_end && interval_end > kv_start) {
                has_any_overlap = true;
            }
        }
    }

    // Check remaining intervals: [F_{2i-1}, F_{2i}) for i = 1, 2, ...
    #pragma unroll
    for (int i = 1; i < num_intervals; i++) {
        int interval_start = func_regs[2 * i - 1];
        int interval_end = func_regs[2 * i];

        if (interval_end <= interval_start) {
            continue;
        }

        if (interval_start <= kv_start && interval_end >= kv_end) {
            is_fully_covered = true;
        } else if (interval_start < kv_end && interval_end > kv_start) {
            has_any_overlap = true;
        }
    }

    if (is_fully_covered) {
        return STATE_FULL;
    } else if (has_any_overlap) {
        return STATE_MASK;
    } else {
        return STATE_EMPTY;
    }
}

/**
 * Block-level reduction to compute kv_block traversal range.
 *
 * Each thread provides its q_token's min/max kv positions. This function performs
 * a block-level reduction to find the overall min/max, then computes the kv_block
 * range that needs to be traversed.
 *
 * @param thread_min_kv: This thread's minimum kv position (INT_MAX if no valid interval)
 * @param thread_max_kv: This thread's maximum kv position (0 if no valid interval)
 * @param KV_BLOCK_SIZE: Size of each kv_block
 * @param num_kv_blocks: Total number of kv_blocks
 * @param warp_min_results: Shared memory array for warp-level min results [MAX_NUM_WARPS]
 * @param warp_max_results: Shared memory array for warp-level max results [MAX_NUM_WARPS]
 * @param out_kv_block_start: Output start of kv_block range (inclusive)
 * @param out_kv_block_end: Output end of kv_block range (exclusive)
 */
__device__ __forceinline__ void reduce_kv_block_range(
    int thread_min_kv,
    int thread_max_kv,
    int KV_BLOCK_SIZE,
    int num_kv_blocks,
    int* warp_min_results,  // shared memory [MAX_NUM_WARPS]
    int* warp_max_results,  // shared memory [MAX_NUM_WARPS]
    int& out_kv_block_start,
    int& out_kv_block_end
) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Warp-level min/max reduction
    int warp_min = __reduce_min_sync(0xFFFFFFFF, thread_min_kv);
    int warp_max = __reduce_max_sync(0xFFFFFFFF, thread_max_kv);

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        warp_min_results[warp_id] = warp_min;
        warp_max_results[warp_id] = warp_max;
    }
    __syncthreads();

    // Warp 0 performs final reduction and computes kv_block range
    int kv_block_start = 0;
    int kv_block_end = 0;

    if (warp_id == 0) {
        int my_min = (lane_id < num_warps) ? warp_min_results[lane_id] : INT_MAX;
        int my_max = (lane_id < num_warps) ? warp_max_results[lane_id] : 0;
        int block_min_kv = __reduce_min_sync(0xFFFFFFFF, my_min);
        int block_max_kv = __reduce_max_sync(0xFFFFFFFF, my_max);

        // Compute kv_block range (only lane 0's result will be used)
        if (block_max_kv > 0 && block_min_kv != INT_MAX) {
            kv_block_start = block_min_kv / KV_BLOCK_SIZE;
            kv_block_end = min(DIVUP(block_max_kv, KV_BLOCK_SIZE), num_kv_blocks);
        }
    }

    // Broadcast result using shared memory (reuse warp_min_results)
    if (tid == 0) {
        warp_min_results[0] = kv_block_start;
        warp_max_results[0] = kv_block_end;
    }
    __syncthreads();

    out_kv_block_start = warp_min_results[0];
    out_kv_block_end = warp_max_results[0];
}

/**
 * Block-level reduction to compute raw min/max kv positions.
 *
 * Similar to reduce_kv_block_range but outputs raw min/max values instead of block indices.
 * Used by the K2Q preprocessing kernel.
 *
 * @param thread_min_kv: This thread's minimum kv position (INT_MAX if no valid interval)
 * @param thread_max_kv: This thread's maximum kv position (0 if no valid interval)
 * @param warp_min_results: Shared memory array for warp-level min results [MAX_NUM_WARPS]
 * @param warp_max_results: Shared memory array for warp-level max results [MAX_NUM_WARPS]
 * @param out_min_kv: Output minimum kv position for this block
 * @param out_max_kv: Output maximum kv position for this block
 */
__device__ __forceinline__ void reduce_kv_range_raw(
    int thread_min_kv,
    int thread_max_kv,
    int* warp_min_results,  // shared memory [MAX_NUM_WARPS]
    int* warp_max_results,  // shared memory [MAX_NUM_WARPS]
    int& out_min_kv,
    int& out_max_kv
) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Warp-level min/max reduction
    int warp_min = __reduce_min_sync(0xFFFFFFFF, thread_min_kv);
    int warp_max = __reduce_max_sync(0xFFFFFFFF, thread_max_kv);

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        warp_min_results[warp_id] = warp_min;
        warp_max_results[warp_id] = warp_max;
    }
    __syncthreads();

    // Warp 0 performs final reduction
    int block_min_kv = INT_MAX;
    int block_max_kv = 0;

    if (warp_id == 0) {
        int my_min = (lane_id < num_warps) ? warp_min_results[lane_id] : INT_MAX;
        int my_max = (lane_id < num_warps) ? warp_max_results[lane_id] : 0;
        block_min_kv = __reduce_min_sync(0xFFFFFFFF, my_min);
        block_max_kv = __reduce_max_sync(0xFFFFFFFF, my_max);
    }

    // Broadcast result using shared memory
    if (tid == 0) {
        warp_min_results[0] = block_min_kv;
        warp_max_results[0] = block_max_kv;
    }
    __syncthreads();

    out_min_kv = warp_min_results[0];
    out_max_kv = warp_max_results[0];
}

/**
 * Combine two states into one.
 * Rules:
 * - Same state -> keep same
 * - One is MASK -> MASK
 * - FULL + EMPTY -> MASK
 */
__device__ __forceinline__ int combine_states(int state1, int state2) {
    if (state1 == state2) {
        return state1;
    }
    // Different states -> MASK
    return STATE_MASK;
}

/**
 * Convert combined state bitmask to state value.
 *
 * Bitmask encoding: EMPTY(0)->0b001, MASK(1)->0b010, FULL(2)->0b100
 * - 0b001 (1) -> all EMPTY -> STATE_EMPTY
 * - 0b100 (4) -> all FULL -> STATE_FULL
 * - otherwise -> mixed states -> STATE_MASK
 */
__device__ __forceinline__ int bitmask_to_state(unsigned int mask) {
    if (mask == 1u) return STATE_EMPTY;      // Only EMPTY present
    if (mask == 4u) return STATE_FULL;       // Only FULL present
    return STATE_MASK;                       // Mixed or has MASK
}

/**
 * Block-level reduction to determine the final state of a block.
 *
 * Rules:
 * - All threads FULL -> FULL block
 * - All threads EMPTY -> EMPTY block (skip)
 * - Any thread MASK, or (some FULL and some EMPTY) -> MASK block
 *
 * Optimized with redux.sync (PTX instruction) for both warp-level and
 * cross-warp reduction, minimizing synchronization overhead.
 *
 * Note: Inactive threads should have their state pre-set before calling:
 * - check_q_boundary=false: inactive threads contribute STATE_FULL
 * - check_q_boundary=true: inactive threads contribute STATE_EMPTY
 */
__device__ __forceinline__ int reduce_block_state(
    int thread_state,
    int* warp_results  // shared memory array of size MAX_NUM_WARPS
) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Convert state to bitmask: EMPTY(0)->0b001, MASK(1)->0b010, FULL(2)->0b100
    unsigned int state_mask = 1u << thread_state;

    // Phase 1: Warp-level reduction using redux.sync (single PTX instruction)
    unsigned int warp_mask = __reduce_or_sync(0xFFFFFFFF, state_mask);

    // Phase 2: Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = static_cast<int>(warp_mask);
    }
    __syncthreads();

    // Phase 3: Warp 0 performs final reduction using redux.sync
    // Each thread in warp 0 reads one warp result, then reduce across the warp
    unsigned int final_mask = 0;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        // Lane i reads warp_results[i] if valid, else 0 (neutral for OR)
        unsigned int my_result = (lane_id < num_warps) ?
            static_cast<unsigned int>(warp_results[lane_id]) : 0u;
        final_mask = __reduce_or_sync(0xFFFFFFFF, my_result);
    }

    // Convert bitmask back to state (only thread 0's result matters to caller)
    return bitmask_to_state(final_mask);
}

// =============================================================================
// Unified state reduction helper
// =============================================================================

/**
 * Warp-level only reduction for state bitmask (used when ENABLE_WARP_LEVEL_OPT is enabled).
 *
 * @param local_state_mask: This thread's combined state bitmask
 * @return: Final state (STATE_FULL, STATE_MASK, or STATE_EMPTY)
 */
__device__ __forceinline__ int warp_reduce_state_only(unsigned int local_state_mask) {
    unsigned int final_mask = __reduce_or_sync(0xFFFFFFFF, local_state_mask);
    return bitmask_to_state(final_mask);
}

/**
 * Block-level reduction for state bitmask (warp reduce + cross-warp reduce).
 * Used when ENABLE_WARP_LEVEL_OPT is NOT enabled.
 *
 * @param local_state_mask: This thread's state bitmask
 * @param warp_results: Shared memory array for cross-warp communication
 * @return: Final state (STATE_FULL, STATE_MASK, or STATE_EMPTY)
 */
__device__ __forceinline__ int block_reduce_state_bitmask(
    unsigned int local_state_mask,
    int* warp_results  // shared memory array of size MAX_NUM_WARPS
) {
    constexpr int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Phase 1: Warp-level reduction
    unsigned int warp_mask = __reduce_or_sync(0xFFFFFFFF, local_state_mask);

    // Phase 2: Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = static_cast<int>(warp_mask);
    }
    __syncthreads();

    // Phase 3: Warp 0 performs final reduction
    unsigned int final_mask = 0;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        unsigned int my_result = (lane_id < num_warps) ?
            static_cast<unsigned int>(warp_results[lane_id]) : 0u;
        final_mask = __reduce_or_sync(0xFFFFFFFF, my_result);
    }

    return bitmask_to_state(final_mask);
}

// =============================================================================
// Q2K Kernel (Forward): fix q_block, loop kv_blocks
// =============================================================================

/**
 * Q2K Kernel: Create block sparse tensors from function encoding.
 *
 * Design (controlled by ENABLE_WARP_LEVEL_OPT macro):
 *
 * Block-level mode (default):
 * - One block processes one (b, h, q_block) with Q_BLOCK_SIZE threads
 * - Each thread handles one q_token (TOKENS_PER_THREAD = 1)
 * - Uses block-level reduction (warp reduce + cross-warp reduce via shared memory)
 *
 * Warp-level mode (ENABLE_WARP_LEVEL_OPT defined):
 * - One block processes one (b, h, q_block) with 32 threads (one warp)
 * - Each thread handles Q_BLOCK_SIZE/32 q_tokens
 * - Uses warp-level reduction only (no shared memory for state reduction)
 *
 * Grid: dim3(num_q_blocks, H, B)
 *
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @tparam Q_BLOCK_SIZE_T: Q block size (compile-time constant, 0 means use runtime value)
 * @tparam KV_BLOCK_SIZE_T: KV block size (compile-time constant, 0 means use runtime value)
 * @param func_tensor: [B, H, n_func, func_q_len], with strides (stride_b, stride_h, stride_f, stride_q)
 */
template <int N_FUNC, int Q_BLOCK_SIZE_T = 0, int KV_BLOCK_SIZE_T = 0>
__global__ void create_q2k_block_sparse_from_func_kernel(
    const int* __restrict__ func_tensor,  // [B, H, n_func, func_q_len]
    int stride_b, int stride_h, int stride_f, int stride_q,  // strides for func_tensor
    int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE_RT, int KV_BLOCK_SIZE_RT,  // Runtime values (used when template param is 0)
    int num_q_blocks, int num_kv_blocks,
    bool check_q_boundary,  // If true, partial q_blocks cannot have FULL kv_blocks
    int* __restrict__ mask_block_cnt,     // [B, H, num_q_blocks]
    int* __restrict__ full_block_cnt,     // [B, H, num_q_blocks]
    int* __restrict__ block_idx           // [B, H, num_q_blocks, num_kv_blocks]
) {
    // Use compile-time constant if available, otherwise use runtime value
    const int Q_BLOCK_SIZE = (Q_BLOCK_SIZE_T > 0) ? Q_BLOCK_SIZE_T : Q_BLOCK_SIZE_RT;
    const int KV_BLOCK_SIZE = (KV_BLOCK_SIZE_T > 0) ? KV_BLOCK_SIZE_T : KV_BLOCK_SIZE_RT;

    // Determine if warp-level optimization is enabled AND valid for this template instantiation
    // Warp-level requires compile-time Q_BLOCK_SIZE_T > 0 and divisible by 32
#ifdef ENABLE_WARP_LEVEL_OPT
    constexpr bool USE_WARP_LEVEL = (Q_BLOCK_SIZE_T > 0) && (Q_BLOCK_SIZE_T % WARP_SIZE_CONST == 0);
#else
    constexpr bool USE_WARP_LEVEL = false;
#endif

    // TOKENS_PER_THREAD: how many q_tokens each thread processes
    // - Warp-level mode: Q_BLOCK_SIZE_T / 32 (e.g., 8 for 256, 4 for 128)
    // - Block-level mode: 1 (each thread handles 1 token)
    constexpr int TOKENS_PER_THREAD = USE_WARP_LEVEL ? (Q_BLOCK_SIZE_T / WARP_SIZE_CONST) : 1;

    // Shared memory for block-level reduction (only needed when not using warp-level)
    __shared__ int warp_results[USE_WARP_LEVEL ? 1 : MAX_NUM_WARPS];
    __shared__ int warp_min_results[USE_WARP_LEVEL ? 1 : MAX_NUM_WARPS];
    __shared__ int warp_max_results[USE_WARP_LEVEL ? 1 : MAX_NUM_WARPS];

    // Get block indices
    int q_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE_CONST;

    // Calculate q range for this block
    int q_start = q_block * Q_BLOCK_SIZE;
    int q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
    int num_active_tokens = q_end - q_start;

    // Check if this q_block is full (has Q_BLOCK_SIZE active tokens)
    bool is_q_block_full = (num_active_tokens == Q_BLOCK_SIZE);

    // Base pointer for func_tensor at [b, h, 0, 0]
    const int* base_func_ptr = func_tensor + b * stride_b + h * stride_h;

    // =========================================================================
    // Load func values into registers
    // Warp-level: each thread loads TOKENS_PER_THREAD tokens
    // Block-level: each thread loads 1 token
    // =========================================================================

    // Common pointer for this thread's primary q_token (used in block-level mode)
    int q_idx_primary = q_start + tid;
    bool is_active_primary = (tid < num_active_tokens);
    const int* my_func_ptr = base_func_ptr + q_idx_primary * stride_q;

    // Register arrays for func values (sized based on mode)
    int func_regs[TOKENS_PER_THREAD][N_FUNC > 0 ? N_FUNC : 1];
    bool token_active[TOKENS_PER_THREAD];

#ifndef DISABLE_REG_CACHE
    // Load func values into registers
    #pragma unroll
    for (int t = 0; t < TOKENS_PER_THREAD; t++) {
        int local_q_idx;
        if constexpr (USE_WARP_LEVEL) {
            local_q_idx = lane_id * TOKENS_PER_THREAD + t;  // Sequential distribution for warp mode
        } else {
            local_q_idx = tid;  // One token per thread for block mode
        }
        int q_idx = q_start + local_q_idx;
        token_active[t] = (local_q_idx < num_active_tokens);

        if (token_active[t]) {
            const int* func_ptr = base_func_ptr + q_idx * stride_q;
            if constexpr (N_FUNC > 0) {
                #pragma unroll
                for (int f = 0; f < N_FUNC; f++) {
                    func_regs[t][f] = func_ptr[f * stride_f];
                }
            }
        }
    }
#else
    // Without register caching, just initialize token_active
    token_active[0] = is_active_primary;
#endif

    // =========================================================================
    // Compute kv_block traversal range
    // =========================================================================
    int kv_block_start, kv_block_end;

#ifdef DISABLE_KV_RANGE_OPT
    kv_block_start = 0;
    kv_block_end = num_kv_blocks;
#else
    {
        int thread_min_kv = INT_MAX;
        int thread_max_kv = 0;

        if constexpr (USE_WARP_LEVEL) {
            // Warp-level: local reduce across this thread's tokens
            #pragma unroll
            for (int t = 0; t < TOKENS_PER_THREAD; t++) {
                if (token_active[t]) {
                    int token_min_kv, token_max_kv;
                    get_token_kv_range_from_regs<N_FUNC>(func_regs[t], token_min_kv, token_max_kv);
                    thread_min_kv = min(thread_min_kv, token_min_kv);
                    thread_max_kv = max(thread_max_kv, token_max_kv);
                }
            }
            // Warp reduce to get block's kv range
            int block_min_kv = __reduce_min_sync(0xFFFFFFFF, thread_min_kv);
            int block_max_kv = __reduce_max_sync(0xFFFFFFFF, thread_max_kv);

            if (block_max_kv > 0 && block_min_kv != INT_MAX) {
                kv_block_start = block_min_kv / KV_BLOCK_SIZE;
                kv_block_end = min(DIVUP(block_max_kv, KV_BLOCK_SIZE), num_kv_blocks);
            } else {
                kv_block_start = 0;
                kv_block_end = 0;
            }
        } else {
            // Block-level: each thread computes for its single token
            if (is_active_primary) {
#ifndef DISABLE_REG_CACHE
                if constexpr (N_FUNC > 0) {
                    get_token_kv_range_from_regs<N_FUNC>(func_regs[0], thread_min_kv, thread_max_kv);
                } else {
                    get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, thread_min_kv, thread_max_kv);
                }
#else
                get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, thread_min_kv, thread_max_kv);
#endif
            }
            // Block-level reduction
            reduce_kv_block_range(thread_min_kv, thread_max_kv, KV_BLOCK_SIZE, num_kv_blocks,
                                  warp_min_results, warp_max_results, kv_block_start, kv_block_end);
        }
    }
#endif  // DISABLE_KV_RANGE_OPT

    // Output tensor strides (contiguous layout)
    int out_cnt_stride_b = H * num_q_blocks;
    int out_cnt_stride_h = num_q_blocks;
    int out_idx_stride_b = H * num_q_blocks * num_kv_blocks;
    int out_idx_stride_h = num_q_blocks * num_kv_blocks;
    int out_idx_stride_q = num_kv_blocks;

    // Output offsets
    int cnt_offset = b * out_cnt_stride_b + h * out_cnt_stride_h + q_block;
    int idx_base = b * out_idx_stride_b + h * out_idx_stride_h + q_block * out_idx_stride_q;

    // Counters
    int mask_count = 0;
    int full_count = 0;

    // Process each kv_block within the computed range
    for (int kv_block = kv_block_start; kv_block < kv_block_end; kv_block++) {
        int kv_start = kv_block * KV_BLOCK_SIZE;
        int kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);

        // Check if this is a partial kv_block
        bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);

        // =====================================================================
        // Each thread computes state for its tokens and accumulates into bitmask
        // =====================================================================
        unsigned int local_state_mask = 0;

        #pragma unroll
        for (int t = 0; t < TOKENS_PER_THREAD; t++) {
            int token_state;
#ifndef DISABLE_REG_CACHE
            if (token_active[t]) {
                if constexpr (N_FUNC > 0) {
                    token_state = get_token_kv_block_state_from_regs<N_FUNC>(func_regs[t], kv_start, kv_end);
                } else {
                    int local_q_idx;
                    if constexpr (USE_WARP_LEVEL) {
                        local_q_idx = lane_id * TOKENS_PER_THREAD + t;
                    } else {
                        local_q_idx = tid;
                    }
                    const int* ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                    token_state = get_token_kv_block_state<N_FUNC>(ptr, n_func, stride_f, kv_start, kv_end);
                }
            } else {
                token_state = check_q_boundary ? STATE_EMPTY : STATE_FULL;
            }
#else
            // Block-level without register caching
            if (is_active_primary) {
                token_state = get_token_kv_block_state<N_FUNC>(my_func_ptr, n_func, stride_f, kv_start, kv_end);
            } else {
                token_state = check_q_boundary ? STATE_EMPTY : STATE_FULL;
            }
#endif
            // Accumulate state into bitmask: EMPTY->0b001, MASK->0b010, FULL->0b100
            local_state_mask |= (1u << token_state);
        }

        // =====================================================================
        // Reduce state across all tokens in the q_block
        // =====================================================================
        int kv_block_state;
        if constexpr (USE_WARP_LEVEL) {
            // Warp-level reduction only (32 threads)
            kv_block_state = warp_reduce_state_only(local_state_mask);
        } else {
            // Block-level reduction (warp reduce + cross-warp reduce)
            kv_block_state = block_reduce_state_bitmask(local_state_mask, warp_results);
        }

        // Record the result (thread 0 / lane 0)
        int leader_id = USE_WARP_LEVEL ? lane_id : tid;
        if (leader_id == 0) {
            bool can_be_full = (!check_q_boundary || is_q_block_full) && is_kv_block_full;

            if (kv_block_state == STATE_FULL && can_be_full) {
                block_idx[idx_base + full_count] = kv_block;
                full_count++;
            } else if (kv_block_state == STATE_FULL || kv_block_state == STATE_MASK) {
                block_idx[idx_base + (num_kv_blocks - 1 - mask_count)] = kv_block;
                mask_count++;
            }
        }

        if constexpr (!USE_WARP_LEVEL) {
            __syncthreads();
        }
    }

    // Write final counts
    int final_leader_id = USE_WARP_LEVEL ? lane_id : tid;
    if (final_leader_id == 0) {
        mask_block_cnt[cnt_offset] = mask_count;
        full_block_cnt[cnt_offset] = full_count;
    }
}

// =============================================================================
// K2Q Preprocessing Kernel: Compute kv range for each q_block
// =============================================================================

/**
 * K2Q Preprocessing Kernel: Compute min/max kv positions for each q_block.
 *
 * This kernel precomputes the kv range that each q_block can access,
 * which allows the main K2Q kernel to skip q_blocks that don't overlap
 * with the current kv_block.
 *
 * Design:
 * - One block processes one (b, h, q_block)
 * - Each block has Q_BLOCK_SIZE threads, one per q_token
 * - Grid: dim3(num_q_blocks, H, B)
 *
 * Output:
 * - q_block_kv_min[B, H, num_q_blocks]: minimum kv position for each q_block
 * - q_block_kv_max[B, H, num_q_blocks]: maximum kv position for each q_block (exclusive)
 *
 * If a q_block has no valid intervals, min=INT_MAX and max=0.
 *
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 */
template <int N_FUNC>
__global__ void compute_q_block_kv_range_kernel(
    const int* __restrict__ func_tensor,  // [B, H, n_func, func_q_len]
    int stride_b, int stride_h, int stride_f, int stride_q,  // strides for func_tensor
    int Q_LEN, int n_func,
    int Q_BLOCK_SIZE,
    int num_q_blocks,
    int* __restrict__ q_block_kv_min,     // [B, H, num_q_blocks]
    int* __restrict__ q_block_kv_max      // [B, H, num_q_blocks]
) {
    // Shared memory for warp-level reduction
    __shared__ int warp_min_results[MAX_NUM_WARPS];
    __shared__ int warp_max_results[MAX_NUM_WARPS];

    // Get block indices
    int q_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int tid = threadIdx.x;

    // Calculate q range for this block
    int q_start = q_block * Q_BLOCK_SIZE;
    int q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
    int num_active_threads = q_end - q_start;

    // This thread's q_token index
    int q_idx = q_start + tid;
    bool is_active = (tid < num_active_threads);

    // Pointer to func_tensor at [b, h, 0, q_idx]
    const int* my_func_ptr = func_tensor + b * stride_b + h * stride_h + q_idx * stride_q;

    // Each thread computes min/max kv for its q_token
    // Inactive threads use INT_MAX/0 so they don't affect the reduce
    int thread_min_kv = INT_MAX;
    int thread_max_kv = 0;

    if (is_active) {
        get_token_kv_range<N_FUNC>(my_func_ptr, n_func, stride_f, thread_min_kv, thread_max_kv);
    }

    // Block-level reduction to get q_block's min/max kv range
    int block_min_kv, block_max_kv;
    reduce_kv_range_raw(thread_min_kv, thread_max_kv,
                        warp_min_results, warp_max_results,
                        block_min_kv, block_max_kv);

    // Thread 0 writes the result
    if (tid == 0) {
        int out_offset = b * H * num_q_blocks + h * num_q_blocks + q_block;
        q_block_kv_min[out_offset] = block_min_kv;
        q_block_kv_max[out_offset] = block_max_kv;
    }
}

// =============================================================================
// K2Q Kernel (Backward): fix kv_block, loop q_blocks
// =============================================================================

/**
 * K2Q Kernel: Create block sparse tensors from function encoding for backward pass.
 *
 * Design (controlled by ENABLE_WARP_LEVEL_OPT macro):
 *
 * Block-level mode (default):
 * - One block processes one (b, h, kv_block) with Q_BLOCK_SIZE threads
 * - Each thread handles one q_token per q_block (TOKENS_PER_THREAD = 1)
 * - Uses block-level reduction (warp reduce + cross-warp reduce via shared memory)
 *
 * Warp-level mode (ENABLE_WARP_LEVEL_OPT defined):
 * - One block processes one (b, h, kv_block) with 32 threads (one warp)
 * - Each thread handles Q_BLOCK_SIZE/32 q_tokens per q_block
 * - Uses warp-level reduction only (no shared memory for state reduction)
 *
 * Grid: dim3(num_kv_blocks, H, B)
 *
 * Note: Always performs boundary checking (partial q_blocks cannot have FULL status).
 *
 * @tparam N_FUNC: number of function values (compile-time constant, 0 means use runtime n_func)
 * @tparam Q_BLOCK_SIZE_T: Q block size (compile-time constant, 0 means use runtime value)
 * @tparam KV_BLOCK_SIZE_T: KV block size (compile-time constant, 0 means use runtime value)
 * @param func_tensor: [B, H, n_func, func_q_len], with strides (stride_b, stride_h, stride_f, stride_q)
 * @param q_block_kv_min: [B, H, num_q_blocks] precomputed min kv for each q_block (nullptr to disable optimization)
 * @param q_block_kv_max: [B, H, num_q_blocks] precomputed max kv for each q_block (nullptr to disable optimization)
 */
template <int N_FUNC, int Q_BLOCK_SIZE_T = 0, int KV_BLOCK_SIZE_T = 0>
__global__ void create_k2q_block_sparse_from_func_kernel(
    const int* __restrict__ func_tensor,  // [B, H, n_func, func_q_len]
    int stride_b, int stride_h, int stride_f, int stride_q,  // strides for func_tensor
    int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE_RT, int KV_BLOCK_SIZE_RT,  // Runtime values (used when template param is 0)
    int num_q_blocks, int num_kv_blocks,
    const int* __restrict__ q_block_kv_min,  // [B, H, num_q_blocks] precomputed (can be nullptr)
    const int* __restrict__ q_block_kv_max,  // [B, H, num_q_blocks] precomputed (can be nullptr)
    int* __restrict__ mask_block_cnt,     // [B, H, num_kv_blocks]
    int* __restrict__ full_block_cnt,     // [B, H, num_kv_blocks]
    int* __restrict__ block_idx           // [B, H, num_kv_blocks, num_q_blocks]
) {
    // Use compile-time constant if available, otherwise use runtime value
    const int Q_BLOCK_SIZE = (Q_BLOCK_SIZE_T > 0) ? Q_BLOCK_SIZE_T : Q_BLOCK_SIZE_RT;
    const int KV_BLOCK_SIZE = (KV_BLOCK_SIZE_T > 0) ? KV_BLOCK_SIZE_T : KV_BLOCK_SIZE_RT;

    // Determine if warp-level optimization is enabled AND valid for this template instantiation
#ifdef ENABLE_WARP_LEVEL_OPT
    constexpr bool USE_WARP_LEVEL_K2Q = (Q_BLOCK_SIZE_T > 0) && (Q_BLOCK_SIZE_T % WARP_SIZE_CONST == 0);
#else
    constexpr bool USE_WARP_LEVEL_K2Q = false;
#endif

    constexpr int TOKENS_PER_THREAD_K2Q = USE_WARP_LEVEL_K2Q ? (Q_BLOCK_SIZE_T / WARP_SIZE_CONST) : 1;

    // Shared memory for block-level reduction (only needed when not using warp-level)
    __shared__ int warp_results_k2q[USE_WARP_LEVEL_K2Q ? 1 : MAX_NUM_WARPS];

    // Get block indices
    int kv_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int H = gridDim.y;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE_CONST;

    // Calculate kv range for this block
    int kv_start = kv_block * KV_BLOCK_SIZE;
    int kv_end = min(kv_start + KV_BLOCK_SIZE, KV_LEN);

    // Check if this is a partial kv_block
    bool is_kv_block_full = (kv_end == kv_start + KV_BLOCK_SIZE);

    // Base pointer for func_tensor at [b, h, 0, 0]
    const int* base_func_ptr = func_tensor + b * stride_b + h * stride_h;

    // Output tensor strides (contiguous layout)
    int out_cnt_stride_b = H * num_kv_blocks;
    int out_cnt_stride_h = num_kv_blocks;
    int out_idx_stride_b = H * num_kv_blocks * num_q_blocks;
    int out_idx_stride_h = num_kv_blocks * num_q_blocks;
    int out_idx_stride_kv = num_q_blocks;

    // Output offsets
    int cnt_offset = b * out_cnt_stride_b + h * out_cnt_stride_h + kv_block;
    int idx_base = b * out_idx_stride_b + h * out_idx_stride_h + kv_block * out_idx_stride_kv;

    // Precomputed q_block kv range offset (if available)
    int precompute_base = b * H * num_q_blocks + h * num_q_blocks;
#ifndef DISABLE_KV_RANGE_OPT
    // Default (optimization enabled): use precompute for K2Q
    // q_block kv ranges are precomputed by a separate kernel
    // Using constexpr to reduce branch instructions
    constexpr bool use_precompute = true;
#else
    // Optimization disabled: don't use precompute, iterate all q_blocks
    constexpr bool use_precompute = false;
#endif

    // Counters
    int mask_count = 0;
    int full_count = 0;

    // Register array for func values
    int func_regs_k2q[TOKENS_PER_THREAD_K2Q][N_FUNC > 0 ? N_FUNC : 1];

    // Process each q_block
    for (int q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
        // =========================================================================
        // Optimization: Skip q_blocks whose kv range doesn't overlap
        // =========================================================================
        if constexpr (use_precompute) {
            int q_min_kv = q_block_kv_min[precompute_base + q_block_idx];
            int q_max_kv = q_block_kv_max[precompute_base + q_block_idx];

            if (q_max_kv <= kv_start || q_min_kv >= kv_end) {
                continue;
            }
        }

        // Calculate q range for this q_block
        int q_start = q_block_idx * Q_BLOCK_SIZE;
        int q_end = min(q_start + Q_BLOCK_SIZE, Q_LEN);
        int num_active_tokens = q_end - q_start;

        // Check if this q_block is full
        bool is_q_block_full = (num_active_tokens == Q_BLOCK_SIZE);

        // =====================================================================
        // Load func values and compute state
        // =====================================================================
        unsigned int local_state_mask = 0;
        bool token_active_k2q[TOKENS_PER_THREAD_K2Q];

        // Load func values
        #pragma unroll
        for (int t = 0; t < TOKENS_PER_THREAD_K2Q; t++) {
            int local_q_idx;
            if constexpr (USE_WARP_LEVEL_K2Q) {
                local_q_idx = lane_id * TOKENS_PER_THREAD_K2Q + t;
            } else {
                local_q_idx = tid;
            }
            int q_idx = q_start + local_q_idx;
            token_active_k2q[t] = (local_q_idx < num_active_tokens);

            if (token_active_k2q[t]) {
                const int* my_func_ptr = base_func_ptr + q_idx * stride_q;
                if constexpr (N_FUNC > 0) {
                    #pragma unroll
                    for (int f = 0; f < N_FUNC; f++) {
                        func_regs_k2q[t][f] = my_func_ptr[f * stride_f];
                    }
                }
            }
        }

        // Compute state for each token
        #pragma unroll
        for (int t = 0; t < TOKENS_PER_THREAD_K2Q; t++) {
            int token_state;
            if (token_active_k2q[t]) {
                if constexpr (N_FUNC > 0) {
                    token_state = get_token_kv_block_state_from_regs<N_FUNC>(func_regs_k2q[t], kv_start, kv_end);
                } else {
                    int local_q_idx;
                    if constexpr (USE_WARP_LEVEL_K2Q) {
                        local_q_idx = lane_id * TOKENS_PER_THREAD_K2Q + t;
                    } else {
                        local_q_idx = tid;
                    }
                    const int* ptr = base_func_ptr + (q_start + local_q_idx) * stride_q;
                    token_state = get_token_kv_block_state<N_FUNC>(ptr, n_func, stride_f, kv_start, kv_end);
                }
            } else {
                // K2Q always checks q_boundary: inactive tokens contribute STATE_EMPTY
                token_state = STATE_EMPTY;
            }
            local_state_mask |= (1u << token_state);
        }

        // =====================================================================
        // Reduce state across all tokens in the q_block
        // =====================================================================
        int q_block_state;
        if constexpr (USE_WARP_LEVEL_K2Q) {
            // Warp-level reduction only (32 threads)
            q_block_state = warp_reduce_state_only(local_state_mask);
        } else {
            // Block-level reduction (warp reduce + cross-warp reduce)
            q_block_state = block_reduce_state_bitmask(local_state_mask, warp_results_k2q);
        }

        // Record the result
        int leader_id_k2q = USE_WARP_LEVEL_K2Q ? lane_id : tid;
        if (leader_id_k2q == 0) {
            bool can_be_full = is_q_block_full && is_kv_block_full;

            if (q_block_state == STATE_FULL && can_be_full) {
                block_idx[idx_base + full_count] = q_block_idx;
                full_count++;
            } else if (q_block_state == STATE_FULL || q_block_state == STATE_MASK) {
                block_idx[idx_base + (num_q_blocks - 1 - mask_count)] = q_block_idx;
                mask_count++;
            }
        }

        if constexpr (!USE_WARP_LEVEL_K2Q) {
            __syncthreads();
        }
    }

    // Write final counts
    int final_leader_id_k2q = USE_WARP_LEVEL_K2Q ? lane_id : tid;
    if (final_leader_id_k2q == 0) {
        mask_block_cnt[cnt_offset] = mask_count;
        full_block_cnt[cnt_offset] = full_count;
    }
}

// =============================================================================
// Launch functions
// =============================================================================

// Macro to dispatch Q2K kernel based on N_FUNC and block size template parameters
// Q_BS and KV_BS are compile-time block sizes (0 means use runtime value)
#define DISPATCH_Q2K_KERNEL_FULL(N_FUNC_VAL, Q_BS, KV_BS) \
    create_q2k_block_sparse_from_func_kernel<N_FUNC_VAL, Q_BS, KV_BS><<<grid, block_size, 0, stream>>>( \
        d_func_tensor, \
        stride_b, stride_h, stride_f, stride_q, \
        Q_LEN, KV_LEN, n_func, \
        Q_BLOCK_SIZE, KV_BLOCK_SIZE, \
        num_q_blocks, num_kv_blocks, \
        check_q_boundary, \
        d_mask_block_cnt, d_full_block_cnt, \
        d_block_idx \
    )

// Shorthand macro for runtime block sizes
#define DISPATCH_Q2K_KERNEL(N_FUNC_VAL) \
    DISPATCH_Q2K_KERNEL_FULL(N_FUNC_VAL, 0, 0)

// Macro to dispatch K2Q preprocess kernel based on N_FUNC template parameter
#define DISPATCH_K2Q_PREPROCESS_KERNEL(N_FUNC_VAL) \
    compute_q_block_kv_range_kernel<N_FUNC_VAL><<<grid, block_size, 0, stream>>>( \
        d_func_tensor, \
        stride_b, stride_h, stride_f, stride_q, \
        Q_LEN, n_func, \
        Q_BLOCK_SIZE, \
        num_q_blocks, \
        d_q_block_kv_min, d_q_block_kv_max \
    )

// Macro to dispatch K2Q kernel based on N_FUNC and block size template parameters
#define DISPATCH_K2Q_KERNEL_FULL(N_FUNC_VAL, Q_BS, KV_BS) \
    create_k2q_block_sparse_from_func_kernel<N_FUNC_VAL, Q_BS, KV_BS><<<grid, block_size, 0, stream>>>( \
        d_func_tensor, \
        stride_b, stride_h, stride_f, stride_q, \
        Q_LEN, KV_LEN, n_func, \
        Q_BLOCK_SIZE, KV_BLOCK_SIZE, \
        num_q_blocks, num_kv_blocks, \
        d_q_block_kv_min, d_q_block_kv_max, \
        d_mask_block_cnt, d_full_block_cnt, \
        d_block_idx \
    )

// Shorthand macro for runtime block sizes
#define DISPATCH_K2Q_KERNEL(N_FUNC_VAL) \
    DISPATCH_K2Q_KERNEL_FULL(N_FUNC_VAL, 0, 0)

// Helper macro for Q2K dispatch with block size specialization
// For common block size combinations, use compile-time constants
#define DISPATCH_Q2K_BY_NFUNC(N_FUNC_VAL, Q_BS, KV_BS) \
    switch (n_func) { \
        case 1:  DISPATCH_Q2K_KERNEL_FULL(1,  Q_BS, KV_BS); break; \
        case 3:  DISPATCH_Q2K_KERNEL_FULL(3,  Q_BS, KV_BS); break; \
        case 5:  DISPATCH_Q2K_KERNEL_FULL(5,  Q_BS, KV_BS); break; \
        case 7:  DISPATCH_Q2K_KERNEL_FULL(7,  Q_BS, KV_BS); break; \
        case 9:  DISPATCH_Q2K_KERNEL_FULL(9,  Q_BS, KV_BS); break; \
        case 11: DISPATCH_Q2K_KERNEL_FULL(11, Q_BS, KV_BS); break; \
        case 13: DISPATCH_Q2K_KERNEL_FULL(13, Q_BS, KV_BS); break; \
        case 15: DISPATCH_Q2K_KERNEL_FULL(15, Q_BS, KV_BS); break; \
        case 17: DISPATCH_Q2K_KERNEL_FULL(17, Q_BS, KV_BS); break; \
        case 19: DISPATCH_Q2K_KERNEL_FULL(19, Q_BS, KV_BS); break; \
        case 21: DISPATCH_Q2K_KERNEL_FULL(21, Q_BS, KV_BS); break; \
        case 23: DISPATCH_Q2K_KERNEL_FULL(23, Q_BS, KV_BS); break; \
        case 25: DISPATCH_Q2K_KERNEL_FULL(25, Q_BS, KV_BS); break; \
        case 27: DISPATCH_Q2K_KERNEL_FULL(27, Q_BS, KV_BS); break; \
        case 29: DISPATCH_Q2K_KERNEL_FULL(29, Q_BS, KV_BS); break; \
        case 31: DISPATCH_Q2K_KERNEL_FULL(31, Q_BS, KV_BS); break; \
        case 33: DISPATCH_Q2K_KERNEL_FULL(33, Q_BS, KV_BS); break; \
        default: DISPATCH_Q2K_KERNEL_FULL(0,  Q_BS, KV_BS); break; \
    }

void launch_create_q2k_block_sparse_from_func(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B, int H, int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    bool check_q_boundary,
    int* d_mask_block_cnt,
    int* d_full_block_cnt,
    int* d_block_idx,
    cudaStream_t stream
) {
    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);

    // 3D grid: (num_q_blocks, H, B)
    dim3 grid(num_q_blocks, H, B);

#ifdef ENABLE_WARP_LEVEL_OPT
    // =========================================================================
    // Warp-level optimization: use only 32 threads per block
    // Each thread processes Q_BLOCK_SIZE/32 q_tokens
    // Requires: Q_BLOCK_SIZE % 32 == 0, n_func in [1,33] (odd)
    // =========================================================================
    int block_size = WARP_SIZE_CONST;  // 32 threads per block

    if (Q_BLOCK_SIZE == 256 && KV_BLOCK_SIZE == 128) {
        DISPATCH_Q2K_BY_NFUNC(n_func, 256, 128);
    } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 128) {
        DISPATCH_Q2K_BY_NFUNC(n_func, 128, 128);
    } else {
        // Unsupported configuration for warp-level: Q_BLOCK_SIZE must be multiple of 32
        // with compile-time known block sizes
        // This should not happen in normal usage
        block_size = Q_BLOCK_SIZE;
        DISPATCH_Q2K_BY_NFUNC(n_func, 0, 0);
    }
#else
    // =========================================================================
    // Block-level baseline: Q_BLOCK_SIZE threads per block
    // Each thread processes 1 q_token
    // =========================================================================
    int block_size = Q_BLOCK_SIZE;

#ifdef DISABLE_BLOCK_SIZE_TEMPLATE
    // Block size template optimization disabled - always use runtime values
    DISPATCH_Q2K_BY_NFUNC(n_func, 0, 0);
#else
    if (Q_BLOCK_SIZE == 256 && KV_BLOCK_SIZE == 128) {
        // Forward pass default: 256x128
        DISPATCH_Q2K_BY_NFUNC(n_func, 256, 128);
    } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 128) {
        // Backward pass default: 128x128
        DISPATCH_Q2K_BY_NFUNC(n_func, 128, 128);
    } else if (Q_BLOCK_SIZE == 64 && KV_BLOCK_SIZE == 64) {
        // Small blocks: 64x64
        DISPATCH_Q2K_BY_NFUNC(n_func, 64, 64);
    } else {
        // Fallback: use runtime block sizes
        DISPATCH_Q2K_BY_NFUNC(n_func, 0, 0);
    }
#endif
#endif  // ENABLE_WARP_LEVEL_OPT
}

void launch_compute_q_block_kv_range(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B, int H, int Q_LEN, int n_func,
    int Q_BLOCK_SIZE,
    int* d_q_block_kv_min,
    int* d_q_block_kv_max,
    cudaStream_t stream
) {
    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);

    // 3D grid: (num_q_blocks, H, B)
    dim3 grid(num_q_blocks, H, B);
    int block_size = Q_BLOCK_SIZE;

    // Dispatch based on n_func value
    switch (n_func) {
        case 1:  DISPATCH_K2Q_PREPROCESS_KERNEL(1);  break;
        case 3:  DISPATCH_K2Q_PREPROCESS_KERNEL(3);  break;
        case 5:  DISPATCH_K2Q_PREPROCESS_KERNEL(5);  break;
        case 7:  DISPATCH_K2Q_PREPROCESS_KERNEL(7);  break;
        case 9:  DISPATCH_K2Q_PREPROCESS_KERNEL(9);  break;
        case 11: DISPATCH_K2Q_PREPROCESS_KERNEL(11); break;
        case 13: DISPATCH_K2Q_PREPROCESS_KERNEL(13); break;
        case 15: DISPATCH_K2Q_PREPROCESS_KERNEL(15); break;
        case 17: DISPATCH_K2Q_PREPROCESS_KERNEL(17); break;
        case 19: DISPATCH_K2Q_PREPROCESS_KERNEL(19); break;
        case 21: DISPATCH_K2Q_PREPROCESS_KERNEL(21); break;
        case 23: DISPATCH_K2Q_PREPROCESS_KERNEL(23); break;
        case 25: DISPATCH_K2Q_PREPROCESS_KERNEL(25); break;
        case 27: DISPATCH_K2Q_PREPROCESS_KERNEL(27); break;
        case 29: DISPATCH_K2Q_PREPROCESS_KERNEL(29); break;
        case 31: DISPATCH_K2Q_PREPROCESS_KERNEL(31); break;
        case 33: DISPATCH_K2Q_PREPROCESS_KERNEL(33); break;
        default: DISPATCH_K2Q_PREPROCESS_KERNEL(0);  break;
    }
}

// Helper macro for K2Q dispatch with block size specialization
#define DISPATCH_K2Q_BY_NFUNC(N_FUNC_VAL, Q_BS, KV_BS) \
    switch (n_func) { \
        case 1:  DISPATCH_K2Q_KERNEL_FULL(1,  Q_BS, KV_BS); break; \
        case 3:  DISPATCH_K2Q_KERNEL_FULL(3,  Q_BS, KV_BS); break; \
        case 5:  DISPATCH_K2Q_KERNEL_FULL(5,  Q_BS, KV_BS); break; \
        case 7:  DISPATCH_K2Q_KERNEL_FULL(7,  Q_BS, KV_BS); break; \
        case 9:  DISPATCH_K2Q_KERNEL_FULL(9,  Q_BS, KV_BS); break; \
        case 11: DISPATCH_K2Q_KERNEL_FULL(11, Q_BS, KV_BS); break; \
        case 13: DISPATCH_K2Q_KERNEL_FULL(13, Q_BS, KV_BS); break; \
        case 15: DISPATCH_K2Q_KERNEL_FULL(15, Q_BS, KV_BS); break; \
        case 17: DISPATCH_K2Q_KERNEL_FULL(17, Q_BS, KV_BS); break; \
        case 19: DISPATCH_K2Q_KERNEL_FULL(19, Q_BS, KV_BS); break; \
        case 21: DISPATCH_K2Q_KERNEL_FULL(21, Q_BS, KV_BS); break; \
        case 23: DISPATCH_K2Q_KERNEL_FULL(23, Q_BS, KV_BS); break; \
        case 25: DISPATCH_K2Q_KERNEL_FULL(25, Q_BS, KV_BS); break; \
        case 27: DISPATCH_K2Q_KERNEL_FULL(27, Q_BS, KV_BS); break; \
        case 29: DISPATCH_K2Q_KERNEL_FULL(29, Q_BS, KV_BS); break; \
        case 31: DISPATCH_K2Q_KERNEL_FULL(31, Q_BS, KV_BS); break; \
        case 33: DISPATCH_K2Q_KERNEL_FULL(33, Q_BS, KV_BS); break; \
        default: DISPATCH_K2Q_KERNEL_FULL(0,  Q_BS, KV_BS); break; \
    }

void launch_create_k2q_block_sparse_from_func(
    const int* d_func_tensor,
    int stride_b, int stride_h, int stride_f, int stride_q,
    int B, int H, int Q_LEN, int KV_LEN, int n_func,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    const int* d_q_block_kv_min,  // precomputed min kv for each q_block (can be nullptr)
    const int* d_q_block_kv_max,  // precomputed max kv for each q_block (can be nullptr)
    int* d_mask_block_cnt,
    int* d_full_block_cnt,
    int* d_block_idx,
    cudaStream_t stream
) {
    int num_q_blocks = DIVUP(Q_LEN, Q_BLOCK_SIZE);
    int num_kv_blocks = DIVUP(KV_LEN, KV_BLOCK_SIZE);

    // 3D grid: (num_kv_blocks, H, B) - note: kv_blocks is the outer loop
    dim3 grid(num_kv_blocks, H, B);

#ifdef ENABLE_WARP_LEVEL_OPT
    // =========================================================================
    // Warp-level optimization: use only 32 threads per block
    // Each thread processes Q_BLOCK_SIZE/32 q_tokens
    // =========================================================================
    int block_size = WARP_SIZE_CONST;  // 32 threads per block

    if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 128) {
        DISPATCH_K2Q_BY_NFUNC(n_func, 128, 128);
    } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 256) {
        DISPATCH_K2Q_BY_NFUNC(n_func, 128, 256);
    } else if (Q_BLOCK_SIZE == 256 && KV_BLOCK_SIZE == 128) {
        DISPATCH_K2Q_BY_NFUNC(n_func, 256, 128);
    } else if (Q_BLOCK_SIZE == 64 && KV_BLOCK_SIZE == 64) {
        DISPATCH_K2Q_BY_NFUNC(n_func, 64, 64);
    } else if (Q_BLOCK_SIZE == 64 && KV_BLOCK_SIZE == 128) {
        DISPATCH_K2Q_BY_NFUNC(n_func, 64, 128);
    } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 64) {
        DISPATCH_K2Q_BY_NFUNC(n_func, 128, 64);
    } else {
        // Unsupported configuration for warp-level
        block_size = Q_BLOCK_SIZE;
        DISPATCH_K2Q_BY_NFUNC(n_func, 0, 0);
    }
#else
    // =========================================================================
    // Block-level baseline: Q_BLOCK_SIZE threads per block
    // =========================================================================
    int block_size = Q_BLOCK_SIZE;

#ifdef DISABLE_BLOCK_SIZE_TEMPLATE
    // Block size template optimization disabled - always use runtime values
    DISPATCH_K2Q_BY_NFUNC(n_func, 0, 0);
#else
    if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 128) {
        // Backward pass default: 128x128
        DISPATCH_K2Q_BY_NFUNC(n_func, 128, 128);
    } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 256) {
        // SM100/SM110 2-CTA linear CSR backward
        DISPATCH_K2Q_BY_NFUNC(n_func, 128, 256);
    } else if (Q_BLOCK_SIZE == 256 && KV_BLOCK_SIZE == 128) {
        // 256x128 configuration
        DISPATCH_K2Q_BY_NFUNC(n_func, 256, 128);
    } else if (Q_BLOCK_SIZE == 64 && KV_BLOCK_SIZE == 64) {
        // Small blocks: 64x64
        DISPATCH_K2Q_BY_NFUNC(n_func, 64, 64);
    } else {
        // Fallback: use runtime block sizes
        DISPATCH_K2Q_BY_NFUNC(n_func, 0, 0);
    }
#endif
#endif  // ENABLE_WARP_LEVEL_OPT
}

#undef DISPATCH_Q2K_KERNEL
#undef DISPATCH_Q2K_KERNEL_FULL
#undef DISPATCH_Q2K_BY_NFUNC
#undef DISPATCH_Q2K_WARP_KERNEL
#undef DISPATCH_Q2K_WARP_BY_NFUNC
#undef DISPATCH_K2Q_PREPROCESS_KERNEL
#undef DISPATCH_K2Q_KERNEL
#undef DISPATCH_K2Q_KERNEL_FULL
#undef DISPATCH_K2Q_BY_NFUNC
#undef DISPATCH_K2Q_WARP_KERNEL
#undef DISPATCH_K2Q_WARP_BY_NFUNC

// =============================================================================
// Compact Block Idx Kernels
// =============================================================================

/**
 * Kernel to extract compact indices from block_idx.
 *
 * Each block handles one row of block_idx.
 * - Full indices: stored left-to-right at block_idx[row, 0:full_cnt]
 * - Mask indices: stored right-to-left at block_idx[row, max_blocks-mask_cnt:max_blocks]
 *   Need to reverse when extracting.
 */
__global__ void extract_compact_indices_kernel(
    const int* __restrict__ block_idx,        // [n_blocks, max_blocks]
    const int* __restrict__ mask_cnt,         // [n_blocks]
    const int* __restrict__ full_cnt,         // [n_blocks]
    const int* __restrict__ mask_offset,      // [n_blocks + 1]
    const int* __restrict__ full_offset,      // [n_blocks + 1]
    int n_blocks,
    int max_blocks,
    int* __restrict__ mask_idx_compact,       // [total_mask_blocks]
    int* __restrict__ full_idx_compact        // [total_full_blocks]
) {
    int row = blockIdx.x;
    if (row >= n_blocks) return;

    int tid = threadIdx.x;

    int fcnt = full_cnt[row];
    int mcnt = mask_cnt[row];
    int foff = full_offset[row];
    int moff = mask_offset[row];

    const int* row_ptr = block_idx + row * max_blocks;

    // Extract full indices (left-to-right, no reversal needed)
    for (int i = tid; i < fcnt; i += blockDim.x) {
        full_idx_compact[foff + i] = row_ptr[i];
    }

    // Extract mask indices (right-to-left, need to reverse)
    // Original: row_ptr[max_blocks - mcnt + i] for i in [0, mcnt)
    // Reversed: row_ptr[max_blocks - 1 - i] for i in [0, mcnt)
    for (int i = tid; i < mcnt; i += blockDim.x) {
        mask_idx_compact[moff + i] = row_ptr[max_blocks - 1 - i];
    }
}

void launch_extract_compact_indices(
    const int* d_block_idx,
    const int* d_mask_block_cnt,
    const int* d_full_block_cnt,
    const int* d_mask_block_offset,
    const int* d_full_block_offset,
    int n_blocks,
    int max_blocks,
    int* d_mask_block_idx,
    int* d_full_block_idx,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    extract_compact_indices_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
        d_block_idx,
        d_mask_block_cnt, d_full_block_cnt,
        d_mask_block_offset, d_full_block_offset,
        n_blocks, max_blocks,
        d_mask_block_idx, d_full_block_idx
    );
}

// ============================================================================
// Dual Inclusive Sum - Single CTA scan using CUB BlockScan
// One CTA processes all B*H*num_blocks_plus_one elements
// ============================================================================

// Block size for scan kernel (must be power of 2 for CUB BlockScan)
#define SCAN_BLOCK_SIZE 256

// Custom int2 addition operator for CUB
struct Int2Sum {
    __device__ __forceinline__ int2 operator()(const int2& a, const int2& b) const {
        return make_int2(a.x + b.x, a.y + b.y);
    }
};

/**
 * Dual inclusive sum kernel with exclusive prefix sum output format.
 *
 * Input:  cnt[0..n-1]
 * Output: offset[0..n] where offset[0]=0, offset[i+1]=sum(cnt[0..i])
 *
 * This produces exclusive prefix sum format: [0, c0, c0+c1, ..., total]
 */
__global__ void dual_inclusive_sum_kernel(
    const int* __restrict__ mask_cnt,      // [n_elements]
    const int* __restrict__ full_cnt,      // [n_elements]
    int* __restrict__ mask_offset,         // [n_elements + 1]
    int* __restrict__ full_offset,         // [n_elements + 1]
    int n_elements                         // Number of count elements (B * H * num_blocks)
) {
    // CUB BlockScan type
    typedef cub::BlockScan<int2, SCAN_BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int2 shared_prefix;

    int tid = threadIdx.x;

    // Write leading zeros (offset[0] = 0)
    if (tid == 0) {
        mask_offset[0] = 0;
        full_offset[0] = 0;
    }

    // Process all elements in tiles using a single CTA
    int2 running_prefix = make_int2(0, 0);

    for (int tile_start = 0; tile_start < n_elements; tile_start += SCAN_BLOCK_SIZE) {
        int idx = tile_start + tid;

        // Load data (use 0 for out-of-bounds)
        int2 thread_data;
        if (idx < n_elements) {
            thread_data = make_int2(mask_cnt[idx], full_cnt[idx]);
        } else {
            thread_data = make_int2(0, 0);
        }

        // Perform block-level inclusive scan
        int2 thread_result;
        BlockScan(temp_storage).InclusiveScan(thread_data, thread_result, Int2Sum());
        __syncthreads();

        // Add running prefix from previous tiles
        thread_result.x += running_prefix.x;
        thread_result.y += running_prefix.y;

        // Write result to offset[idx + 1] (exclusive prefix sum format)
        if (idx < n_elements) {
            mask_offset[idx + 1] = thread_result.x;
            full_offset[idx + 1] = thread_result.y;
        }

        // Broadcast running_prefix to all threads for next tile
        // The last valid thread in this tile writes to shared memory
        int last_tid_in_tile = min(SCAN_BLOCK_SIZE - 1, n_elements - 1 - tile_start);
        if (tid == last_tid_in_tile) {
            shared_prefix = thread_result;
        }
        __syncthreads();
        running_prefix = shared_prefix;
    }
}

void launch_dual_inclusive_sum(
    const int* d_mask_cnt,
    const int* d_full_cnt,
    int* d_mask_offset,
    int* d_full_offset,
    int n_elements,         // Number of count elements (B * H * num_blocks)
    cudaStream_t stream
) {
    // Launch single CTA to process all elements
    // Kernel writes offset[0] = 0 and offset[i+1] = inclusive_sum(cnt[0..i])
    dual_inclusive_sum_kernel<<<1, SCAN_BLOCK_SIZE, 0, stream>>>(
        d_mask_cnt, d_full_cnt,
        d_mask_offset, d_full_offset,
        n_elements
    );
}
