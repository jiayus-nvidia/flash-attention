/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flash {

// Validate the value-dependent part of the deterministic K2Q contract before
// launching a kernel that may spin on dq_write_order.  This is intentionally a
// host-side, one-shot validator for the raw C++ op.  The public Python wrapper
// performs the same work in its explicit metadata preprocessing step and uses
// the internal skip flag only after obtaining that validation certificate.
template <typename Tensor>
void validate_deterministic_k2q_semantics(
    Tensor const& mask_cnt,
    Tensor const& mask_offset,
    Tensor const& mask_idx,
    Tensor const& full_cnt,
    Tensor const& full_offset,
    Tensor const& full_idx,
    Tensor const& mask_rank,
    Tensor const* full_rank,
    int64_t num_n_blocks,
    int64_t num_m_blocks,
    cudaStream_t stream) {
    if (num_n_blocks <= 0 || num_m_blocks <= 0) {
        throw std::invalid_argument(
            "deterministic K2Q metadata requires positive n-block and m-block counts");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t const capture_query_status =
        cudaStreamIsCapturing(stream, &capture_status);
    if (capture_query_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("failed to query deterministic K2Q validation stream: ") +
            cudaGetErrorString(capture_query_status));
    }
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument(
            "raw deterministic K2Q semantic validation is not supported during "
            "CUDA Graph capture; use the public Python attention wrapper, or "
            "preprocess metadata before capture and explicitly set "
            "unsafe_skip_block_sparse_semantic_validation=True on the raw op");
    }

    std::vector<int32_t> h_mask_cnt(mask_cnt.numel());
    std::vector<int32_t> h_mask_offset(mask_offset.numel());
    std::vector<int32_t> h_mask_idx(mask_idx.numel());
    std::vector<int32_t> h_full_cnt(full_cnt.numel());
    std::vector<int32_t> h_full_offset(full_offset.numel());
    std::vector<int32_t> h_full_idx(full_idx.numel());
    std::vector<int32_t> h_mask_rank(mask_rank.numel());
    std::vector<int32_t> h_full_rank(full_rank == nullptr ? 0 : full_rank->numel());

    auto copy_to_host = [&](Tensor const& tensor, std::vector<int32_t>& host) {
        if (host.empty()) {
            return;
        }
        cudaError_t const status = cudaMemcpyAsync(
            host.data(),
            tensor.data_ptr(),
            host.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("failed to copy deterministic K2Q metadata to host: ") +
                cudaGetErrorString(status));
        }
    };
    copy_to_host(mask_cnt, h_mask_cnt);
    copy_to_host(mask_offset, h_mask_offset);
    copy_to_host(mask_idx, h_mask_idx);
    copy_to_host(full_cnt, h_full_cnt);
    copy_to_host(full_offset, h_full_offset);
    copy_to_host(full_idx, h_full_idx);
    copy_to_host(mask_rank, h_mask_rank);
    if (full_rank != nullptr) {
        copy_to_host(*full_rank, h_full_rank);
    }
    cudaError_t const sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("failed to synchronize deterministic K2Q validation: ") +
            cudaGetErrorString(sync_status));
    }

    int64_t const num_rows = static_cast<int64_t>(h_mask_cnt.size());
    if (num_rows % num_n_blocks != 0) {
        throw std::invalid_argument(
            "deterministic K2Q count rows must be divisible by num_n_blocks");
    }

    auto validate_csr = [&](char const* kind,
                            std::vector<int32_t> const& counts,
                            std::vector<int32_t> const& offsets,
                            std::vector<int32_t> const& indices) {
        if (offsets.empty() || offsets.front() != 0) {
            throw std::invalid_argument(
                std::string("deterministic K2Q ") + kind +
                " CSR offset must start at 0");
        }
        if (offsets.back() != static_cast<int64_t>(indices.size())) {
            throw std::invalid_argument(
                std::string("deterministic K2Q ") + kind +
                " CSR final offset must equal idx.numel()");
        }
        for (int64_t row = 0; row < num_rows; ++row) {
            int64_t const begin = offsets[row];
            int64_t const end = offsets[row + 1];
            if (counts[row] < 0 || begin < 0 || end < begin ||
                end - begin != counts[row]) {
                throw std::invalid_argument(
                    std::string("deterministic K2Q ") + kind +
                    " CSR offset delta must equal a non-negative count");
            }
        }
    };
    validate_csr("partial", h_mask_cnt, h_mask_offset, h_mask_idx);
    validate_csr("full", h_full_cnt, h_full_offset, h_full_idx);

    using Contributor = std::pair<int32_t, int32_t>;  // (n_block, rank)
    std::unordered_set<uint64_t> edges;
    std::unordered_map<uint64_t, std::vector<Contributor>> contributors;
    edges.reserve(h_mask_idx.size() + h_full_idx.size());
    contributors.reserve(h_mask_idx.size() + h_full_idx.size());

    auto add_edges = [&](char const* kind,
                         std::vector<int32_t> const& offsets,
                         std::vector<int32_t> const& indices,
                         std::vector<int32_t> const& ranks) {
        for (int64_t row = 0; row < num_rows; ++row) {
            int64_t const n_block = row % num_n_blocks;
            int64_t const metadata_bh = row / num_n_blocks;
            for (int64_t pos = offsets[row]; pos < offsets[row + 1]; ++pos) {
                int64_t const m_block = indices[pos];
                if (m_block < 0 || m_block >= num_m_blocks) {
                    throw std::invalid_argument(
                        std::string("deterministic K2Q ") + kind +
                        " CSR contains out-of-range m_block");
                }
                uint64_t const edge_key =
                    static_cast<uint64_t>(row) * static_cast<uint64_t>(num_m_blocks) +
                    static_cast<uint64_t>(m_block);
                if (!edges.insert(edge_key).second) {
                    throw std::invalid_argument(
                        "deterministic K2Q CSR contains a duplicate edge or the "
                        "same edge in partial/full");
                }
                uint64_t const group_key =
                    static_cast<uint64_t>(metadata_bh) *
                        static_cast<uint64_t>(num_m_blocks) +
                    static_cast<uint64_t>(m_block);
                contributors[group_key].emplace_back(
                    static_cast<int32_t>(n_block), ranks[pos]);
            }
        }
    };
    add_edges("partial", h_mask_offset, h_mask_idx, h_mask_rank);
    add_edges("full", h_full_offset, h_full_idx, h_full_rank);

    for (auto& entry : contributors) {
        auto& values = entry.second;
        std::sort(
            values.begin(),
            values.end(),
            [](Contributor const& lhs, Contributor const& rhs) {
                return lhs.first > rhs.first;
            });
        for (int64_t rank = 0; rank < static_cast<int64_t>(values.size()); ++rank) {
            if ((rank > 0 && values[rank - 1].first == values[rank].first) ||
                values[rank].second != rank) {
                throw std::invalid_argument(
                    "deterministic K2Q dq_write_order must be the contiguous "
                    "rank of descending n_block across partial/full contributors");
            }
        }
    }
}

}  // namespace flash
