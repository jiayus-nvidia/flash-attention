"""
Block-sparsity utilities for FlexAttention
"""

from typing import Callable, NamedTuple, Tuple

import cutlass.cute as cute
import torch

from flash_attn_cute.cute_dsl_utils import get_broadcast_dims, to_cute_tensor
from flash_attn_cute.testing import is_fake_mode


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None = None
    full_block_idx: cute.Tensor | None = None
    cu_total_m_blocks: cute.Tensor | None = None
    cu_block_idx_offsets: cute.Tensor | None = None
    dq_write_order: cute.Tensor | None = None
    dq_write_order_full: cute.Tensor | None = None
    mask_block_offset: cute.Tensor | None = None
    full_block_offset: cute.Tensor | None = None
    mask_block_masks: cute.Tensor | None = None

    def __new_from_mlir_values__(self, values):
        new_fields = []
        idx = 0
        for original in self:
            if original is None:
                new_fields.append(None)
            else:
                new_fields.append(values[idx])
                idx += 1
        return BlockSparseTensors(*new_fields)


class BlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    cu_total_m_blocks: torch.Tensor | None = None
    cu_block_idx_offsets: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: torch.Tensor | None = None
    dq_write_order_full: torch.Tensor | None = None
    spt: bool | None = None
    mask_block_offset: torch.Tensor | None = None
    full_block_offset: torch.Tensor | None = None
    mask_block_masks: torch.Tensor | None = None
    pack_gqa: bool | None = None
    bwd_tensors: "BlockSparseTensorsTorch | None" = None


class LinearBlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_offset: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None = None
    full_block_offset: cute.Tensor | None = None
    full_block_idx: cute.Tensor | None = None

    def __new_from_mlir_values__(self, values):
        if len(values) == 3:
            values = (*values, None, None, None)
        return LinearBlockSparseTensors(*values)


class LinearBlockSparseTensorsTorch(NamedTuple):
    mask_block_cnt: torch.Tensor
    mask_block_offset: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: torch.Tensor | None = None
    full_block_offset: torch.Tensor | None = None
    full_block_idx: torch.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: torch.Tensor | None = None
    dq_write_order_full: torch.Tensor | None = None
    spt: bool | None = None


def is_linear_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
) -> bool:
    return tensors.mask_block_offset is not None


def bhqk_to_linear_sparse_tensors(
    bhqk_tensors: BlockSparseTensorsTorch,
) -> LinearBlockSparseTensorsTorch:
    """Convert 4D block-sparse indices to the compact CSR-like representation."""
    mask_block_cnt = bhqk_tensors.mask_block_cnt
    batch, nheads, n_blocks = mask_block_cnt.shape
    mask_block_offset = torch.cat(
        [
            torch.zeros(1, device=mask_block_cnt.device, dtype=torch.int32),
            torch.cumsum(mask_block_cnt.flatten(), dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    mask_block_idx_parts = [
        bhqk_tensors.mask_block_idx[i, j, k, : mask_block_cnt[i, j, k].item()]
        for i in range(batch)
        for j in range(nheads)
        for k in range(n_blocks)
    ]
    mask_block_idx = torch.cat(mask_block_idx_parts, dim=0)

    full_block_cnt = bhqk_tensors.full_block_cnt
    full_block_offset = None
    full_block_idx = None
    if full_block_cnt is not None:
        full_block_offset = torch.cat(
            [
                torch.zeros(1, device=full_block_cnt.device, dtype=torch.int32),
                torch.cumsum(full_block_cnt.flatten(), dim=0, dtype=torch.int32),
            ],
            dim=0,
        )
        if bhqk_tensors.full_block_idx is not None:
            full_block_idx_parts = [
                bhqk_tensors.full_block_idx[i, j, k, : full_block_cnt[i, j, k].item()]
                for i in range(batch)
                for j in range(nheads)
                for k in range(n_blocks)
            ]
            full_block_idx = torch.cat(full_block_idx_parts, dim=0)

    return LinearBlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_offset=mask_block_offset,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_offset=full_block_offset,
        full_block_idx=full_block_idx,
        block_size=bhqk_tensors.block_size,
        dq_write_order=bhqk_tensors.dq_write_order,
        dq_write_order_full=bhqk_tensors.dq_write_order_full,
        spt=bhqk_tensors.spt,
    )


def _ordered_to_dense_simple(
    num_blocks: torch.Tensor,
    indices: torch.Tensor,
    num_cols: int,
) -> torch.Tensor:
    """Convert ordered sparse representation to dense binary matrix.

    Args:
        num_blocks: [B, H, num_rows] count of valid entries per row
        indices: [B, H, num_rows, max_entries] column indices (valid entries packed left)
        num_cols: total number of columns

    Returns:
        dense: [B, H, num_rows, num_cols] binary int32 matrix
    """
    B, H, num_rows, max_entries = indices.shape
    device = indices.device
    dense = torch.zeros(B, H, num_rows, num_cols + 1, dtype=torch.int32, device=device)
    col_range = torch.arange(max_entries, device=device)
    valid = col_range[None, None, None, :] < num_blocks[:, :, :, None]
    safe_indices = torch.where(valid, indices.long(), num_cols)
    row_idx = torch.arange(num_rows, device=device)[None, None, :, None].expand_as(indices)
    b_idx = torch.arange(B, device=device)[:, None, None, None].expand_as(indices)
    h_idx = torch.arange(H, device=device)[None, :, None, None].expand_as(indices)
    dense[b_idx, h_idx, row_idx, safe_indices] = 1
    return dense[:, :, :, :num_cols]


def compute_dq_write_order(
    fwd_mask_cnt: torch.Tensor,
    fwd_mask_idx: torch.Tensor,
    fwd_full_cnt: torch.Tensor | None,
    fwd_full_idx: torch.Tensor | None,
    bwd_mask_cnt: torch.Tensor,
    bwd_mask_idx: torch.Tensor,
    bwd_full_cnt: torch.Tensor | None,
    bwd_full_idx: torch.Tensor | None,
    spt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute dQ write-order metadata for deterministic block-sparse backward.

    For each (n_block, i) in the backward iteration, computes the semaphore
    lock value: the rank of n_block in the combined (partial + full) sorted
    contributor list for the target m_block.

    Lock values are assigned in ascending n_block order (or descending if spt=True)
    to guarantee deadlock-freedom with the CTA scheduling order.

    Args:
        fwd_mask_cnt: [B, H, num_m_blocks] partial contributor counts per m_block
        fwd_mask_idx: [B, H, num_m_blocks, max_kv] partial contributor n_block indices (ascending)
        fwd_full_cnt: [B, H, num_m_blocks] full contributor counts per m_block (optional)
        fwd_full_idx: [B, H, num_m_blocks, max_kv] full contributor n_block indices (optional)
        bwd_mask_cnt: [B, H, num_n_blocks] partial iteration counts per n_block
        bwd_mask_idx: [B, H, num_n_blocks, max_q] partial iteration m_block indices
        bwd_full_cnt: [B, H, num_n_blocks] full iteration counts per n_block (optional)
        bwd_full_idx: [B, H, num_n_blocks, max_q] full iteration m_block indices (optional)
        spt: if True, reverse ordering (highest n_block gets lock_value=0)

    Returns:
        (dq_write_order, dq_write_order_full): tensors parallel to bwd_mask_idx
        and bwd_full_idx respectively, containing lock values.
    """
    device = fwd_mask_idx.device
    B, H, num_m, max_kv_partial = fwd_mask_idx.shape
    _, _, num_n, max_q_partial = bwd_mask_idx.shape

    has_full = fwd_full_cnt is not None and fwd_full_idx is not None

    dense_partial = _ordered_to_dense_simple(fwd_mask_cnt, fwd_mask_idx, num_n)
    if has_full:
        dense_full = _ordered_to_dense_simple(fwd_full_cnt, fwd_full_idx, num_n)
        dense = (dense_partial + dense_full).clamp(max=1)
    else:
        dense = dense_partial

    cumsum = dense.cumsum(dim=-1)
    rank_table = (cumsum - dense).to(torch.int32)

    if spt:
        total_per_m = cumsum[:, :, :, -1:]
        rank_table = (total_per_m - 1 - rank_table).to(torch.int32)

    def _gather_write_order(bwd_idx, bwd_cnt):
        b_i = torch.arange(B, device=device)[:, None, None, None].expand_as(bwd_idx)
        h_i = torch.arange(H, device=device)[None, :, None, None].expand_as(bwd_idx)
        n_i = torch.arange(bwd_idx.shape[2], device=device)[None, None, :, None].expand_as(bwd_idx)
        m_vals = bwd_idx.long().clamp(0, num_m - 1)
        return rank_table[b_i, h_i, m_vals, n_i].to(torch.int32)

    dq_write_order = _gather_write_order(bwd_mask_idx, bwd_mask_cnt)

    dq_write_order_full = None
    if has_full and bwd_full_cnt is not None and bwd_full_idx is not None:
        dq_write_order_full = _gather_write_order(bwd_full_idx, bwd_full_cnt)

    return dq_write_order, dq_write_order_full


def compute_dq_write_order_from_block_mask(
    block_mask,
    spt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = block_mask.as_tuple()
    return compute_dq_write_order(
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        spt=spt,
    )


def compute_dq_write_order_from_linear_csr(
    tensors: LinearBlockSparseTensorsTorch,
    spt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute dQ write-order metadata for linear CSR backward sparsity.

    The input CSR is expected to be the Q-direction/backward sparsity:
    each compact row is an n_block and each entry is an m_block written by that
    n_block. The returned tensors are compact 1D arrays parallel to
    mask_block_idx and full_block_idx.
    """
    if tensors.mask_block_offset is None:
        raise ValueError("linear CSR dq_write_order computation requires mask_block_offset")
    if tensors.mask_block_idx is None:
        raise ValueError("linear CSR dq_write_order computation requires mask_block_idx")
    if tensors.mask_block_cnt.ndim != 3:
        raise ValueError("linear CSR mask_block_cnt must have shape (B, H, N)")

    mask_cnt = tensors.mask_block_cnt
    mask_offset = tensors.mask_block_offset
    mask_idx = tensors.mask_block_idx
    device = mask_idx.device
    _, _, num_n_blocks = mask_cnt.shape
    num_rows = mask_cnt.numel()
    if mask_offset.numel() != num_rows + 1:
        raise ValueError(
            "linear CSR mask_block_offset must have one more element than mask_block_cnt"
        )

    has_full = any(
        x is not None
        for x in (tensors.full_block_cnt, tensors.full_block_offset, tensors.full_block_idx)
    )
    if has_full and not all(
        x is not None
        for x in (tensors.full_block_cnt, tensors.full_block_offset, tensors.full_block_idx)
    ):
        raise ValueError(
            "full_block_cnt, full_block_offset, and full_block_idx must be provided together"
        )

    def _entry_metadata(offset: torch.Tensor, idx: torch.Tensor):
        positions = torch.arange(idx.numel(), device=device, dtype=torch.int64)
        row = torch.searchsorted(offset.to(torch.int64), positions, right=True) - 1
        bh = row // num_n_blocks
        n_block = row - bh * num_n_blocks
        return bh, n_block, idx.to(torch.int64)

    mask_bh, mask_n, mask_m = _entry_metadata(mask_offset, mask_idx)
    all_bh = [mask_bh]
    all_n = [mask_n]
    all_m = [mask_m]

    full_len = 0
    if has_full:
        full_cnt = tensors.full_block_cnt
        full_offset = tensors.full_block_offset
        full_idx = tensors.full_block_idx
        assert full_cnt is not None
        assert full_offset is not None
        assert full_idx is not None
        if full_cnt.shape != mask_cnt.shape:
            raise ValueError("linear CSR full_block_cnt must match mask_block_cnt shape")
        if full_offset.numel() != num_rows + 1:
            raise ValueError(
                "linear CSR full_block_offset must have one more element than full_block_cnt"
            )
        full_len = full_idx.numel()
        full_bh, full_n, full_m = _entry_metadata(full_offset, full_idx)
        all_bh.append(full_bh)
        all_n.append(full_n)
        all_m.append(full_m)

    total = mask_idx.numel() + full_len
    if total == 0:
        return (
            torch.zeros_like(mask_idx),
            torch.zeros_like(tensors.full_block_idx) if has_full else None,
        )

    flat_bh = torch.cat(all_bh)
    flat_n = torch.cat(all_n)
    flat_m = torch.cat(all_m)
    max_m = torch.max(flat_m)
    group_key = flat_bh * (max_m + 1) + flat_m
    n_order = (num_n_blocks - 1 - flat_n) if spt else flat_n
    sort_key = group_key * num_n_blocks + n_order
    sorted_pos = torch.argsort(sort_key, stable=True)
    sorted_group = group_key[sorted_pos]

    pos = torch.arange(total, device=device, dtype=torch.int64)
    boundary_pos = torch.full_like(pos, -1)
    boundary_pos[0] = 0
    if total > 1:
        boundary_pos[1:] = torch.where(
            sorted_group[1:] != sorted_group[:-1],
            pos[1:],
            torch.full_like(pos[1:], -1),
        )
    last_boundary, _ = torch.cummax(boundary_pos, dim=0)
    sorted_rank = (pos - last_boundary).to(torch.int32)

    flat_rank = torch.empty(total, device=device, dtype=torch.int32)
    flat_rank[sorted_pos] = sorted_rank
    mask_rank = flat_rank[: mask_idx.numel()]
    full_rank = flat_rank[mask_idx.numel() :] if has_full else None
    return mask_rank, full_rank


def get_sparse_q_block_size(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch | None,
    seqlen_q: int,
) -> int | None:
    """Return the Q sparse block size, or None when sparsity is unset or ambiguous."""
    if tensors is None:
        return None
    if tensors.block_size is not None:
        return tensors.block_size[0]
    if is_linear_block_sparse_tensors(tensors):
        num_m_blocks = tensors.mask_block_cnt.shape[2]
    else:
        num_m_blocks = tensors.mask_block_idx.shape[2]
    min_block_size = ceildiv(seqlen_q, num_m_blocks)
    max_block_size = seqlen_q if num_m_blocks == 1 else (seqlen_q - 1) // (num_m_blocks - 1)
    if min_block_size != max_block_size:
        return None
    return min_block_size


def _expand_sparsity_tensor(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    tensor_name: str,
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> torch.Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = tensor.shape != expected_shape
    if not needs_expand:
        return tensor
    can_expand = all(map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape))
    if not can_expand:
        context_clause = f" ({context})" if context else ""
        resolved_hint = hint() if callable(hint) else hint
        hint_clause = f" Hint: {resolved_hint}" if resolved_hint else ""
        raise ValueError(
            f"{tensor_name}{context_clause} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
            f"{hint_clause}"
        )
    return tensor.expand(*expected_shape)


def _check_and_expand_block(
    name: str,
    cnt: torch.Tensor | None,
    idx: torch.Tensor | None,
    expected_count_shape: Tuple[int, ...],
    expected_index_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    if cnt.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError(f"{name}_block tensors must have dtype torch.int32")
    if cnt.device != idx.device:
        raise ValueError(f"{name}_block_cnt and {name}_block_idx must be on the same device")
    if not cnt.is_cuda or not idx.is_cuda:
        raise ValueError(f"{name}_block tensors must live on CUDA")
    expanded_cnt = _expand_sparsity_tensor(
        cnt, expected_count_shape, f"{name}_block_cnt", context, hint
    )
    # [Note] Allow Compact block sparse indices
    # Allow the last dimension (n_blocks) of idx to be <= expected, since
    # FA4 only accesses indices 0..cnt-1 per query tile. This enables compact
    # index tensors that avoid O(N^2) memory at long sequence lengths.
    if idx.ndim == 4 and idx.shape[3] <= expected_index_shape[3]:
        expected_index_shape = (*expected_index_shape[:3], idx.shape[3])
    expanded_idx = _expand_sparsity_tensor(
        idx, expected_index_shape, f"{name}_block_idx", context, hint
    )
    return expanded_cnt, expanded_idx


def _check_and_expand_metadata_tensor(
    name: str,
    tensor: torch.Tensor | None,
    expected_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as block sparse tensors")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must live on CUDA")
    return _expand_sparsity_tensor(tensor, expected_shape, name, context, hint)


def _check_linear_block(
    name: str,
    cnt: torch.Tensor | None,
    offset: torch.Tensor | None,
    idx: torch.Tensor | None,
    expected_count_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
    expected_base_shape: Tuple[int, ...] | None = None,
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if cnt is None and offset is None and idx is None:
        return None, None, None
    if cnt is None or offset is None or idx is None:
        raise ValueError(
            f"{name}_block_cnt, {name}_block_offset, and {name}_block_idx "
            "must all be provided or all be None for linear block sparsity"
        )
    if cnt.dtype != torch.int32 or offset.dtype != torch.int32 or idx.dtype != torch.int32:
        raise ValueError(f"{name}_block tensors must have dtype torch.int32")
    if cnt.device != offset.device or cnt.device != idx.device:
        raise ValueError(f"{name}_block tensors must be on the same device")
    if not cnt.is_cuda or not offset.is_cuda or not idx.is_cuda:
        raise ValueError(f"{name}_block tensors must live on CUDA")
    if cnt.ndim != 3 or offset.ndim != 1 or idx.ndim != 1:
        raise ValueError(
            f"{name}_block tensors{f' ({context})' if context else ''} must have shapes "
            "(B, H, M/N), (B * H * M/N + 1), and (nnz,)."
        )
    if expected_base_shape is not None and tuple(cnt.shape) != expected_base_shape:
        raise ValueError(
            f"{name}_block_cnt{f' ({context})' if context else ''} must have shape "
            f"{expected_base_shape} to share CSR offsets with mask_block_cnt."
        )
    for dim_name, cur, tgt in (
        ("batch", cnt.shape[0], expected_count_shape[0]),
        ("head", cnt.shape[1], expected_count_shape[1]),
    ):
        if cur != tgt and cur != 1:
            resolved_hint = hint() if callable(hint) else hint
            hint_clause = f" Hint: {resolved_hint}" if resolved_hint else ""
            raise ValueError(
                f"{name}_block_cnt{f' ({context})' if context else ''} {dim_name} dim "
                f"must be {tgt} or 1.{hint_clause}"
            )
    if cnt.shape[2] != expected_count_shape[2]:
        raise ValueError(
            f"{name}_block_cnt{f' ({context})' if context else ''} block dimension "
            f"{cnt.shape[2]} does not match expected {expected_count_shape[2]}."
        )
    expected_offset_size = cnt.shape[0] * cnt.shape[1] * cnt.shape[2] + 1
    if offset.numel() != expected_offset_size:
        raise ValueError(
            f"{name}_block_offset{f' ({context})' if context else ''} must have "
            f"{expected_offset_size} elements for count shape {tuple(cnt.shape)}, "
            f"got {offset.numel()}."
        )
    return cnt, offset, idx


def _check_linear_metadata_tensor(
    name: str,
    tensor: torch.Tensor | None,
    index_tensor: torch.Tensor | None,
    context: str | None,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if index_tensor is None:
        raise ValueError(f"{name} was provided but the corresponding block index tensor is None")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as block sparse tensors")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must live on CUDA")
    if tensor.ndim != 1 or tensor.shape != index_tensor.shape:
        context_clause = f" ({context})" if context else ""
        raise ValueError(
            f"{name}{context_clause} must be a 1D compact tensor with shape "
            f"{tuple(index_tensor.shape)}, got {tuple(tensor.shape)}."
        )
    return tensor


def get_block_sparse_expected_shapes(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for block sparse normalization."""
    m_block_size_effective = q_stage * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, m_block_size_effective)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)
    return expected_count_shape, expected_index_shape


def infer_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], int]:
    """Infer shapes and scaling for block-sparse tensors.

    Expectations:
    - mask_block_cnt is (B, H, M) and mask_block_idx is (B, H, M, N).
    - Batch/head dims may be 1 for broadcast, or match the requested sizes.
    - sparse_block_size_kv must match tile_n.
    - sparse_block_size_q must be a multiple of q_stage * tile_m.
    - If sparse_block_size_q is omitted and seqlen_q/num_m_blocks is ambiguous,
      the caller must provide block_size to disambiguate. TODO will make this required in a future PR.
    """
    base_m_block = q_stage * m_block_size
    base_n_block = n_block_size
    if sparse_block_size_kv is None:
        sparse_block_size_kv = base_n_block
    if sparse_block_size_kv != base_n_block:
        raise ValueError(f"Block sparse tensors{context} require BLOCK_SIZE_KV={base_n_block}.")
    if tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")
    num_m_blocks = tensors.mask_block_idx.shape[2]

    if sparse_block_size_q is None:
        sparse_block_size_q = get_sparse_q_block_size(tensors, seqlen_q)
        if sparse_block_size_q is None and base_m_block != 1:
            raise ValueError(
                f"Block sparse tensors{context} require explicit sparse_block_size[0] "
                f"to disambiguate block size for seqlen_q={seqlen_q} and num_m_blocks={num_m_blocks}."
            )
        if sparse_block_size_q is None:
            sparse_block_size_q = ceildiv(seqlen_q, num_m_blocks)

    if sparse_block_size_q % base_m_block != 0:
        raise ValueError(
            f"Block sparse tensors{context} have block size {sparse_block_size_q}, "
            f"which must be a multiple of {base_m_block}."
        )

    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, sparse_block_size_kv)
    q_subtile_factor = sparse_block_size_q // base_m_block
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)

    mask_block_cnt = tensors.mask_block_cnt
    mask_block_idx = tensors.mask_block_idx
    if mask_block_cnt is None or mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")
    if mask_block_cnt.ndim != 3 or mask_block_idx.ndim != 4:
        raise ValueError(
            f"Block sparse tensors{context} must have shapes (B, H, M) and (B, H, M, N)."
        )
    for dim_name, cur, tgt in (
        ("batch", mask_block_cnt.shape[0], expected_count_shape[0]),
        ("head", mask_block_cnt.shape[1], expected_count_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1.")
    for dim_name, cur, tgt in (
        ("batch", mask_block_idx.shape[0], expected_index_shape[0]),
        ("head", mask_block_idx.shape[1], expected_index_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1.")
    if mask_block_cnt.shape[2] != mask_block_idx.shape[2]:
        raise ValueError(f"Block sparse tensors{context} must share the same m-block dimension.")
    # [Note] Allow Compact block sparse indices: FA4 only accesses indices 0..cnt-1
    # per query tile, so idx.shape[3] can be <= expected_n_blocks.
    if mask_block_idx.shape[3] > expected_n_blocks:
        raise ValueError(
            f"Block sparse tensors{context} n-block dimension must be <= {expected_n_blocks}."
        )
    if expected_m_blocks != num_m_blocks:
        raise ValueError(
            f"Block sparse tensors{context} m-block dimension {num_m_blocks} does not match "
            f"sparse_block_size_q={sparse_block_size_q}. "
            f"Set BlockSparseTensorsTorch.block_size to match the BlockMask BLOCK_SIZE."
        )
    return expected_count_shape, expected_index_shape, q_subtile_factor


def infer_linear_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], int]:
    base_m_block = q_stage * m_block_size
    base_n_block = n_block_size
    if sparse_block_size_kv is None:
        sparse_block_size_kv = base_n_block
    if sparse_block_size_kv != base_n_block:
        raise ValueError(
            f"Linear block sparse tensors {context} require BLOCK_SIZE_KV={base_n_block}."
        )

    num_m_blocks = tensors.mask_block_cnt.shape[2]
    if sparse_block_size_q is None:
        sparse_block_size_q = get_sparse_q_block_size(tensors, seqlen_q)
        if sparse_block_size_q is None:
            raise ValueError(
                f"Linear block sparse tensors {context} require explicit block_size[0] "
                f"to disambiguate block size for seqlen_q={seqlen_q} and num_blocks={num_m_blocks}."
            )
    if sparse_block_size_q % base_m_block != 0:
        raise ValueError(
            f"Linear block sparse tensors {context} have block size {sparse_block_size_q}, "
            f"which must be a multiple of {base_m_block}."
        )

    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, sparse_block_size_kv)
    q_subtile_factor = sparse_block_size_q // base_m_block
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)
    if expected_m_blocks != num_m_blocks:
        raise ValueError(
            f"Linear block sparse tensors {context} block dimension {num_m_blocks} does not match "
            f"sparse_block_size_q={sparse_block_size_q}."
        )
    return expected_count_shape, expected_index_shape, q_subtile_factor


def get_block_sparse_expected_shapes_bwd(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for backward block sparse normalization.

    Backward uses Q-direction indexing (transposed from forward), where shapes are
    indexed by N-blocks first, then M-blocks. The sparse_block_size_q is determined
    by subtile_factor * m_block_size.
    """
    sparse_block_size_q = subtile_factor * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_n_blocks)
    expected_index_shape = (batch_size, num_head, expected_n_blocks, expected_m_blocks)
    return expected_count_shape, expected_index_shape


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch,
    *,
    expected_count_shape: Tuple[int, ...],
    expected_index_shape: Tuple[int, ...],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsTorch:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if full_cnt is not None and mask_cnt.device != full_cnt.device:
        raise ValueError("All block sparse tensors must be on the same device")

    dq_write_order = _check_and_expand_metadata_tensor(
        "dq_write_order",
        tensors.dq_write_order,
        tuple(mask_idx.shape),
        context,
        hint,
        mask_cnt.device,
    )
    dq_write_order_full = _check_and_expand_metadata_tensor(
        "dq_write_order_full",
        tensors.dq_write_order_full,
        tuple(full_idx.shape) if full_idx is not None else expected_index_shape,
        context,
        hint,
        mask_cnt.device,
    )
    spt = tensors.spt
    if spt is not None and not isinstance(spt, bool):
        raise ValueError("spt must be a bool when provided")
    if spt is not None and dq_write_order is None:
        raise ValueError("spt requires dq_write_order to be provided")

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        cu_total_m_blocks=tensors.cu_total_m_blocks,
        cu_block_idx_offsets=tensors.cu_block_idx_offsets,
        block_size=tensors.block_size,
        dq_write_order=dq_write_order,
        dq_write_order_full=dq_write_order_full,
        spt=spt,
        mask_block_offset=tensors.mask_block_offset,
        full_block_offset=tensors.full_block_offset,
        mask_block_masks=tensors.mask_block_masks,
        pack_gqa=tensors.pack_gqa,
        bwd_tensors=tensors.bwd_tensors,
    )


def normalize_linear_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
    *,
    expected_count_shape: Tuple[int, ...],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsTorch:
    if (
        tensors.mask_block_cnt is None
        or tensors.mask_block_offset is None
        or tensors.mask_block_idx is None
    ):
        raise ValueError(
            "mask_block_cnt, mask_block_offset, and mask_block_idx must be provided for linear block sparsity."
        )

    mask_cnt, mask_offset, mask_idx = _check_linear_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_offset,
        tensors.mask_block_idx,
        expected_count_shape,
        context,
        hint,
    )
    if mask_cnt is None or mask_offset is None or mask_idx is None:
        raise ValueError("mask block tensors must be provided for linear block sparsity.")

    full_cnt, full_offset, full_idx = _check_linear_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_offset,
        tensors.full_block_idx,
        expected_count_shape,
        context,
        hint,
        expected_base_shape=tuple(mask_cnt.shape),
    )

    dq_write_order = _check_linear_metadata_tensor(
        "dq_write_order",
        tensors.dq_write_order,
        mask_idx,
        context,
        mask_cnt.device,
    )
    dq_write_order_full = _check_linear_metadata_tensor(
        "dq_write_order_full",
        tensors.dq_write_order_full,
        full_idx,
        context,
        mask_cnt.device,
    )
    spt = tensors.spt
    if spt is not None and not isinstance(spt, bool):
        raise ValueError("spt must be a bool when provided")
    if spt is not None and dq_write_order is None:
        raise ValueError("spt requires dq_write_order to be provided")

    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=tensors.block_size,
        dq_write_order=dq_write_order,
        dq_write_order_full=dq_write_order_full,
        spt=spt,
        mask_block_offset=mask_offset,
        full_block_offset=full_offset,
        mask_block_masks=getattr(tensors, "mask_block_masks", None),
        pack_gqa=getattr(tensors, "pack_gqa", None),
        bwd_tensors=getattr(tensors, "bwd_tensors", None),
    )


def normalize_arbitrary_block_sparse_config(
    tensors: BlockSparseTensorsTorch,
    *,
    device: torch.device,
    batch_size: int,
    num_q_heads: int,
    is_varlen: bool,
    block_size: tuple[int, int],
    pack_gqa: bool,
    num_mask_payload_groups: int,
    payload_padded_words: int,
    expected_fixed_total_m_blocks: int | None = None,
) -> BlockSparseTensorsTorch:
    """Validate the compact arbitrary Q2K plan without synchronizing the device."""

    if not isinstance(tensors, BlockSparseTensorsTorch):
        raise TypeError(
            "arbitrary attention requires BlockSparseTensorsTorch returned by "
            "create_arbitrary_block_sparse_tensors"
        )
    if tensors.block_size != block_size:
        raise ValueError(
            f"arbitrary plan block_size={tensors.block_size} does not match "
            f"the resolved consumer block_size={block_size}"
        )
    if tensors.pack_gqa is None or tensors.pack_gqa != pack_gqa:
        raise ValueError(
            f"arbitrary plan pack_gqa={tensors.pack_gqa} does not match "
            f"the resolved consumer pack_gqa={pack_gqa}"
        )

    required = {
        "mask_block_cnt": (tensors.mask_block_cnt, torch.int32, 2),
        "mask_block_offset": (tensors.mask_block_offset, torch.int32, 1),
        "mask_block_idx": (tensors.mask_block_idx, torch.int32, 1),
        "full_block_cnt": (tensors.full_block_cnt, torch.int32, 2),
        "full_block_offset": (tensors.full_block_offset, torch.int32, 1),
        "full_block_idx": (tensors.full_block_idx, torch.int32, 1),
        "mask_block_masks": (tensors.mask_block_masks, torch.uint32, 4),
    }
    for name, (tensor, dtype, ndim) in required.items():
        if tensor is None:
            raise ValueError(f"arbitrary plan requires {name}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}; got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}; got shape {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}; got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    mask_cnt = tensors.mask_block_cnt
    full_cnt = tensors.full_block_cnt
    hmask, total_m_blocks = mask_cnt.shape
    if hmask not in (1, num_q_heads):
        raise ValueError(f"arbitrary plan Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    if tuple(full_cnt.shape) != tuple(mask_cnt.shape):
        raise ValueError("mask_block_cnt and full_block_cnt must have identical shapes")
    num_plan_rows = hmask * total_m_blocks
    if tensors.mask_block_offset.numel() != num_plan_rows + 1:
        raise ValueError("mask_block_offset must have Hmask * total_m_blocks + 1 elements")
    if tensors.full_block_offset.numel() != num_plan_rows + 1:
        raise ValueError("full_block_offset must have Hmask * total_m_blocks + 1 elements")
    if (
        expected_fixed_total_m_blocks is not None
        and total_m_blocks != expected_fixed_total_m_blocks
    ):
        raise ValueError(
            f"fixed arbitrary plan has total_m_blocks={total_m_blocks}; "
            f"expected {expected_fixed_total_m_blocks}"
        )

    expected_payload_shape = (
        tensors.mask_block_idx.numel(),
        1,
        num_mask_payload_groups,
        payload_padded_words,
    )
    if tuple(tensors.mask_block_masks.shape) != expected_payload_shape:
        raise ValueError(
            "mask_block_masks has incompatible consumer layout: expected "
            f"{expected_payload_shape}, got {tuple(tensors.mask_block_masks.shape)}"
        )
    if not is_fake_mode() and tensors.mask_block_masks.data_ptr() % 16 != 0:
        raise ValueError("mask_block_masks must be 16-byte aligned")

    if is_varlen:
        cu_total_m_blocks = tensors.cu_total_m_blocks
        if cu_total_m_blocks is None:
            raise ValueError("varlen arbitrary plan requires cu_total_m_blocks")
        if (
            cu_total_m_blocks.dtype != torch.int32
            or cu_total_m_blocks.device != device
            or cu_total_m_blocks.ndim != 1
            or cu_total_m_blocks.numel() != batch_size + 1
            or not cu_total_m_blocks.is_contiguous()
        ):
            raise ValueError("cu_total_m_blocks must be contiguous int32 CUDA [B + 1]")
    elif tensors.cu_total_m_blocks is not None:
        raise ValueError("fixed arbitrary plan must not provide cu_total_m_blocks")

    for name in (
        "cu_block_idx_offsets",
        "dq_write_order",
        "dq_write_order_full",
        "spt",
    ):
        if getattr(tensors, name) is not None:
            raise ValueError(f"arbitrary forward plan requires {name}=None")
    return tensors


def normalize_arbitrary_block_sparse_config_bwd(
    tensors: BlockSparseTensorsTorch,
    *,
    device: torch.device,
    batch_size: int,
    num_q_heads: int,
    is_varlen: bool,
    block_size: tuple[int, int],
    subtile_factor: int,
    num_mma_threads: int,
    payload_padded_words: int,
    expected_hmask: int | None = None,
    expected_spt: bool = False,
    expected_fixed_total_n_blocks: int | None = None,
) -> BlockSparseTensorsTorch:
    """Validate the compact arbitrary K2Q plan without synchronizing the device."""

    if not isinstance(tensors, BlockSparseTensorsTorch):
        raise TypeError(
            "arbitrary backward requires BlockSparseTensorsTorch returned by "
            "create_arbitrary_block_sparse_tensors(..., build_backward=True)"
        )
    if tensors.block_size != block_size:
        raise ValueError(
            f"arbitrary backward plan block_size={tensors.block_size} does not match "
            f"the resolved consumer block_size={block_size}"
        )
    if tensors.pack_gqa is not None:
        raise ValueError("arbitrary backward plan requires pack_gqa=None")
    if tensors.bwd_tensors is not None:
        raise ValueError("nested arbitrary backward plans are not supported")

    required = {
        "mask_block_cnt": (tensors.mask_block_cnt, torch.int32, 2),
        "mask_block_offset": (tensors.mask_block_offset, torch.int32, 1),
        "mask_block_idx": (tensors.mask_block_idx, torch.int32, 1),
        "full_block_cnt": (tensors.full_block_cnt, torch.int32, 2),
        "full_block_offset": (tensors.full_block_offset, torch.int32, 1),
        "full_block_idx": (tensors.full_block_idx, torch.int32, 1),
        "dq_write_order": (tensors.dq_write_order, torch.int32, 1),
        "dq_write_order_full": (tensors.dq_write_order_full, torch.int32, 1),
        "mask_block_masks": (tensors.mask_block_masks, torch.uint32, 4),
    }
    for name, (tensor, dtype, ndim) in required.items():
        if tensor is None:
            raise ValueError(f"arbitrary backward plan requires {name}")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}; got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}; got shape {tuple(tensor.shape)}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}; got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    mask_cnt = tensors.mask_block_cnt
    full_cnt = tensors.full_block_cnt
    hmask, total_n_blocks = mask_cnt.shape
    if hmask not in (1, num_q_heads):
        raise ValueError(
            f"arbitrary backward plan Hmask must be 1 or Hq ({num_q_heads}); got {hmask}"
        )
    if expected_hmask is not None and hmask != expected_hmask:
        raise ValueError(
            f"arbitrary backward plan Hmask={hmask} does not match forward plan Hmask={expected_hmask}"
        )
    if tuple(full_cnt.shape) != tuple(mask_cnt.shape):
        raise ValueError("mask_block_cnt and full_block_cnt must have identical shapes")
    if (
        expected_fixed_total_n_blocks is not None
        and total_n_blocks != expected_fixed_total_n_blocks
    ):
        raise ValueError(
            f"fixed arbitrary backward plan has total_n_blocks={total_n_blocks}; "
            f"expected {expected_fixed_total_n_blocks}"
        )

    num_plan_rows = hmask * total_n_blocks
    if tensors.mask_block_offset.numel() != num_plan_rows + 1:
        raise ValueError("mask_block_offset must have Hmask * total_n_blocks + 1 elements")
    if tensors.full_block_offset.numel() != num_plan_rows + 1:
        raise ValueError("full_block_offset must have Hmask * total_n_blocks + 1 elements")
    if tensors.dq_write_order.shape != tensors.mask_block_idx.shape:
        raise ValueError("dq_write_order must be parallel to mask_block_idx")
    if tensors.dq_write_order_full.shape != tensors.full_block_idx.shape:
        raise ValueError("dq_write_order_full must be parallel to full_block_idx")

    expected_payload_shape = (
        tensors.mask_block_idx.numel(),
        subtile_factor,
        num_mma_threads,
        payload_padded_words,
    )
    if tuple(tensors.mask_block_masks.shape) != expected_payload_shape:
        raise ValueError(
            "mask_block_masks has incompatible backward consumer layout: expected "
            f"{expected_payload_shape}, got {tuple(tensors.mask_block_masks.shape)}"
        )
    if not is_fake_mode() and tensors.mask_block_masks.data_ptr() % 16 != 0:
        raise ValueError("mask_block_masks must be 16-byte aligned")

    if is_varlen:
        cu_total_m_blocks = tensors.cu_total_m_blocks
        if cu_total_m_blocks is None:
            raise ValueError("varlen arbitrary backward plan requires cu_total_m_blocks")
        if (
            cu_total_m_blocks.dtype != torch.int32
            or cu_total_m_blocks.device != device
            or cu_total_m_blocks.ndim != 1
            or cu_total_m_blocks.numel() != batch_size + 1
            or not cu_total_m_blocks.is_contiguous()
        ):
            raise ValueError("backward cu_total_m_blocks must be contiguous int32 CUDA [B + 1]")
    elif tensors.cu_total_m_blocks is not None:
        raise ValueError("fixed arbitrary backward plan must not provide cu_total_m_blocks")
    if tensors.cu_block_idx_offsets is not None:
        raise ValueError("arbitrary backward plan requires cu_block_idx_offsets=None")
    if tensors.spt is None or tensors.spt != expected_spt:
        raise ValueError(
            f"arbitrary backward plan spt={tensors.spt} does not match consumer spt={expected_spt}"
        )
    return tensors


def is_block_sparsity_enabled(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def get_block_sparse_broadcast_pattern(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
) -> Tuple[Tuple[bool, ...], ...] | None:
    """Return broadcast pattern for block sparse tensors by checking actual strides.

    Returns a tuple of broadcast patterns (one per tensor) where each pattern
    is a tuple of bools indicating which dims have stride=0.
    This is used in compile keys to ensure kernels are recompiled when
    broadcast patterns change, since CuTe's mark_layout_dynamic() keeps
    stride=0 as static.

    The tensors should already be expanded/normalized before calling this function.

    Returns None if block sparsity is not enabled.
    """
    if not is_block_sparsity_enabled(tensors):
        return None

    patterns = []
    for tensor in (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
        tensors.dq_write_order,
        tensors.dq_write_order_full,
        tensors.mask_block_offset,
        tensors.full_block_offset,
        getattr(tensors, "mask_block_masks", None),
    ):
        if tensor is not None:
            if is_linear_block_sparse_tensors(tensors) and tensor.ndim == 3:
                patterns.append((tensor.shape[0] == 1, tensor.shape[1] == 1, False))
            else:
                patterns.append(get_broadcast_dims(tensor))
        else:
            patterns.append(None)
    return tuple(patterns)


def normalize_block_sparse_config(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    q_stage: int,
) -> tuple[BlockSparseTensorsTorch, Tuple[Tuple[bool, ...], ...] | None, int]:
    """Validate the block-sparse config, infer expected shapes, and normalize.

    Handles both fixed-length (3D `[B, H, M]` / 4D `[B, H, M, N]`) and varlen
    (2D `[H, total_m_blocks]` / `[H, total_n_blocks]`) layouts. Varlen is
    detected by `tensors.cu_total_m_blocks is not None` and forces
    `q_subtile_factor == 1` (TODO: potentially remove this restriction).
    """
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = None, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity requires sparse_block_size[1]={n_block_size} to match tile_n."
        )
    if is_linear_block_sparse_tensors(tensors):
        expected_count_shape, _, q_subtile_factor = infer_linear_block_sparse_expected_shapes(
            tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            context="forward",
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_kv=sparse_block_size_kv,
        )
        normalized_tensors = normalize_linear_block_sparse_tensors(
            tensors,
            expected_count_shape=expected_count_shape,
            context="_flash_attn_fwd",
            hint=lambda: (
                "Forward expects Q2K CSR tensors (mask/full KV lists per Q block) "
                f"with BLOCK_SIZE=({sparse_block_size_q or q_stage * m_block_size}, {n_block_size})."
            ),
        )
        return (
            normalized_tensors,
            get_block_sparse_broadcast_pattern(normalized_tensors),
            q_subtile_factor,
        )
    if tensors.cu_total_m_blocks is not None:
        base_m_block = q_stage * m_block_size
        if sparse_block_size_q is not None and sparse_block_size_q != base_m_block:
            raise ValueError(
                f"Varlen block sparsity requires sparse_block_size[0]={base_m_block} "
                f"(= q_stage * tile_m); got {sparse_block_size_q}."
            )
        total_m_blocks = tensors.mask_block_cnt.shape[-1]
        total_n_blocks = tensors.mask_block_idx.shape[-1]
        expected_count_shape = (num_head, total_m_blocks)
        expected_index_shape = (num_head, total_n_blocks)
        q_subtile_factor = 1
    else:
        expected_count_shape, expected_index_shape, q_subtile_factor = (
            infer_block_sparse_expected_shapes(
                tensors,
                batch_size=batch_size,
                num_head=num_head,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                q_stage=q_stage,
                context="forward",
                sparse_block_size_q=sparse_block_size_q,
                sparse_block_size_kv=sparse_block_size_kv,
            )
        )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
    )
    return (
        normalized_tensors,
        get_block_sparse_broadcast_pattern(normalized_tensors),
        q_subtile_factor,
    )


def normalize_block_sparse_config_bwd(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    subtile_factor: int,
) -> tuple[BlockSparseTensorsTorch, Tuple[Tuple[bool, ...], ...] | None]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = subtile_factor * m_block_size, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_q != subtile_factor * m_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size_q={subtile_factor * m_block_size} "
            f"for subtile_factor={subtile_factor}."
        )
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
        batch_size,
        num_head,
        seqlen_q,
        seqlen_k,
        m_block_size,
        n_block_size,
        subtile_factor,
    )
    if is_linear_block_sparse_tensors(tensors):
        normalized_tensors = normalize_linear_block_sparse_tensors(
            tensors,
            expected_count_shape=expected_count_shape,
            context="_flash_attn_bwd",
            hint=lambda: (
                f"Backward expects K2Q CSR tensors (mask/full Q lists per KV block). "
                f"Regenerate with BLOCK_SIZE=({subtile_factor * m_block_size}, {n_block_size})."
            ),
        )
        return normalized_tensors, get_block_sparse_broadcast_pattern(normalized_tensors)
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
        context="_flash_attn_bwd",
        hint=lambda: (
            f"Backward expects Q-direction block-sparse tensors (q_mask_cnt/q_mask_idx, "
            f"and optionally full_q_cnt/full_q_idx). Regenerate the backward BlockMask with "
            f"BLOCK_SIZE=({subtile_factor * m_block_size}, {n_block_size})."
        ),
    )
    return normalized_tensors, get_block_sparse_broadcast_pattern(normalized_tensors)


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsTorch | LinearBlockSparseTensorsTorch,
    enable_tvm_ffi: bool = True,
) -> BlockSparseTensors | None:
    """Convert torch block sparsity tensors to CuTe tensors, optionally for tvm ffi"""
    if not is_block_sparsity_enabled(tensors):
        return None
    mask_block_cnt_tensor, mask_block_idx_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        for t in (tensors.mask_block_cnt, tensors.mask_block_idx)
    ]
    full_block_cnt_tensor, full_block_idx_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.full_block_cnt, tensors.full_block_idx)
    ]
    cu_total_m_blocks_tensor, cu_block_idx_offsets_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.cu_total_m_blocks, tensors.cu_block_idx_offsets)
    ]
    dq_write_order_tensor, dq_write_order_full_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.dq_write_order, tensors.dq_write_order_full)
    ]
    mask_block_offset_tensor, full_block_offset_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.mask_block_offset, tensors.full_block_offset)
    ]
    mask_block_masks_tensor = (
        to_cute_tensor(
            tensors.mask_block_masks,
            assumed_align=16,
            leading_dim=-1,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        if getattr(tensors, "mask_block_masks", None) is not None
        else None
    )

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
        cu_total_m_blocks_tensor,
        cu_block_idx_offsets_tensor,
        dq_write_order_tensor,
        dq_write_order_full_tensor,
        mask_block_offset_tensor,
        full_block_offset_tensor,
        mask_block_masks_tensor,
    )


def to_cute_linear_block_sparse_tensors(
    tensors: LinearBlockSparseTensorsTorch,
    enable_tvm_ffi: bool = True,
) -> BlockSparseTensors | None:
    return to_cute_block_sparse_tensors(tensors, enable_tvm_ffi=enable_tvm_ffi)


def fast_sampling(mask_mod):
    """Convenience decorator to mark mask_mod as safe for 5-point fast sampling"""
    mask_mod.use_fast_sampling = True
    return mask_mod
