# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union, List, Tuple, NamedTuple

import os
import torch
import torch.nn as nn
import warnings
import weakref


# ============================================================================
# Block Sparsity Data Structures
# ============================================================================

class LinearBlockSparseTensors(NamedTuple):
    """
    Block sparsity tensors in CSR (Compressed Sparse Row) format.

    For each m_block (query block), we have lists of n_blocks (key blocks) to process:
    - mask_block: blocks that require element-level masking (partial blocks)
    - full_block: blocks that don't require masking (full blocks)

    Data layout:
    - cnt: [B, H, num_m_blocks] 3D counts (B, H can be 1 for broadcasting)
    - offset: [B * H * num_m_blocks + 1] CSR-style exclusive prefix sum (starts with 0) (B, H can be 1 for broadcasting)
    - idx: [total_blocks] compact n_block indices (csr format)

    This structure is compatible with PyTorch's create_block_mask output format.
    """
    mask_block_cnt: torch.Tensor       # [B, H, num_m_blocks]: count of mask blocks per m_block, supports broadcasting (B, H can be 1)
    mask_block_offset: torch.Tensor    # [B*H*num_m_blocks+1]: cumulative offset into mask_idx, supports broadcasting (B, H can be 1)
    mask_block_idx: torch.Tensor       # [total_mask_blocks]: indices of mask blocks (csr format)
    full_block_cnt: Optional[torch.Tensor] = None       # [B, H, num_m_blocks]: count of full blocks, supports broadcasting (B, H can be 1)
    full_block_offset: Optional[torch.Tensor] = None    # [B*H*num_m_blocks+1]: cumulative offset into full_idx, supports broadcasting (B, H can be 1)
    full_block_idx: Optional[torch.Tensor] = None       # [total_full_blocks]: indices of full blocks (csr format)
    # Optional deterministic K2Q/backward metadata.  These fields are appended
    # to preserve the first six positional fields used by existing callers.
    dq_write_order: Optional[torch.Tensor] = None
    dq_write_order_full: Optional[torch.Tensor] = None

USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if not USE_TRITON_ROCM and getattr(torch.version, 'hip', None) is not None:
    try:
        import flash_attn_3._C
    except ImportError:
        warnings.warn("flash_attn_3._C (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
        USE_TRITON_ROCM = True

if USE_TRITON_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3 as flash_attn_3_gpu
else:
    # isort: off
    # We need to import the CUDA kernels after importing torch
    import flash_attn_3._C # Registers operators with PyTorch

    # isort: on

    flash_attn_3_gpu = torch.ops.flash_attn_3

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


def _ceildiv(x, y):
    return (x + y - 1) // y


def _device_capability_major(device):
    capability = torch.cuda.get_device_capability(device)
    return capability[0], capability[1]


def _dense_fwd_block_size_sm8x(
    head_dim,
    head_dim_v,
    element_size=2,
    sm86_or_89=False,
    paged_kv_non_tma=False,
    varlen_and_split=False,
    append_kv=False,
):
    # Keep this in sync with tile_size_fwd_sm8x(..., is_arbitrary=true).
    if element_size != 2:
        return 128, 64
    if head_dim <= 64:
        return 128, 80 if varlen_and_split else 96
    if head_dim <= 96:
        return 128, 48
    if head_dim <= 128:
        use_8_warps = sm86_or_89 or varlen_and_split
        if use_8_warps:
            return 128, 96 if varlen_and_split or sm86_or_89 else 48
        return 128, 48
    if head_dim <= 192:
        return 128, 64
    if sm86_or_89:
        return 128, 32 if append_kv else 48
    return 128, 48 if append_kv else 64


def _dense_fwd_block_size_sm90(
    head_dim,
    head_dim_v,
    element_size=2,
    softcap=False,
    paged_kv_non_tma=False,
):
    # Keep this in sync with tile_size_fwd_sm90(..., is_arbitrary=true).
    if element_size == 2:
        if head_dim <= 64:
            if head_dim_v == 512:
                return 64, 64
            if head_dim_v == 256:
                return 128, 96
            return 192, 128
        if head_dim <= 96:
            return 192, 128
        if head_dim <= 128:
            return 128, 128
        if head_dim <= 192:
            return 128, 96
        return 128, 64
    if head_dim <= 64:
        return 192, 160
    if head_dim <= 96:
        return 192, 128
    if head_dim <= 128:
        return 128, 160 if paged_kv_non_tma else (192 if softcap else 224)
    if head_dim <= 192:
        return 128, 128 if (paged_kv_non_tma or softcap) else 160
    return 128, 64


def _dense_fwd_block_size(
    q,
    v,
    softcap=False,
    paged_kv_non_tma=False,
    varlen_and_split=False,
    append_kv=False,
):
    major, minor = _device_capability_major(q.device)
    if major == 8:
        return _dense_fwd_block_size_sm8x(
            q.shape[-1],
            v.shape[-1],
            element_size=q.element_size(),
            sm86_or_89=minor in (6, 9),
            paged_kv_non_tma=paged_kv_non_tma,
            varlen_and_split=varlen_and_split,
            append_kv=append_kv,
        )
    return _dense_fwd_block_size_sm90(
        q.shape[-1],
        v.shape[-1],
        element_size=q.element_size(),
        softcap=softcap,
        paged_kv_non_tma=paged_kv_non_tma,
    )


def _dense_bwd_block_size_sm90(head_dim):
    if head_dim <= 64:
        return 128, 128
    if head_dim <= 96:
        return 64, 128
    if head_dim <= 128:
        return 64, 128
    if head_dim <= 192:
        return 64, 96
    return 64, 80


def _dense_bwd_block_size_sm8x(head_dim, sm86_or_89=False):
    # Keep this in sync with tile_size_bwd_sm8x(..., is_arbitrary=true).
    if sm86_or_89:
        if head_dim <= 64:
            return 64, 128
        if head_dim <= 96:
            return 64, 128
        if head_dim <= 128:
            return 64, 96
        if head_dim <= 192:
            return 64, 64
        return 32, 64
    if head_dim <= 64:
        return 128, 128
    if head_dim <= 96:
        return 64, 128
    if head_dim <= 128:
        return 64, 128
    if head_dim <= 192:
        return 64, 80
    return 64, 64


def _dense_bwd_block_size(q, v=None):
    major, minor = _device_capability_major(q.device)
    head_dim = round_up_headdim(
        max(q.shape[-1], v.shape[-1] if v is not None else q.shape[-1])
    )
    if major == 8:
        return _dense_bwd_block_size_sm8x(head_dim, sm86_or_89=minor in (6, 9))
    return _dense_bwd_block_size_sm90(head_dim)


def _make_dense_linear_block_sparse(
    num_outer_blocks,
    num_inner_blocks,
    device,
    *,
    deterministic_k2q=False,
):
    mask_block_cnt = torch.full(
        (1, 1, num_outer_blocks), num_inner_blocks, dtype=torch.int32, device=device
    )
    mask_block_offset = torch.arange(
        num_outer_blocks + 1, dtype=torch.int32, device=device
    ) * num_inner_blocks
    mask_block_idx = torch.arange(
        num_inner_blocks, dtype=torch.int32, device=device
    ).repeat(num_outer_blocks)
    full_block_cnt = torch.zeros_like(mask_block_cnt)
    full_block_offset = torch.zeros(num_outer_blocks + 1, dtype=torch.int32, device=device)
    full_block_idx = torch.empty(0, dtype=torch.int32, device=device)
    dq_write_order = None
    dq_write_order_full = None
    if deterministic_k2q:
        # K2Q rows are n_blocks.  The C++ SPT scheduler visits n in
        # descending order, so every edge in row n has rank N - 1 - n.
        dq_write_order = torch.arange(
            num_outer_blocks - 1, -1, -1, dtype=torch.int32, device=device
        ).repeat_interleave(num_inner_blocks)
        dq_write_order_full = torch.empty(0, dtype=torch.int32, device=device)
    tensors = LinearBlockSparseTensors(
        mask_block_cnt,
        mask_block_offset,
        mask_block_idx,
        full_block_cnt,
        full_block_offset,
        full_block_idx,
        dq_write_order,
        dq_write_order_full,
    )
    if deterministic_k2q:
        # Construction proves the generic CSR/rank invariants, so avoid a
        # redundant device-to-host semantic scan on the first launch.
        _register_trusted_deterministic_k2q_semantics(
            tensors, num_inner_blocks - 1 if num_inner_blocks > 0 else -1
        )
    return tensors


def _as_linear_block_sparse_tensors(tensors, name="block_sparse"):
    if tensors is None:
        return None
    if isinstance(tensors, LinearBlockSparseTensors):
        return tensors
    if hasattr(tensors, "mask_block_cnt"):
        return LinearBlockSparseTensors(
            tensors.mask_block_cnt,
            tensors.mask_block_offset,
            tensors.mask_block_idx,
            tensors.full_block_cnt,
            tensors.full_block_offset,
            tensors.full_block_idx,
            getattr(tensors, "dq_write_order", None),
            getattr(tensors, "dq_write_order_full", None),
        )
    if not isinstance(tensors, (tuple, list)) or len(tensors) not in (6, 8):
        raise ValueError(
            f"{name} must be LinearBlockSparseTensors or a 6/8-item tuple"
        )
    return LinearBlockSparseTensors(*tensors)


def _validate_linear_csr_structure(tensors, name="k2q_block_sparse"):
    tensors = _as_linear_block_sparse_tensors(tensors, name)
    assert tensors is not None
    base = (
        tensors.mask_block_cnt,
        tensors.mask_block_offset,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_offset,
        tensors.full_block_idx,
    )
    if any(x is None for x in base):
        raise ValueError(
            f"{name} requires all six mask/full CSR tensors; partial metadata is not supported"
        )
    for field, tensor, ndim in (
        ("mask_block_cnt", tensors.mask_block_cnt, 3),
        ("mask_block_offset", tensors.mask_block_offset, 1),
        ("mask_block_idx", tensors.mask_block_idx, 1),
        ("full_block_cnt", tensors.full_block_cnt, 3),
        ("full_block_offset", tensors.full_block_offset, 1),
        ("full_block_idx", tensors.full_block_idx, 1),
    ):
        assert tensor is not None
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name}.{field} must have dtype torch.int32")
        if tensor.ndim != ndim:
            raise ValueError(f"{name}.{field} must be {ndim}D")
        if not tensor.is_contiguous():
            raise ValueError(f"{name}.{field} must be contiguous")
    assert tensors.full_block_cnt is not None
    assert tensors.mask_block_offset is not None
    assert tensors.full_block_offset is not None
    if tensors.full_block_cnt.shape != tensors.mask_block_cnt.shape:
        raise ValueError(f"{name} mask/full count tensors must have the same shape")
    num_rows = tensors.mask_block_cnt.numel()
    if tensors.mask_block_offset.numel() != num_rows + 1:
        raise ValueError(f"{name}.mask_block_offset must have count.numel() + 1 entries")
    if tensors.full_block_offset.numel() != num_rows + 1:
        raise ValueError(f"{name}.full_block_offset must have count.numel() + 1 entries")
    return tensors


def compute_dq_write_order_from_linear_csr(tensors):
    """Compute compact deterministic dQ write ranks for K2Q linear CSR.

    Partial and full contributors are ranked together for each
    ``(metadata_batch, metadata_head, m_block)``.  The highest contributing
    n_block receives rank 0, matching the C++ SM8x/SM90 deterministic scheduler.
    """
    tensors = _validate_linear_csr_structure(tensors)
    mask_cnt = tensors.mask_block_cnt
    mask_offset = tensors.mask_block_offset
    mask_idx = tensors.mask_block_idx
    full_cnt = tensors.full_block_cnt
    full_offset = tensors.full_block_offset
    full_idx = tensors.full_block_idx
    assert mask_offset is not None and full_cnt is not None
    assert full_offset is not None and full_idx is not None

    device = mask_idx.device
    num_n_blocks = mask_cnt.shape[2]

    def _entry_metadata(offset, idx):
        positions = torch.arange(idx.numel(), device=device, dtype=torch.int64)
        row = torch.searchsorted(offset.to(torch.int64), positions, right=True) - 1
        bh = row // num_n_blocks
        n_block = row - bh * num_n_blocks
        return bh, n_block, idx.to(torch.int64)

    mask_bh, mask_n, mask_m = _entry_metadata(mask_offset, mask_idx)
    full_bh, full_n, full_m = _entry_metadata(full_offset, full_idx)
    total = mask_idx.numel() + full_idx.numel()
    if total == 0:
        return torch.zeros_like(mask_idx), torch.zeros_like(full_idx)

    flat_bh = torch.cat((mask_bh, full_bh))
    flat_n = torch.cat((mask_n, full_n))
    flat_m = torch.cat((mask_m, full_m))
    if bool((flat_m < 0).any()):
        raise ValueError("k2q_block_sparse m_block indices must be non-negative")
    max_m = torch.max(flat_m)
    group_key = flat_bh * (max_m + 1) + flat_m
    n_order = num_n_blocks - 1 - flat_n
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
    return flat_rank[: mask_idx.numel()], flat_rank[mask_idx.numel() :]


def prepare_deterministic_k2q_metadata(tensors):
    """Return validated K2Q CSR metadata for SM8x/SM90 deterministic backward.

    The CSR is assumed to have been built with the backend's queried backward
    tile size; the tile is derived again at launch and is not stored here.
    The returned CSR and rank tensors must be treated as immutable.  Normal
    in-place PyTorch mutations invalidate the cached validation certificate,
    but writes through ``.data`` or unregistered raw CUDA kernels cannot be
    detected and may make the metadata unsafe for a deterministic launch.
    """
    tensors = _validate_linear_csr_structure(tensors)
    dq_write_order, dq_write_order_full = compute_dq_write_order_from_linear_csr(tensors)
    prepared = tensors._replace(
        dq_write_order=dq_write_order,
        dq_write_order_full=dq_write_order_full,
    )
    _certify_deterministic_k2q_semantics(prepared)
    return prepared


_validated_deterministic_k2q = {}


def _tensor_validation_token(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors intentionally do not expose a version counter, so
        # they cannot participate in the one-time validation cache safely.
        return None
    return (
        tensor.data_ptr(),
        version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device,
    )


def _validation_cache_key(tensors):
    return tuple(id(tensor) for tensor in tensors)


def _validation_cache_hit(cache_key, tensors):
    cached = _validated_deterministic_k2q.get(cache_key)
    if cached is None:
        return None
    refs, tokens, max_m_block = cached
    current_tokens = tuple(_tensor_validation_token(tensor) for tensor in tensors)
    valid = (
        all(token is not None for token in current_tokens)
        and all(ref() is tensor for ref, tensor in zip(refs, tensors))
        and tokens == current_tokens
    )
    return max_m_block if valid else None


def _cache_validated_metadata(cache_key, tensors, max_m_block):
    tokens = tuple(_tensor_validation_token(tensor) for tensor in tensors)
    if any(token is None for token in tokens):
        return
    if len(_validated_deterministic_k2q) >= 256:
        # Keep this bounded without retaining Tensor objects.  Weak references
        # make allocator pointer/id reuse fail closed instead of accidentally
        # treating a new metadata object as already validated.
        _validated_deterministic_k2q.pop(next(iter(_validated_deterministic_k2q)))
    _validated_deterministic_k2q[cache_key] = (
        tuple(weakref.ref(tensor) for tensor in tensors),
        tokens,
        max_m_block,
    )


def _deterministic_k2q_certificate_tensors(tensors):
    tensors = _validate_linear_csr_structure(tensors)
    if tensors.dq_write_order is None:
        raise ValueError("deterministic K2Q metadata requires dq_write_order")
    assert tensors.mask_block_idx is not None
    assert tensors.full_block_idx is not None
    if tensors.full_block_idx.numel() != 0 and tensors.dq_write_order_full is None:
        raise ValueError(
            "deterministic K2Q metadata requires dq_write_order_full "
            "for non-empty full CSR"
        )
    for name, rank, idx in (
        ("dq_write_order", tensors.dq_write_order, tensors.mask_block_idx),
        ("dq_write_order_full", tensors.dq_write_order_full, tensors.full_block_idx),
    ):
        if rank is None:
            continue
        if rank.dtype != torch.int32 or rank.ndim != 1 or not rank.is_contiguous():
            raise ValueError(f"k2q_block_sparse.{name} must be contiguous 1D torch.int32")
        if rank.numel() != idx.numel():
            raise ValueError(f"k2q_block_sparse.{name} must be parallel to its compact idx")

    required_tensors = (
        tensors.mask_block_cnt,
        tensors.mask_block_offset,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_offset,
        tensors.full_block_idx,
        tensors.dq_write_order,
    )
    all_tensors = required_tensors + (
        () if tensors.dq_write_order_full is None else (tensors.dq_write_order_full,)
    )
    metadata_device = all_tensors[0].device
    if any(tensor.device != metadata_device for tensor in all_tensors):
        raise ValueError("all deterministic K2Q metadata tensors must be on one device")
    return tensors, all_tensors


def _certify_deterministic_k2q_semantics(tensors):
    """Validate launch-independent CSR/rank invariants and cache max m_block."""
    tensors, all_tensors = _deterministic_k2q_certificate_tensors(tensors)
    cache_key = _validation_cache_key(all_tensors)
    max_m_block = _validation_cache_hit(cache_key, all_tensors)
    if max_m_block is not None:
        return max_m_block

    mask_cnt = tensors.mask_block_cnt.detach().cpu().reshape(-1).tolist()
    mask_offset = tensors.mask_block_offset.detach().cpu().tolist()
    mask_idx = tensors.mask_block_idx.detach().cpu().tolist()
    full_cnt = tensors.full_block_cnt.detach().cpu().reshape(-1).tolist()
    full_offset = tensors.full_block_offset.detach().cpu().tolist()
    full_idx = tensors.full_block_idx.detach().cpu().tolist()
    mask_rank = tensors.dq_write_order.detach().cpu().tolist()
    full_rank = (
        []
        if tensors.dq_write_order_full is None
        else tensors.dq_write_order_full.detach().cpu().tolist()
    )

    def _validate_csr(kind, counts, offsets, indices):
        if not offsets or offsets[0] != 0:
            raise ValueError(f"K2Q {kind} CSR offset must start at 0")
        if offsets[-1] != len(indices):
            raise ValueError(f"K2Q {kind} CSR final offset must equal idx.numel()")
        for row, count in enumerate(counts):
            if count < 0 or offsets[row + 1] < offsets[row]:
                raise ValueError(f"K2Q {kind} CSR counts/offsets must be non-negative")
            if offsets[row + 1] - offsets[row] != count:
                raise ValueError(f"K2Q {kind} CSR offset delta must equal count")

    _validate_csr("partial", mask_cnt, mask_offset, mask_idx)
    _validate_csr("full", full_cnt, full_offset, full_idx)

    num_n_blocks = tensors.mask_block_cnt.shape[2]
    max_m_block = -1
    contributors = {}
    edges_per_row = [set() for _ in range(len(mask_cnt))]
    for kind, offsets, indices, ranks in (
        ("partial", mask_offset, mask_idx, mask_rank),
        ("full", full_offset, full_idx, full_rank),
    ):
        for row in range(len(mask_cnt)):
            n_block = row % num_n_blocks
            bh = row // num_n_blocks
            for pos in range(offsets[row], offsets[row + 1]):
                m_block = indices[pos]
                if m_block < 0:
                    raise ValueError(f"K2Q {kind} CSR contains a negative m_block")
                max_m_block = max(max_m_block, m_block)
                if m_block in edges_per_row[row]:
                    raise ValueError(
                        "K2Q CSR contains a duplicate edge or the same edge in partial/full"
                    )
                edges_per_row[row].add(m_block)
                contributors.setdefault((bh, m_block), []).append(
                    (n_block, ranks[pos])
                )
    for values in contributors.values():
        n_blocks = [n for n, _ in values]
        if len(n_blocks) != len(set(n_blocks)):
            raise ValueError("K2Q CSR contains duplicate contributors")
        expected = {n: rank for rank, n in enumerate(sorted(n_blocks, reverse=True))}
        if any(rank != expected[n] for n, rank in values):
            raise ValueError(
                "K2Q dq_write_order must be the contiguous rank of descending n_block "
                "across partial and full contributors"
            )

    _cache_validated_metadata(cache_key, all_tensors, max_m_block)
    return max_m_block


def _register_trusted_deterministic_k2q_semantics(tensors, max_m_block):
    """Register metadata whose construction proves the generic invariants."""
    _, all_tensors = _deterministic_k2q_certificate_tensors(tensors)
    if torch.compiler.is_compiling():
        return
    from torch._subclasses.fake_tensor import is_fake
    if any(tensor.device.type == "meta" or is_fake(tensor) for tensor in all_tensors):
        return
    _cache_validated_metadata(
        _validation_cache_key(all_tensors), all_tensors, max_m_block
    )


def _validate_deterministic_k2q_metadata(
    tensors,
    q,
    k,
    v,
    seqlen_q,
    seqlen_k,
):
    tensors, all_tensors = _deterministic_k2q_certificate_tensors(tensors)
    for tensor in all_tensors:
        if tensor.device != q.device:
            raise ValueError("deterministic K2Q metadata must be on the same CUDA device as q")

    batch_size, num_heads = q.shape[0], q.shape[-2]
    metadata_batch, metadata_heads, num_n_blocks = tensors.mask_block_cnt.shape
    if metadata_batch not in (1, batch_size):
        raise ValueError("K2Q metadata batch dimension must be 1 or match q batch")
    if metadata_heads not in (1, num_heads):
        raise ValueError("K2Q metadata head dimension must be 1 or match q heads")

    # Fake tensors have no values to inspect.  Real metadata is semantically
    # validated once per tensor version and then cached for repeated launches.
    if torch.compiler.is_compiling():
        # The opaque backward custom op repeats this validation at runtime.
        # Avoid data_ptr/value inspection while Dynamo/AOT is tracing the
        # public wrapper around it.
        return tensors
    from torch._subclasses.fake_tensor import is_fake
    if (
        q.device.type == "meta"
        or is_fake(q)
        or any(t.device.type == "meta" or is_fake(t) for t in all_tensors)
    ):
        return tensors
    max_m_block = _certify_deterministic_k2q_semantics(tensors)
    if q.device.type != "cuda":
        return tensors
    block_m, block_n = _dense_bwd_block_size(q, v)
    expected_n_blocks = _ceildiv(seqlen_k, block_n)
    if num_n_blocks != expected_n_blocks:
        raise ValueError(
            f"K2Q metadata has {num_n_blocks} n-block rows, expected {expected_n_blocks}"
        )
    num_m_blocks = _ceildiv(seqlen_q, block_m)
    if max_m_block >= num_m_blocks:
        raise ValueError("K2Q CSR contains out-of-range m_block")
    return tensors


def _warn_arbitrary_dense_fallback(missing_names):
    if torch.compiler.is_compiling():
        return
    missing = " and ".join(missing_names)
    warnings.warn(
        f"arbitrary_func was provided without {missing}; FlashAttention will generate dense "
        "block-sparse fallback metadata. This preserves correctness but can be much slower "
        "because the kernel may visit blocks that the arbitrary mask later rejects. For better "
        "performance, generate and pass block sparsity metadata from the same mask.",
        RuntimeWarning,
        stacklevel=3,
    )


def _validate_arbitrary_mask_mode(
    arbitrary_func,
    causal,
    window_size_left,
    window_size_right,
    attention_chunk=0,
):
    if arbitrary_func is not None and (
        causal
        or window_size_left >= 0
        or window_size_right >= 0
        or attention_chunk >= 1
    ):
        raise ValueError(
            "arbitrary mask cannot be combined with native causal/local flags; "
            "encode the constraint in arbitrary_func/CSR"
        )


def _prepare_arbitrary_block_sparse(
    q,
    k,
    v,
    arbitrary_func,
    q2k_block_sparse,
    k2q_block_sparse,
    seqlen_q,
    seqlen_k,
    softcap=0.0,
    paged_kv_non_tma=False,
    varlen_and_split=False,
    append_kv=False,
    prepare_k2q=True,
    deterministic=False,
    is_varlen=False,
):
    if arbitrary_func is None:
        return q2k_block_sparse, k2q_block_sparse
    needs_backward_sparse = any(t is not None and t.requires_grad for t in (q, k, v))
    deterministic_backward = deterministic and needs_backward_sparse
    if deterministic_backward:
        if not torch.compiler.is_compiling():
            major, minor = _device_capability_major(q.device)
            if (major, minor) not in ((8, 0), (8, 6), (8, 9), (9, 0)):
                raise NotImplementedError(
                    "C++ arbitrary deterministic backward is currently supported only on "
                    "SM80, SM86, SM89, and SM90"
                )
        if is_varlen or q.ndim != 4 or k.ndim != 4:
            raise NotImplementedError(
                "C++ arbitrary deterministic backward currently supports fixed-length tensors only"
            )
        if round_up_headdim(max(q.shape[-1], v.shape[-1])) > 192:
            raise ValueError(
                "C++ arbitrary deterministic backward does not support the rounded hdim-256 bucket"
            )
    missing_sparse = []
    if q2k_block_sparse is None:
        missing_sparse.append("q2k_block_sparse")
    if prepare_k2q and k2q_block_sparse is None and needs_backward_sparse:
        missing_sparse.append("k2q_block_sparse")
    if missing_sparse:
        _warn_arbitrary_dense_fallback(missing_sparse)
    if q2k_block_sparse is None:
        block_m, block_n = _dense_fwd_block_size(
            q,
            v,
            softcap=softcap > 0.0,
            paged_kv_non_tma=paged_kv_non_tma,
            varlen_and_split=varlen_and_split,
            append_kv=append_kv,
        )
        q2k_block_sparse = _make_dense_linear_block_sparse(
            _ceildiv(seqlen_q, block_m),
            _ceildiv(seqlen_k, block_n),
            q.device,
        )
    else:
        q2k_block_sparse = _as_linear_block_sparse_tensors(
            q2k_block_sparse, "q2k_block_sparse"
        )
    if prepare_k2q and k2q_block_sparse is None and needs_backward_sparse:
        block_m, block_n = _dense_bwd_block_size(q, v)
        k2q_block_sparse = _make_dense_linear_block_sparse(
            _ceildiv(seqlen_k, block_n),
            _ceildiv(seqlen_q, block_m),
            q.device,
            deterministic_k2q=deterministic_backward,
        )
    elif prepare_k2q and k2q_block_sparse is not None:
        k2q_block_sparse = _as_linear_block_sparse_tensors(
            k2q_block_sparse, "k2q_block_sparse"
        )
    if deterministic_backward:
        if k2q_block_sparse is None:
            raise ValueError("arbitrary deterministic backward requires K2Q CSR metadata")
        k2q_block_sparse = _validate_deterministic_k2q_metadata(
            k2q_block_sparse,
            q,
            k,
            v,
            seqlen_q,
            seqlen_k,
        )
    return q2k_block_sparse, k2q_block_sparse


def _save_k2q_metadata(ctx, tensors):
    tensors = _as_linear_block_sparse_tensors(tensors, "k2q_block_sparse")
    if tensors is None:
        return
    ctx.k2q_mask_cnt = tensors.mask_block_cnt
    ctx.k2q_mask_offset = tensors.mask_block_offset
    ctx.k2q_mask_idx = tensors.mask_block_idx
    ctx.k2q_full_cnt = tensors.full_block_cnt
    ctx.k2q_full_offset = tensors.full_block_offset
    ctx.k2q_full_idx = tensors.full_block_idx
    ctx.k2q_dq_write_order = tensors.dq_write_order
    ctx.k2q_dq_write_order_full = tensors.dq_write_order_full


def _k2q_metadata_args_from_ctx(ctx):
    return (
        getattr(ctx, "k2q_mask_cnt", None),
        getattr(ctx, "k2q_mask_offset", None),
        getattr(ctx, "k2q_mask_idx", None),
        getattr(ctx, "k2q_full_cnt", None),
        getattr(ctx, "k2q_full_offset", None),
        getattr(ctx, "k2q_full_idx", None),
        getattr(ctx, "k2q_dq_write_order", None),
        getattr(ctx, "k2q_dq_write_order_full", None),
    )


def _k2q_metadata_kwargs(tensors):
    tensors = _as_linear_block_sparse_tensors(tensors, "k2q_block_sparse")
    if tensors is None:
        return {}
    return {
        "k2q_block_sparse_mask_cnt": tensors.mask_block_cnt,
        "k2q_block_sparse_mask_offset": tensors.mask_block_offset,
        "k2q_block_sparse_mask_idx": tensors.mask_block_idx,
        "k2q_block_sparse_full_cnt": tensors.full_block_cnt,
        "k2q_block_sparse_full_offset": tensors.full_block_offset,
        "k2q_block_sparse_full_idx": tensors.full_block_idx,
        "k2q_block_sparse_dq_write_order": tensors.dq_write_order,
        "k2q_block_sparse_dq_write_order_full": tensors.dq_write_order_full,
    }


def round_up_headdim(head_size: int) -> int:
    from flash_attn_config import CONFIG

    if not CONFIG["build_flags"]["FLASHATTENTION_DISABLE_HDIM64"]:
        if head_size <= 64:
            return 64
    if not CONFIG["build_flags"]["FLASHATTENTION_DISABLE_HDIM96"]:
        if head_size <= 96:
            return 96
    if not CONFIG["build_flags"]["FLASHATTENTION_DISABLE_HDIM128"]:
        if head_size <= 128:
            return 128
    if not CONFIG["build_flags"]["FLASHATTENTION_DISABLE_HDIM192"]:
        if head_size <= 192:
            return 192
    if not CONFIG["build_flags"]["FLASHATTENTION_DISABLE_HDIM256"]:
        if head_size <= 256:
            return 256
    return 256


@torch.library.custom_op("flash_attn_3::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    # Block sparsity parameters (Q2K direction)
    block_sparse_mask_cnt: Optional[torch.Tensor] = None,
    block_sparse_mask_offset: Optional[torch.Tensor] = None,
    block_sparse_mask_idx: Optional[torch.Tensor] = None,
    block_sparse_full_cnt: Optional[torch.Tensor] = None,
    block_sparse_full_offset: Optional[torch.Tensor] = None,
    block_sparse_full_idx: Optional[torch.Tensor] = None,
    # Arbitrary mask function tensor for element-level masking
    arbitrary_func: Optional[torch.Tensor] = None,
    # K2Q block sparsity parameters (for backward, stored for autograd)
    k2q_block_sparse_mask_cnt: Optional[torch.Tensor] = None,
    k2q_block_sparse_mask_offset: Optional[torch.Tensor] = None,
    k2q_block_sparse_mask_idx: Optional[torch.Tensor] = None,
    k2q_block_sparse_full_cnt: Optional[torch.Tensor] = None,
    k2q_block_sparse_full_offset: Optional[torch.Tensor] = None,
    k2q_block_sparse_full_idx: Optional[torch.Tensor] = None,
    k2q_block_sparse_dq_write_order: Optional[torch.Tensor] = None,
    k2q_block_sparse_dq_write_order_full: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Note: k2q_block_sparse_* parameters are not used in forward pass.
    # They are saved by setup_context for use in backward pass.
    _validate_arbitrary_mask_mode(
        arbitrary_func,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
    )
    q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, out_accum, softmax_lse_accum = flash_attn_3_gpu.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        block_sparse_mask_cnt,
        block_sparse_mask_offset,
        block_sparse_mask_idx,
        block_sparse_full_cnt,
        block_sparse_full_offset,
        block_sparse_full_idx,
        arbitrary_func,
    )

    if out_accum is None:
        out_accum = torch.tensor([], device=out.device)

    if softmax_lse_accum is None:
        softmax_lse_accum = torch.tensor([], device=out.device)

    return out, softmax_lse, out_accum, softmax_lse_accum


@torch.library.register_fake("flash_attn_3::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    # Block sparsity parameters (Q2K direction)
    block_sparse_mask_cnt: Optional[torch.Tensor] = None,
    block_sparse_mask_offset: Optional[torch.Tensor] = None,
    block_sparse_mask_idx: Optional[torch.Tensor] = None,
    block_sparse_full_cnt: Optional[torch.Tensor] = None,
    block_sparse_full_offset: Optional[torch.Tensor] = None,
    block_sparse_full_idx: Optional[torch.Tensor] = None,
    # Arbitrary mask function tensor for element-level masking
    arbitrary_func: Optional[torch.Tensor] = None,
    # K2Q block sparsity parameters (for backward, stored for autograd)
    k2q_block_sparse_mask_cnt: Optional[torch.Tensor] = None,
    k2q_block_sparse_mask_offset: Optional[torch.Tensor] = None,
    k2q_block_sparse_mask_idx: Optional[torch.Tensor] = None,
    k2q_block_sparse_full_cnt: Optional[torch.Tensor] = None,
    k2q_block_sparse_full_offset: Optional[torch.Tensor] = None,
    k2q_block_sparse_full_idx: Optional[torch.Tensor] = None,
    k2q_block_sparse_dq_write_order: Optional[torch.Tensor] = None,
    k2q_block_sparse_dq_write_order_full: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symbolic fake implementation of flash attention forward.
    Returns tensors with the correct shapes and dtypes without actual computation.
    """
    _validate_arbitrary_mask_mode(
        arbitrary_func,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
    )

    # Determine if we're in varlen mode
    is_varlen_q = cu_seqlens_q is not None

    # Get dimensions from query tensor
    if is_varlen_q:
        # varlen mode: q is (total_q, num_heads, head_size)
        total_q, num_heads, head_size = q.shape
        batch_size = cu_seqlens_q.shape[0] - 1

        if max_seqlen_q is None:
            raise ValueError("max_seqlen_q must be provided if cu_seqlens_q is provided")
        seqlen_q = max_seqlen_q
    else:
        # batch mode: q is (batch_size, seqlen_q, num_heads, head_size)
        batch_size, seqlen_q, num_heads, head_size = q.shape
        total_q = batch_size * q.shape[1]
    # Get value head dimension
    head_size_v = v.shape[-1]

    # Determine output dtype (FP8 inputs produce BF16 outputs)
    q_type = q.dtype
    if q_type == torch.float8_e4m3fn:
        out_dtype = torch.bfloat16
    else:
        out_dtype = q_type

    # Create output tensor
    if out_ is not None:
        # If out_ is provided, _flash_attn_forward becomes non-functional
        raise TypeError("Tracing (torch.compile/torch.export) with pre-allocated output tensor is not supported.")

    if is_varlen_q:
        out = torch.empty((total_q, num_heads, head_size_v), dtype=out_dtype, device=q.device)
    else:
        out = torch.empty((batch_size, seqlen_q, num_heads, head_size_v), dtype=out_dtype, device=q.device)

    # Create softmax_lse tensor
    if is_varlen_q:
        softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
    else:
        softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device)

    # TODO(guilhermeleobas): Implement "get_num_splits"
    # There's an heuristic to compute num_splits when "num_splits <= 0"
    # assert that num_splits is > 0 for now
    if num_splits <= 0:
        raise ValueError(f"tracing (torch.compile/torch.export) with num_splits <= 0 not supported. Got {num_splits=}")

    if num_splits > 1:
        if is_varlen_q:
            out_accum = torch.empty((num_splits, num_heads, total_q, head_size_v), dtype=torch.float32, device=q.device)
            softmax_lse_accum = torch.empty((num_splits, num_heads, total_q), dtype=torch.float32, device=q.device)
        else:
            out_accum = torch.empty((num_splits, batch_size, num_heads, seqlen_q, head_size_v), dtype=torch.float32, device=q.device)
            softmax_lse_accum = torch.empty((num_splits, batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device)
    else:
        # Tensors are not set when num_splits < 1
        out_accum = torch.tensor([], device=out.device)
        softmax_lse_accum = torch.tensor([], device=out.device)

    return out, softmax_lse, out_accum, softmax_lse_accum


@torch.library.custom_op("flash_attn_3::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    sequed_q: Optional[torch.Tensor] = None,
    sequed_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    # Arbitrary mask function tensor for element-level masking
    arbitrary_func: Optional[torch.Tensor] = None,
    # Block sparsity parameters (K2Q direction for backward)
    block_sparse_mask_cnt: Optional[torch.Tensor] = None,
    block_sparse_mask_offset: Optional[torch.Tensor] = None,
    block_sparse_mask_idx: Optional[torch.Tensor] = None,
    block_sparse_full_cnt: Optional[torch.Tensor] = None,
    block_sparse_full_offset: Optional[torch.Tensor] = None,
    block_sparse_full_idx: Optional[torch.Tensor] = None,
    block_sparse_dq_write_order: Optional[torch.Tensor] = None,
    block_sparse_dq_write_order_full: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    metadata_prevalidated = False
    if deterministic and arbitrary_func is not None:
        if any(
            x is not None
            for x in (cu_seqlens_q, cu_seqlens_k, sequed_q, sequed_k)
        ):
            raise NotImplementedError(
                "C++ arbitrary deterministic backward currently supports fixed-length tensors only"
            )
        metadata = LinearBlockSparseTensors(
            block_sparse_mask_cnt,
            block_sparse_mask_offset,
            block_sparse_mask_idx,
            block_sparse_full_cnt,
            block_sparse_full_offset,
            block_sparse_full_idx,
            block_sparse_dq_write_order,
            block_sparse_dq_write_order_full,
        )
        _validate_deterministic_k2q_metadata(
            metadata, q, k, v, q.shape[1], k.shape[1]
        )
        metadata_prevalidated = True
    softmax_d, *rest = flash_attn_3_gpu.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        sm_margin,
        arbitrary_func,
        block_sparse_mask_cnt,
        block_sparse_mask_offset,
        block_sparse_mask_idx,
        block_sparse_full_cnt,
        block_sparse_full_offset,
        block_sparse_full_idx,
        block_sparse_dq_write_order,
        block_sparse_dq_write_order_full,
        metadata_prevalidated,
    )
    # The return is an internal autograd sequencing token; dq/dk/dv are
    # produced through the declared mutations above.
    return softmax_d.new_empty((0,))


@torch.library.register_fake("flash_attn_3::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    sequed_q: Optional[torch.Tensor] = None,
    sequed_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    # Arbitrary mask function tensor for element-level masking
    arbitrary_func: Optional[torch.Tensor] = None,
    # Block sparsity parameters (K2Q direction for backward)
    block_sparse_mask_cnt: Optional[torch.Tensor] = None,
    block_sparse_mask_offset: Optional[torch.Tensor] = None,
    block_sparse_mask_idx: Optional[torch.Tensor] = None,
    block_sparse_full_cnt: Optional[torch.Tensor] = None,
    block_sparse_full_offset: Optional[torch.Tensor] = None,
    block_sparse_full_idx: Optional[torch.Tensor] = None,
    block_sparse_dq_write_order: Optional[torch.Tensor] = None,
    block_sparse_dq_write_order_full: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    is_varlen = is_varlen_q or is_varlen_k or sequed_q is not None or sequed_k is not None

    if not is_varlen_q:
        seqlen_q = q.size(1)
    else:
        if max_seqlen_q is None:
            raise ValueError(
                "max_seqlen_q must be provided if cu_seqlens_q is provided"
            )
        seqlen_q = max_seqlen_q
    if is_varlen_k:
        if max_seqlen_k is None:
            raise ValueError(
                "max_seqlen_k must be provided if cu_seqlens_k is provided"
            )
        seqlen_k = max_seqlen_k
    else:
        seqlen_k = k.size(1)

    head_size = q.size(-1)
    head_size_v = v.size(-1)
    head_size_rounded = round_up_headdim(max(head_size, head_size_v))

    is_arbitrary = arbitrary_func is not None
    if is_arbitrary and deterministic:
        # The output is a fixed internal token, so Fake/AOT shape inference
        # does not need a target-specific tile.  A known target is still
        # checked against the runtime support set.
        if q.device.type == "meta" or not torch.cuda.is_available():
            arch = None
        else:
            try:
                cap = torch.cuda.get_device_capability(q.device)
            except (AssertionError, RuntimeError):
                # CUDA FakeTensors are valid on CPU-only tracing hosts.  Their
                # target architecture is unknown until deployment/runtime.
                arch = None
            else:
                arch = cap[0] * 10 + cap[1]
        if arch is not None and arch not in (80, 86, 89, 90):
            raise NotImplementedError(
                "C++ arbitrary deterministic backward is currently supported only on "
                "SM80, SM86, SM89, and SM90"
            )
        if is_varlen or q.ndim != 4 or k.ndim != 4:
            raise NotImplementedError(
                "C++ arbitrary deterministic backward currently supports fixed-length tensors only"
            )
        if head_size_rounded > 192:
            raise ValueError(
                "C++ arbitrary deterministic backward does not support the rounded hdim-256 bucket"
            )
        metadata = LinearBlockSparseTensors(
            block_sparse_mask_cnt,
            block_sparse_mask_offset,
            block_sparse_mask_idx,
            block_sparse_full_cnt,
            block_sparse_full_offset,
            block_sparse_full_idx,
            block_sparse_dq_write_order,
            block_sparse_dq_write_order_full,
        )
        _validate_deterministic_k2q_metadata(
            metadata,
            q,
            k,
            v,
            seqlen_q,
            seqlen_k,
        )
    return torch.empty((0,), dtype=torch.float32, device=q.device)


def setup_context(ctx, inputs, output):
    q, k, v = inputs[:3]
    out, softmax_lse, _, _ = output
    ctx.save_for_backward(q, k, v, out, softmax_lse)
    # K2Q metadata and deterministic were appended to the original inputs.
    ctx.softmax_scale = inputs[-27]
    ctx.causal = inputs[-26]
    ctx.window_size = [inputs[-25], inputs[-24]]
    ctx.attention_chunk = inputs[-23]
    ctx.softcap = inputs[-22]
    ctx.sm_margin = inputs[-17]
    ctx.arbitrary_func = inputs[-10]
    ctx.k2q_mask_cnt = inputs[-9]
    ctx.k2q_mask_offset = inputs[-8]
    ctx.k2q_mask_idx = inputs[-7]
    ctx.k2q_full_cnt = inputs[-6]
    ctx.k2q_full_offset = inputs[-5]
    ctx.k2q_full_idx = inputs[-4]
    ctx.k2q_dq_write_order = inputs[-3]
    ctx.k2q_dq_write_order_full = inputs[-2]
    ctx.deterministic = inputs[-1]


def _backward(ctx, dout, *grads):
    q, k, v, out, softmax_lse = ctx.saved_tensors
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        None, None, # cu_seqlens_q, cu_seqlens_k,
        None, None, # sequed_q, sequed_k,
        None, None, # max_seqlen_q, max_seqlen_k,
        dq,
        dk,
        dv,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size[0],
        ctx.window_size[1],
        ctx.softcap,
        ctx.deterministic,
        ctx.sm_margin,
        ctx.arbitrary_func if hasattr(ctx, 'arbitrary_func') else None,
        *_k2q_metadata_args_from_ctx(ctx),
    )
    # _flash_attn_forward has 50 parameters: q, k, v + 47 others.
    return dq, dk, dv, *((None,) * 47)


_flash_attn_forward.register_autograd(_backward, setup_context=setup_context)



class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
        return_softmax=False,
        # Arbitrary mask and block sparse support
        arbitrary_func=None,  # [batch, head_q, func_num, seqlen_q+256], supports broadcasting (batch/head_q can be 1)
        # Q2K block sparse (for forward)
        q2k_block_sparse=None,  # LinearBlockSparseTensors or tuple of 6 tensors
        # K2Q block sparse (for backward, stored to ctx)
        k2q_block_sparse=None,  # LinearBlockSparseTensors or tuple of 6/8 tensors
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)

        q2k_block_sparse, k2q_block_sparse = _prepare_arbitrary_block_sparse(
            q,
            k,
            v,
            arbitrary_func,
            q2k_block_sparse,
            k2q_block_sparse,
            q.shape[1],
            k.shape[1],
            softcap=softcap,
            deterministic=deterministic,
        )

        # Extract q2k block sparse tensors for forward
        q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx = None, None, None
        q2k_full_cnt, q2k_full_offset, q2k_full_idx = None, None, None
        if q2k_block_sparse is not None:
            if hasattr(q2k_block_sparse, 'mask_block_cnt'):
                # LinearBlockSparseTensors object
                q2k_mask_cnt = q2k_block_sparse.mask_block_cnt
                q2k_mask_offset = q2k_block_sparse.mask_block_offset
                q2k_mask_idx = q2k_block_sparse.mask_block_idx
                q2k_full_cnt = q2k_block_sparse.full_block_cnt
                q2k_full_offset = q2k_block_sparse.full_block_offset
                q2k_full_idx = q2k_block_sparse.full_block_idx
            else:
                # Tuple of 6 tensors
                (q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx,
                 q2k_full_cnt, q2k_full_offset, q2k_full_idx) = q2k_block_sparse

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            None,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
            # Q2K block sparse for forward
            block_sparse_mask_cnt=q2k_mask_cnt,
            block_sparse_mask_offset=q2k_mask_offset,
            block_sparse_mask_idx=q2k_mask_idx,
            block_sparse_full_cnt=q2k_full_cnt,
            block_sparse_full_offset=q2k_full_offset,
            block_sparse_full_idx=q2k_full_idx,
            arbitrary_func=arbitrary_func,
            deterministic=deterministic,
            **_k2q_metadata_kwargs(k2q_block_sparse),
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        ctx.sm_margin = sm_margin
        # Save arbitrary_func for backward
        ctx.arbitrary_func = arbitrary_func
        _save_k2q_metadata(ctx, k2q_block_sparse)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            ctx.arbitrary_func if hasattr(ctx, 'arbitrary_func') else None,
            *_k2q_metadata_args_from_ctx(ctx),
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        # Return gradients for: qkv, softmax_scale, causal, q_descale, k_descale, v_descale,
        # window_size, attention_chunk, softcap, deterministic, num_heads_q, sm_margin,
        # return_softmax, arbitrary_func, q2k_block_sparse, k2q_block_sparse (total 16)
        return dqkv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
        # Arbitrary mask and block sparse support
        arbitrary_func=None,  # [batch, head_q, func_num, seqlen_q+256], supports broadcasting (batch/head_q can be 1)
        # Q2K block sparse (for forward)
        q2k_block_sparse=None,  # LinearBlockSparseTensors or tuple of 6 tensors
        # K2Q block sparse (for backward, stored to ctx)
        k2q_block_sparse=None,  # LinearBlockSparseTensors or tuple of 6/8 tensors
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)

        q2k_block_sparse, k2q_block_sparse = _prepare_arbitrary_block_sparse(
            q,
            k,
            v,
            arbitrary_func,
            q2k_block_sparse,
            k2q_block_sparse,
            q.shape[1],
            k.shape[1],
            softcap=softcap,
            deterministic=deterministic,
        )

        # Extract q2k block sparse tensors for forward
        q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx = None, None, None
        q2k_full_cnt, q2k_full_offset, q2k_full_idx = None, None, None
        if q2k_block_sparse is not None:
            if hasattr(q2k_block_sparse, 'mask_block_cnt'):
                # LinearBlockSparseTensors object
                q2k_mask_cnt = q2k_block_sparse.mask_block_cnt
                q2k_mask_offset = q2k_block_sparse.mask_block_offset
                q2k_mask_idx = q2k_block_sparse.mask_block_idx
                q2k_full_cnt = q2k_block_sparse.full_block_cnt
                q2k_full_offset = q2k_block_sparse.full_block_offset
                q2k_full_idx = q2k_block_sparse.full_block_idx
            else:
                # Tuple of 6 tensors
                (q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx,
                 q2k_full_cnt, q2k_full_offset, q2k_full_idx) = q2k_block_sparse

        # out, q, k, v, out_padded, softmax_lse = _flash_attn_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            # Q2K block sparse for forward
            block_sparse_mask_cnt=q2k_mask_cnt,
            block_sparse_mask_offset=q2k_mask_offset,
            block_sparse_mask_idx=q2k_mask_idx,
            block_sparse_full_cnt=q2k_full_cnt,
            block_sparse_full_offset=q2k_full_offset,
            block_sparse_full_idx=q2k_full_idx,
            arbitrary_func=arbitrary_func,
            deterministic=deterministic,
            **_k2q_metadata_kwargs(k2q_block_sparse),
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        # Save arbitrary_func for backward
        ctx.arbitrary_func = arbitrary_func
        _save_k2q_metadata(ctx, k2q_block_sparse)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            ctx.arbitrary_func if hasattr(ctx, 'arbitrary_func') else None,
            *_k2q_metadata_args_from_ctx(ctx),
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        # Return gradients for: q, k, v, softmax_scale, causal, qv, q_descale, k_descale, v_descale,
        # window_size, attention_chunk, softcap, num_splits, pack_gqa, deterministic, sm_margin,
        # return_softmax, arbitrary_func, q2k_block_sparse, k2q_block_sparse (total 20)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
        # Arbitrary mask and block sparse support
        arbitrary_func=None,  # [batch, head_q, func_num, seqlen_q+256], supports broadcasting
        q2k_block_sparse=None,  # LinearBlockSparseTensors or tuple of 6 tensors
        k2q_block_sparse=None,  # LinearBlockSparseTensors or tuple of 6/8 tensors
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)

        q2k_block_sparse, k2q_block_sparse = _prepare_arbitrary_block_sparse(
            q,
            k,
            v,
            arbitrary_func,
            q2k_block_sparse,
            k2q_block_sparse,
            max_seqlen_q,
            max_seqlen_k,
            softcap=softcap,
            varlen_and_split=num_splits > 1,
            deterministic=deterministic,
            is_varlen=True,
        )

        # Extract q2k block sparse tensors for forward
        q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx = None, None, None
        q2k_full_cnt, q2k_full_offset, q2k_full_idx = None, None, None
        if q2k_block_sparse is not None:
            if hasattr(q2k_block_sparse, 'mask_block_cnt'):
                # LinearBlockSparseTensors object
                q2k_mask_cnt = q2k_block_sparse.mask_block_cnt
                q2k_mask_offset = q2k_block_sparse.mask_block_offset
                q2k_mask_idx = q2k_block_sparse.mask_block_idx
                q2k_full_cnt = q2k_block_sparse.full_block_cnt
                q2k_full_offset = q2k_block_sparse.full_block_offset
                q2k_full_idx = q2k_block_sparse.full_block_idx
            else:
                # Tuple of 6 tensors
                (q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx,
                 q2k_full_cnt, q2k_full_offset, q2k_full_idx) = q2k_block_sparse

        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            # Q2K block sparse for forward
            block_sparse_mask_cnt=q2k_mask_cnt,
            block_sparse_mask_offset=q2k_mask_offset,
            block_sparse_mask_idx=q2k_mask_idx,
            block_sparse_full_cnt=q2k_full_cnt,
            block_sparse_full_offset=q2k_full_offset,
            block_sparse_full_idx=q2k_full_idx,
            arbitrary_func=arbitrary_func,
            deterministic=deterministic,
            **_k2q_metadata_kwargs(k2q_block_sparse),
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        # Save arbitrary_func for backward
        ctx.arbitrary_func = arbitrary_func
        _save_k2q_metadata(ctx, k2q_block_sparse)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            ctx.arbitrary_func if hasattr(ctx, 'arbitrary_func') else None,
            *_k2q_metadata_args_from_ctx(ctx),
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        # Return gradients for: q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
        # max_seqlen_q, max_seqlen_k, softmax_scale, causal, qv, q_descale, k_descale, v_descale,
        # window_size, attention_chunk, softcap, num_splits, pack_gqa, deterministic, sm_margin,
        # return_softmax, arbitrary_func, q2k_block_sparse, k2q_block_sparse (total 26)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
    return_attn_probs=False,
    arbitrary_func: Optional[torch.Tensor] = None,
    q2k_block_sparse: Optional[LinearBlockSparseTensors] = None,  # Q2K direction for forward
    k2q_block_sparse: Optional[LinearBlockSparseTensors] = None,  # K2Q direction for backward
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        arbitrary_func: Optional[torch.Tensor]. Func tensor for arbitrary mask support.
            Shape: [batch, head_q, func_num, seqlen_q+256], supports broadcasting (batch/head_q can be 1).
        q2k_block_sparse: Optional[LinearBlockSparseTensors]. Block sparse pattern for Q2K direction (forward).
        k2q_block_sparse: Optional[LinearBlockSparseTensors]. Block sparse pattern for K2Q direction (backward).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
        return_attn_probs,
        arbitrary_func,
        q2k_block_sparse,
        k2q_block_sparse,
    )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
    # Arbitrary mask and block sparse support
    arbitrary_func: Optional[torch.Tensor] = None,
    q2k_block_sparse: Optional[LinearBlockSparseTensors] = None,  # Q2K direction for forward
    k2q_block_sparse: Optional[LinearBlockSparseTensors] = None,  # K2Q direction for backward
):
    """Flash Attention with optional arbitrary mask and block sparsity support.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        arbitrary_func: Optional tensor for arbitrary mask function.
            Shape: [batch, head_q, func_num, seqlen_q+256], supports broadcasting (batch/head_q can be 1)
        q2k_block_sparse: Optional LinearBlockSparseTensors for Q2K block sparsity (forward pass).
        k2q_block_sparse: Optional LinearBlockSparseTensors for K2Q block sparsity (backward pass).

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
        arbitrary_func,
        q2k_block_sparse,
        k2q_block_sparse,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
    # Arbitrary mask and block sparse support
    arbitrary_func: Optional[torch.Tensor] = None,
    q2k_block_sparse: Optional[LinearBlockSparseTensors] = None,  # Q2K direction for forward
    k2q_block_sparse: Optional[LinearBlockSparseTensors] = None,  # K2Q direction for backward
):
    """Flash Attention for variable-length sequences with optional arbitrary mask and block sparsity.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = sum of seqlen_q for all sequences
        k: (total_k, nheads_k, headdim), where total_k = sum of seqlen_k for all sequences
        v: (total_k, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), cumulative sequence lengths for Q
        cu_seqlens_k: (batch_size + 1,), cumulative sequence lengths for K/V
        max_seqlen_q: Maximum sequence length for Q
        max_seqlen_k: Maximum sequence length for K/V
        seqused_q: (batch_size,), optional, actual sequence lengths for Q
        seqused_k: (batch_size,), optional, actual sequence lengths for K/V
        softmax_scale: float. Default to 1 / sqrt(headdim)
        causal: bool. Whether to apply causal attention mask
        window_size: (left, right). Sliding window local attention
        deterministic: bool. Use deterministic backward pass
        return_attn_probs: bool. Return softmax_lse
        arbitrary_func: Optional tensor for arbitrary mask function
        q2k_block_sparse: Optional LinearBlockSparseTensors for Q2K block sparsity (forward)
        k2q_block_sparse: Optional LinearBlockSparseTensors for K2Q block sparsity (backward)

    Return:
        out: (total_q, nheads, headdim)
        softmax_lse [optional]: (batch_size, nheads, max_seqlen_q)
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
        arbitrary_func,
        q2k_block_sparse,
        k2q_block_sparse,
    )


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_3_gpu.fwd_combine(out_partial, lse_partial, out, out_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    # Arbitrary mask and block sparse support (forward only, no backward)
    arbitrary_func: Optional[torch.Tensor] = None,
    q2k_block_sparse: Optional[LinearBlockSparseTensors] = None,  # Q2K direction for forward
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.).
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    if arbitrary_func is not None and q2k_block_sparse is None:
        max_k_cache_len = (
            k_cache.shape[1] if page_table is None else page_table.shape[1] * k_cache.shape[1]
        )
        q2k_block_sparse, _ = _prepare_arbitrary_block_sparse(
            q,
            k_cache,
            v_cache,
            arbitrary_func,
            q2k_block_sparse,
            None,
            max_seqlen_q or q.shape[1],
            max_k_cache_len,
            softcap=softcap,
            paged_kv_non_tma=page_table is not None,
            append_kv=k is not None,
            prepare_k2q=False,
        )

    # Extract q2k block sparse tensors for forward
    q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx = None, None, None
    q2k_full_cnt, q2k_full_offset, q2k_full_idx = None, None, None
    if q2k_block_sparse is not None:
        if hasattr(q2k_block_sparse, 'mask_block_cnt'):
            # LinearBlockSparseTensors object
            q2k_mask_cnt = q2k_block_sparse.mask_block_cnt
            q2k_mask_offset = q2k_block_sparse.mask_block_offset
            q2k_mask_idx = q2k_block_sparse.mask_block_idx
            q2k_full_cnt = q2k_block_sparse.full_block_cnt
            q2k_full_offset = q2k_block_sparse.full_block_offset
            q2k_full_idx = q2k_block_sparse.full_block_idx
        else:
            # Tuple of 6 tensors
            (q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx,
             q2k_full_cnt, q2k_full_offset, q2k_full_idx) = q2k_block_sparse

    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale, k_descale, v_descale,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        # Q2K block sparse for forward
        block_sparse_mask_cnt=q2k_mask_cnt,
        block_sparse_mask_offset=q2k_mask_offset,
        block_sparse_mask_idx=q2k_mask_idx,
        block_sparse_full_cnt=q2k_full_cnt,
        block_sparse_full_offset=q2k_full_offset,
        block_sparse_full_idx=q2k_full_idx,
        arbitrary_func=arbitrary_func,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = flash_attn_3_gpu.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return scheduler_metadata
