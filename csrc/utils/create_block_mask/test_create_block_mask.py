"""
Test script for create_block_mask CUDA kernels.

Tests two CUDA kernels:
  - create_q2k_block_sparse_from_func: Q2K (Forward)  - fix q_block, loop kv_blocks
  - create_k2q_block_sparse_from_func: K2Q (Backward) - fix kv_block, loop q_blocks

Compares the output of our CUDA kernels with PyTorch reference implementations.

Memory-efficient format:
  - block_idx: [B, H, num_blocks, max_blocks] - combined tensor
  - full blocks stored left-to-right (indices 0, 1, 2, ...)
  - mask blocks stored right-to-left (indices max_blocks-1, max_blocks-2, ...)

Usage:
    # Build with Makefile (from utils directory)
    make block_mask

    # Or build directly
    cd create_block_mask && pip install -e . --no-build-isolation

    # Run tests
    python test_create_block_mask.py
"""
import math

import torch
import pytest
from torch.nn.attention.flex_attention import create_block_mask

# Import our CUDA kernel
try:
    import create_block_mask_cuda
except ImportError:
    create_block_mask_cuda = None
    print("Warning: create_block_mask_cuda not found. Please build it first with: pip install -e .")

def flex_arbitrary_mask(b, h, q_idx, kv_idx, arbitrary_func):
    """
    Reference flex attention mask function.

    arbitrary_func: [B, H, n_func, seqlen_q + 256]
    Valid intervals are [0, F0), [F1, F2), [F3, F4), ...
    """
    zero = h * 0
    value_valid = kv_idx < arbitrary_func[b, zero, zero, q_idx]
    n_func = arbitrary_func.shape[2]
    for i in range(n_func // 2):
        in_range = (kv_idx >= arbitrary_func[b, zero, zero + (2*i+1), q_idx]) & (kv_idx < arbitrary_func[b, zero, zero + (2*i+2), q_idx])
        value_valid = value_valid | in_range
    return value_valid


def create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Create reference block mask using PyTorch's create_block_mask.

    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension

    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx
    """
    frozen_af = func_tensor

    def mask_mod(b, h, q_idx, kv_idx, af=frozen_af):
        return flex_arbitrary_mask(b, h, q_idx, kv_idx, af)

    bm = create_block_mask(
        mask_mod,
        1,  # B
        1,  # H
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
    )

    # Extract tensors from BlockMask
    # Format: (Q_LEN, KV_LEN, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
    #          q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, Q_BLOCK_SIZE, KV_BLOCK_SIZE, mask_mod)
    bm_tuple = bm.as_tuple()
    mask_block_cnt = bm_tuple[2]   # kv_num_blocks
    mask_block_idx = bm_tuple[3]   # kv_indices
    full_block_cnt = bm_tuple[4]   # full_kv_num_blocks
    full_block_idx = bm_tuple[5]   # full_kv_indices

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def create_reference_k2q_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Create reference K2Q (backward) block mask using PyTorch's create_block_mask.

    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension

    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx
        Note: For K2Q, these are q_num_blocks, q_indices, full_q_num_blocks, full_q_indices
    """
    frozen_af = func_tensor

    def mask_mod(b, h, q_idx, kv_idx, af=frozen_af):
        return flex_arbitrary_mask(b, h, q_idx, kv_idx, af)

    bm = create_block_mask(
        mask_mod,
        1,  # B
        1,  # H
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
    )

    # Extract K2Q tensors from BlockMask
    # Format: (Q_LEN, KV_LEN, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
    #          q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, Q_BLOCK_SIZE, KV_BLOCK_SIZE, mask_mod)
    bm_tuple = bm.as_tuple()
    mask_block_cnt = bm_tuple[6]   # q_num_blocks
    mask_block_idx = bm_tuple[7]   # q_indices
    full_block_cnt = bm_tuple[8]   # full_q_num_blocks
    full_block_idx = bm_tuple[9]   # full_q_indices

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx


def extract_indices_from_combined(mask_cnt, full_cnt, block_idx):
    """
    Extract mask_idx and full_idx from the combined block_idx tensor.

    The combined block_idx tensor layout:
    - full blocks: stored left-to-right at indices 0, 1, 2, ...
    - mask blocks: stored right-to-left at indices max_blocks-1, max_blocks-2, ...

    Args:
        mask_cnt: [B, H, num_blocks], count of mask blocks per row
        full_cnt: [B, H, num_blocks], count of full blocks per row
        block_idx: [B, H, num_blocks, max_blocks], combined indices

    Returns:
        mask_idx: [B, H, num_blocks, max_blocks], mask block indices (extracted and reversed)
        full_idx: [B, H, num_blocks, max_blocks], full block indices (extracted)
    """
    B, H, num_blocks, max_blocks = block_idx.shape

    # Create output tensors initialized to -1
    mask_idx = torch.full_like(block_idx, -1)
    full_idx = torch.full_like(block_idx, -1)

    for b in range(B):
        for h in range(H):
            for blk in range(num_blocks):
                fcnt = full_cnt[b, h, blk].item()
                mcnt = mask_cnt[b, h, blk].item()

                # Extract full indices (left-to-right)
                if fcnt > 0:
                    full_idx[b, h, blk, :fcnt] = block_idx[b, h, blk, :fcnt]

                # Extract mask indices (right-to-left, then reverse)
                if mcnt > 0:
                    # Mask blocks are stored at positions [max_blocks-mcnt, max_blocks)
                    # We need to reverse them to match the reference order
                    mask_indices = block_idx[b, h, blk, max_blocks-mcnt:max_blocks]
                    mask_idx[b, h, blk, :mcnt] = mask_indices.flip(0)

    return mask_idx, full_idx


def create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=False):
    """
    Create Q2K (forward) block mask using our CUDA kernel.

    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension
        check_q_boundary: if True (FlexAttention mode), partial q_blocks cannot have FULL kv_blocks

    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_idx
        Output shapes: [B, H, num_q_blocks, ...] for idx tensors
    """
    with torch.cuda.nvtx.range("create_q2k_kernel_block_mask"):
        mask_block_cnt, full_block_cnt, block_idx = \
            create_block_mask_cuda.create_q2k_block_sparse_from_func(
                func_tensor,
                seqlen_q,
                seqlen_k,
                Q_BLOCK_SIZE,
                KV_BLOCK_SIZE,
                check_q_boundary,
                debug=True  # Initialize to -1 for testing
            )

    # Extract mask and full indices from combined tensor
    mask_block_idx, full_block_idx = extract_indices_from_combined(
        mask_block_cnt, full_block_cnt, block_idx
    )

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_idx


def create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """
    Create K2Q (backward) block mask using our CUDA kernel.

    Args:
        func_tensor: [B, H, n_func, seqlen_q + 256], int32
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        Q_BLOCK_SIZE: block size for query dimension
        KV_BLOCK_SIZE: block size for key dimension

    Returns:
        mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_idx
        Output shapes: [B, H, num_kv_blocks, ...] for idx tensors
    """
    mask_block_cnt, full_block_cnt, block_idx = \
        create_block_mask_cuda.create_k2q_block_sparse_from_func(
            func_tensor,
            seqlen_q,
            seqlen_k,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            debug=True  # Initialize to -1 for testing
        )

    # Extract mask and full indices from combined tensor
    mask_block_idx, full_block_idx = extract_indices_from_combined(
        mask_block_cnt, full_block_cnt, block_idx
    )

    return mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx, block_idx


def compare_block_masks(ref_cnt, ref_idx, kernel_cnt, kernel_idx, name=""):
    """
    Compare block mask tensors.

    The idx tensors may have different order but should contain the same values.
    """
    # Compare counts
    cnt_match = torch.equal(ref_cnt, kernel_cnt)
    if not cnt_match:
        print(f"{name} count mismatch!")
        print(f"  Reference: {ref_cnt}")
        print(f"  Kernel:    {kernel_cnt}")
        return False

    # Compare indices (considering they might be in different order)
    B, H, num_blocks = ref_cnt.shape
    for b in range(B):
        for h in range(H):
            for block_idx in range(num_blocks):
                cnt = ref_cnt[b, h, block_idx].item()
                if cnt > 0:
                    ref_indices = set(ref_idx[b, h, block_idx, :cnt].tolist())
                    kernel_indices = set(kernel_idx[b, h, block_idx, :cnt].tolist())
                    if ref_indices != kernel_indices:
                        print(f"{name} index mismatch at (b={b}, h={h}, block={block_idx})!")
                        print(f"  Reference: {sorted(ref_indices)}")
                        print(f"  Kernel:    {sorted(kernel_indices)}")
                        return False

    return True

def generate_func_tensor(seqlen_q, seqlen_k, n_func=3, device="cuda"):
    """
    Generate func_tensor with multiple intervals.

    Returns:
        func_tensor: [1, 1, n_func, seqlen_q + 256]
    """
    B, H = 1, 1
    func_tensor = torch.zeros(B, H, n_func, seqlen_q + 256, dtype=torch.int32, device=device)

    pattern = "random"
    if pattern == "random":
        # random pattern
        coef = 1.0 / n_func
        for i in range(n_func):
            low = int(i * coef * seqlen_k)
            high = int((i + 1) * coef * seqlen_k)
            if high <= low:
                high = low + 1
            func_tensor[:, :, i, :seqlen_q] = torch.randint(low, high, size=(B, H, seqlen_q), dtype=torch.int32, device=device)
    elif pattern == "causal":
        # causal mask pattern
        for q_idx in range(seqlen_q + 256):
            if q_idx < seqlen_q:
                func_tensor[0, 0, 0, q_idx] = min(q_idx + 1, seqlen_k)

    return func_tensor.contiguous()


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8192, 8192),
    (16384, 16384),
    (16666, 16666),
    (32768, 32768),
    (33333, 33333),
    (61111, 61111),
])
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
    (256, 128),
])
def test_q2k_random_mask(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test Q2K (forward) random arbitrary mask patterns."""
    print(f"\nTesting Q2K random mask: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Reference
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (Q2K)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, "Mask block mismatch"
    assert full_ok, "Full block mismatch"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8192, 8192),
    (16384, 16384),
    (16666, 16666),
    (32768, 32768),
    (33333, 33333),
    (61111, 61111),
])
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
    (128, 256),
    (256, 128),
])
def test_k2q_random_mask(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test K2Q (backward) random arbitrary mask patterns."""
    print(f"\nTesting K2Q random mask: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Reference (K2Q)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_k2q_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (K2Q)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, "Mask block mismatch"
    assert full_ok, "Full block mismatch"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize(
    "headdim,headdim_v,expected",
    [
        (128, 128, (128, 256)),
        (192, 128, (128, 256)),
        (128, 64, (128, 128)),
        (256, 256, (256, 128)),
    ],
)
def test_sm100_k2q_auto_tile_sizes(headdim, headdim_v, expected):
    assert create_block_mask_cuda.get_bwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=100,
        headdim_v=headdim_v,
    ) == expected


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("arch", [80, 86, 89])
@pytest.mark.parametrize("headdim", [64, 96, 128, 192, 256])
def test_sm8x_q2k_auto_tile_sizes(arch, headdim):
    rounded_headdim = create_block_mask_cuda.round_up_hopper_headdim(headdim)
    if rounded_headdim <= 64:
        expected = (128, 96)
    elif rounded_headdim <= 96:
        expected = (128, 48)
    elif rounded_headdim <= 128:
        expected = (128, 96) if arch in (86, 89) else (128, 48)
    elif rounded_headdim <= 192:
        expected = (128, 64)
    else:
        expected = (128, 48) if arch in (86, 89) else (128, 64)
    assert create_block_mask_cuda.get_fwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=arch,
    ) == expected


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("query_name", ["get_fwd_tile_sizes", "get_bwd_tile_sizes"])
@pytest.mark.parametrize("backend", [None, ""])
def test_sm90_tile_backend_is_required(query_name, backend):
    query = getattr(create_block_mask_cuda, query_name)
    with pytest.raises(RuntimeError, match=r"backend is required for SM90"):
        query(
            128,
            is_arbitrary=True,
            arch=90,
            backend=backend,
        )


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("query_name", ["get_fwd_tile_sizes", "get_bwd_tile_sizes"])
@pytest.mark.parametrize("arch", [90, 100])
def test_tile_backend_rejects_unknown_values(query_name, arch):
    query = getattr(create_block_mask_cuda, query_name)
    with pytest.raises(RuntimeError, match=r"backend must be 'cpp' or 'dsl'"):
        query(
            128,
            is_arbitrary=True,
            arch=arch,
            backend="unknown",
        )


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize(
    "arch,backend,error",
    [
        (80, "dsl", r"backend='dsl' is not supported on SM80"),
        (100, "cpp", r"backend='cpp' is not supported on SM100"),
    ],
)
def test_tile_backend_rejects_unsupported_arch_combinations(arch, backend, error):
    with pytest.raises(RuntimeError, match=error):
        create_block_mask_cuda.get_fwd_tile_sizes(
            128,
            is_arbitrary=True,
            arch=arch,
            backend=backend,
        )


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("backend", ["cpp", "dsl"])
@pytest.mark.parametrize("headdim", [64, 96, 128, 192, 256])
def test_sm90_q2k_auto_tile_sizes(headdim, backend):
    if backend == "cpp":
        rounded_headdim = create_block_mask_cuda.round_up_hopper_headdim(headdim)
        expected = (
            (192, 128)
            if rounded_headdim <= 96
            else (128, 128)
            if rounded_headdim <= 128
            else (128, 96)
            if rounded_headdim <= 192
            else (128, 64)
        )
    else:
        expected = (
            (192, 128)
            if headdim <= 64
            else (192, 144)
            if headdim <= 96
            else (128, 128)
            if headdim <= 128
            else (128, 112)
            if headdim <= 192
            else (128, 80)
        )
    assert create_block_mask_cuda.get_fwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=90,
        backend=backend,
    ) == expected


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("arch", [80, 86, 89])
@pytest.mark.parametrize(
    "headdim",
    [
        64,
        96,
        128,
        192,
    ],
)
def test_sm8x_k2q_auto_tile_sizes(arch, headdim):
    rounded_headdim = create_block_mask_cuda.round_up_hopper_headdim(headdim)
    if arch == 80:
        expected = (
            (128, 128)
            if rounded_headdim <= 64
            else (64, 128)
            if rounded_headdim <= 128
            else (64, 80)
            if rounded_headdim <= 192
            else (64, 64)
        )
    else:
        expected = (
            (64, 128)
            if rounded_headdim <= 96
            else (64, 96)
            if rounded_headdim <= 128
            else (64, 64)
            if rounded_headdim <= 192
            else (32, 64)
        )
    assert create_block_mask_cuda.get_bwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=arch,
    ) == expected


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("backend", ["cpp", "dsl"])
@pytest.mark.parametrize("headdim", [64, 96, 128, 192, 256])
def test_sm90_k2q_auto_tile_sizes(headdim, backend):
    if backend == "cpp":
        rounded_headdim = create_block_mask_cuda.round_up_hopper_headdim(headdim)
        expected = (
            (128, 128)
            if rounded_headdim <= 64
            else (64, 128)
            if rounded_headdim <= 128
            else (64, 96)
            if rounded_headdim <= 192
            else (64, 80)
        )
    else:
        expected = (
            (128, 128)
            if headdim <= 128
            else (128, 96)
            if headdim <= 192
            else (128, 64)
        )
    assert create_block_mask_cuda.get_bwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=90,
        backend=backend,
    ) == expected


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
def test_q2k_auto_csr_uses_queried_tile_sizes():
    q_len, kv_len, headdim = 257, 289, 128
    arch = create_block_mask_cuda.get_gpu_arch()
    backend = "cpp" if arch == 90 else None
    func_tensor = torch.full(
        (1, 1, 1, q_len + 256),
        kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    result = create_block_mask_cuda.create_q2k_csr_sparse_auto(
        func_tensor,
        q_len,
        kv_len,
        headdim,
        is_arbitrary=True,
        check_q_boundary=True,
        backend=backend,
    )
    queried_tile = create_block_mask_cuda.get_fwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        backend=backend,
    )
    assert result[-2:] == queried_tile
    mask_cnt, mask_offset, _, full_cnt, full_offset, _, block_m, _ = result
    assert mask_cnt.shape == full_cnt.shape == (
        1,
        1,
        math.ceil(q_len / block_m),
    )
    assert mask_offset.numel() == full_offset.numel() == mask_cnt.numel() + 1


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
def test_k2q_auto_csr_uses_queried_tile_sizes():
    q_len, kv_len, headdim = 257, 289, 128
    arch = create_block_mask_cuda.get_gpu_arch()
    backend = "cpp" if arch == 90 else None
    func_tensor = torch.full(
        (1, 1, 1, q_len + 256),
        kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    result = create_block_mask_cuda.create_k2q_csr_sparse_auto(
        func_tensor,
        q_len,
        kv_len,
        headdim,
        is_arbitrary=True,
        backend=backend,
    )
    queried_tile = create_block_mask_cuda.get_bwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        backend=backend,
    )
    assert result[-2:] == queried_tile
    mask_cnt, mask_offset, _, full_cnt, full_offset, _, _, block_n = result
    assert mask_cnt.shape == full_cnt.shape == (
        1,
        1,
        math.ceil(kv_len / block_n),
    )
    assert mask_offset.numel() == full_offset.numel() == mask_cnt.numel() + 1


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize(
    "auto_name",
    ["create_q2k_csr_sparse_auto", "create_k2q_csr_sparse_auto"],
)
def test_sm90_auto_csr_backend_is_required(auto_name):
    if create_block_mask_cuda.get_gpu_arch() != 90:
        pytest.skip("SM90 auto backend validation requires an SM90 GPU")
    q_len, kv_len, headdim = 65, 97, 128
    func_tensor = torch.full(
        (1, 1, 1, q_len + 256),
        kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    auto = getattr(create_block_mask_cuda, auto_name)
    with pytest.raises(RuntimeError, match=r"backend is required for SM90"):
        auto(func_tensor, q_len, kv_len, headdim)


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("backend", ["cpp", "dsl"])
def test_sm90_q2k_auto_csr_forwards_backend(backend):
    if create_block_mask_cuda.get_gpu_arch() != 90:
        pytest.skip("SM90 auto backend dispatch requires an SM90 GPU")
    q_len, kv_len, headdim = 193, 257, 96
    func_tensor = torch.full(
        (1, 1, 1, q_len + 256),
        kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    result = create_block_mask_cuda.create_q2k_csr_sparse_auto(
        func_tensor,
        q_len,
        kv_len,
        headdim,
        is_arbitrary=True,
        check_q_boundary=True,
        backend=backend,
    )
    assert result[-2:] == create_block_mask_cuda.get_fwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=90,
        backend=backend,
    )


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("backend", ["cpp", "dsl"])
def test_sm90_k2q_auto_csr_forwards_backend(backend):
    if create_block_mask_cuda.get_gpu_arch() != 90:
        pytest.skip("SM90 auto backend dispatch requires an SM90 GPU")
    q_len, kv_len, headdim = 193, 257, 256
    func_tensor = torch.full(
        (1, 1, 1, q_len + 256),
        kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    result = create_block_mask_cuda.create_k2q_csr_sparse_auto(
        func_tensor,
        q_len,
        kv_len,
        headdim,
        is_arbitrary=True,
        headdim_v=headdim,
        backend=backend,
    )
    assert result[-2:] == create_block_mask_cuda.get_bwd_tile_sizes(
        headdim,
        is_arbitrary=True,
        arch=90,
        headdim_v=headdim,
        backend=backend,
    )
    mask_cnt, _, _, full_cnt, _, _, _, block_n = result
    assert mask_cnt.shape == full_cnt.shape == (
        1,
        1,
        math.ceil(kv_len / block_n),
    )


# =============================================================================
# Special Pattern Tests
# =============================================================================

def generate_special_func_tensor(seqlen_q, seqlen_k, pattern, n_func=3, block_size=128, device="cuda"):
    """
    Generate func_tensor for special attention patterns.

    Args:
        seqlen_q: query sequence length
        seqlen_k: key sequence length
        pattern: one of "empty", "full", "causal", "diagonal", "sliding_window"
        n_func: number of function values (must be odd: 1, 3, 5, ...)
        block_size: block size for diagonal pattern
        device: cuda device

    Returns:
        func_tensor: [1, 1, n_func, seqlen_q + 256]
    """
    B, H = 1, 1
    func_tensor = torch.zeros(B, H, n_func, seqlen_q + 256, dtype=torch.int32, device=device)
    q_indices = torch.arange(seqlen_q, dtype=torch.int32, device=device)

    if pattern == "empty":
        # All positions invalid: [0, 0) for all q_idx
        # F0 = 0 means first interval [0, 0) is empty
        func_tensor[:, :, 0, :seqlen_q] = 0

    elif pattern == "full":
        # All positions valid: [0, seqlen_k) for all q_idx
        func_tensor[:, :, 0, :seqlen_q] = seqlen_k

    elif pattern == "causal":
        # Causal mask: [0, q_idx + 1) for each q_idx
        func_tensor[:, :, 0, :seqlen_q] = torch.minimum(
            q_indices + 1,
            torch.full_like(q_indices, seqlen_k)
        )

    elif pattern == "anti_causal":
        # Anti-causal mask: [q_idx, seqlen_k) for each q_idx
        # Using n_func=3: F0=0 (first interval empty), F1=q_idx, F2=seqlen_k
        func_tensor[:, :, 0, :seqlen_q] = 0  # First interval [0, 0) is empty
        func_tensor[:, :, 1, :seqlen_q] = torch.minimum(q_indices, torch.full_like(q_indices, seqlen_k))
        func_tensor[:, :, 2, :seqlen_q] = seqlen_k

    elif pattern == "diagonal":
        # Block-level diagonal: each q_block only attends to corresponding kv_block
        # For q_idx in [block_start, block_end), valid kv range is [block_start, block_end)
        block_start = (q_indices // block_size) * block_size
        block_end = torch.minimum(
            block_start + block_size,
            torch.full_like(block_start, seqlen_k)
        )
        # Using n_func=3: F0=0 (first interval empty), F1=block_start, F2=block_end
        func_tensor[:, :, 0, :seqlen_q] = 0
        func_tensor[:, :, 1, :seqlen_q] = block_start
        func_tensor[:, :, 2, :seqlen_q] = block_end

    elif pattern == "sliding_window":
        # Sliding window with window_size = 2 * block_size
        window_size = 2 * block_size
        window_start = torch.maximum(
            q_indices - window_size // 2,
            torch.zeros_like(q_indices)
        )
        window_end = torch.minimum(
            q_indices + window_size // 2,
            torch.full_like(q_indices, seqlen_k)
        )
        # Using n_func=3: F0=0, F1=window_start, F2=window_end
        func_tensor[:, :, 0, :seqlen_q] = 0
        func_tensor[:, :, 1, :seqlen_q] = window_start
        func_tensor[:, :, 2, :seqlen_q] = window_end

    elif pattern == "checkerboard":
        # Checkerboard pattern: alternating blocks
        block_idx_q = q_indices // block_size
        for q_idx in range(seqlen_q):
            blk_q = q_idx // block_size
            # Even q_blocks attend to even kv_blocks, odd to odd
            if blk_q % 2 == 0:
                # Attend to even kv_blocks
                func_tensor[0, 0, 0, q_idx] = min(block_size, seqlen_k)  # First block
                if n_func >= 3:
                    func_tensor[0, 0, 1, q_idx] = min(2 * block_size, seqlen_k)
                    func_tensor[0, 0, 2, q_idx] = min(3 * block_size, seqlen_k)
            else:
                # Attend to odd kv_blocks
                func_tensor[0, 0, 0, q_idx] = 0  # First interval empty
                if n_func >= 3:
                    func_tensor[0, 0, 1, q_idx] = min(block_size, seqlen_k)
                    func_tensor[0, 0, 2, q_idx] = min(2 * block_size, seqlen_k)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return func_tensor.contiguous()


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("pattern", ["empty", "full", "causal", "anti_causal", "diagonal", "sliding_window"])
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 4096, 4111])
def test_special_patterns_q2k(pattern, seqlen):
    """Test Q2K kernel with special attention patterns."""
    print(f"\nTesting Q2K special pattern: {pattern}, seqlen={seqlen}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 3 if pattern in ["anti_causal", "diagonal", "sliding_window", "checkerboard"] else 1

    func_tensor = generate_special_func_tensor(seqlen, seqlen, pattern, n_func=n_func, block_size=KV_BLOCK_SIZE)

    # Reference
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (Q2K)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_q2k_kernel_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    # Print statistics
    num_blocks = (seqlen + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    total_mask = kernel_mask_cnt.sum().item()
    total_full = kernel_full_cnt.sum().item()
    total_blocks = num_blocks * num_blocks
    print(f"  Pattern: {pattern}, mask={total_mask}, full={total_full}, empty={total_blocks - total_mask - total_full}")

    assert mask_ok, f"Mask block mismatch for pattern={pattern}"
    assert full_ok, f"Full block mismatch for pattern={pattern}"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("pattern", ["empty", "full", "causal", "anti_causal", "diagonal", "sliding_window"])
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 4096, 4111])
def test_special_patterns_k2q(pattern, seqlen):
    """Test K2Q kernel with special attention patterns."""
    print(f"\nTesting K2Q special pattern: {pattern}, seqlen={seqlen}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 3 if pattern in ["anti_causal", "diagonal", "sliding_window", "checkerboard"] else 1

    func_tensor = generate_special_func_tensor(seqlen, seqlen, pattern, n_func=n_func, block_size=KV_BLOCK_SIZE)

    # Reference (K2Q)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_k2q_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (K2Q)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_k2q_kernel_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, f"Mask block mismatch for pattern={pattern}"
    assert full_ok, f"Full block mismatch for pattern={pattern}"
    print("  PASSED!")


# =============================================================================
# Boundary Condition Tests
# =============================================================================

@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen", [1, 2, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_small_seqlen_q2k(seqlen):
    """Test Q2K kernel with small sequence lengths (boundary conditions)."""
    print(f"\nTesting Q2K small seqlen: {seqlen}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 1

    # Use causal pattern for meaningful test
    func_tensor = generate_special_func_tensor(seqlen, seqlen, "causal", n_func=n_func)

    # Reference
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (Q2K)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_q2k_kernel_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, f"Mask block mismatch for seqlen={seqlen}"
    assert full_ok, f"Full block mismatch for seqlen={seqlen}"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen", [1, 2, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_small_seqlen_k2q(seqlen):
    """Test K2Q kernel with small sequence lengths (boundary conditions)."""
    print(f"\nTesting K2Q small seqlen: {seqlen}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 1

    # Use causal pattern for meaningful test
    func_tensor = generate_special_func_tensor(seqlen, seqlen, "causal", n_func=n_func)

    # Reference (K2Q)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_k2q_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (K2Q)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_k2q_kernel_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, f"Mask block mismatch for seqlen={seqlen}"
    assert full_ok, f"Full block mismatch for seqlen={seqlen}"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (128, 256),   # Q shorter than K
    (256, 128),   # Q longer than K
    (100, 300),   # Non-aligned, Q < K
    (300, 100),   # Non-aligned, Q > K
    (1, 1000),    # Single Q token
    (1000, 1),    # Single K token
    (127, 129),   # Just under/over block boundary
    (129, 127),   # Swapped
    (255, 257),   # 2x block boundary
    (257, 255),   # Swapped
])
def test_asymmetric_seqlen_q2k(seqlen_q, seqlen_k):
    """Test Q2K kernel with asymmetric Q/K lengths."""
    print(f"\nTesting Q2K asymmetric: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 1

    # Use full pattern to cover all cases
    func_tensor = generate_special_func_tensor(seqlen_q, seqlen_k, "full", n_func=n_func)

    # Reference
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (Q2K)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, f"Mask block mismatch for seqlen_q={seqlen_q}, seqlen_k={seqlen_k}"
    assert full_ok, f"Full block mismatch for seqlen_q={seqlen_q}, seqlen_k={seqlen_k}"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (128, 256),   # Q shorter than K
    (256, 128),   # Q longer than K
    (100, 300),   # Non-aligned, Q < K
    (300, 100),   # Non-aligned, Q > K
    (1, 1000),    # Single Q token
    (1000, 1),    # Single K token
    (127, 129),   # Just under/over block boundary
    (129, 127),   # Swapped
])
def test_asymmetric_seqlen_k2q(seqlen_q, seqlen_k):
    """Test K2Q kernel with asymmetric Q/K lengths."""
    print(f"\nTesting K2Q asymmetric: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    n_func = 1

    # Use full pattern
    func_tensor = generate_special_func_tensor(seqlen_q, seqlen_k, "full", n_func=n_func)

    # Reference (K2Q)
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_k2q_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (K2Q)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, f"Mask block mismatch for seqlen_q={seqlen_q}, seqlen_k={seqlen_k}"
    assert full_ok, f"Full block mismatch for seqlen_q={seqlen_q}, seqlen_k={seqlen_k}"
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (64, 64),
    (64, 128),
    (128, 64),
    (128, 256),
    (256, 128),
    (256, 256),
])
def test_different_block_sizes(Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test with different Q and KV block sizes."""
    seqlen = 1024
    n_func = 1
    print(f"\nTesting block sizes: Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    func_tensor = generate_special_func_tensor(seqlen, seqlen, "causal", n_func=n_func)

    # Reference
    ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
        create_reference_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    # Kernel (Q2K)
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, _ = \
        create_q2k_kernel_block_mask(func_tensor, seqlen, seqlen, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary=True)

    # Compare
    mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
    full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

    assert mask_ok, f"Mask block mismatch for Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}"
    assert full_ok, f"Full block mismatch for Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}"
    print("  PASSED!")


def create_reference_compact_block_idx(mask_block_cnt, full_block_cnt, block_idx):
    """
    Reference implementation of compact_block_idx using PyTorch.

    Converts BHQK format to linear sparse format (CSR-like).

    Args:
        mask_block_cnt: [B, H, num_blocks]
        full_block_cnt: [B, H, num_blocks]
        block_idx: [B, H, num_blocks, max_blocks]

    Returns:
        mask_block_cnt: [B, H, num_blocks] (unchanged)
        mask_block_offset: [B * H * num_blocks + 1] (exclusive prefix sum, starts with 0)
        mask_block_idx_compact: [total_mask]
        full_block_cnt: [B, H, num_blocks] (unchanged)
        full_block_offset: [B * H * num_blocks + 1]
        full_block_idx_compact: [total_full]
    """
    B, H, num_blocks, max_blocks = block_idx.shape
    n_blocks_flat = B * H * num_blocks
    offset_size = n_blocks_flat + 1

    # Flatten counts for cumsum
    mask_cnt_flat = mask_block_cnt.flatten()
    full_cnt_flat = full_block_cnt.flatten()

    # Compute offsets (exclusive prefix sum: [0, c0, c0+c1, ..., total])
    mask_block_offset = torch.zeros(offset_size, dtype=torch.int32, device=mask_block_cnt.device)
    full_block_offset = torch.zeros(offset_size, dtype=torch.int32, device=full_block_cnt.device)
    mask_block_offset[1:] = torch.cumsum(mask_cnt_flat, dim=0).to(torch.int32)
    full_block_offset[1:] = torch.cumsum(full_cnt_flat, dim=0).to(torch.int32)

    # Extract compact indices
    block_idx_flat = block_idx.view(n_blocks_flat, max_blocks)

    mask_block_idx_list = []
    full_block_idx_list = []

    for i in range(n_blocks_flat):
        fcnt = full_cnt_flat[i].item()
        mcnt = mask_cnt_flat[i].item()

        # Full indices: left-to-right
        if fcnt > 0:
            full_block_idx_list.append(block_idx_flat[i, :fcnt])

        # Mask indices: right-to-left, reversed
        if mcnt > 0:
            mask_indices = block_idx_flat[i, max_blocks - mcnt:max_blocks]
            mask_block_idx_list.append(mask_indices.flip(0))

    mask_block_idx_compact = torch.cat(mask_block_idx_list, dim=0) if mask_block_idx_list else torch.empty(0, dtype=torch.int32, device=mask_block_cnt.device)
    full_block_idx_compact = torch.cat(full_block_idx_list, dim=0) if full_block_idx_list else torch.empty(0, dtype=torch.int32, device=full_block_cnt.device)

    return (mask_block_cnt, mask_block_offset, mask_block_idx_compact,
            full_block_cnt, full_block_offset, full_block_idx_compact)


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8192, 8192),
    (16384, 16384),
    (16666, 16666),
    (32768, 32768),
    (33333, 33333),
    (61111, 61111),
])
@pytest.mark.parametrize("n_func", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
    (256, 128),
])
def test_compact_block_idx(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test compact_block_idx converts BHQK format to linear sparse format correctly."""
    print(f"\nTesting compact_block_idx: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Get kernel output (returns 3 tensors)
    mask_block_cnt, full_block_cnt, block_idx = \
        create_block_mask_cuda.create_q2k_block_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary=True, debug=True
        )

    # Compact using our function
    (kernel_mask_cnt, kernel_mask_offset, kernel_mask_idx,
    kernel_full_cnt, kernel_full_offset, kernel_full_idx) = \
        create_block_mask_cuda.compact_block_idx(
            mask_block_cnt, full_block_cnt, block_idx
        )


    # Reference compact
    (ref_mask_cnt, ref_mask_offset, ref_mask_idx,
     ref_full_cnt, ref_full_offset, ref_full_idx) = \
        create_reference_compact_block_idx(mask_block_cnt, full_block_cnt, block_idx)

    # Compare
    assert torch.equal(kernel_mask_cnt, ref_mask_cnt), "mask_cnt mismatch"
    assert torch.equal(kernel_full_cnt, ref_full_cnt), "full_cnt mismatch"
    assert torch.equal(kernel_mask_offset, ref_mask_offset), "mask_offset mismatch"
    assert torch.equal(kernel_full_offset, ref_full_offset), "full_offset mismatch"
    assert torch.equal(kernel_mask_idx, ref_mask_idx), f"mask_idx mismatch: kernel={kernel_mask_idx}, ref={ref_mask_idx}"
    assert torch.equal(kernel_full_idx, ref_full_idx), f"full_idx mismatch: kernel={kernel_full_idx}, ref={ref_full_idx}"

    # Print memory savings
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    original_size = num_q_blocks * num_kv_blocks * 4  # bytes (int32)
    compact_size = (len(kernel_mask_idx) + len(kernel_full_idx)) * 4
    print(f"  Memory: {original_size} -> {compact_size} bytes ({100*compact_size/original_size:.1f}%)")
    print("  PASSED!")


# =============================================================================
# Tests for CSR Sparse Format APIs (combined create + compact)
# =============================================================================

@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8192, 8192),
    (16384, 16384),
    (16666, 16666),
])
@pytest.mark.parametrize("n_func", [1, 3, 5])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
    (256, 128),
])
def test_q2k_csr_sparse_from_func(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test create_q2k_csr_sparse_from_func returns same result as separate calls."""
    print(f"\nTesting Q2K CSR sparse: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Method 1: Use the combined CSR API
    (csr_mask_cnt, csr_mask_offset, csr_mask_idx,
     csr_full_cnt, csr_full_offset, csr_full_idx) = \
        create_block_mask_cuda.create_q2k_csr_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary=True
        )

    # Method 2: Use separate calls
    mask_block_cnt, full_block_cnt, block_idx = \
        create_block_mask_cuda.create_q2k_block_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary=True, debug=False
        )

    (sep_mask_cnt, sep_mask_offset, sep_mask_idx,
     sep_full_cnt, sep_full_offset, sep_full_idx) = \
        create_block_mask_cuda.compact_block_idx(
            mask_block_cnt, full_block_cnt, block_idx
        )

    # Compare results
    assert torch.equal(csr_mask_cnt, sep_mask_cnt), "mask_cnt mismatch"
    assert torch.equal(csr_full_cnt, sep_full_cnt), "full_cnt mismatch"
    assert torch.equal(csr_mask_offset, sep_mask_offset), "mask_offset mismatch"
    assert torch.equal(csr_full_offset, sep_full_offset), "full_offset mismatch"
    assert torch.equal(csr_mask_idx, sep_mask_idx), "mask_idx mismatch"
    assert torch.equal(csr_full_idx, sep_full_idx), "full_idx mismatch"

    # Also verify with reference
    (ref_mask_cnt, ref_mask_offset, ref_mask_idx,
     ref_full_cnt, ref_full_offset, ref_full_idx) = \
        create_reference_compact_block_idx(mask_block_cnt, full_block_cnt, block_idx)

    assert torch.equal(csr_mask_offset, ref_mask_offset), "mask_offset vs ref mismatch"
    assert torch.equal(csr_full_offset, ref_full_offset), "full_offset vs ref mismatch"
    assert torch.equal(csr_mask_idx, ref_mask_idx), "mask_idx vs ref mismatch"
    assert torch.equal(csr_full_idx, ref_full_idx), "full_idx vs ref mismatch"

    print(f"  Total mask blocks: {len(csr_mask_idx)}")
    print(f"  Total full blocks: {len(csr_full_idx)}")
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (4096, 4096),
    (4111, 4111),
    (8192, 8192),
    (16384, 16384),
    (16666, 16666),
])
@pytest.mark.parametrize("n_func", [1, 3, 5])
@pytest.mark.parametrize("Q_BLOCK_SIZE,KV_BLOCK_SIZE", [
    (128, 128),
])
def test_k2q_csr_sparse_from_func(seqlen_q, seqlen_k, n_func, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    """Test create_k2q_csr_sparse_from_func returns same result as separate calls."""
    print(f"\nTesting K2Q CSR sparse: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"n_func={n_func}, Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Method 1: Use the combined CSR API
    (csr_mask_cnt, csr_mask_offset, csr_mask_idx,
     csr_full_cnt, csr_full_offset, csr_full_idx) = \
        create_block_mask_cuda.create_k2q_csr_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE
        )

    # Method 2: Use separate calls
    mask_block_cnt, full_block_cnt, block_idx = \
        create_block_mask_cuda.create_k2q_block_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            debug=False
        )

    (sep_mask_cnt, sep_mask_offset, sep_mask_idx,
     sep_full_cnt, sep_full_offset, sep_full_idx) = \
        create_block_mask_cuda.compact_block_idx(
            mask_block_cnt, full_block_cnt, block_idx
        )

    # Compare results
    assert torch.equal(csr_mask_cnt, sep_mask_cnt), "mask_cnt mismatch"
    assert torch.equal(csr_full_cnt, sep_full_cnt), "full_cnt mismatch"
    assert torch.equal(csr_mask_offset, sep_mask_offset), "mask_offset mismatch"
    assert torch.equal(csr_full_offset, sep_full_offset), "full_offset mismatch"
    assert torch.equal(csr_mask_idx, sep_mask_idx), "mask_idx mismatch"
    assert torch.equal(csr_full_idx, sep_full_idx), "full_idx mismatch"

    # Also verify with reference
    (ref_mask_cnt, ref_mask_offset, ref_mask_idx,
     ref_full_cnt, ref_full_offset, ref_full_idx) = \
        create_reference_compact_block_idx(mask_block_cnt, full_block_cnt, block_idx)

    assert torch.equal(csr_mask_offset, ref_mask_offset), "mask_offset vs ref mismatch"
    assert torch.equal(csr_full_offset, ref_full_offset), "full_offset vs ref mismatch"
    assert torch.equal(csr_mask_idx, ref_mask_idx), "mask_idx vs ref mismatch"
    assert torch.equal(csr_full_idx, ref_full_idx), "full_idx vs ref mismatch"

    print(f"  Total mask blocks: {len(csr_mask_idx)}")
    print(f"  Total full blocks: {len(csr_full_idx)}")
    print("  PASSED!")


@pytest.mark.skipif(create_block_mask_cuda is None, reason="CUDA kernel not built")
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (2048, 4096),   # seqlen_q < seqlen_k
    (4096, 2048),   # seqlen_q > seqlen_k
    (3333, 5555),   # non-divisible, q < k
    (5555, 3333),   # non-divisible, q > k
])
@pytest.mark.parametrize("n_func", [1, 3])
def test_csr_sparse_asymmetric_seqlen(seqlen_q, seqlen_k, n_func):
    """Test CSR sparse APIs with asymmetric sequence lengths."""
    print(f"\nTesting CSR sparse asymmetric: seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, n_func={n_func}")

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = 128, 128
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Test Q2K CSR
    (q2k_mask_cnt, q2k_mask_offset, q2k_mask_idx,
     q2k_full_cnt, q2k_full_offset, q2k_full_idx) = \
        create_block_mask_cuda.create_q2k_csr_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary=True
        )

    # Test K2Q CSR
    (k2q_mask_cnt, k2q_mask_offset, k2q_mask_idx,
     k2q_full_cnt, k2q_full_offset, k2q_full_idx) = \
        create_block_mask_cuda.create_k2q_csr_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE
        )

    # Verify shapes
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    # Q2K: cnt shape is [B, H, num_q_blocks]
    assert q2k_mask_cnt.shape == (1, 1, num_q_blocks), f"Q2K mask_cnt shape mismatch: {q2k_mask_cnt.shape}"
    assert q2k_full_cnt.shape == (1, 1, num_q_blocks), f"Q2K full_cnt shape mismatch: {q2k_full_cnt.shape}"

    # K2Q: cnt shape is [B, H, num_kv_blocks]
    assert k2q_mask_cnt.shape == (1, 1, num_kv_blocks), f"K2Q mask_cnt shape mismatch: {k2q_mask_cnt.shape}"
    assert k2q_full_cnt.shape == (1, 1, num_kv_blocks), f"K2Q full_cnt shape mismatch: {k2q_full_cnt.shape}"

    print(f"  Q2K: {len(q2k_mask_idx)} mask + {len(q2k_full_idx)} full blocks")
    print(f"  K2Q: {len(k2q_mask_idx)} mask + {len(k2q_full_idx)} full blocks")
    print("  PASSED!")


def run_q2k_csr_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """Run a single Q2K CSR test for debugging."""
    print(f"Running Q2K CSR test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        return False

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Use CSR API
    (mask_cnt, mask_offset, mask_idx,
     full_cnt, full_offset, full_idx) = \
        create_block_mask_cuda.create_q2k_csr_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary=True
        )

    print(f"\nCSR Output:")
    print(f"  mask_cnt shape: {mask_cnt.shape}")
    print(f"  mask_offset shape: {mask_offset.shape}")
    print(f"  mask_idx shape: {mask_idx.shape}")
    print(f"  full_cnt shape: {full_cnt.shape}")
    print(f"  full_offset shape: {full_offset.shape}")
    print(f"  full_idx shape: {full_idx.shape}")

    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_blocks = num_q_blocks * num_kv_blocks

    print(f"\nStatistics:")
    print(f"  Total possible blocks: {total_blocks}")
    print(f"  Mask blocks: {len(mask_idx)} ({100*len(mask_idx)/total_blocks:.1f}%)")
    print(f"  Full blocks: {len(full_idx)} ({100*len(full_idx)/total_blocks:.1f}%)")
    print(f"  Empty blocks: {total_blocks - len(mask_idx) - len(full_idx)}")

    print("\nPASS Q2K CSR test executed successfully!")
    return True


def run_k2q_csr_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """Run a single K2Q CSR test for debugging."""
    print(f"Running K2Q CSR test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        return False

    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Use CSR API
    (mask_cnt, mask_offset, mask_idx,
     full_cnt, full_offset, full_idx) = \
        create_block_mask_cuda.create_k2q_csr_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE
        )

    print(f"\nCSR Output:")
    print(f"  mask_cnt shape: {mask_cnt.shape}")
    print(f"  mask_offset shape: {mask_offset.shape}")
    print(f"  mask_idx shape: {mask_idx.shape}")
    print(f"  full_cnt shape: {full_cnt.shape}")
    print(f"  full_offset shape: {full_offset.shape}")
    print(f"  full_idx shape: {full_idx.shape}")

    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_blocks = num_q_blocks * num_kv_blocks

    print(f"\nStatistics:")
    print(f"  Total possible blocks: {total_blocks}")
    print(f"  Mask blocks: {len(mask_idx)} ({100*len(mask_idx)/total_blocks:.1f}%)")
    print(f"  Full blocks: {len(full_idx)} ({100*len(full_idx)/total_blocks:.1f}%)")
    print(f"  Empty blocks: {total_blocks - len(mask_idx) - len(full_idx)}")

    print("\nPASS K2Q CSR test executed successfully!")
    return True


def run_q2k_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128, check_q_boundary=True):
    """
    Run a single Q2K (forward) test for debugging.
    """
    print(f"Running Q2K test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    print(f"  check_q_boundary={check_q_boundary}")

    # Generate random func_tensor
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    print(f"func_tensor: {func_tensor}")

    print(f"\nFunc tensor shape: {func_tensor.shape}")

    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        print("Please build with: pip install -e .")
        return False

    # Kernel
    print("\nComputing Q2K kernel...")
    kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, block_idx = \
        create_q2k_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE, check_q_boundary)

    if check_q_boundary:
        # Compare with PyTorch reference (FlexAttention mode)
        print("\nComputing PyTorch reference (check_q_boundary=True)...")
        ref_mask_cnt, ref_mask_idx, ref_full_cnt, ref_full_idx = \
            create_reference_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

        # Compare
        print("\nComparing results...")
        mask_ok = compare_block_masks(ref_mask_cnt, ref_mask_idx, kernel_mask_cnt, kernel_mask_idx, "mask_block")
        full_ok = compare_block_masks(ref_full_cnt, ref_full_idx, kernel_full_cnt, kernel_full_idx, "full_block")

        if mask_ok and full_ok:
            print("\nPASS Q2K test passed!")
        else:
            print("\nFAIL Q2K test failed!")
    else:
        mask_ok = True
        full_ok = True

    # Print some statistics
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_mask_blocks = kernel_mask_cnt.sum().item()
    total_full_blocks = kernel_full_cnt.sum().item()
    total_blocks = num_q_blocks * num_kv_blocks

    print(f"\nQ2K Statistics:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Mask blocks: {total_mask_blocks} ({100*total_mask_blocks/total_blocks:.1f}%)")
    print(f"  Full blocks: {total_full_blocks} ({100*total_full_blocks/total_blocks:.1f}%)")
    print(f"  Empty blocks: {total_blocks - total_mask_blocks - total_full_blocks}")

    return mask_ok and full_ok


def run_k2q_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """
    Run a single K2Q (backward) test for debugging.
    """
    print(f"Running K2Q test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    # Generate random func_tensor
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    print(f"\nFunc tensor shape: {func_tensor.shape}")

    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        print("Please build with: pip install -e .")
        return False

    # Kernel
    print("\nComputing K2Q kernel...")
    with torch.cuda.nvtx.range("create_k2q_kernel_block_mask"):
        kernel_mask_cnt, kernel_mask_idx, kernel_full_cnt, kernel_full_idx, block_idx = \
            create_k2q_kernel_block_mask(func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE)

    print(f"K2Q mask_block_cnt shape: {kernel_mask_cnt.shape}")
    print(f"K2Q full_block_cnt shape: {kernel_full_cnt.shape}")

    # Print some statistics
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    total_mask_blocks = kernel_mask_cnt.sum().item()
    total_full_blocks = kernel_full_cnt.sum().item()
    total_blocks = num_q_blocks * num_kv_blocks

    print(f"\nK2Q Statistics:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Mask blocks: {total_mask_blocks} ({100*total_mask_blocks/total_blocks:.1f}%)")
    print(f"  Full blocks: {total_full_blocks} ({100*total_full_blocks/total_blocks:.1f}%)")
    print(f"  Empty blocks: {total_blocks - total_mask_blocks - total_full_blocks}")

    print("\nPASS K2Q kernel executed successfully!")
    return True


def run_compact_test(seqlen_q=1024, seqlen_k=1024, n_func=1, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128):
    """
    Run a single compact_block_idx test for debugging.
    """
    print(f"Running compact_block_idx test:")
    print(f"  seqlen_q={seqlen_q}, seqlen_k={seqlen_k}")
    print(f"  n_func={n_func}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")

    if create_block_mask_cuda is None:
        print("\nCUDA kernel not built. Skipping kernel test.")
        print("Please build with: pip install -e .")
        return False

    # Generate random func_tensor
    func_tensor = generate_func_tensor(seqlen_q, seqlen_k, n_func)

    # Get kernel output (returns 3 tensors)
    mask_block_cnt, full_block_cnt, block_idx = \
        create_block_mask_cuda.create_q2k_block_sparse_from_func(
            func_tensor, seqlen_q, seqlen_k, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
            check_q_boundary=True, debug=True
        )

    print(f"\nOriginal tensors:")
    print(f"  mask_block_cnt shape: {mask_block_cnt.shape}")
    print(f"  full_block_cnt shape: {full_block_cnt.shape}")
    print(f"  block_idx shape: {block_idx.shape}")

    # Compact using our function
    with torch.cuda.nvtx.range("compact_block_idx"):
        (kernel_mask_cnt, kernel_mask_offset, kernel_mask_idx,
        kernel_full_cnt, kernel_full_offset, kernel_full_idx) = \
            create_block_mask_cuda.compact_block_idx(
                mask_block_cnt, full_block_cnt, block_idx
            )

    print(f"\nCompact tensors:")
    print(f"  mask_block_cnt shape: {kernel_mask_cnt.shape}")
    print(f"  mask_block_offset shape: {kernel_mask_offset.shape}")
    print(f"  mask_block_idx shape: {kernel_mask_idx.shape}")
    print(f"  full_block_cnt shape: {kernel_full_cnt.shape}")
    print(f"  full_block_offset shape: {kernel_full_offset.shape}")
    print(f"  full_block_idx shape: {kernel_full_idx.shape}")

    # Reference compact
    (ref_mask_cnt, ref_mask_offset, ref_mask_idx,
     ref_full_cnt, ref_full_offset, ref_full_idx) = \
        create_reference_compact_block_idx(mask_block_cnt, full_block_cnt, block_idx)

    # Compare
    all_ok = True
    if not torch.equal(kernel_mask_cnt, ref_mask_cnt):
        print("FAIL mask_cnt mismatch")
        all_ok = False
    if not torch.equal(kernel_full_cnt, ref_full_cnt):
        print("FAIL full_cnt mismatch")
        all_ok = False
    if not torch.equal(kernel_mask_offset, ref_mask_offset):
        print("FAIL mask_offset mismatch")
        all_ok = False
    if not torch.equal(kernel_full_offset, ref_full_offset):
        print("FAIL full_offset mismatch")
        all_ok = False
    if not torch.equal(kernel_mask_idx, ref_mask_idx):
        print("FAIL mask_idx mismatch")
        all_ok = False
    if not torch.equal(kernel_full_idx, ref_full_idx):
        print("FAIL full_idx mismatch")
        all_ok = False

    # Print memory savings
    num_q_blocks = (seqlen_q + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    original_size = num_q_blocks * num_kv_blocks * 4  # bytes (int32)
    compact_size = (len(kernel_mask_idx) + len(kernel_full_idx)) * 4

    print(f"\nMemory savings:")
    print(f"  Original block_idx: {original_size:,} bytes")
    print(f"  Compact indices: {compact_size:,} bytes")
    print(f"  Savings: {100*(1-compact_size/original_size):.1f}%")

    if all_ok:
        print("\nPASS compact_block_idx test passed!")
    else:
        print("\nFAIL compact_block_idx test failed!")

    return all_ok


if __name__ == "__main__":
    # Run Q2K tests
    print("=" * 60)
    print("Test 1: Q2K (Forward) with check_q_boundary=True")
    print("=" * 60)
    run_q2k_test(seqlen_q=40480, seqlen_k=40480, n_func=3, Q_BLOCK_SIZE=256, KV_BLOCK_SIZE=128, check_q_boundary=True)

    print("\n" + "=" * 60)
    print("Test 2: Q2K Non-divisible seqlen")
    print("=" * 60)
    run_q2k_test(seqlen_q=21111, seqlen_k=41111, n_func=3, Q_BLOCK_SIZE=256, KV_BLOCK_SIZE=128, check_q_boundary=True)

    # Run K2Q tests
    print("\n" + "=" * 60)
    print("Test 3: K2Q (Backward)")
    print("=" * 60)
    run_k2q_test(seqlen_q=20480, seqlen_k=40480, n_func=3, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128)

    print("\n" + "=" * 60)
    print("Test 4: K2Q Non-divisible seqlen")
    print("=" * 60)
    run_k2q_test(seqlen_q=21111, seqlen_k=41111, n_func=3, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128)

    # Test compact_block_idx
    print("\n" + "=" * 60)
    print("Test 5: compact_block_idx")
    print("=" * 60)
    run_compact_test(seqlen_q=40480, seqlen_k=40480, n_func=3, Q_BLOCK_SIZE=256, KV_BLOCK_SIZE=128)

    # Test CSR APIs
    print("\n" + "=" * 60)
    print("Test 6: Q2K CSR Sparse")
    print("=" * 60)
    run_q2k_csr_test(seqlen_q=40480, seqlen_k=40480, n_func=3, Q_BLOCK_SIZE=256, KV_BLOCK_SIZE=128)

    print("\n" + "=" * 60)
    print("Test 7: K2Q CSR Sparse")
    print("=" * 60)
    run_k2q_csr_test(seqlen_q=40480, seqlen_k=40480, n_func=3, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128)
