# create_block_mask

CUDA extension for converting arbitrary interval-mask metadata into compact block-sparse CSR tensors.

Build from the repository root:

```bash
make create_block_mask
```

or from `csrc/utils`:

```bash
make block_mask
```

or directly from this directory:

```bash
cd csrc/utils/create_block_mask
pip install --no-user -e . --no-build-isolation
```

Run the module tests:

```bash
make test_create_block_mask
```

Run the FA4 arbitrary-mask CSR integration tests:

```bash
make test_arbitrary_mask_csr
```

Forward uses Q2K metadata and backward uses K2Q metadata:

```python
import create_block_mask_cuda
from flash_attn_cute import flash_attn_func
from flash_attn_cute.block_sparsity import LinearBlockSparseTensorsTorch

q2k = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
    arbitrary_func, seqlen_q, seqlen_k, q_block_size, kv_block_size
)
k2q = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
    arbitrary_func, seqlen_q, seqlen_k, q_block_size_bwd, kv_block_size_bwd
)

linear_k = LinearBlockSparseTensorsTorch(*q2k, block_size=(q_block_size, kv_block_size))
linear_q = LinearBlockSparseTensorsTorch(
    *k2q, block_size=(q_block_size_bwd, kv_block_size_bwd)
)

out, lse = flash_attn_func(
    q,
    k,
    v,
    arbitrary=True,
    aux_tensors=[arbitrary_func],
    linear_k_block_sparse_tensors=linear_k,
    linear_q_block_sparse_tensors=linear_q,
    return_lse=True,
)
```

The CSR tuple order is:

```python
(
    mask_block_cnt,
    mask_block_offset,
    mask_block_idx,
    full_block_cnt,
    full_block_offset,
    full_block_idx,
)
```

Use `create_q2k_csr_sparse_auto` and `create_k2q_csr_sparse_auto` to get the
selected tile sizes back with the CSR tensors. Pass `headdim_v` to the backward
helper when it differs from `headdim`; the default SM100/SM110 2-CTA linear CSR
paths for `128/128` and `192/128` use K2Q blocks of `128x256`.

All public APIs that query tile sizes accept an optional `backend` argument:

```text
get_fwd_tile_sizes(..., backend=None)
get_bwd_tile_sizes(..., backend=None)
create_q2k_csr_sparse_auto(..., backend=None)
create_k2q_csr_sparse_auto(..., backend=None)
```

On SM90, both the Hopper C++ and CuTeDSL implementations are available, so
`backend` is required and must be either `"cpp"` or `"dsl"`. On SM8x the
backend defaults to C++, and on SM100+ it defaults to DSL.

```python
fwd_block_size = create_block_mask_cuda.get_fwd_tile_sizes(
    headdim,
    is_arbitrary=True,
    backend="cpp",  # Hopper C++ on SM90
)
bwd_block_size = create_block_mask_cuda.get_bwd_tile_sizes(
    headdim,
    is_arbitrary=True,
    headdim_v=headdim_v,
    backend="cpp",
)
```
