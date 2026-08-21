# Flex Attention

FlexAttention provides high-performance arbitrary-mask attention through interval mask plans
and packed predicate masks.

## Supported Configurations

- GPUs: SM90, SM100, and SM103.
- Layouts: fixed-length BSHD and true-variable-length THD.
- Data types: FP16 and BF16.
- Modes: forward, backward, MHA, GQA, and MQA.
- Head dimensions: the SM100/SM103 generic kernels independently support `Dqk` and `Dv`
  from `{8,16,...,128}`; `(192,128)` is also supported, while `(256,256)` uses a dedicated
  kernel.
- Masks: unions of intervals in the form `[0,F0), [F1,F2), [F3,F4), ...`.

Paged KV, SplitKV, MLA, FP8, SM80, and SM120 paths are not currently supported.

## Installation

```bash
python -m pip install -e '.[dev]'
```

The project requires `nvidia-cutlass-dsl>=4.5.2` and pins `quack-kernels==0.5.0`.
Release compatibility is validated with NVIDIA CuTe DSL 4.7.0.

## API

```python
from flex_attn import create_mask_plan, flex_attn_func

# q: [B, Sq, Hq, Dqk]
# k: [B, Sk, Hkv, Dqk]
# v: [B, Sk, Hkv, Dv]
# mask_func: contiguous CUDA int32 [Hmask, nfunc, B * Sq]
plan = create_mask_plan(mask_func, q, k, v)
out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
```

Each `mask_func` endpoint uses sample-local K coordinates. The public tensor does not require
planner padding. After construction, `MaskPlan` does not retain `mask_func`; for variable-length
inputs, it owns copies of the sequence-prefix tensors.

With `return_lse=False`, the API returns `out`. With `return_lse=True`, it returns `(out, lse)`.

Variable-length geometry is provided only when constructing the plan:

```python
from flex_attn import create_mask_plan, flex_attn_varlen_func

plan = create_mask_plan(
    mask_func,
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
out, lse = flex_attn_varlen_func(q, k, v, mask_plan=plan, return_lse=True)
```

## Design Documentation

- [Arbitrary Mask Design (English)](docs/design.md)
- [Arbitrary Mask Design (Simplified Chinese)](docs/design_zh.md)

## Testing

```bash
PYTHONPATH=src:. python -m pytest tests/unit
PYTHONPATH=src:. python -m pytest tests/gpu --run-gpu
PYTHONPATH=src:. python -m pytest tests/gpu --run-gpu --full-random-cases
```

The default GPU smoke suite contains the original 144 randomly stratified representatives plus
10 directed SM100 head-dimension cases, for a total of 154 cases. `--full-random-cases` runs all
1,024 cases generated with the fixed seed.

## Standard Static-Mask Benchmark

### Benchmark Cases

![Static attention mask shapes](docs/assets/static_mask_shapes.png)

### Benchmark Results

![Static-mask attention performance on NVIDIA GB300](docs/assets/static_mask_benchmark.png)

## Attribution and License

Thanks to FlashAttention. The core implementation is derived from the FlashAttention CuTe DSL
code and retains the original authors' copyright notices. See [NOTICE](NOTICE) for attribution
details. This project is licensed under the BSD 3-Clause License.
