# Flex Attention

本项目提供基于 NVIDIA CuTe DSL 的训练态 FlexAttention backend。它从
FlashAttention CuTe DSL 的 SM90/SM100 实现演化而来，只保留 interval mask plan
与 packed predicate mask 路径。

## 支持范围

- GPU：SM90、SM100、SM103；
- 布局：定长 BSHD、真实变长 THD；
- 数据类型：FP16、BF16；
- 模式：forward、backward、MHA、GQA、MQA；
- head 维度：SM100/SM103 generic 支持 Dqk、Dv 独立取
  `{8,16,...,128}`，另支持 `(192,128)`；`(256,256)` 使用专用 kernel；
- mask：`[0,F0), [F1,F2), [F3,F4), ...` interval union。

不提供 score-mod、mask-mod callable、paged KV、SplitKV、MLA、FP8、SM80 或
SM120 路径。

## 安装

```bash
python -m pip install -e '.[dev]'
```

项目固定使用 `nvidia-cutlass-dsl==4.5.2` 和 `quack-kernels==0.5.0`。

## 接口

```python
from flex_attn import create_mask_plan, flex_attn_func

# q: [B, Sq, Hq, Dqk]
# k: [B, Sk, Hkv, Dqk]
# v: [B, Sk, Hkv, Dv]
# mask_func: contiguous CUDA int32 [Hmask, nfunc, B * Sq]
plan = create_mask_plan(mask_func, q, k, v)
out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
```

`mask_func` endpoint 是样本内 local-K 坐标，公开 tensor 不需要 planner padding。
`MaskPlan` 构建后不保留 `mask_func`，并自行持有变长 prefix 的副本。
`return_lse=False` 时接口返回 `out`；设为 `True` 时返回 `(out, lse)`。

变长 geometry 只在构建 plan 时提供：

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

## 测试

```bash
PYTHONPATH=src:. python -m pytest tests/unit
PYTHONPATH=src:. python -m pytest tests/gpu --run-gpu
PYTHONPATH=src:. python -m pytest tests/gpu --run-gpu --full-random-cases
```

默认 GPU smoke 包含原 144 个随机 strata 代表和 10 个 SM100 head-dim directed
case，共 154 个。`--full-random-cases` 仍运行固定 seed 生成的原 1024 个 case。

## 来源与许可证

核心 kernel 来源于 FlashAttention CuTe DSL，并保留原作者版权声明。详细迁移边界
见 [docs/design.md](docs/design.md) 和 [NOTICE](NOTICE)。项目使用 BSD 3-Clause
License。
