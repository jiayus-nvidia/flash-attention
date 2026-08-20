# Flex Attention backend 设计

## 范围

本项目是训练态 FlexAttention backend，只保留 interval mask plan、packed predicate
mask、普通 QKV attention forward/backward，以及 fixed BSHD 和 true-varlen THD 两种
数据路径。

不提供 score-mod、mask-mod callable、paged KV、SplitKV、MLA、FP8、SM80、SM120
或 D512。SM90 保留 Hopper 实现；SM100 与 SM103 共用 Blackwell consumer family。

## Mask 语义

公开 `mask_func` 是 contiguous CUDA `int32` tensor，形状为
`[Hmask,nfunc,total_q]`。`nfunc` 为奇数，每一行表示以下 sample-local KV interval
并集：

```text
[0, F0), [F1, F2), [F3, F4), ...
```

endpoint 必须位于当前 sample 的 local-K 范围内并保持非递减。builder 内部把 endpoint
转换为 planner 坐标并添加 256 行安全 padding；临时表示不进入公开 `MaskPlan`。

## Plan 构建

```text
sample-local mask_func
        |
        v
输入验证、坐标转换和 padding
        |
        v
Q2K classify -> empty / partial / full
        |
        +--> partial/full count -> exclusive offsets
        |
        v
Q2K materialize -> raw CSR indices + packed partial payload
        |
        +--> K2Q count/materialize（仅 backward）
        |
        +--> SM100 forward schedule preprocess
        |
        v
MaskPlan
```

Q2K 与 K2Q 均使用两组相互独立的 raw CSR：

- partial block：`mask_block_offset`、`mask_block_idx`；
- full block：`full_block_offset`、`full_block_idx`。

consumer 不假设 block index 连续，不生成 full-block run，也不动态合并 partial/full
两组索引。只有 partial block 携带 `mask_block_masks`；full block 不加载 mask payload。

变长 plan 克隆并持有 `cu_seqlens_q/k`。attention 调用使用 plan 持有的 prefix，调用方
后续原地修改输入 prefix 不会改变 plan。plan runtime binding 同时记录 tensor identity、
version 和 geometry，防止错误复用。

## SM100 forward schedule

SM100/SM103 的所有 forward 均使用 Blackwell 硬件 CLC persistent scheduler。fixed、
true-varlen、generic、D192 和 D256 都不得静默回退 static；无法构造合法 CLC launch
时必须显式报错。

plan preprocess 生成：

```text
sequence_desc[B,8]（true-varlen 与 D256 fixed 使用）
  0 q_offset
  1 k_offset
  2 q_len
  3 k_len
  4 q_plan_row_begin
  5 q_plan_row_count
  6 num_k_blocks
  7 reserved

fwd_work_desc[T,4]
  0 m_block
  1 head_idx
  2 batch_idx
  3 q_valid_rows
```

`fwd_work_desc` 让一个 CLC work item 精确对应一个有效 Q plan row 和一个调度 head。
true-varlen main kernel 从 descriptor 取得 sample offset、长度和尾 tile 边界，不重新
枚举 sample task。

task cost 定义为当前 plan row 的 `partial_count + full_count`，即 full 与 partial
block 等价。任务先按 sample/KV-head 的 L2 section 分组，再执行稳定 LPT 排序；零
cost task 保留在队尾，由 forward 写出 `O=0`、`LSE=-inf`。

## SM100 forward kernel

支持范围：

```text
generic:   Dqk,Dv 独立取 {8,16,...,128}
generic:   (Dqk,Dv) = (192,128)
dedicated: (Dqk,Dv) = (256,256)
```

generic kernel 的静态 topology 为：

```text
默认  (q_stage=2, cta_group_size=1)
可选  (q_stage=1, cta_group_size=1)
可选  (q_stage=1, cta_group_size=2)
```

generic forward 按 pipeline 拆分，同时只保留一个公共 kernel 外壳：

```text
sm100/fwd/
├── forward.py             # launch、warp roles、公共 softmax/epilogue/helpers
├── forward_config.py      # generic compile-time 配置
├── forward_qstage1.py     # qstage1 load/MMA/correction，统一 1CTA/2CTA 类
├── forward_qstage2.py     # qstage2 load/MMA/correction
├── forward_hd256.py       # D256 专用实现
├── forward_config_hd256.py
└── named_barrier.py
```

qstage1 的 1CTA/2CTA 不使用两个 wrapper class；同一个
`FlexAttentionForwardQStage1Sm100` 由 compile key 中的静态
`use_2cta_instrs` 参数化。该拆分只改变代码归属，不改变 pipeline、barrier、TMEM、
CLC 或 mask plan 协议。

默认 plan 始终选择 `(2,1)`。qstage1 两个候选必须通过内部 plan variant 显式构建，
不会根据 causal/local、sequence length 或具体 mask shape 自动选择。qstage1 只允许
使用当前 N-direction pipeline；旧的单流 qstage1 pipeline 不可构造。

D256 使用独立实现，固定 `q_stage=1`，维护 1CTA/2CTA 两种 compile-key 变体；默认
为 1CTA。generic 与 D256 都先遍历 partial CSR 并应用 packed mask，再遍历 full CSR
且不应用 mask：

```text
for partial block:
    softmax_step(apply_mask=True)

for full block:
    softmax_step(apply_mask=False)
```

fixed 输出使用 TMA O store；true-varlen 使用带完整尾维保护的 register-to-GMEM
路径。变长禁止复用 fixed fast path。

## SM100 backward kernel

generic backward 负责标准 head dim 与 `(192,128)`，D256 使用独立 dQ、dKdV kernel。
backward 保持各 kernel 既有调度，不套用 forward 的统一 CLC 规则。

K2Q plan 让 dKdV 以 KV block 为 task 直接遍历关联 Q block。所有 backward consumer
都分别读取 `mask_block_idx` 与 `full_block_idx`；singleton-heavy mask 不存在额外 raw
index 旁路。

需要 head padding 或 head reduction 的 dK/dV 写入 FP32 workspace，再由公共
postprocess 按真实 head dim 裁剪并转换输出。fixed 无 padding MHA 可以使用既有 TMA
direct-store；true-varlen 保持 ragged 输出路径。

## 精度

QK、PV、dP、dQ、dK、dV 的 Tensor Core accumulator 均为 FP32。softmax、scale、
归约及 CUDA Core 特殊函数均在 FP32 上执行，最终再显式转换到输出 dtype。

并行调度导致的合法 FP32 累加顺序变化不要求 bitwise-identical；正确性以 CUDA eager
FP32 reference 为 gate，低精度 reordered PyTorch 结果只用于误差基准。

## Compile key

compile key 只包含影响 codegen 的静态配置，例如 architecture、dtype、head dim、
layout、kernel topology、plan signature、payload layout 和是否 true-varlen。

以下运行时值不得直接进入 compile key：batch size、`total_q`、`total_k`、
`max_seqlen_q`、`max_seqlen_k`、具体 `cu_seqlens` 内容、task 数和 sparse nnz。
qstage1 1CTA 可以消费 plan 归约得到的 `narrow_workset` 布尔提示，选择同一 kernel
内部的静态 overlap 策略；它不根据 causal/local 名称分派，也不改变 topology。

## 测试

默认 GPU smoke 包含 144 个随机 strata 代表和 10 个 SM100 directed head-dim case，
共 154 个。`--full-random-cases` 运行固定 seed 生成的 1024 个随机 case。fixed 与
true-varlen 共用 case 规范，测试覆盖 forward、LSE、dQ、dK、dV、MHA/GQA/MQA、
空 mask、full mask、discontiguous full block、deterministic 和三个 generic forward
topology。

## TODO

- SM100/SM103 D256 fixed-causal forward 在 GB300、BF16、B=8、Sq=Sk=8192、
  Hq/Hkv=16/4 下为 `3360.212 us`，FA4 为 `2834.209 us`，ratio `1.1856`。
  当前正确性与统一 CLC 约束均已满足，后续需要继续优化到 `flex/fa <= 1.05`。
