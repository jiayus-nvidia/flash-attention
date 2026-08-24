# Arbitrary Mask 设计

## 1. 设计目标与范围

FlexAttention 使用 interval endpoints 表达静态 arbitrary mask。planner 将 endpoints 转换为
可复用的 `MaskPlan`，attention kernel 直接消费 plan，不需要在运行时执行 Python mask
function，也不需要重新判断 block visibility。

本文定义以下内容：

- 公开 mask 的表示方法和坐标系；
- planner 的构建阶段及中间数据；
- partial/full block topology；
- 与具体 consumer 绑定的 packed predicate payload；
- forward/backward 的 plan 消费协议；
- plan 的所有权、兼容性与正确性约束。

通用 Q/K/V pipeline、scheduler 策略、DIM 调优、benchmark 和优化历史不属于本文范围。

## 2. 公开 Mask 表示

公开参数 `mask_func` 是 contiguous CUDA `int32` tensor：

```text
mask_func[Hmask, nfunc, total_q]
```

`mask_func` 是沿用的 API 名称，实际输入是 tensor，不是 callable。

- `Hmask` 为 `1` 或 `Hq`；`Hmask=1` 表示所有 Q heads 共享一份 mask。
- `nfunc` 是正奇数。
- fixed-length BSHD 的 `total_q` 为 `B * Sq`；Varlen THD 的 `total_q` 为所有 Q token
  的总数。
- 每个 endpoint 都是当前 Q row 所属 sample 内的 local-K 位置。

对任意 Q row，endpoints `F0, F1, ..., F(nfunc-1)` 表示以下半开区间并集：

```text
visible(q) = [0, F0) U [F1, F2) U [F3, F4) U ...
```

例如：

```text
endpoints = [32, 64, 96]
visible K = [0, 32) U [64, 96)
```

每行 endpoints 必须有序，并且不能超过当前 sample 的 K 长度：

```text
0 <= F0 <= F1 <= ... <= F(nfunc-1) <= sample_k_len
```

相邻 endpoints 相等时，对应一个空区间。公开接口不要求调用方构造 block mask 或 packed
bit tensor。

## 3. 坐标系

设计中明确区分逻辑坐标与物理存储地址。

### 3.1 Fixed Length

fixed-length BSHD 的每个 sample 使用相同的 `Sq` 和 `Sk`。`mask_func` 中的 Q rows 按
batch-major 顺序展平，但 endpoint 的值始终是 sample-local K 位置。

### 3.2 Varlen

Varlen THD 使用 `cu_seqlens_q` 和 `cu_seqlens_k` 定义 sample 边界。一个 Q token 对应的
公开 endpoint 仍然是该 sample 内的 K 坐标，不是 flattened K tensor 的全局 offset。

host 侧验证两组 prefix tensor：

- 为 rank-1、contiguous CUDA `int32` tensor，并与 Q 位于同一 device；
- shape 相同。

prefix 的数值内容由 planner 在 GPU 上验证：必须从 0 开始，以 `total_q`/`total_k`
结束，单调非递减，并满足 `max_seqlen_q`/`max_seqlen_k`。验证结果通过 4.5 节的精确分配
header 返回，因此非法输入仍会同步抛出公开异常，但不再单独把整段 prefix D2H 两次。

公开 builder 会 clone prefix tensor，并由 `MaskPlan` 持有。调用方之后原地修改最初传入的
tensor，不会改变现有 plan 的 sample 划分。

### 3.3 Planner 与 Consumer 坐标

planner kernel 直接读取公开的 sample-local endpoints。row 对应的 sequence descriptor
提供 `q_offset`、`k_offset`、`q_len` 和 `k_len`；endpoint 只与该 row 的 local `k_len`
比较并完成 clamp/validation。planner 不再给 endpoint 添加物理 K offset，也不再构造
padded endpoint tensor。

最终 plan 中的 compact topology 重新保存 sample-local block index。consumer 按以下方式
计算实际地址：

```text
sample physical offset + sample-local block offset + in-block coordinate
```

因此，任意 plan row 都不能访问另一个 sample 的 Q/K/V 数据。

## 4. Planner 总体流程

planner 必须先确定具体 consumer，再生成 packed payload，因为 block geometry 和 predicate
bit ownership 都与目标 kernel 有关。

```text
sample-local interval endpoints
              |
              v
输入结构校验
              |
              v
解析 consumer topology，构建 compact Varlen row prefixes
              |
              v
Q2K classify -> visible_bits、full_bits、partial/full counts
              |
              +-------------------------------+
              | build_backward=True           |
              v                               |
根据 backward Q2K bitsets 统计 K2Q counts     |
              |                               |
              v                               |
scan counts，按精确大小分配 compact outputs  |
              |                               |
              v                               |
materialize Q2K/K2Q CSR 与 packed payloads    |
              |                               |
              v                               |
materialize 架构通用 FWD schedule            |
              |                               |
              +-------------------------------+
              v
MaskPlan
```

### 4.1 输入校验与 Consumer 解析

host 首先校验 tensor rank、dtype、device、layout、Q/K/V geometry、head 数量、endpoint
shape，以及 Fixed/Varlen 参数。endpoint 和 Varlen prefix 的数值内容由 classifier 与
allocation-header 路径校验，避免额外的 elementwise validation kernel 和 D2H wait。随后
根据 architecture 和计算方向解析 consumer config。

consumer config 向 planner 提供：

- 逻辑 Q/K block size；
- `q_stage` 和 CTA-group topology；
- PackGQA mapping；
- MMA/register ownership；
- physical payload subtile 数量；
- payload group 数量；
- 每个 payload group padding 后的 `uint32` word 数量。

这里需要区分两层设计：sparse block relation 与架构无关，但 packed predicate payload 与
具体 consumer 绑定。

### 4.2 Compact Row Prefixes

fixed-length plan 的 Q/K plan rows 可以用矩形上界表示。Varlen plan 则先计算每个 sample
实际拥有的 Q/K block rows，再生成 prefix sums：

```text
cu_total_q_plan_rows[B + 1]
cu_total_k_plan_rows[B + 1]
```

这些 prefixes 会去掉不存在的 rows，也能把 compact outer row 直接映射回所属 sample。
classifier 和 consumer 因而不需要扫描 `cu_seqlens` 来重新判断 row ownership。

forward 和 backward 的 tile geometry 可以不同，因此两者可以使用不同的 row prefixes。

一个架构通用的 `VarlenGeometry` kernel 根据 `cu_seqlens_q/k` 构建 FWD 以及可选 BWD 所需
的全部 block prefixes。该 kernel 同时检查 prefix 的起点、终点、顺序和最大长度，并写出
device error flag。SM90、SM100 和 SM103 使用同一套算法，区别仅在 consumer tile size 与
编译目标。

### 4.3 Q2K Classification

Q2K classifier 检查每个有效 Q plan row 与每个 sample-local K block 的组合，并将其唯一地
归入以下三类：

- `empty`：没有任何有效 score element 可见；
- `full`：完整 Q/K tile 均在有效范围内，并且所有 score elements 可见；
- `partial`：至少存在一个可见元素，但 tile 不是全可见，或者需要处理 Q/K tail。

classifier 写出四组临时数据：

```text
visible_bits[Hmask, upper_q_rows, words_for_k_blocks]
full_bits[Hmask, upper_q_rows, words_for_k_blocks]
partial_counts_tmp[Hmask, upper_q_rows]
full_counts_tmp[Hmask, upper_q_rows]
```

`visible_bits` 记录所有 non-empty Q/K block pairs；`full_bits` 是其中全可见 pair 的子集。
在有效 K-block 范围内：

```text
partial = visible_bits & ~full_bits
empty   = ~visible_bits
```

bitsets 是一种紧凑的中间表示，既可以按 Q-major 读取，也可以按 K-major 读取。它们只是
planner workspace，不进入最终 `MaskPlan`。

classifier 会检查每个 endpoint 是否位于所属 sample 的 local-K 范围，并检查完整 endpoint
序列是否单调。interval 或 Varlen prefix 错误统一写入精确分配阶段使用的 device error
word。host 只读取一次该错误值，并在 materialization 前抛出异常，非法 metadata 不会生成
plan。

### 4.4 Backward K2Q Counting

当 `build_backward=True` 时，planner 使用 backward consumer 的 tile geometry 再执行一次
classification。生成的 Q-major bitsets 表示 backward block 粒度下的同一个数学 mask。

K2Q count kernel 按 K row 读取这些 bitsets，并统计每个 K block 对应的 partial/full Q
contributors。这是 sparse transpose 的 count 阶段：它不会重新遍历所有 endpoint
intervals，也不会在此时写入最终 K2Q indices。

### 4.5 Count、Scan 与精确分配

classification 阶段使用 upper-bound workspace，因为 GPU kernel 完成前无法确定 active
blocks 的精确数量。planner 将 counts 汇总为一个很小的 header：

```text
forward Q rows、forward partial nnz、forward full nnz、
backward Q rows、backward K rows、backward partial nnz、backward full nnz、
error flags
```

host 读取一次该 8-value header。该 readback 用于确定精确分配大小，并将 device 侧的
metadata error 转换为公开异常。

SM90、SM100 和 SM103 的 fixed 与 Varlen plan 都使用架构通用的 scan/header kernel。
FWD/BWD classify 仍然独立，两种模式也都保留一次 allocation-header D2H。

fixed-length 下，`FixedScanHeader` 扫描 FWD 以及可选 K2Q/dedicated-dQ count arrays。临时
counts 已经具有最终保留的 shape，因此 plan 直接复用它们，同时由该 kernel 写出 exact
exclusive CSR offsets 和 allocation header。

Varlen 下，`VarlenScanHeader` 扫描 upper-bound count arrays，保留 inclusive scans，并写出
相同的 allocation header。header 确定精确 row 数和 NNZ 后，一个
`VarlenCompactMetadata` kernel 统一复制每个 head 的有效 count prefix，并将 inclusive
scans 转换为最终 exact offsets：

```text
partial_offset[0] = 0
partial_offset[i + 1] = partial_offset[i] + partial_count[i]

full_offset[0] = 0
full_offset[i + 1] = full_offset[i] + full_count[i]
```

该 compact materializer 在一次 launch 中处理 FWD、可选 K2Q 以及可选 dedicated-dQ
metadata。它不会为最终 CSR 添加 padding，也不会改成 capacity-based allocation。
partial 与 full outputs 继续独立分配，empty block 不占用 compact storage。

### 4.6 Q2K Materialization

Q2K materializer 再次访问 classification workspace 中置位的 bits，并写出两套独立 CSR：

```text
partial CSR                         full CSR
mask_block_cnt                      full_block_cnt
mask_block_offset                   full_block_offset
mask_block_idx                      full_block_idx
mask_block_masks
```

`mask_block_idx` 和 `full_block_idx` 保存 sample-local K block index。每个 partial entry
同时生成一份 consumer-specific packed predicate payload；full entry 不带 payload。

设计不假设 index 连续，不生成 full-block runs，也不把 partial/full blocks 合并到同一个
index array。

### 4.7 K2Q Materialization

K2Q materializer 完成 sparse transpose 的 write 阶段。对于每个 K-major row，它写出
4.4 节已经统计过的 sample-local Q block indices，并继续使用独立的 partial/full CSR。

K2Q payload 按 backward consumer 的 accumulator layout 生成。generic backward 还可以
生成 `dq_write_order` 和 `dq_write_order_full`；它们只定义并行 dQ accumulation 的合法
顺序，不改变 mask visibility。

如果 dedicated dQ kernel 需要独立的 Q-major layout，plan 还可以额外保存一份为该
consumer materialize 的 Q2K view。

### 4.8 架构通用的 Forward Schedule Materialization

SM90、SM100 和 SM103 forward 共用一份由 plan 持有的 task schedule。planner 为每个
有效 Q plan row 和 scheduled head 生成一个 task：

```text
num_forward_tasks = valid Q plan rows * scheduled heads
```

所有受支持的 forward plan 都持有 `fwd_work_desc`。Varlen 以及需要显式 sequence
descriptor 的 consumer family 还会持有 `sequence_desc`：

| Descriptor | 内容 | 作用 |
|---|---|---|
| `sequence_desc[B, 8]` | `q_offset`、`k_offset`、`q_len`、`k_len`、Q-plan-row begin/count、有效 K-block count、reserved field | 将逻辑 row 映射到 sample，并提供地址和 tail 边界。 |
| `fwd_work_desc[num_forward_tasks, 4]` | `m_block`、scheduled head、`batch_idx`、`q_valid_rows` | 给出每个 scheduled task 的完整 Q-tile identity。 |

planner 使用以下 cost：

```text
task_cost = partial_block_count + full_block_count
```

partial 与 full block 的 cost 相同。positive-cost tasks 按 L2 section、task cost 降序以及
既有的 head/Q-block tie order 执行 stable ordering；zero-cost tasks 随后按
batch/head/Q-block 排列。planner 使用 bounded counting 实现该顺序：第一个 kernel 统计
`(section, task_cost)` histogram，一个 CTA 计算稳定的 section rank 与 bucket offsets，随后
每个 section 由一个 warp 按 tie order scatter 32-task chunks。该实现保持完全相同的
lexicographic order，同时不再使用通用 GPU sort 链或 data-dependent host read。kernel 最终
消费 prepared work queue，不再枚举整个 batch 的 Q tiles，也不通过 `cu_seqlens` 重建 task
ownership。

descriptor layout 和 ordering 与 architecture 无关，只有 queue backend 不同：

- SM90 使用 `PlanDynamicPersistentTileSchedulerSm90`，backend 是调用级 software atomic
  counter。Fixed 和 Varlen forward 使用同一套 queue 机制，所有 task 都从 dynamic
  queue 中取得。
- SM100/SM103 使用 `PlanClcPersistentTileSchedulerSm100`，backend 是硬件 CLC queue。

SM90 counter 属于 runtime workspace，不属于 plan storage。这样同一个 immutable plan
可以在不同 CUDA stream 上并发使用，而不会共享可变 scheduler state。

完成调度后，`q_len`、`k_len` 和 `q_valid_rows` 仍然有必要：它们用于保护 sequence tail
处的物理 load/store，而不是重新判断有哪些 task。

## 5. Packed Predicate Payload

只有 partial block 携带 `mask_block_masks`，其逻辑 shape 为：

```text
mask_block_masks[
    partial_nnz,
    physical_subtile,
    payload_group,
    uint32_word,
]
```

该 tensor 不是 score tile 的 row-major bitmap。materializer 按目标 consumer 的 MMA
accumulator ownership，将每个 score predicate 写到最终 register fragment 对应的位置：

```text
bit = 1  -> 保留该 score
bit = 0  -> 将该 score 设为 -inf
```

attention kernel 因此不需要重新解释 interval endpoints，也不需要执行通用 mask modifier。
kernel 只加载当前 payload group 的 `uint32` words，并对 score accumulator 执行编译期展开的
bit selection。

### 5.1 R2P 语义

R2P 表示 register fragment 与 packed predicate 结合后生成 predicated register values 的
数据流，并不要求硬件或源码中存在一个名为 `R2P` 的 intrinsic。

- SM100/SM103 从 TMEM 加载 score fragment，然后执行 constexpr bit selection。
- SM90 对 WGMMA register fragment 使用相同的 predicate 语义。

最后一个 payload word 可以包含 padding bits。consumer 只为 accumulator fragment 中实际
存在的 constexpr elements 生成访问，因此 padding 不会造成越界 register access。

## 6. Kernel 消费协议

### 6.1 Forward

forward 分别遍历 partial/full CSR：

```text
for block in partial_range:
    load K/V
    compute score
    load packed predicate
    softmax_step(apply_mask=True)

for block in full_range:
    load K/V
    compute score
    softmax_step(apply_mask=False)
```

empty block 不进入任何 range。full block 不加载 packed predicate。没有 active block 的
plan row 写出 `O=0`、`LSE=-inf`。

### 6.2 Two-Stage SMEM Mask Pipeline

SM90 forward 和 generic SM100/SM103 qstage1 + 2CTA forward kernel 使用 two-stage
SMEM packed-mask pipeline 暂存 partial payload。每个 stage 持有一个 CTA-native payload
slot 和对应 barrier state。对于 M128xN128 non-PackGQA consumer，每个 CTA 共使用 4 KB
mask SMEM：

```text
SM90:             256 payload groups * 2 uint32 words * 4 bytes = 2 KB per stage
SM100/SM103 2CTA: 128 payload groups * 4 uint32 words * 4 bytes = 2 KB per stage
```

处理 partial block 时，load warp 将 CTA-native payload plane 从 GMEM bulk-copy 到选定的
SMEM stage。MMA/softmax consumers 等待该 stage，将各自的 payload words 复制到
register，释放 stage，然后应用 packed predicate。PackGQA 可以减少 payload group 数量，
但不改变 pipeline 协议。

每个 partial block 在所属 stage 上恰好推进一次 pipeline state。full 和 empty blocks
既不加载 payload，也不推进 mask pipeline。Fixed、Varlen、PackGQA 和 non-PackGQA 的
generic shapes 共用该协议。

SM90 D256 使用 M128xN64 tile。较小的 N 维度为 two-stage mask pipeline 留出足够的
shared memory；non-PackGQA payload 中每个 consumer thread 对应一个 `uint32` word，两个
stage 共使用 2 KB。generic SM100/SM103 qstage1 + 1CTA、qstage2 + 1CTA 和 dedicated
SM100/SM103 D256 forward 仍直接将 payload 从 GMEM 加载到 register。delivery method
不改变 payload ABI 或 partial/full topology。

### 6.3 Backward

backward consumer 根据输出方向选择 traversal：

- Q-major consumer 使用 Q2K view；
- K-major dK/dV consumer 使用 K2Q view；
- partial entry 加载自己的 packed predicate；
- full entry 使用 `full_block_idx`，不加载 packed predicate；只有 consumer 明确需要时才读取
  write-order metadata。

具体 kernel 可以选择 loop order，但必须保持 partial/full 的独立语义。不存在针对
singleton-heavy mask 的 raw-index bypass。

## 7. 架构与 Plan 兼容性

endpoint 语义和 partial/full CSR 组织方式在 SM90、SM100、SM103 上保持一致。但
materialized topology 和 payload 属于一个具体 consumer，因为 WGMMA、tcgen05/TMEM、
direction 和 CTA topology 的 block geometry 与 accumulator ownership 不同。

每个 materialized view 都携带 `ArbitraryPlanSignature`，记录 architecture family、
direction、kernel family、tile geometry、`q_stage`、CTA-group size、PackGQA、MMA layout、
payload layout、scheduler layout 和 dQ-order format。

dispatch 在 launch 前逐字段比较 signature。任何 mismatch 都显式报错；runtime 不会将一个
consumer 的 payload 当成另一个 consumer 的 payload 解释。

因此，同一个公开 mask 可以针对不同 architecture 或 kernel topology 分别建 plan。一个
`MaskPlan` 可以同时持有 forward、backward 和 dedicated dQ views，但每个 view 都保持
consumer-specific。

## 8. Plan 所有权与复用

`MaskPlan` 是 opaque、read-only 对象，持有 compact topology、packed payload、runtime
geometry 和可选 schedule metadata。

- Q/K/V tensor identity 不属于绑定条件；geometry 不变时，可以复用 plan 处理新数据。
- fixed-length plan 绑定 batch size、`Sq`、`Sk`、head 数量、head dimensions、dtype、
  device、architecture 和 consumer topology。
- Varlen plan 额外持有 cloned Q/K prefix tensors 及其 versioned runtime binding。
- training 调用必须使用包含 backward views 的 plan。
- `debug_snapshot()` 返回 tensor clones，不暴露可变的内部存储。

## 9. 设计不变量

1. 公开 endpoints 与最终 CSR indices 始终使用 sample-local 坐标。
2. 每个有效 Q/K block pair 必须且只能属于 `empty`、`partial`、`full` 中的一类。
3. partial/full 使用独立 CSR，不要求 indices 连续。
4. 只有 partial block 携带 packed predicate payload。
5. payload bit order 必须与目标 consumer 的 accumulator ownership 完全一致。
6. full block 不加载 payload、不执行 bit-select，也不推进 mask pipeline。
7. Q2K 与 K2Q 在各自 consumer geometry 下表示同一个数学 block relation。
8. Varlen 不得使用 fixed-length addressing。
9. plan-signature mismatch 必须报错，不能作为 fallback 条件。
10. direct delivery 与 SMEM-staged delivery 必须保持相同 payload ABI 和 mask 语义。
