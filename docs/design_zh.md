# Arbitrary Mask 设计

## 文档范围

本文只描述 arbitrary mask 的公开语义、plan 构建、packed predicate payload 以及
forward/backward consumer 的使用协议。

以下内容不属于本文范围：kernel pipeline、warp role、CLC scheduler、qstage、
1CTA/2CTA、DIM 性能调优、benchmark 和测试矩阵。

## 公开 Mask 语义

公开输入 `mask_func` 是 contiguous CUDA `int32` tensor：

```text
mask_func[Hmask, nfunc, total_q]
```

- `Hmask` 只能为 `1` 或 `Hq`；`1` 表示所有 Q heads 共享 mask。
- `nfunc` 必须为正奇数。
- `total_q` 是 fixed BSHD 展平后的 `B * Sq`，或 true-varlen THD 的总 Q token 数。
- 每个 endpoint 都使用当前 sample 内的 local-K 坐标，而不是整个 THD tensor 的全局
  K 坐标。

对任意一行 Q，endpoint `F0, F1, ..., F(nfunc-1)` 表示以下半开区间并集：

```text
[0, F0) U [F1, F2) U [F3, F4) U ...
```

例如：

```text
endpoints = [32, 64, 96]
visible K = [0, 32) U [64, 96)
```

每行 endpoint 必须满足：

```text
0 <= F0 <= F1 <= ... <= F(nfunc-1) <= sample_k_len
```

空区间由相等 endpoint 自然表达。公开接口不接收 Python callable mask-mod，也不要求
调用方构造 block mask 或 packed bit tensor。

## 坐标与所有权

### Fixed length

fixed BSHD 的每个 sample 具有相同 `Sq`、`Sk`。`mask_func` 的最后一维仍按 batch-major
Q row 展平，但 endpoint 始终是 sample-local K 坐标。

### True varlen

true-varlen THD 通过 `cu_seqlens_q/k` 定义 sample 边界。builder 先验证 prefix：

- rank-1、contiguous CUDA `int32`；
- 从 0 开始，以 `total_q/total_k` 结束；
- 单调非递减；
- 每个 sample 长度不超过对应 `max_seqlen`。

随后 builder 克隆 `cu_seqlens_q/k` 并由 `MaskPlan` 持有。调用方之后原地修改最初传入
的 prefix，不会改变已构建 plan 的 sample 划分。

### Planner 内部坐标

公开 endpoint 在进入 GPU planner 前会加上当前 sample 的全局 K offset，从
sample-local 坐标转换为 planner 内部坐标。内部 `mask_func` 的 Q 维额外保留 256 行零
padding，用于 planner 的安全读取；该临时 tensor 不进入公开 `MaskPlan`。

最终 CSR 中的 block index 重新保持 sample-local。consumer 使用 sample offset 加上
local block index 形成实际 Q/K/V 地址，因此一个 plan row 不会跨越 sample 边界。

## Plan 构建流程

```text
sample-local interval endpoints
            |
            v
输入验证、prefix 克隆、内部坐标转换与 padding
            |
            v
Q2K classify
  每个 Q plan row x K block -> empty / partial / full
            |
            v
partial/full count -> exclusive CSR offsets
            |
            v
Q2K compact materialize
  partial/full indices + partial packed payload
            |
            +-----------------------------+
            | build_backward=True         |
            v                             |
K2Q count + compact materialize           |
  K-major indices + backward payload      |
            |                             |
            +-----------------------------+
            v
MaskPlan
```

### Block 分类

plan tile 的大小由目标 consumer 决定。对每个有效 Q plan row 和 sample-local K block：

- `empty`：有效 attention 元素全部不可见；
- `full`：Q/K tile 完整位于当前 sample 的有效范围内，并且 tile 内所有 attention
  元素都可见；
- `partial`：tile 内存在可见元素，但 mask 不是全可见，或者该 tile 是需要 Q/K
  边界保护的尾块。

分类阶段生成临时 `visible_bits`、`full_bits` 和每行 partial/full count。bitset 只用于
planner 中间计算，不作为 attention consumer 的公开输入。

### 两套独立 CSR

Q2K 与 K2Q 都使用两套相互独立的 CSR：

```text
partial CSR
  mask_block_cnt
  mask_block_offset
  mask_block_idx

full CSR
  full_block_cnt
  full_block_offset
  full_block_idx
```

其中：

- `*_cnt[mask_head, plan_row]` 是当前 row 的 block 数；
- `*_offset` 是展平 CSR 的 exclusive offset；
- `*_idx` 保存 sample-local block index。

设计不假设 block index 连续，不生成 full-block run，也不把 partial/full 动态合并为
单一索引流。`empty` block 不进入 CSR。

## Q2K 与 K2Q

### Q2K

Q2K 以 Q plan row 为外层 row，每行列出该 Q tile 需要访问的 K blocks。forward 使用
Q2K plan；需要独立 Q-major traversal 的 backward consumer 也可以拥有自己的 Q2K view。

### K2Q

K2Q 是同一 backward topology 下 Q2K sparse pair 的转置视图，以 K block 为外层
row，每行列出与其相关的 Q blocks。dK/dV consumer 使用 K2Q plan，使每个 KV task
只遍历真实关联的 Q blocks，而不重新扫描所有 Q rows。

K2Q 可以附带 `dq_write_order` 和 `dq_write_order_full`。它们只描述并行 dQ 写回顺序，
不改变 mask 的数学语义。

## Packed Predicate Payload

只有 partial block 携带 `mask_block_masks`；full block 不分配、不加载 mask payload。

逻辑形状为：

```text
mask_block_masks[
    partial_nnz,
    physical_subtile,
    payload_group,
    uint32_word,
]
```

payload 不是通用的 row-major bitmap。planner 根据目标 consumer 的 MMA accumulator
ownership，把每个 score 元素映射到对应线程或 payload group，并按 consumer 最终读取
register fragment 的线性顺序写入 bit：

```text
bit = 1  -> 保留该 score
bit = 0  -> 将该 score 设为 -inf
```

因此 attention kernel 不需要重新执行 interval 判断、坐标比较或通用 mask-mod。
consumer 只需加载属于自己的 packed `uint32`，再对 accumulator 做 constexpr bit-select。

这里的 R2P 表示“register fragment 按 packed predicate 选择”的数据流语义；源码不要求
存在一个名为 `R2P` 的独立 intrinsic。SM100 的 TMEM-to-register load 与随后展开的
bit-select 共同实现该路径，SM90 则对 WGMMA register fragment 使用相同语义。

最后一个 `uint32` 可能覆盖超过 fragment 实际元素数的 padding bits。consumer 只对
`col < size(accumulator_fragment)` 的 constexpr 元素生成访问，禁止读取越界 accumulator。

## Consumer 协议

### Forward

forward 从 Q2K row 分别取得 partial/full CSR：

```text
for block in partial CSR:
    load K/V
    compute score
    load packed predicate
    masked softmax step

for block in full CSR:
    load K/V
    compute score
    unmasked softmax step
```

零 block 的 row 写出 `O=0`、`LSE=-inf`。sequence tail 已编码在 plan/payload 与 consumer
边界保护中，不允许访问下一个 sample。

### Backward

backward 根据 kernel 数学方向消费 Q2K 或 K2Q view：

- partial block 读取自己的 packed predicate；
- full block 只读取 `full_block_idx`；
- partial/full 的遍历顺序可以由具体 kernel pipeline 决定，但两套 CSR 的语义不变；
- 不存在 singleton-heavy raw-index 旁路。

## 架构与 Consumer 边界

arbitrary mask 的 interval 语义和 CSR 逻辑在 SM90、SM100、SM103 上一致，但已
materialize 的 payload 与具体 consumer 绑定：

- SM90 payload 按 WGMMA accumulator partition 生成；
- SM100/SM103 payload 按 tcgen05/TMEM load 后的 register ownership 生成；
- forward、generic backward、D256 dQ、D256 dKdV 可以具有不同 tile、subtile、CTA group
  和 payload word 数。

`MaskPlan` 携带 `ArbitraryPlanSignature`，记录 architecture family、direction、tile
topology、MMA layout、payload layout、PackGQA 和 dQ order format。dispatch 必须将该
signature 与即将启动的 consumer 配置逐项匹配。

结论是：同一个公开 `mask_func` 可以为不同架构或 kernel topology 重新建 plan。
同一个 `MaskPlan` 可以同时保存 forward 和 backward 各自的 materialized view，但
任何一个 view 都不能被当成另一方向或不兼容 consumer 的数据解释；整个 `MaskPlan`
也不能跨架构或跨不兼容的 tile geometry 复用。

## Plan 生命周期

`MaskPlan` 是 opaque、consumer-specific 的只读对象：

- Q/K/V tensor identity 不属于绑定条件；相同 geometry 下可以复用 plan 处理新的数值；
- fixed plan 绑定 batch、Sq、Sk、heads、head dims、dtype 和 device；
- varlen plan 额外持有自己的 `cu_seqlens_q/k` clone；
- runtime binding 记录 geometry、prefix tensor identity 和 version，拒绝 stale prefix；
- 训练调用必须使用包含 backward payload 的 plan；
- `debug_snapshot()` 只返回 tensor clone，不暴露可变的内部 plan。

## 设计不变量

1. 公开 endpoint 永远使用 sample-local K 坐标。
2. CSR block index 永远使用 sample-local block 坐标。
3. partial 与 full 始终是两套独立 CSR。
4. 只有 partial block 携带 packed predicate payload。
5. payload bit 顺序必须与目标 consumer 的 accumulator ownership 完全一致。
6. full block 不执行 packed-mask load 或 bit-select。
7. consumer 不依赖 block 连续性，不生成或消费 full-block run。
8. true-varlen 不得复用 fixed-length 地址路径。
9. plan signature 不匹配时必须显式报错，不能静默解释为另一种 layout。
10. packed payload 的尾部 padding bit 不得产生越界 accumulator 访问。
