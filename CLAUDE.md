# Flex Attention

## 协作前置条件

在实施前，必须明确需求、方案和目标。存在任何不清楚或歧义时，不得开始行动，
必须先向用户确认清楚。

## 算法约束

- **变长不可走定长 fast path**：varlen 场景禁止复用 fixed-length fast path；
  varlen benchmark 必须使用真实变长代码路径，不得用定长代码近似。

## 编码规范

- **张量前缀**：`m*`（GMEM）、`g*`（GMEM tile）、`s*`（SMEM）、
  `t*`（线程视图）、`acc_*`（累加器）。
- **命名规范**：遵循 FlashAttention CuTe DSL（FA/FA4）风格；变量和函数优先使用
  描述性 `snake_case`，类使用 PascalCase，架构后缀使用 `Sm100` 等清晰命名。
- **Kernel 类结构**：`__init__`（host 配置）→ `@cute.jit __call__`（launch）→
  `@cute.kernel`（设备主体）。
- **代码格式**：注释使用英文，代码使用 4 空格缩进；类型标注使用
  `cute.Tensor`、`Optional[cute.Tensor]`、`cutlass.Constexpr[...]`。
- **导入顺序**：`cutlass` → `cutlass.cute as cute` → `cutlass.cute.nvgpu` → `cuda.bindings`

FA4 风格参考：

- `/home/scratch.cjerry_sw/ref4agent/attention/flash-attention/flash_attn/cute/flash_fwd_sm100.py`
- `/home/scratch.cjerry_sw/ref4agent/attention/flash-attention/flash_attn/cute/flash_fwd_sm100.py`
- `/home/scratch.cjerry_sw/ref4agent/attention/flash-attention/flash_attn/cute/flash_bwd_sm100.py`
- `/home/scratch.cjerry_sw/ref4agent/attention/flash-attention/flash_attn/cute/interface.py`

## 计算精度规范

- **Tensor Core GEMM**：accumulator 必须为 FP32。
- **CUDA Core 计算**：乘加及 `sin`、`cos`、`tan`、`ex2` 等特殊函数必须在
  FP32 上执行；禁止在 CUDA Core 上用 FP16/BF16 做算术或 transcendental 计算。
- **数据流精度**：GMEM/SMEM 中的数据可以是 BF16/FP16 或更低精度量化格式；
  允许先转换为 FP32，再在 CUDA Core 上执行 FP32 计算。该“低精度存储 + FP32
  计算”路径合规。
- **输出精度**：按各路径既有约定（如 O 为 MXFP8）在 FP32 计算完成后显式
  downcast；不得在累加或归约阶段提前截断到低精度。
- **非确定性累加**：只要数学语义、数据类型和 FP32 累加精度不变，
  online-softmax、reduce、atomic、NVLS/AllReduce 等路径因并行调度或输入处理
  顺序不同而改变 FP32 累加顺序，不属于精度问题。正确性测试不要求确定性累加
  顺序或 bitwise-identical 输出，不得仅因合法的累加顺序变化判定为 bug。

## 验证流程

1. 修改 kernel 后，**先运行对应单元测试**。
2. 确认**全部用例 PASS** 后再 commit。
3. 若修改共享组件，必须运行全部测试。
4. 新增功能必须有测试覆盖；优先扩展已有测试函数，无必要不要新增测试函数。
5. 完成 code clean：
   - 不保留 debug 代码或冗余分支。
   - 正确性和性能不得回退。
   - 完成编码规范审查；变量命名、函数命名、warp-specialization 写法和 interface
     接口写法必须与 FA/FA4 风格保持一致。
   - compile key 只包含影响 codegen 的静态配置，不得包含 batch size、
     `total_q`、`total_kv`、`max_seqlen_q`、`max_seqlen_kv` 或具体
     `cu_seqlens` 值等运行时变量。
   - 尽可能使用 CuTe DSL 已有封装以及 tiled/cute copy；TopK indices、LSE、
     `cu_seqlens` 等 metadata tensor 可以不使用 tiled/cute copy。

## 开发测试规范

- **编译计时**：每次 `cute.compile(...)` 后必须打印编译耗时，区分编译慢和 kernel 死锁：

  ```python
  import time

  t0 = time.time()
  compiled = cute.compile(demo, ...)
  print(f"Compiled in {time.time() - t0:.1f}s")
  compiled(...)  # If this times out, treat it as a deadlock.
  ```

- **超时 30s = 死锁**：kernel 运行超过 30s 认定为死锁，不是编译慢（编译通常 < 10s）
- **提交前删除 `printf`**：kernel/device 侧 `printf` 及临时调试输出仅允许用于
  本地开发和 debug，commit 前必须全部删除。用于记录 `cute.compile(...)` 耗时的
  host 侧 `print(...)` 必须保留。

## 文件组织规范

- **统一 agent 工作目录**：agent 创建的临时测试、benchmark、profile、分析脚本、
  日志、JSON/CSV 汇总和 NCU/NSYS 报告均放在
  `/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/agent/`，
  不得散落在仓库顶层或源码目录。
- **Git 跟踪规则**：整个 `agent/` 目录仅用于本地 agent 工作，不得被 Git 跟踪或提交。
- **测试文件**：`/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/agent/tests`
- **Benchmark 文件**：`/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/agent/agent_benchmark`
- **Profile 文件**：`/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/agent/profile`
- **记忆总结目录**：`/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/agent/memory`

## 性能优化 Skill

- **Codex Skill 根目录**：`/home/cjerry/.codex/skills`
  - 更新 `ref4agent` 或 DKG 后，运行：
    `bash /home/cjerry/.codex/sync-kernel-skills.sh`
  - 若 Codex 目录缺失某个 skill，再回退到
    `/home/scratch.cjerry_sw/ref4agent` 下的 canonical 源路径读取对应
    `SKILL.md`。
- **架构设计**：`/home/cjerry/.codex/skills/design-cutedsl/SKILL.md`
- **现有 kernel 理解**：`/home/cjerry/.codex/skills/ask-cutedsl/SKILL.md`
- **实现与更新**：
  `/home/cjerry/.codex/skills/generate-cutedsl/SKILL.md`、
  `/home/cjerry/.codex/skills/update-cutedsl/SKILL.md`
- **源码级性能优化**：
  `/home/cjerry/.codex/skills/optimize-cutedsl/SKILL.md`
- **pipeline timeline 分析**：
  `/home/cjerry/.codex/skills/iket-cutedsl/SKILL.md`
- **寄存器 spill 诊断**：
  `/home/cjerry/.codex/skills/diagnose-register-spilling/SKILL.md`
- 不得直接调用 `cutlass-ir/cudeepy`；由上述 action skill 按需加载。
- **KAT**：`/home/cjerry/.codex/skills/kat/SKILL.md`，用于 kernel 自动调优、
  PIC/perf 分析和优化循环。
- **kernel-optimization-agent**：
  `/home/cjerry/.codex/skills/sass-optimize/SKILL.md`，用于 SASS 级分析、
  patch、校验和测速。
- **ncu-report-skill**：
  `/home/cjerry/.codex/skills/ncu-report-skill/SKILL.md`，用于 Nsight Compute
  profile、指标解析和性能报告。

## 性能调优 Reference

- **MiniMax MSA forward**：
  `/home/scratch.cjerry_sw/MiniMax-infer/agent/worktrees/msa_v2/msa_v2/attention/fwd/atten_fwd.py`
- **BSA SM100 blk64 CuTe DSL**：
  `/home/scratch.cjerry_sw/BSA/csrc/fwd/sm100_blk64/cutedsl`
- **FlashAttention CuTe DSL（FA/FA4）**：作为命名、softmax、mask、scheduler、
  TMA/UMMA 和 warp specialization 的主要参考；重点阅读：
  - `/home/scratch.cjerry_sw/ref4agent/attention/flash-attention/flash_attn/cute/flash_fwd_sm100.py`
  - `/home/scratch.cjerry_sw/ref4agent/attention/flash-attention/flash_attn/cute/flash_bwd_sm100.py`
- **FlashMLA DSA sparse forward head64**：
  `/home/scratch.cjerry_sw/ref4agent/attention/FlashMLA/csrc/sm100/prefill/sparse/fwd/head64`

参考实现与本项目方法差异较大，只提取局部优化思路并通过实测验证。

## 性能优化记录规范

- 进行性能优化时，必须记录每一项优化对性能的影响，不得只记录最终结果。
- 所有候选必须使用统一初始版本作为基线，同时记录相对初始基线的累计提升和
  相对上一有效版本的增量提升。
- 每项优化记录：原因、具体改动、benchmark case、测试环境、优化前后 runtime、
  TFLOPS、cycles，以及提升或下降百分比。
- 未采用或正确性失败的候选也要记录，并注明拒绝原因。功耗、频率或测量窗口
  不一致时必须单独说明，不得直接比较不同口径的数据。
- 多项优化不得捆绑后只报告整体提升；若必须同时修改，补充逐项 ablation
  benchmark，确保收益可以独立归因。

## 性能分析规范

### NCU 性能指标
- NCU的路径，请注意 arch 的路径: `/home/scratch.cjerry_sw/tools`
- 优化判断以 `gpu__cycles_elapsed.avg` 为主。
- 同时输出并关注 Tensor Core SOL / TC active。
- 使用 ncu report skill：
  `/home/cjerry/.codex/skills/ncu-report-skill/SKILL.md`。

### 优化检查项

- 不应出现 scalar load/store，以及 `STS`、`LDS`、`STG`、`LDG.16` 等指令；
  尽可能使用 tiled copy 和 cute copy。
- 允许少量编译器产生的寄存器 spill 及对应 `LDL`/`STL`，但必须量化并通过
  benchmark 证明没有性能回退；禁止 stack object、local array 或动态 stack
  allocation。
- 不应出现非合并访存。
- 不应出现大量共享内存冲突。
- pipeline 编排必须合理，尽可能隐藏 load、softmax、`atomicAdd` 延迟。

### Runtime TFLOPS 口径

- Sparse dense-equivalent causal case 的 FLOPs 必须按实际 causal attention
  计算量统计。