# Arbitrary Mask Design

## Scope

This document describes only the public arbitrary-mask semantics, plan construction, packed
predicate payload, and the protocol used by forward and backward consumers.

Kernel pipelines, warp roles, CLC scheduling, Q stages, 1CTA/2CTA topologies, head-dimension
performance tuning, benchmarks, and test matrices are outside its scope.

## Public Mask Semantics

The public `mask_func` input is a contiguous CUDA `int32` tensor:

```text
mask_func[Hmask, nfunc, total_q]
```

- `Hmask` must be either `1` or `Hq`; `1` means that all Q heads share the mask.
- `nfunc` must be a positive odd integer.
- `total_q` is `B * Sq` for flattened fixed-length BSHD inputs, or the total number of Q tokens
  for true-variable-length THD inputs.
- Every endpoint uses local-K coordinates within its sample, rather than global K coordinates in
  the flattened THD tensor.

For each Q row, endpoints `F0, F1, ..., F(nfunc-1)` represent the following union of half-open
intervals:

```text
[0, F0) U [F1, F2) U [F3, F4) U ...
```

For example:

```text
endpoints = [32, 64, 96]
visible K = [0, 32) U [64, 96)
```

The endpoints of every row must satisfy:

```text
0 <= F0 <= F1 <= ... <= F(nfunc-1) <= sample_k_len
```

Equal adjacent endpoints naturally represent an empty interval. The public API accepts neither a
Python callable mask modifier nor a block mask or packed bit tensor constructed by the caller.

## Coordinates and Ownership

### Fixed Length

Every sample in fixed-length BSHD has the same `Sq` and `Sk`. The last dimension of `mask_func`
still flattens Q rows in batch-major order, but every endpoint remains a sample-local K coordinate.

### True Variable Length

True-variable-length THD inputs use `cu_seqlens_q/k` to define sample boundaries. The builder
first validates that each prefix tensor:

- is a rank-1, contiguous CUDA `int32` tensor;
- starts at 0 and ends at `total_q/total_k`;
- is monotonically nondecreasing; and
- gives each sample a length no greater than the corresponding `max_seqlen`.

The builder then clones `cu_seqlens_q/k`, and the resulting `MaskPlan` owns the clones. Mutating
the originally supplied prefix tensors in place therefore cannot alter the sample partition of an
existing plan.

### Internal Planner Coordinates

Before GPU planning, the builder adds the current sample's global K offset to every public
endpoint. This converts sample-local coordinates into the planner's internal coordinate system.
The internal `mask_func` also has 256 zero-padded Q rows for safe planner reads; this temporary
tensor is not retained in the public `MaskPlan`.

Block indices in the final CSR remain sample-local. A consumer forms actual Q/K/V addresses by
combining a sample offset with a local block index, so a plan row cannot cross a sample boundary.

## Plan Construction

```text
sample-local interval endpoints
            |
            v
input validation, prefix cloning, internal coordinate conversion, and padding
            |
            v
Q2K classify
  each Q plan row x K block -> empty / partial / full
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

### Block Classification

The target consumer determines the plan tile size. For every valid Q plan row and sample-local K
block, the classifier assigns one of three states:

- `empty`: all valid attention elements are invisible;
- `full`: the Q and K tiles lie entirely within the current sample's valid range, and every
  attention element in the tile is visible; or
- `partial`: the tile contains visible elements, but the mask is not fully visible or the tile is
  a Q/K tail that requires boundary protection.

Classification produces temporary `visible_bits`, `full_bits`, and per-row partial/full counts.
These bitsets are planner intermediates rather than public attention-consumer inputs.

### Independent CSR Streams

Both Q2K and K2Q use two independent CSR streams:

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

For each stream:

- `*_cnt[mask_head, plan_row]` is the number of blocks in the row;
- `*_offset` is the exclusive offset into the flattened CSR data; and
- `*_idx` stores sample-local block indices.

The design does not assume that block indices are contiguous, does not generate full-block runs,
and does not dynamically combine partial and full blocks into one index stream. Empty blocks do
not enter either CSR.

## Q2K and K2Q

### Q2K

Q2K is Q-major: each outer row lists the K blocks visited by one Q tile. Forward uses a Q2K plan;
a backward consumer requiring an independent Q-major traversal may own a separate Q2K view.

### K2Q

Within the same backward topology, K2Q is the transpose of the Q2K sparse block pairs. Each
K-major row lists the related Q blocks. The dK/dV consumer uses the K2Q plan to visit only the Q
blocks actually related to each KV task, without rescanning every Q row.

K2Q may also carry `dq_write_order` and `dq_write_order_full`. They describe only the ordering of
parallel dQ writes and do not change the mathematical mask semantics.

## Packed Predicate Payload

Only partial blocks carry `mask_block_masks`; full blocks neither allocate nor load a mask payload.

The logical shape is:

```text
mask_block_masks[
    partial_nnz,
    physical_subtile,
    payload_group,
    uint32_word,
]
```

The payload is not a generic row-major bitmap. The planner uses the target consumer's MMA
accumulator ownership to map every score element to its thread or payload group, then writes its
bit in the linear order in which the consumer will read the final register fragment:

```text
bit = 1  -> keep the score
bit = 0  -> replace the score with -inf
```

The attention kernel therefore does not need to reevaluate intervals, compare coordinates, or
execute a generic mask modifier. The consumer only loads its packed `uint32` words and applies a
constexpr bit-select to the accumulator.

Here, R2P names the data-flow semantics of selecting a register fragment with a packed predicate;
the source does not require a distinct intrinsic literally named `R2P`. On SM100, a
TMEM-to-register load followed by the unrolled bit-select implements this path. SM90 applies the
same semantics to a WGMMA register fragment.

The final `uint32` can contain padding bits beyond the fragment's actual element count. The
consumer generates accesses only for constexpr elements satisfying
`col < size(accumulator_fragment)`, preventing an out-of-bounds accumulator access.

## Consumer Protocol

### Forward

Forward reads the partial and full CSR streams of a Q2K row separately:

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

A row with zero blocks writes `O=0` and `LSE=-inf`. Sequence tails are encoded by the plan and
payload together with consumer-side boundary protection; the consumer must not access the next
sample.

### Backward

A backward kernel consumes a Q2K or K2Q view according to its mathematical traversal:

- a partial block loads its packed predicate;
- a full block reads only `full_block_idx`;
- a concrete kernel pipeline may choose its traversal order, but the two CSR streams retain their
  independent semantics; and
- no singleton-heavy raw-index bypass exists.

## Architecture and Consumer Boundary

The arbitrary-mask interval semantics and CSR organization are identical on SM90, SM100, and
SM103, but a materialized payload is tied to its concrete consumer:

- an SM90 payload follows WGMMA accumulator partitioning;
- an SM100/SM103 payload follows register ownership after a tcgen05/TMEM load; and
- forward, generic backward, D256 dQ, and D256 dKdV may use different tiles, subtiles, CTA groups,
  and payload word counts.

`MaskPlan` carries an `ArbitraryPlanSignature` that records the architecture family, direction,
tile topology, MMA layout, payload layout, PackGQA configuration, and dQ order format. Dispatch
must compare this signature field by field with the consumer about to launch.

The same public `mask_func` may therefore be planned separately for different architectures or
kernel topologies. One `MaskPlan` may contain distinct materialized views for forward and
backward, but no view may be interpreted as another direction or by an incompatible consumer.
The complete `MaskPlan` also cannot be reused across architectures or incompatible tile
geometries.

## Plan Lifecycle

`MaskPlan` is an opaque, consumer-specific, read-only object:

- Q/K/V tensor identity is not a binding condition, so a plan can process new values with the
  same geometry;
- a fixed-length plan binds batch size, Sq, Sk, head counts, head dimensions, dtype, and device;
- a variable-length plan additionally owns clones of `cu_seqlens_q/k`;
- runtime binding records geometry plus prefix-tensor identity and version, and rejects stale
  prefixes;
- a training call requires a plan that contains backward payloads; and
- `debug_snapshot()` returns only tensor clones and never exposes mutable internal plan storage.

## Design Invariants

1. Public endpoints always use sample-local K coordinates.
2. CSR block indices always use sample-local block coordinates.
3. Partial and full blocks always use independent CSR streams.
4. Only partial blocks carry packed predicate payloads.
5. Payload bit order must exactly match the target consumer's accumulator ownership.
6. Full blocks perform neither packed-mask loads nor bit-selects.
7. Consumers do not rely on block contiguity and neither generate nor consume full-block runs.
8. True-variable-length inputs never reuse fixed-length addressing paths.
9. A plan-signature mismatch must raise an explicit error rather than silently reinterpret a
   different layout.
10. Tail padding bits in a packed payload must never cause an out-of-bounds accumulator access.
