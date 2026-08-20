"""Deterministic correctness-case generation for FlexAttention tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import random


LENGTH_RANGES = {
    "128": (103, 153),
    "1k": (820, 1228),
    "4k": (3277, 4915),
    "8k": (6554, 9830),
    "16k": (13108, 16384),
}
SYMMETRIC_BUCKETS = tuple((name, name) for name in ("128", "1k", "4k", "8k", "16k"))
ASYMMETRIC_BUCKETS = tuple((name, "16k") for name in ("128", "1k", "4k", "8k"))
LENGTH_BUCKETS = SYMMETRIC_BUCKETS + ASYMMETRIC_BUCKETS
HEAD_DIMS = ((64, 64), (128, 128), (192, 128), (256, 256))
SM100_GENERIC_HEAD_DIM_SMOKE_DIMS = (
    (8, 8),
    (8, 128),
    (128, 8),
    (24, 40),
    (120, 72),
)


@dataclass(frozen=True)
class AttentionCase:
    case_id: int
    mode: str
    q_bucket: str
    k_bucket: str
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]
    head_dim: int
    head_dim_v: int
    deterministic: bool
    nfunc: int
    seed: int
    batch_size: int = 8
    dtype: str = "bfloat16"
    num_q_heads: int = 16
    num_kv_heads: int = 4
    hmask: int = 1
    mask_kind: str = "random"
    full_reference: bool = False

    @property
    def stratum(self) -> tuple[str, str, str, int, int, bool]:
        return (
            self.mode,
            self.q_bucket,
            self.k_bucket,
            self.head_dim,
            self.head_dim_v,
            self.deterministic,
        )

    @property
    def id(self) -> str:
        return (
            f"{self.case_id:04d}-{self.mode}-{self.q_bucket}x{self.k_bucket}-"
            f"d{self.head_dim}v{self.head_dim_v}-det{int(self.deterministic)}-"
            f"nf{self.nfunc}-{self.mask_kind}"
        )


def _extra_strata() -> set[tuple[str, str, str, int, int, bool]]:
    """Select 16 unique extra strata with exact mode/determinism/dim balance."""

    rng = random.Random(0)
    selected = set()
    for mode, deterministic, dims_pair in itertools.product(
        ("fixed", "varlen"), (False, True), HEAD_DIMS
    ):
        q_bucket, k_bucket = rng.choice(LENGTH_BUCKETS)
        selected.add((mode, q_bucket, k_bucket, *dims_pair, deterministic))
    assert len(selected) == 16
    return selected


def _draw_lengths(
    rng: random.Random,
    *,
    mode: str,
    q_bucket: str,
    k_bucket: str,
    batch_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = 1 if mode == "fixed" else batch_size
    q_range = LENGTH_RANGES[q_bucket]
    k_range = LENGTH_RANGES[k_bucket]
    q_lengths = []
    k_lengths = []
    for _ in range(count):
        q_len = rng.randint(*q_range)
        if q_bucket == k_bucket:
            k_len = q_len
        else:
            k_len = rng.randint(*k_range)
            q_len = min(q_len, k_len)
        q_lengths.append(q_len)
        k_lengths.append(k_len)
    if mode == "fixed":
        q_lengths *= batch_size
        k_lengths *= batch_size
    return tuple(q_lengths), tuple(k_lengths)


def generate_random_cases(seed: int = 20250815) -> tuple[AttentionCase, ...]:
    """Generate exactly 1024 balanced and reproducible case records."""

    rng = random.Random(seed)
    extra = _extra_strata()
    records = []
    case_id = 0
    for mode, buckets, dims, deterministic in itertools.product(
        ("fixed", "varlen"), LENGTH_BUCKETS, HEAD_DIMS, (False, True)
    ):
        q_bucket, k_bucket = buckets
        stratum = (mode, q_bucket, k_bucket, *dims, deterministic)
        repetitions = 8 if stratum in extra else 7
        for _ in range(repetitions):
            q_lengths, k_lengths = _draw_lengths(
                rng,
                mode=mode,
                q_bucket=q_bucket,
                k_bucket=k_bucket,
                batch_size=8,
            )
            records.append(
                AttentionCase(
                    case_id=case_id,
                    mode=mode,
                    q_bucket=q_bucket,
                    k_bucket=k_bucket,
                    q_lengths=q_lengths,
                    k_lengths=k_lengths,
                    head_dim=dims[0],
                    head_dim_v=dims[1],
                    deterministic=deterministic,
                    nfunc=rng.randrange(1, 33, 2),
                    seed=rng.randrange(2**31),
                )
            )
            case_id += 1

    assert len(records) == 1024
    records = _inject_directed_coverage(records)
    _assert_case_balance(records)
    return tuple(records)


def _inject_directed_coverage(records: list[AttentionCase]) -> list[AttentionCase]:
    directed = (
        {"dtype": "float16", "mask_kind": "full"},
        {"num_kv_heads": 16, "mask_kind": "random"},
        {"num_kv_heads": 1, "mask_kind": "random"},
        {"hmask": 16, "mask_kind": "random"},
        {"nfunc": 1, "mask_kind": "causal"},
        {"nfunc": 31, "mask_kind": "random"},
        {"nfunc": 1, "mask_kind": "empty"},
        {"nfunc": 3, "mask_kind": "tile_boundary"},
    )
    for mode in ("fixed", "varlen"):
        used_strata = set()
        for changes in directed:
            index = next(
                idx
                for idx, case in enumerate(records)
                if case.mode == mode
                and case.q_bucket == "128"
                and case.stratum not in used_strata
                and case.head_dim != 256
            )
            used_strata.add(records[index].stratum)
            records[index] = replace(records[index], **changes)

    for mode, head_dim in itertools.product(("fixed", "varlen"), (128, 192)):
        index = next(
            idx
            for idx, case in enumerate(records)
            if case.mode == mode
            and case.q_bucket == "1k"
            and case.k_bucket == "16k"
            and case.head_dim == head_dim
            and not case.deterministic
        )
        records[index] = replace(
            records[index],
            mask_kind="discontiguous_full",
            nfunc=7,
        )

    full_reference_index = next(
        idx
        for idx, case in enumerate(records)
        if case.mode == "fixed"
        and case.q_bucket == "1k"
        and case.k_bucket == "1k"
        and case.head_dim == 128
        and not case.deterministic
    )
    records[full_reference_index] = replace(
        records[full_reference_index],
        mask_kind="full",
        full_reference=True,
    )
    return records


def _assert_case_balance(records: list[AttentionCase]) -> None:
    assert sum(case.mode == "fixed" for case in records) == 512
    assert sum(case.mode == "varlen" for case in records) == 512
    assert sum(not case.deterministic for case in records) == 512
    assert sum(case.deterministic for case in records) == 512
    dim_counts = [
        sum((case.head_dim, case.head_dim_v) == dims for case in records)
        for dims in HEAD_DIMS
    ]
    assert dim_counts == [256, 256, 256, 256]
    strata_counts: dict[tuple, int] = {}
    for case in records:
        strata_counts[case.stratum] = strata_counts.get(case.stratum, 0) + 1
    assert len(strata_counts) == 144
    assert set(strata_counts.values()) <= {7, 8}


def _make_sm100_generic_head_dim_smoke_cases() -> tuple[AttentionCase, ...]:
    """Build short fixed and true-varlen cases for generic SM100 head shapes."""

    fixed_q_lengths = (137,) * 8
    fixed_k_lengths = (149,) * 8
    varlen_q_lengths = (103, 111, 119, 127, 135, 143, 151, 153)
    varlen_k_lengths = (109, 120, 129, 140, 148, 156, 160, 163)
    cases = []
    case_id = 1024
    for mode in ("fixed", "varlen"):
        q_lengths = fixed_q_lengths if mode == "fixed" else varlen_q_lengths
        k_lengths = fixed_k_lengths if mode == "fixed" else varlen_k_lengths
        for head_dim, head_dim_v in SM100_GENERIC_HEAD_DIM_SMOKE_DIMS:
            test_mha = (head_dim, head_dim_v) in ((8, 8), (24, 40))
            mask_kind = "empty" if (head_dim, head_dim_v) == (8, 8) else "random"
            dtype = (
                "float16"
                if (head_dim, head_dim_v) in ((8, 128), (24, 40))
                else "bfloat16"
            )
            cases.append(
                AttentionCase(
                    case_id=case_id,
                    mode=mode,
                    q_bucket="head-dim",
                    k_bucket="head-dim",
                    q_lengths=q_lengths,
                    k_lengths=k_lengths,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    deterministic=(head_dim, head_dim_v) == (8, 8),
                    nfunc=3,
                    seed=20260819 + case_id,
                    dtype=dtype,
                    num_kv_heads=16 if test_mha else 4,
                    mask_kind=mask_kind,
                )
            )
            case_id += 1
    return tuple(cases)


RANDOM_CASES = generate_random_cases()
SM100_GENERIC_HEAD_DIM_SMOKE_CASES = _make_sm100_generic_head_dim_smoke_cases()


def smoke_cases(mode: str) -> tuple[AttentionCase, ...]:
    """Return one deterministic representative from every stratum of one mode."""

    selected = {}
    for case in RANDOM_CASES:
        if case.mode == mode:
            selected.setdefault(case.stratum, case)
    generic_head_dim_cases = tuple(
        case for case in SM100_GENERIC_HEAD_DIM_SMOKE_CASES if case.mode == mode
    )
    return (*selected.values(), *generic_head_dim_cases)


__all__ = [
    "AttentionCase",
    "HEAD_DIMS",
    "LENGTH_BUCKETS",
    "LENGTH_RANGES",
    "RANDOM_CASES",
    "SM100_GENERIC_HEAD_DIM_SMOKE_CASES",
    "SM100_GENERIC_HEAD_DIM_SMOKE_DIMS",
    "generate_random_cases",
    "smoke_cases",
]
