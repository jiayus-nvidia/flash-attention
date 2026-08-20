"""36-entry FlexAttention versus FA4 training benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.metadata
import itertools
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from flex_attn import create_mask_plan, flex_attn_func, flex_attn_varlen_func  # noqa: E402


REFERENCE_ROOT = Path(
    "/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/"
    "agent/agent_benchmark/fa4-beta19"
)
REFERENCE_COMMIT = "940cd9680f3315f2f06b43ab5bea2c2cf2d96806"
VARLEN_LENGTHS = (6592, 6992, 7392, 7792, 8592, 8992, 9392, 9792)
assert len(set(VARLEN_LENGTHS)) == 8 and sum(VARLEN_LENGTHS) == 8 * 8192


@dataclass(frozen=True)
class BenchmarkCase:
    mode: str
    mask_kind: str
    head_dim: int
    head_dim_v: int
    phase: str

    @property
    def id(self) -> str:
        return (
            f"{self.mode}-{self.mask_kind}-d{self.head_dim}v{self.head_dim_v}-"
            f"{self.phase}"
        )

    @property
    def reference_comparable(self) -> bool:
        return not (
            self.mask_kind == "local"
            and self.head_dim == 256
            and self.head_dim_v == 256
        )


CASES = tuple(
    BenchmarkCase(mode, mask_kind, *dims, phase)
    for dims, mode, mask_kind, phase in itertools.product(
        ((128, 128), (192, 128), (256, 256)),
        ("fixed", "varlen"),
        ("causal", "local"),
        ("forward", "backward", "combined"),
    )
)
assert len(CASES) == 36
assert sum(case.reference_comparable for case in CASES) == 30


def _prefix(lengths: tuple[int, ...]) -> torch.Tensor:
    values = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return torch.cat(
        (torch.zeros(1, dtype=torch.int32, device="cuda"), values.cumsum(0, dtype=torch.int32))
    )


def _make_fixed_mask(mask_kind: str) -> torch.Tensor:
    q_idx = torch.arange(8192, dtype=torch.int32, device="cuda")
    end = q_idx + 1
    if mask_kind == "causal":
        return end.repeat(8).view(1, 1, -1).contiguous()
    begin = torch.clamp(q_idx - 512, min=0)
    per_sample = torch.stack((torch.zeros_like(begin), begin, end), dim=0)
    return per_sample.repeat(1, 8).view(1, 3, -1).contiguous()


def _make_varlen_mask(mask_kind: str, lengths: tuple[int, ...]) -> torch.Tensor:
    parts = []
    for seqlen in lengths:
        q_idx = torch.arange(seqlen, dtype=torch.int32, device="cuda")
        end = q_idx + 1
        if mask_kind == "causal":
            parts.append(end.view(1, -1))
        else:
            parts.append(
                torch.stack(
                    (torch.zeros_like(end), torch.clamp(q_idx - 512, min=0), end)
                )
            )
    return torch.cat(parts, dim=1).unsqueeze(0).contiguous()


def _load_reference(root: Path, *, allow_mismatch: bool):
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_COMMIT and not allow_mismatch:
        raise RuntimeError(
            f"FA reference must be at {REFERENCE_COMMIT}; got {commit}. "
            "Use a separate worktree or pass --allow-reference-mismatch for diagnostics only."
        )
    sys.path.insert(0, str(root))
    module = importlib.import_module("flash_attn.cute")
    return module, commit


def _make_inputs(case: BenchmarkCase):
    torch.manual_seed(0)
    requires_grad = case.phase != "forward"
    if case.mode == "fixed":
        q = torch.randn(
            8, 8192, 16, case.head_dim,
            dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad,
        )
        k = torch.randn(
            8, 8192, 4, case.head_dim,
            dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad,
        )
        v = torch.randn(
            8, 8192, 4, case.head_dim_v,
            dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad,
        )
        cu_q = cu_k = None
        mask_func = _make_fixed_mask(case.mask_kind)
    else:
        cu_q = _prefix(VARLEN_LENGTHS)
        cu_k = _prefix(VARLEN_LENGTHS)
        total = sum(VARLEN_LENGTHS)
        q = torch.randn(
            total, 16, case.head_dim,
            dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad,
        )
        k = torch.randn(
            total, 4, case.head_dim,
            dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad,
        )
        v = torch.randn(
            total, 4, case.head_dim_v,
            dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad,
        )
        mask_func = _make_varlen_mask(case.mask_kind, VARLEN_LENGTHS)
    dout = torch.randn((*q.shape[:-1], case.head_dim_v), dtype=q.dtype, device=q.device)
    return q, k, v, dout, mask_func, cu_q, cu_k


def _make_calls(case: BenchmarkCase, reference, inputs):
    q, k, v, dout, mask_func, cu_q, cu_k = inputs
    if case.mode == "fixed":
        create_plan = lambda: create_mask_plan(mask_func, q, k, v)
    else:
        create_plan = lambda: create_mask_plan(
            mask_func, q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max(VARLEN_LENGTHS), max_seqlen_k=max(VARLEN_LENGTHS),
        )

    # Compile planner kernels before measuring the runtime plan construction.
    create_plan()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    plan = create_plan()
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - started_at) * 1e3

    if case.mode == "fixed":
        flex_forward = lambda: flex_attn_func(q, k, v, mask_plan=plan)
        reference_forward = lambda: reference.flash_attn_func(
            q,
            k,
            v,
            causal=case.mask_kind == "causal",
            window_size=(512, 0) if case.mask_kind == "local" else (None, None),
        )
    else:
        flex_forward = lambda: flex_attn_varlen_func(q, k, v, mask_plan=plan)
        reference_forward = lambda: reference.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(VARLEN_LENGTHS),
            max_seqlen_k=max(VARLEN_LENGTHS),
            causal=case.mask_kind == "causal",
            window_size=(512, 0) if case.mask_kind == "local" else (None, None),
        )

    def phase_call(forward):
        if case.phase == "forward":
            return forward
        if case.phase == "backward":
            out = forward()
            if isinstance(out, tuple):
                out = out[0]

            def backward():
                torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)

            return backward

        def combined():
            out = forward()
            if isinstance(out, tuple):
                out = out[0]
            torch.autograd.grad(out, (q, k, v), dout)

        return combined

    reference_call = (
        phase_call(reference_forward) if case.reference_comparable else None
    )
    return phase_call(flex_forward), reference_call, plan_ms


def _time_cuda(callable_, *, warmup: int = 10, rounds: int = 5, iterations: int = 50) -> float:
    callable_()
    torch.cuda.synchronize()
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            callable_()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iterations)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    args = parser.parse_args()
    selected = tuple(case for case in CASES if args.case_filter in case.id)
    if args.dry_run:
        for case in selected:
            print(case.id)
        print(f"entries={len(selected)}")
        return

    cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    quack_version = importlib.metadata.version("quack-kernels")
    if cutlass_version != "4.5.2" or quack_version != "0.5.0":
        raise RuntimeError(
            f"benchmark requires CUTLASS DSL 4.5.2 and quack-kernels 0.5.0; "
            f"got {cutlass_version} and {quack_version}"
        )
    reference, commit = _load_reference(
        args.reference_root, allow_mismatch=args.allow_reference_mismatch
    )
    print(f"gpu={torch.cuda.get_device_name()} reference={commit}", flush=True)
    failures = []
    comparable_entries = 0
    unsupported_entries = 0
    for case in selected:
        inputs = _make_inputs(case)
        flex_call, reference_call, plan_ms = _make_calls(case, reference, inputs)
        flex_us = _time_cuda(flex_call)
        if reference_call is None:
            unsupported_entries += 1
            print(
                f"{case.id}: flex={flex_us:.3f}us fa=N/A ratio=N/A "
                f"plan={plan_ms:.3f}ms",
                flush=True,
            )
            continue
        reference_us = _time_cuda(reference_call)
        ratio = flex_us / reference_us
        comparable_entries += 1
        print(
            f"{case.id}: flex={flex_us:.3f}us fa={reference_us:.3f}us "
            f"ratio={ratio:.4f} plan={plan_ms:.3f}ms",
            flush=True,
        )
        if ratio > 1.05:
            failures.append((case.id, ratio))
    print(
        f"summary: entries={len(selected)} comparable={comparable_entries} "
        f"unsupported={unsupported_entries} failures={len(failures)}",
        flush=True,
    )
    if failures:
        raise SystemExit(f"performance gate failed: {failures}")


if __name__ == "__main__":
    main()
