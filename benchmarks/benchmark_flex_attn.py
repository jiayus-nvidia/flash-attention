"""Standard static-mask training benchmark for FlexAttention and its baselines."""

from __future__ import annotations

import argparse
from array import array
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Literal, Sequence

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))


FA4_REPOSITORY = "https://github.com/dao-ailab/flash-attention"
FA4_COMMIT = "0251105a2fb19d2957484b7f023cd8c115286ced"
DEFAULT_FA4_ROOT = Path(
    "/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/"
    "agent/reference/dao-flash-attention-main"
)
MAGI_REPOSITORY = "https://github.com/jiayus-nvidia/flash-attention"
MAGI_COMMIT = "55221a93a8fc415a721502ed68643983dbf67862"
DEFAULT_MAGI_ROOT = Path(
    "/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/"
    "agent/worktrees/flexattention-magi-backend"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/scratch.cjerry_sw/Framework/Keling/sparse_attn-sparse_attention/"
    "agent/agent_benchmark/flex_attn_static_masks"
)
STANDARD_SEQLEN = 128 * 1024
CORRECTNESS_SEQLEN = 8 * 1024
HSTU_DOCUMENT_MIN = 8 * 1024
HSTU_DOCUMENT_MAX = 16 * 1024
DOCUMENT_LENGTHS_128K = (
    11858,
    8270,
    12765,
    11038,
    4578,
    14018,
    11721,
    11988,
    4942,
    8393,
    7541,
    13495,
    10465,
)
MASK_NAMES = (
    "causal",
    "document_causal",
    "local",
    "sink_local",
    "tree_dfs",
    "tree_bfs",
    "longformer",
    "hstu",
)
BACKEND_NAMES = ("flex", "fa4", "magi", "torch")
DEFAULT_BACKEND_NAMES = ("flex", "magi", "torch")
PHASE_NAMES = ("forward", "backward", "combined")
TORCH_RECOMPILE_LIMIT = 64


@dataclass(frozen=True)
class Workload:
    batch_size: int = 1
    seqlen: int = STANDARD_SEQLEN
    num_q_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 128
    head_dim_v: int = 128
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("the standard static-mask benchmark requires batch_size=1")
        if self.seqlen <= 0:
            raise ValueError("seqlen must be positive")
        if self.num_q_heads != self.num_kv_heads:
            raise ValueError("the standard benchmark requires Hq=Hkv")
        if (self.head_dim, self.head_dim_v) != (128, 128):
            raise ValueError("the standard benchmark requires Dqk=Dv=128")
        if self.dtype != torch.bfloat16:
            raise ValueError("the standard benchmark requires BF16")


@dataclass(frozen=True)
class MaskSpec:
    name: str
    title: str
    endpoints: torch.Tensor
    details: dict[str, Any]
    visible_pairs: int

    @property
    def nfunc(self) -> int:
        return self.endpoints.shape[0]

    @property
    def seqlen(self) -> int:
        return self.endpoints.shape[1]

    @property
    def density(self) -> float:
        return self.visible_pairs / (self.seqlen * self.seqlen)


@dataclass
class TimingStats:
    samples_ms: list[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    def to_json(self) -> dict[str, Any]:
        return {
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "samples_ms": self.samples_ms,
        }


@dataclass(frozen=True)
class BlockStats:
    block_size: tuple[int, int]
    partial: int
    full: int
    empty: int

    @property
    def active(self) -> int:
        return self.partial + self.full

    @property
    def total(self) -> int:
        return self.partial + self.full + self.empty

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["active"] = self.active
        result["total"] = self.total
        result["active_ratio"] = self.active / self.total if self.total else 0.0
        result["partial_ratio"] = self.partial / self.total if self.total else 0.0
        result["full_ratio"] = self.full / self.total if self.total else 0.0
        return result


@dataclass
class BackendInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dout: torch.Tensor

    @property
    def qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q, self.k, self.v


@dataclass
class BackendRunner:
    name: str
    inputs: BackendInputs
    build_metadata: Callable[[], Any]
    forward_impl: Callable[[Any, bool], Any]
    normalize_bshd: Callable[[torch.Tensor], torch.Tensor]
    normalize_lse: Callable[[torch.Tensor], torch.Tensor]
    metadata: Any = None
    block_stats: BlockStats | None = None
    compile_ms: dict[str, float] = field(default_factory=dict)
    backward_output: torch.Tensor | None = None

    def forward(self, metadata: Any | None = None, *, return_lse: bool = False):
        return self.forward_impl(self.metadata if metadata is None else metadata, return_lse)

    @staticmethod
    def output_tensor(result: Any) -> torch.Tensor:
        return result[0] if isinstance(result, tuple) else result

    def prepare_backward_graph(self) -> None:
        result = self.forward()
        self.backward_output = self.output_tensor(result)

    def run_forward(self) -> torch.Tensor:
        return self.output_tensor(self.forward())

    def run_backward(self) -> tuple[torch.Tensor, ...]:
        if self.backward_output is None:
            raise RuntimeError("backward graph has not been prepared")
        backward_output = self.backward_output
        self.backward_output = None
        return torch.autograd.grad(
            backward_output,
            self.inputs.qkv,
            self.inputs.dout,
            retain_graph=False,
        )

    def run_combined(self) -> tuple[torch.Tensor, ...]:
        out = self.run_forward()
        return torch.autograd.grad(out, self.inputs.qkv, self.inputs.dout)

    def run_metadata_forward(self) -> torch.Tensor:
        return self.output_tensor(self.forward(self.build_metadata()))

    def run_metadata_combined(self) -> tuple[torch.Tensor, ...]:
        out = self.output_tensor(self.forward(self.build_metadata()))
        return torch.autograd.grad(out, self.inputs.qkv, self.inputs.dout)


def _jittered_lengths(total: int, count: int, *, seed: int, jitter: float) -> list[int]:
    if count <= 0 or total < count:
        raise ValueError("length count must be positive and no greater than total")
    generator = random.Random(seed)
    weights = [generator.uniform(1.0 - jitter, 1.0 + jitter) for _ in range(count)]
    scaled = [weight * total / sum(weights) for weight in weights]
    lengths = [max(1, math.floor(value)) for value in scaled]
    difference = total - sum(lengths)
    fractional_order = sorted(
        range(count), key=lambda idx: scaled[idx] - math.floor(scaled[idx]), reverse=True
    )
    for index in range(difference):
        lengths[fractional_order[index % count]] += 1
    if sum(lengths) != total:
        raise AssertionError("failed to normalize jittered lengths")
    return lengths


def _bounded_partition(
    total: int,
    minimum: int,
    maximum: int,
    *,
    seed: int,
    prefer_multiple: bool = False,
) -> list[int]:
    if total < minimum:
        return [total]
    min_parts = math.ceil(total / maximum)
    max_parts = total // minimum
    count = round(total / ((minimum + maximum) / 2))
    count = min(max(count, min_parts), max_parts)
    if prefer_multiple and count == 1 and max_parts >= 2:
        count = 2
    generator = random.Random(seed)
    lengths = [generator.randint(minimum, maximum) for _ in range(count)]
    difference = total - sum(lengths)
    while difference:
        order = list(range(count))
        generator.shuffle(order)
        progressed = False
        for index in order:
            if difference > 0:
                delta = min(difference, maximum - lengths[index])
            else:
                delta = -min(-difference, lengths[index] - minimum)
            if delta:
                lengths[index] += delta
                difference -= delta
                progressed = True
            if difference == 0:
                break
        if not progressed:
            raise AssertionError("bounded partition cannot satisfy the requested total")
    return lengths


def _hstu_document_lengths(total: int, *, seed: int) -> tuple[list[int], list[int]]:
    contexts = _bounded_partition(
        total // 2,
        HSTU_DOCUMENT_MIN // 2,
        HSTU_DOCUMENT_MAX // 2,
        seed=seed,
    )
    targets = contexts.copy()
    # Standard benchmark lengths are even. Keep odd custom lengths lossless by
    # assigning the unmatched token to the final target segment.
    targets[-1] += total % 2
    return contexts, targets


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for begin, end in sorted(intervals):
        if not 0 <= begin <= end:
            raise ValueError(f"invalid interval [{begin}, {end})")
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return [(begin, end) for begin, end in merged]


def _encode_intervals(
    intervals: Sequence[tuple[int, int]],
    *,
    max_intervals: int,
    starts_at_zero: bool,
) -> list[int]:
    merged = _merge_intervals(intervals)
    if not merged or len(merged) > max_intervals:
        raise ValueError("interval union does not fit the selected nfunc")
    encoded: list[int]
    if starts_at_zero:
        if merged[0][0] != 0:
            raise ValueError("the first interval must start at zero")
        encoded = [merged[0][1]]
        encoded.extend(value for pair in merged[1:] for value in pair)
        expected = 2 * max_intervals - 1
    else:
        encoded = [0]
        encoded.extend(value for pair in merged for value in pair)
        expected = 2 * max_intervals + 1
    encoded.extend([encoded[-1]] * (expected - len(encoded)))
    return encoded


def _validate_endpoints(endpoints: torch.Tensor, seqlen: int) -> None:
    if endpoints.dtype != torch.int32 or endpoints.ndim != 2 or not endpoints.is_contiguous():
        raise ValueError("endpoints must be a contiguous int32 [nfunc, seqlen] tensor")
    if endpoints.shape[1] != seqlen or endpoints.shape[0] % 2 != 1:
        raise ValueError("endpoints must have odd nfunc and exactly seqlen columns")
    if endpoints.shape[0] >= 33:
        raise ValueError("the standard benchmark requires nfunc < 33")
    if bool(((endpoints < 0) | (endpoints > seqlen)).any()):
        raise ValueError("endpoint lies outside the sequence")
    if endpoints.shape[0] > 1 and bool((endpoints[1:] < endpoints[:-1]).any()):
        raise ValueError("endpoints must be nondecreasing for every query")


def visible_pair_count(endpoints: torch.Tensor) -> int:
    counts = endpoints[0].to(torch.int64)
    for index in range(1, endpoints.shape[0], 2):
        counts = counts + endpoints[index + 1] - endpoints[index]
    return int(counts.sum().item())


def endpoint_visible(endpoints: torch.Tensor, q_idx: int, kv_idx: int) -> bool:
    row = endpoints[:, q_idx]
    visible = kv_idx < int(row[0])
    for index in range(1, row.numel(), 2):
        visible |= int(row[index]) <= kv_idx < int(row[index + 1])
    return visible


def _make_causal(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    return torch.arange(1, seqlen + 1, dtype=torch.int32).view(1, -1), {}


def _make_document_causal(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    lengths = (
        list(DOCUMENT_LENGTHS_128K)
        if seqlen == STANDARD_SEQLEN
        else _bounded_partition(seqlen, 4096, 16384, seed=42, prefer_multiple=True)
    )
    endpoints = torch.empty((3, seqlen), dtype=torch.int32)
    offset = 0
    for length in lengths:
        end = offset + length
        endpoints[0, offset:end] = 0
        endpoints[1, offset:end] = offset
        endpoints[2, offset:end] = torch.arange(offset + 1, end + 1, dtype=torch.int32)
        offset = end
    return endpoints, {"document_lengths": lengths, "seed": 42}


def _make_local(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    q_idx = torch.arange(seqlen, dtype=torch.int32)
    end = q_idx + 1
    begin = torch.clamp(q_idx - 512, min=0)
    return torch.stack((torch.zeros_like(begin), begin, end)), {"window_left": 512}


def _make_sink_local(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    q_idx = torch.arange(seqlen, dtype=torch.int32)
    end = q_idx + 1
    sink_end = torch.minimum(end, torch.scalar_tensor(4, dtype=torch.int32))
    local_begin = torch.clamp(q_idx - 512, min=4)
    local_begin = torch.minimum(torch.maximum(local_begin, sink_end), end)
    return torch.stack((sink_end, local_begin, end)), {"sink_tokens": 4, "window_left": 512}


def _tree_order(depth: int, traversal: Literal["dfs", "bfs"]) -> list[int]:
    node_count = 2**depth - 1
    if traversal == "bfs":
        return list(range(node_count))
    order: list[int] = []

    def visit(node: int) -> None:
        if node >= node_count:
            return
        order.append(node)
        visit(2 * node + 1)
        visit(2 * node + 2)

    visit(0)
    return order


def _tree_ancestors(node: int) -> list[int]:
    result = []
    while True:
        result.append(node)
        if node == 0:
            break
        node = (node - 1) // 2
    return list(reversed(result))


def _make_tree(
    seqlen: int, traversal: Literal["dfs", "bfs"]
) -> tuple[torch.Tensor, dict[str, Any]]:
    depth = 7
    node_count = 2**depth - 1
    lengths = _jittered_lengths(seqlen, node_count, seed=42, jitter=0.2)
    order = _tree_order(depth, traversal)
    starts: dict[int, int] = {}
    offset = 0
    for node in order:
        starts[node] = offset
        offset += lengths[node]
    endpoints = torch.empty((2 * depth - 1, seqlen), dtype=torch.int32)
    for node in order:
        start = starts[node]
        end = start + lengths[node]
        row_end = torch.arange(start + 1, end + 1, dtype=torch.int32)
        ancestor_intervals = [
            (starts[ancestor], starts[ancestor] + lengths[ancestor])
            for ancestor in _tree_ancestors(node)[:-1]
        ]
        template = _merge_intervals((*ancestor_intervals, (start, end)))
        encoded = _encode_intervals(template, max_intervals=depth, starts_at_zero=True)
        current_end_slot = 2 * len(template) - 2
        for slot, value in enumerate(encoded):
            if slot >= current_end_slot:
                endpoints[slot, start:end] = row_end
            else:
                endpoints[slot, start:end] = value
    return endpoints, {
        "depth": depth,
        "node_count": node_count,
        "node_lengths": lengths,
        "node_order": order,
        "traversal": traversal,
        "seed": 42,
        "length_jitter": 0.2,
    }


def _make_longformer(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    radius = 256
    global_tokens = [int((index + 0.5) * seqlen / 8) for index in range(8)]
    global_tokens = sorted({min(token, seqlen - 1) for token in global_tokens})
    flat = array("i")
    global_set = set(global_tokens)
    for q_idx in range(seqlen):
        if q_idx in global_set:
            intervals = [(0, seqlen)]
        else:
            intervals = [(max(0, q_idx - radius), min(seqlen, q_idx + radius + 1))]
            intervals.extend((token, token + 1) for token in global_tokens)
        flat.extend(_encode_intervals(intervals, max_intervals=9, starts_at_zero=False))
    endpoints = torch.tensor(flat, dtype=torch.int32).view(seqlen, 19).t().contiguous()
    return endpoints, {
        "local_radius": radius,
        "global_tokens": global_tokens,
        "global_token_count": len(global_tokens),
    }


def _make_hstu(seqlen: int) -> tuple[torch.Tensor, dict[str, Any]]:
    contexts, targets = _hstu_document_lengths(seqlen, seed=42)
    endpoints = torch.empty((5, seqlen), dtype=torch.int32)
    offset = 0
    for context, target in zip(contexts, targets):
        context_end = offset + context
        document_end = context_end + target
        context_rows = torch.arange(offset + 1, context_end + 1, dtype=torch.int32)
        endpoints[0, offset:context_end] = 0
        endpoints[1, offset:context_end] = offset
        endpoints[2, offset:context_end] = context_rows
        endpoints[3, offset:context_end] = context_rows
        endpoints[4, offset:context_end] = context_rows
        target_rows = torch.arange(context_end, document_end, dtype=torch.int32)
        endpoints[0, context_end:document_end] = 0
        endpoints[1, context_end:document_end] = offset
        endpoints[2, context_end:document_end] = context_end
        endpoints[3, context_end:document_end] = target_rows
        endpoints[4, context_end:document_end] = target_rows + 1
        offset = document_end
    return endpoints, {
        "context_lengths": contexts,
        "target_lengths": targets,
        "document_lengths": [c + t for c, t in zip(contexts, targets)],
        "document_length_bounds": [HSTU_DOCUMENT_MIN, HSTU_DOCUMENT_MAX],
        "target_context_ratio": 1,
        "seed": 42,
    }


_MASK_BUILDERS: dict[str, tuple[str, Callable[[int], tuple[torch.Tensor, dict[str, Any]]]]] = {
    "causal": ("Causal", _make_causal),
    "document_causal": ("Varlen document causal", _make_document_causal),
    "local": ("Causal local W=512", _make_local),
    "sink_local": ("Sink S=4 + local W=512", _make_sink_local),
    "tree_dfs": ("Tree attention DFS", lambda n: _make_tree(n, "dfs")),
    "tree_bfs": ("Tree attention BFS", lambda n: _make_tree(n, "bfs")),
    "longformer": ("Longformer", _make_longformer),
    "hstu": ("Packed HSTU context/target", _make_hstu),
}


def make_mask_spec(name: str, seqlen: int) -> MaskSpec:
    if name not in _MASK_BUILDERS:
        raise ValueError(f"unknown mask {name!r}; expected one of {MASK_NAMES}")
    title, builder = _MASK_BUILDERS[name]
    endpoints, details = builder(seqlen)
    endpoints = endpoints.contiguous()
    _validate_endpoints(endpoints, seqlen)
    return MaskSpec(
        name=name,
        title=title,
        endpoints=endpoints,
        details=details,
        visible_pairs=visible_pair_count(endpoints),
    )


@dataclass(frozen=True)
class Fa4Modules:
    root: Path
    commit: str
    flash_attn_func: Callable[..., Any]
    block_sparse_tensors_cls: type
    cutlass: Any
    cute: Any
    utils: Any


@dataclass(frozen=True)
class MagiModules:
    root: Path
    commit: str
    flash_attn_func: Callable[..., Any]
    linear_block_sparse_tensors_cls: type
    create_block_mask: Any


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def load_fa4(root: Path) -> Fa4Modules:
    root = root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"FA4 root is not a git checkout: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != FA4_COMMIT:
        raise RuntimeError(f"FA4 must be at {FA4_COMMIT}; got {commit} in {root}")
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"FA4 checkout has tracked modifications: {root}")
    loaded = sys.modules.get("flash_attn")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if root not in loaded_path.parents:
            raise RuntimeError(f"flash_attn was already imported from {loaded_path}")
    sys.path.insert(0, str(root))
    flash_module = importlib.import_module("flash_attn.cute")
    block_module = importlib.import_module("flash_attn.cute.block_sparsity")
    utils = importlib.import_module("flash_attn.cute.utils")
    cutlass = importlib.import_module("cutlass")
    cute = importlib.import_module("cutlass.cute")
    return Fa4Modules(
        root=root,
        commit=commit,
        flash_attn_func=flash_module.flash_attn_func,
        block_sparse_tensors_cls=block_module.BlockSparseTensorsTorch,
        cutlass=cutlass,
        cute=cute,
        utils=utils,
    )


def _load_source_package(package_name: str, package_root: Path):
    loaded = sys.modules.get(package_name)
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if package_root.resolve() not in loaded_path.parents:
            raise RuntimeError(f"{package_name} was already imported from {loaded_path}")
        return loaded
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {package_name} from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[package_name]
        raise
    return module


def load_magi(root: Path) -> MagiModules:
    root = root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Magi backend root is not a git checkout: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != MAGI_COMMIT:
        raise RuntimeError(f"Magi backend must be at {MAGI_COMMIT}; got {commit} in {root}")
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError(f"Magi backend checkout has tracked modifications: {root}")

    package_root = root / "flash_attn" / "cute"
    flash_module = _load_source_package("flash_attn_cute", package_root)
    block_module = importlib.import_module("flash_attn_cute.block_sparsity")
    extension_root = root / "csrc" / "utils" / "create_block_mask"
    sys.path.insert(0, str(extension_root))
    try:
        create_block_mask = importlib.import_module("create_block_mask_cuda")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Magi backend requires create_block_mask_cuda; build it with "
            "`make create_block_mask` in the Magi backend checkout"
        ) from error
    return MagiModules(
        root=root,
        commit=commit,
        flash_attn_func=flash_module.flash_attn_func,
        linear_block_sparse_tensors_cls=block_module.LinearBlockSparseTensorsTorch,
        create_block_mask=create_block_mask,
    )


def make_torch_mask_mod(
    spec: MaskSpec, endpoints: torch.Tensor | None = None
) -> Callable[..., torch.Tensor]:
    """Build a vmap-compatible predicate from the canonical endpoint tensor."""

    endpoint_tensor = spec.endpoints if endpoints is None else endpoints
    nfunc, seqlen = spec.nfunc, spec.seqlen
    if endpoint_tensor.shape != (nfunc, seqlen):
        raise ValueError("Torch endpoint tensor shape does not match MaskSpec")

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_in_bounds = (q_idx >= 0) & (q_idx < seqlen)
        safe_q_idx = q_idx.clamp(0, seqlen - 1).to(torch.int64)
        gather_idx = safe_q_idx.reshape(1, 1).expand(nfunc, 1)
        row = torch.gather(endpoint_tensor, 1, gather_idx).squeeze(1)
        visible = kv_idx < row[0]
        for endpoint_idx in range(1, nfunc, 2):
            visible = visible | ((kv_idx >= row[endpoint_idx]) & (kv_idx < row[endpoint_idx + 1]))
        return q_in_bounds & visible

    return mask_mod


def make_fa4_mask_mod(endpoints: torch.Tensor, fa4: Fa4Modules) -> Callable[..., Any]:
    """Create a scalar-compatible mask with a packed 32-column SM100 fast path."""

    cutlass = fa4.cutlass
    cute = fa4.cute
    utils = fa4.utils
    nfunc = endpoints.shape[0]

    @cute.jit
    def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        endpoint_tensor = aux_tensors[0]
        q_row = q_idx[0]
        if cutlass.const_expr(cute.size(kv_idx.shape) == 1):
            kv_col = kv_idx[0]
            visible = kv_col < cutlass.Int32(endpoint_tensor[0, q_row])
            for index in cutlass.range_constexpr(1, nfunc, 2):
                begin = cutlass.Int32(endpoint_tensor[index, q_row])
                end = cutlass.Int32(endpoint_tensor[index + 1, q_row])
                visible = visible | ((kv_col >= begin) & (kv_col < end))
            result = cute.make_rmem_tensor(1, dtype=cutlass.Boolean)
            result[0] = visible
            return result.load()

        base = kv_idx[0]
        packed = cutlass.Uint32(0)
        for interval in cutlass.range_constexpr((nfunc + 1) // 2):
            if cutlass.const_expr(interval == 0):
                begin = cutlass.Int32(0)
                end = cutlass.Int32(endpoint_tensor[0, q_row])
            else:
                begin = cutlass.Int32(endpoint_tensor[2 * interval - 1, q_row])
                end = cutlass.Int32(endpoint_tensor[2 * interval, q_row])
            begin_count = min(max(begin - base, cutlass.Int32(0)), cutlass.Int32(32))
            end_count = min(max(end - base, cutlass.Int32(0)), cutlass.Int32(32))
            bits_above_begin = utils.shl_u32(
                cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(begin_count)
            )
            bits_below_end = utils.shr_u32(
                cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(32 - end_count)
            )
            packed = packed | (bits_above_begin & bits_below_end)
        result = cute.make_rmem_tensor(1, dtype=cutlass.Uint32)
        result[0] = packed
        return result.load()

    mask_mod.__vec_size__ = 32
    return mask_mod


def _block_stats_from_counts(
    partial_counts: torch.Tensor,
    full_counts: torch.Tensor | None,
    *,
    block_size: tuple[int, int],
    seqlen: int,
) -> BlockStats:
    partial = int(partial_counts.sum().item())
    full = int(full_counts.sum().item()) if full_counts is not None else 0
    num_n_blocks = math.ceil(seqlen / block_size[1])
    total = partial_counts.numel() * num_n_blocks
    empty = total - partial - full
    if empty < 0:
        raise AssertionError("block metadata contains more active blocks than possible")
    return BlockStats(block_size, partial, full, empty)


def _flex_block_stats(plan: Any, seqlen: int) -> BlockStats:
    packed_plan = plan._runtime_args[0]
    return _block_stats_from_counts(
        packed_plan.mask_block_cnt,
        packed_plan.full_block_cnt,
        block_size=packed_plan.block_size,
        seqlen=seqlen,
    )


def _block_mask_stats(block_mask: Any, block_size: tuple[int, int]) -> BlockStats:
    values = block_mask.as_tuple()
    return _block_stats_from_counts(
        values[2], values[4], block_size=block_size, seqlen=int(values[1])
    )


def _causal_block_stats(seqlen: int, block_size: tuple[int, int]) -> BlockStats:
    block_m, block_n = block_size
    num_m_blocks = math.ceil(seqlen / block_m)
    num_n_blocks = math.ceil(seqlen / block_n)
    partial = 0
    full = 0
    for m_block in range(num_m_blocks):
        q_begin = m_block * block_m
        q_end = min(q_begin + block_m, seqlen)
        for n_block in range(num_n_blocks):
            k_begin = n_block * block_n
            k_end = min(k_begin + block_n, seqlen)
            if k_end - 1 <= q_begin:
                full += 1
            elif k_begin < q_end:
                partial += 1
    return BlockStats(
        block_size=block_size,
        partial=partial,
        full=full,
        empty=num_m_blocks * num_n_blocks - partial - full,
    )


def _fa4_sparse_tensors(block_mask: Any, fa4: Fa4Modules, block_size: tuple[int, int]):
    values = block_mask.as_tuple()
    forward = fa4.block_sparse_tensors_cls(
        mask_block_cnt=values[2],
        mask_block_idx=values[3],
        full_block_cnt=values[4],
        full_block_idx=values[5],
        block_size=block_size,
    )
    backward = fa4.block_sparse_tensors_cls(
        mask_block_cnt=values[6],
        mask_block_idx=values[7],
        full_block_cnt=values[8],
        full_block_idx=values[9],
        block_size=block_size,
    )
    return forward, backward, block_mask


def _make_base_inputs(workload: Workload) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    shape_q = (
        workload.batch_size,
        workload.seqlen,
        workload.num_q_heads,
        workload.head_dim,
    )
    shape_k = (
        workload.batch_size,
        workload.seqlen,
        workload.num_kv_heads,
        workload.head_dim,
    )
    shape_v = (*shape_k[:-1], workload.head_dim_v)
    q = torch.randn(shape_q, dtype=workload.dtype, device="cuda")
    k = torch.randn(shape_k, dtype=workload.dtype, device="cuda")
    v = torch.randn(shape_v, dtype=workload.dtype, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(1)
    dout = torch.randn(shape_v, dtype=workload.dtype, device="cuda", generator=generator)
    return q, k, v, dout


def _clone_inputs(base: tuple[torch.Tensor, ...], *, bhsd: bool) -> BackendInputs:
    tensors = []
    for tensor in base:
        value = tensor.transpose(1, 2).contiguous() if bhsd else tensor.clone()
        tensors.append(value.detach().requires_grad_(tensor is not base[-1]))
    q, k, v, dout = tensors
    dout.requires_grad_(False)
    return BackendInputs(q, k, v, dout)


def _make_flex_runner(
    inputs: BackendInputs,
    mask_func: torch.Tensor,
    workload: Workload,
) -> BackendRunner:
    flex_module = importlib.import_module("flex_attn")

    def build_metadata():
        return flex_module.create_mask_plan(
            mask_func,
            inputs.q,
            inputs.k,
            inputs.v,
            pack_gqa=False,
            build_backward=True,
        )

    def forward(metadata, return_lse: bool):
        return flex_module.flex_attn_func(
            inputs.q,
            inputs.k,
            inputs.v,
            mask_plan=metadata,
            softmax_scale=1.0 / math.sqrt(workload.head_dim),
            deterministic=False,
            return_lse=return_lse,
        )

    return BackendRunner(
        "flex",
        inputs,
        build_metadata,
        forward,
        normalize_bshd=lambda tensor: tensor,
        normalize_lse=lambda tensor: tensor,
    )


def _make_fa4_runner(
    inputs: BackendInputs,
    spec: MaskSpec,
    endpoints: torch.Tensor,
    workload: Workload,
    fa4: Fa4Modules,
    compiled_create_block_mask: Callable[..., Any],
) -> BackendRunner:
    block_size = (256, 128)
    if spec.name == "causal":

        def build_metadata():
            return None

        def forward(metadata, return_lse: bool):
            del metadata
            return fa4.flash_attn_func(
                inputs.q,
                inputs.k,
                inputs.v,
                softmax_scale=1.0 / math.sqrt(workload.head_dim),
                causal=True,
                pack_gqa=False,
                deterministic=False,
                return_lse=return_lse,
            )

        return BackendRunner(
            "fa4",
            inputs,
            build_metadata,
            forward,
            normalize_bshd=lambda tensor: tensor,
            normalize_lse=lambda tensor: tensor,
            block_stats=_causal_block_stats(workload.seqlen, block_size),
        )

    torch_mask_mod = make_torch_mask_mod(spec, endpoints)
    fa4_mask_mod = make_fa4_mask_mod(endpoints, fa4)
    endpoints.__leading_dim__ = 1
    endpoints.__assumed_align__ = 16

    def build_metadata():
        block_mask = compiled_create_block_mask(
            torch_mask_mod,
            None,
            None,
            workload.seqlen,
            workload.seqlen,
            device="cuda",
            BLOCK_SIZE=block_size,
        )
        return _fa4_sparse_tensors(block_mask, fa4, block_size)

    def forward(metadata, return_lse: bool):
        sparse_forward, sparse_backward, _ = metadata
        return fa4.flash_attn_func(
            inputs.q,
            inputs.k,
            inputs.v,
            softmax_scale=1.0 / math.sqrt(workload.head_dim),
            pack_gqa=False,
            deterministic=False,
            mask_mod=fa4_mask_mod,
            aux_tensors=[endpoints],
            block_sparse_tensors=sparse_forward,
            block_sparse_tensors_bwd=sparse_backward,
            return_lse=return_lse,
        )

    return BackendRunner(
        "fa4",
        inputs,
        build_metadata,
        forward,
        normalize_bshd=lambda tensor: tensor,
        normalize_lse=lambda tensor: tensor,
    )


def _magi_linear_from_csr(
    csr_tensors: Sequence[torch.Tensor],
    block_size: tuple[int, int],
    magi: MagiModules,
):
    (
        mask_block_cnt,
        mask_block_offset,
        mask_block_idx,
        full_block_cnt,
        full_block_offset,
        full_block_idx,
    ) = csr_tensors
    return magi.linear_block_sparse_tensors_cls(
        mask_block_cnt=mask_block_cnt,
        mask_block_offset=mask_block_offset,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_offset=full_block_offset,
        full_block_idx=full_block_idx,
        block_size=block_size,
    )


def _make_magi_runner(
    inputs: BackendInputs,
    endpoints: torch.Tensor,
    workload: Workload,
    magi: MagiModules,
) -> BackendRunner:
    q_stage = 2 if workload.seqlen > 128 else 1
    fwd_block_size = (q_stage * 128, 128)
    bwd_block_size = tuple(
        magi.create_block_mask.get_bwd_tile_sizes(
            workload.head_dim,
            is_arbitrary=True,
            headdim_v=workload.head_dim_v,
        )
    )
    if bwd_block_size != (128, 256):
        raise RuntimeError(f"unexpected Magi backend D128 backward tile: {bwd_block_size}")

    arbitrary_func = torch.zeros(
        1,
        1,
        endpoints.shape[0],
        workload.seqlen + 256,
        dtype=torch.int32,
        device="cuda",
    )
    arbitrary_func[0, 0, :, : workload.seqlen].copy_(endpoints)

    def build_metadata():
        q2k_csr = magi.create_block_mask.create_q2k_csr_sparse_from_func(
            arbitrary_func,
            workload.seqlen,
            workload.seqlen,
            Q_BLOCK_SIZE=fwd_block_size[0],
            KV_BLOCK_SIZE=fwd_block_size[1],
            check_q_boundary=True,
        )
        k2q_csr = magi.create_block_mask.create_k2q_csr_sparse_from_func(
            arbitrary_func,
            workload.seqlen,
            workload.seqlen,
            Q_BLOCK_SIZE=bwd_block_size[0],
            KV_BLOCK_SIZE=bwd_block_size[1],
        )
        return (
            _magi_linear_from_csr(q2k_csr, fwd_block_size, magi),
            _magi_linear_from_csr(k2q_csr, bwd_block_size, magi),
        )

    def forward(metadata, return_lse: bool):
        sparse_forward, sparse_backward = metadata
        return magi.flash_attn_func(
            inputs.q,
            inputs.k,
            inputs.v,
            softmax_scale=1.0 / math.sqrt(workload.head_dim),
            pack_gqa=False,
            deterministic=False,
            arbitrary=True,
            aux_tensors=[arbitrary_func],
            linear_k_block_sparse_tensors=sparse_forward,
            linear_q_block_sparse_tensors=sparse_backward,
            return_lse=return_lse,
        )

    return BackendRunner(
        "magi",
        inputs,
        build_metadata,
        forward,
        normalize_bshd=lambda tensor: tensor,
        normalize_lse=lambda tensor: tensor,
    )


def _make_torch_runner(
    inputs: BackendInputs,
    spec: MaskSpec,
    endpoints: torch.Tensor,
    workload: Workload,
    compiled_create_block_mask: Callable[..., Any],
    compiled_flex_attention: Callable[..., Any],
) -> BackendRunner:
    block_size = (128, 128)
    mask_mod = make_torch_mask_mod(spec, endpoints)

    def build_metadata():
        return compiled_create_block_mask(
            mask_mod,
            None,
            None,
            workload.seqlen,
            workload.seqlen,
            device="cuda",
            BLOCK_SIZE=block_size,
        )

    def forward(metadata, return_lse: bool):
        return compiled_flex_attention(
            inputs.q,
            inputs.k,
            inputs.v,
            block_mask=metadata,
            scale=1.0 / math.sqrt(workload.head_dim),
            enable_gqa=False,
            return_lse=return_lse,
            kernel_options=None,
        )

    return BackendRunner(
        "torch",
        inputs,
        build_metadata,
        forward,
        normalize_bshd=lambda tensor: tensor.transpose(1, 2).contiguous(),
        normalize_lse=lambda tensor: tensor,
    )


def make_backend_runners(
    workload: Workload,
    spec: MaskSpec,
    backend_names: Sequence[str],
    *,
    fa4: Fa4Modules | None,
    magi: MagiModules | None,
) -> list[BackendRunner]:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    base = _make_base_inputs(workload)
    endpoints = spec.endpoints.to(device="cuda", non_blocking=True).contiguous()
    mask_func = endpoints.unsqueeze(0).contiguous()
    compiled_create_block_mask = torch.compile(create_block_mask, fullgraph=True, dynamic=False)
    compiled_flex_attention = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    runners = []
    for name in backend_names:
        if name == "flex":
            runners.append(_make_flex_runner(_clone_inputs(base, bhsd=False), mask_func, workload))
        elif name == "fa4":
            if fa4 is None:
                raise RuntimeError("FA4 modules were not loaded")
            runners.append(
                _make_fa4_runner(
                    _clone_inputs(base, bhsd=False),
                    spec,
                    endpoints,
                    workload,
                    fa4,
                    compiled_create_block_mask,
                )
            )
        elif name == "magi":
            if magi is None:
                raise RuntimeError("Magi backend modules were not loaded")
            runners.append(
                _make_magi_runner(
                    _clone_inputs(base, bhsd=False),
                    endpoints,
                    workload,
                    magi,
                )
            )
        elif name == "torch":
            runners.append(
                _make_torch_runner(
                    _clone_inputs(base, bhsd=True),
                    spec,
                    endpoints,
                    workload,
                    compiled_create_block_mask,
                    compiled_flex_attention,
                )
            )
        else:
            raise ValueError(f"unknown backend: {name}")
    del base
    return runners


def _create_l2_flush_buffer() -> torch.Tensor:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = int(getattr(properties, "L2_cache_size", 128 * 1024 * 1024))
    return torch.empty(max(2 * l2_bytes, 256 * 1024 * 1024), dtype=torch.uint8, device="cuda")


def _flush_l2(buffer: torch.Tensor) -> None:
    buffer.zero_()


def _compile_step(callable_: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started_at) * 1e3


def prepare_runner(runner: BackendRunner, workload: Workload, phases: Sequence[str]) -> None:
    runner.metadata, runner.compile_ms["metadata"] = _compile_step(runner.build_metadata)
    if runner.block_stats is None:
        if runner.name == "flex":
            runner.block_stats = _flex_block_stats(runner.metadata, workload.seqlen)
        elif runner.name == "fa4":
            runner.block_stats = _block_mask_stats(runner.metadata[2], (256, 128))
        elif runner.name == "magi":
            sparse_forward = runner.metadata[0]
            runner.block_stats = _block_stats_from_counts(
                sparse_forward.mask_block_cnt,
                sparse_forward.full_block_cnt,
                block_size=sparse_forward.block_size,
                seqlen=workload.seqlen,
            )
        else:
            runner.block_stats = _block_mask_stats(runner.metadata, (128, 128))
    _, runner.compile_ms["forward"] = _compile_step(runner.run_forward)
    if "backward" in phases or "combined" in phases:
        runner.prepare_backward_graph()
        torch.cuda.synchronize()
        _, runner.compile_ms["backward"] = _compile_step(runner.run_backward)
    print(
        f"Compiled {runner.name}: "
        + ", ".join(f"{name}={value:.1f}ms" for name, value in runner.compile_ms.items()),
        flush=True,
    )


def _warmup_one(
    callable_: Callable[[], Any],
    flush_buffer: torch.Tensor,
    warmup: int,
    setup: Callable[[], Any] | None,
) -> None:
    for _ in range(warmup):
        if setup is not None:
            setup()
        _flush_l2(flush_buffer)
        callable_()
        torch.cuda.synchronize()


def _time_one_cuda(callable_: Callable[[], Any], flush_buffer: torch.Tensor) -> float:
    _flush_l2(flush_buffer)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _time_one_wall(callable_: Callable[[], Any], flush_buffer: torch.Tensor) -> float:
    _flush_l2(flush_buffer)
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started_at) * 1e3


def measure_interleaved(
    runners: Sequence[BackendRunner],
    action: Callable[[BackendRunner], Callable[[], Any]],
    *,
    setup: Callable[[BackendRunner], Callable[[], Any]] | None,
    flush_buffer: torch.Tensor,
    warmup: int,
    runs: int,
    wall_clock: bool,
) -> dict[str, TimingStats]:
    samples = {runner.name: [] for runner in runners}
    for runner in runners:
        setup_callable = setup(runner) if setup is not None else None
        _warmup_one(action(runner), flush_buffer, warmup, setup_callable)
    timer = _time_one_wall if wall_clock else _time_one_cuda
    for sample_idx in range(runs):
        offset = sample_idx % len(runners)
        ordered = (*runners[offset:], *runners[:offset])
        for runner in ordered:
            if setup is not None:
                setup(runner)()
            samples[runner.name].append(timer(action(runner), flush_buffer))
    return {name: TimingStats(values) for name, values in samples.items()}


def _phase_flops(workload: Workload, visible_pairs: int, phase: str) -> int:
    head_pairs = workload.num_q_heads * visible_pairs
    if phase == "forward":
        return 2 * head_pairs * (workload.head_dim + workload.head_dim_v)
    if phase == "backward":
        return 2 * head_pairs * (3 * workload.head_dim + 2 * workload.head_dim_v)
    if phase == "combined":
        return 2 * head_pairs * (4 * workload.head_dim + 3 * workload.head_dim_v)
    raise ValueError(f"phase {phase!r} has no FLOP definition")


def _tflops(flops: int, milliseconds: float) -> float:
    return flops / (milliseconds * 1e9)


def _metric_actions(phases: Sequence[str]):
    actions: list[
        tuple[
            str,
            Callable[[BackendRunner], Callable[[], Any]],
            bool,
            Callable[[BackendRunner], Callable[[], Any]] | None,
        ]
    ] = [("metadata", lambda runner: runner.build_metadata, True, None)]
    if "forward" in phases:
        actions.extend(
            (
                ("forward", lambda runner: runner.run_forward, False, None),
                (
                    "metadata_forward",
                    lambda runner: runner.run_metadata_forward,
                    True,
                    None,
                ),
            )
        )
    if "backward" in phases:
        actions.append(
            (
                "backward",
                lambda runner: runner.run_backward,
                False,
                lambda runner: runner.prepare_backward_graph,
            )
        )
    if "combined" in phases:
        actions.extend(
            (
                ("combined", lambda runner: runner.run_combined, False, None),
                (
                    "metadata_combined",
                    lambda runner: runner.run_metadata_combined,
                    True,
                    None,
                ),
            )
        )
    return actions


def benchmark_mask(
    workload: Workload,
    spec: MaskSpec,
    backend_names: Sequence[str],
    phases: Sequence[str],
    *,
    fa4: Fa4Modules | None,
    magi: MagiModules | None,
    warmup: int,
    runs: int,
    metadata_runs: int,
) -> dict[str, Any]:
    print(
        f"\n[{spec.name}] nfunc={spec.nfunc} density={spec.density:.4%} "
        f"visible_pairs={spec.visible_pairs:,}",
        flush=True,
    )
    runners = make_backend_runners(workload, spec, backend_names, fa4=fa4, magi=magi)
    active: list[BackendRunner] = []
    errors: dict[str, str] = {}
    for runner in runners:
        try:
            prepare_runner(runner, workload, phases)
        except Exception as error:  # noqa: BLE001 - preserve unsupported baseline errors
            errors[runner.name] = f"{type(error).__name__}: {error}"
            print(f"{runner.name}: N/A ({errors[runner.name]})", flush=True)
        else:
            active.append(runner)
    if "flex" in backend_names and not any(runner.name == "flex" for runner in active):
        raise RuntimeError(f"FlexAttention failed for {spec.name}: {errors.get('flex')}")
    measurements: dict[str, dict[str, TimingStats]] = {runner.name: {} for runner in active}
    flush_buffer = None
    if active:
        flush_buffer = _create_l2_flush_buffer()
        for metric, action, wall_clock, setup in _metric_actions(phases):
            metric_runs = metadata_runs if metric == "metadata" else runs
            metric_warmup = 0 if metric == "metadata" else warmup
            timed = measure_interleaved(
                active,
                action,
                setup=setup,
                flush_buffer=flush_buffer,
                warmup=metric_warmup,
                runs=metric_runs,
                wall_clock=wall_clock,
            )
            for name, stats in timed.items():
                measurements[name][metric] = stats
    result: dict[str, Any] = {
        "mask": spec.name,
        "title": spec.title,
        "nfunc": spec.nfunc,
        "visible_pairs": spec.visible_pairs,
        "element_density": spec.density,
        "details": spec.details,
        "backends": {},
    }
    for name in backend_names:
        runner = next((candidate for candidate in active if candidate.name == name), None)
        if runner is None:
            result["backends"][name] = {"status": "unsupported", "error": errors.get(name)}
            continue
        metrics = {metric: stats.to_json() for metric, stats in measurements[name].items()}
        for phase in PHASE_NAMES:
            if phase in measurements[name]:
                metrics[phase]["active_tflops"] = _tflops(
                    _phase_flops(workload, spec.visible_pairs, phase),
                    measurements[name][phase].median_ms,
                )
        result["backends"][name] = {
            "status": "ok",
            "compile_ms": runner.compile_ms,
            "block_stats": runner.block_stats.to_json() if runner.block_stats else None,
            "metrics": metrics,
        }
        summary = " ".join(
            f"{metric}={stats.median_ms:.3f}ms" for metric, stats in measurements[name].items()
        )
        print(f"{name}: {summary}", flush=True)
    del flush_buffer, runners, active
    torch.cuda.empty_cache()
    return result


def _visible_matrix(
    endpoints: torch.Tensor,
    q_begin: int,
    q_end: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    rows = endpoints[:, q_begin:q_end].t().to(device=device, non_blocking=True)
    columns = torch.arange(endpoints.shape[1], dtype=torch.int32, device=device)
    visible = columns[None, :] < rows[:, 0, None]
    for index in range(1, endpoints.shape[0], 2):
        visible |= (columns[None, :] >= rows[:, index, None]) & (
            columns[None, :] < rows[:, index + 1, None]
        )
    return visible


def eager_fp32_reference(
    workload: Workload,
    spec: MaskSpec,
    base: tuple[torch.Tensor, ...],
    *,
    q_chunk_size: int = 128,
) -> dict[str, torch.Tensor]:
    q_base, k_base, v_base, dout_base = base
    device = q_base.device
    out = torch.empty(
        workload.batch_size,
        workload.seqlen,
        workload.num_q_heads,
        workload.head_dim_v,
        dtype=torch.float32,
        device=device,
    )
    lse = torch.empty(
        workload.batch_size,
        workload.num_q_heads,
        workload.seqlen,
        dtype=torch.float32,
        device=device,
    )
    dq = torch.empty_like(q_base, dtype=torch.float32)
    dk = torch.empty_like(k_base, dtype=torch.float32)
    dv = torch.empty_like(v_base, dtype=torch.float32)
    scale = 1.0 / math.sqrt(workload.head_dim)
    for head_idx in range(workload.num_q_heads):
        q = q_base[0, :, head_idx].float().detach().requires_grad_(True)
        k = k_base[0, :, head_idx].float().detach().requires_grad_(True)
        v = v_base[0, :, head_idx].float().detach().requires_grad_(True)
        out_chunks = []
        lse_chunks = []
        for q_begin in range(0, workload.seqlen, q_chunk_size):
            q_end = min(q_begin + q_chunk_size, workload.seqlen)
            score = (q[q_begin:q_end] @ k.t()) * scale
            visible = _visible_matrix(spec.endpoints, q_begin, q_end, device=device)
            score = score.masked_fill(~visible, -torch.inf)
            probability = torch.softmax(score, dim=-1)
            out_chunks.append(probability @ v)
            lse_chunks.append(torch.logsumexp(score, dim=-1))
        head_out = torch.cat(out_chunks)
        head_lse = torch.cat(lse_chunks)
        head_grads = torch.autograd.grad(
            head_out,
            (q, k, v),
            dout_base[0, :, head_idx].float(),
        )
        out[0, :, head_idx] = head_out.detach()
        lse[0, head_idx] = head_lse.detach()
        dq[0, :, head_idx] = head_grads[0]
        dk[0, :, head_idx] = head_grads[1]
        dv[0, :, head_idx] = head_grads[2]
    return {"out": out, "lse": lse, "dq": dq, "dk": dk, "dv": dv}


def _candidate_outputs(runner: BackendRunner) -> dict[str, torch.Tensor]:
    result = runner.forward(return_lse=True)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"{runner.name} did not return (out, lse)")
    out, lse = result[:2]
    gradients = torch.autograd.grad(out, runner.inputs.qkv, runner.inputs.dout)
    normalized_lse = runner.normalize_lse(lse)
    return {
        "out": runner.normalize_bshd(out).detach(),
        "lse": normalized_lse.detach(),
        "dq": runner.normalize_bshd(gradients[0]).detach(),
        "dk": runner.normalize_bshd(gradients[1]).detach(),
        "dv": runner.normalize_bshd(gradients[2]).detach(),
    }


def _finite_values(tensor: torch.Tensor, reference: torch.Tensor):
    if not torch.equal(torch.isneginf(tensor), torch.isneginf(reference)):
        raise AssertionError("negative-infinity locations differ")
    if torch.isnan(tensor).any() or torch.isposinf(tensor).any():
        raise AssertionError("candidate contains NaN or positive infinity")
    finite = torch.isfinite(reference)
    return tensor.float()[finite], reference.float()[finite]


def _reference_error_gate(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    pytorch_candidate: torch.Tensor,
) -> dict[str, float]:
    candidate_finite, reference_finite = _finite_values(candidate, reference)
    pytorch_finite, _ = _finite_values(pytorch_candidate, reference)
    if candidate_finite.numel() == 0:
        return {"max_error": 0.0, "pytorch_error": 0.0, "limit": 0.0}
    max_error = float((candidate_finite - reference_finite).abs().max().item())
    pytorch_error = float((pytorch_finite - reference_finite).abs().max().item())
    arithmetic_atol = float(
        (2.0 * (reference_finite + 0.3 - 0.3 - reference_finite).abs().max()).item()
    )
    limit = 2.0 * pytorch_error + arithmetic_atol
    if max_error > limit:
        raise AssertionError(f"max error {max_error:.8g} exceeds {limit:.8g}")
    return {"max_error": max_error, "pytorch_error": pytorch_error, "limit": limit}


def validate_against_fp32(
    workload: Workload,
    spec: MaskSpec,
    backend_names: Sequence[str],
    *,
    fa4: Fa4Modules | None,
    magi: MagiModules | None,
) -> dict[str, Any]:
    print(f"\n[correctness/fp32] {spec.name} N={workload.seqlen}", flush=True)
    base = _make_base_inputs(workload)
    reference = eager_fp32_reference(workload, spec, base)
    del base
    runners = make_backend_runners(workload, spec, backend_names, fa4=fa4, magi=magi)
    candidates: dict[str, dict[str, torch.Tensor]] = {}
    errors: dict[str, str] = {}
    for runner in runners:
        try:
            runner.metadata, runner.compile_ms["metadata"] = _compile_step(runner.build_metadata)
            candidates[runner.name], runner.compile_ms["correctness"] = _compile_step(
                lambda runner=runner: _candidate_outputs(runner)
            )
        except Exception as error:  # noqa: BLE001 - preserve baseline failures
            errors[runner.name] = f"{type(error).__name__}: {error}"
            print(f"{runner.name}: N/A ({errors[runner.name]})", flush=True)
    if "torch" not in candidates:
        raise RuntimeError(f"PyTorch reference baseline failed: {errors.get('torch')}")
    result: dict[str, Any] = {"mask": spec.name, "backends": {}}
    for name in backend_names:
        if name not in candidates:
            result["backends"][name] = {"status": "unsupported", "error": errors.get(name)}
            continue
        tensor_results = {}
        for tensor_name, reference_tensor in reference.items():
            tensor_results[tensor_name] = _reference_error_gate(
                candidates[name][tensor_name],
                reference_tensor,
                candidates["torch"][tensor_name],
            )
        result["backends"][name] = {"status": "pass", "tensors": tensor_results}
        print(
            f"{name}: PASS "
            + " ".join(
                f"{tensor_name}={values['max_error']:.3e}"
                for tensor_name, values in tensor_results.items()
            ),
            flush=True,
        )
    del runners, candidates, reference
    torch.cuda.empty_cache()
    return result


def _cross_error(candidate: torch.Tensor, anchor: torch.Tensor, *, atol: float) -> dict[str, float]:
    candidate_finite, anchor_finite = _finite_values(candidate, anchor)
    if candidate_finite.numel() == 0:
        return {"max_abs": 0.0, "max_rel": 0.0}
    absolute = (candidate_finite - anchor_finite).abs()
    relative = absolute / anchor_finite.abs().clamp_min(1e-5)
    torch.testing.assert_close(candidate_finite, anchor_finite, rtol=2e-2, atol=atol)
    return {"max_abs": float(absolute.max().item()), "max_rel": float(relative.max().item())}


def validate_full_cross_backend(
    workload: Workload,
    spec: MaskSpec,
    backend_names: Sequence[str],
    *,
    fa4: Fa4Modules | None,
    magi: MagiModules | None,
) -> dict[str, Any]:
    print(f"\n[correctness/cross] {spec.name} N={workload.seqlen}", flush=True)
    runners = make_backend_runners(workload, spec, backend_names, fa4=fa4, magi=magi)
    candidates: dict[str, dict[str, torch.Tensor]] = {}
    errors: dict[str, str] = {}
    for runner in runners:
        try:
            runner.metadata, runner.compile_ms["metadata"] = _compile_step(runner.build_metadata)
            candidates[runner.name], runner.compile_ms["correctness"] = _compile_step(
                lambda runner=runner: _candidate_outputs(runner)
            )
        except Exception as error:  # noqa: BLE001 - preserve baseline failures
            errors[runner.name] = f"{type(error).__name__}: {error}"
            print(f"{runner.name}: N/A ({errors[runner.name]})", flush=True)
    if not candidates:
        raise RuntimeError(f"all full cross-check backends failed: {errors}")
    anchor_name = "flex" if "flex" in candidates else "torch"
    if anchor_name not in candidates:
        anchor_name = next(iter(candidates))
    anchor = candidates[anchor_name]
    result: dict[str, Any] = {"mask": spec.name, "anchor": anchor_name, "backends": {}}
    for name in backend_names:
        if name not in candidates:
            result["backends"][name] = {"status": "unsupported", "error": errors.get(name)}
            continue
        tensor_results = {}
        for tensor_name, anchor_tensor in anchor.items():
            atol = 2e-2 if tensor_name in ("out", "lse") else 3e-2
            tensor_results[tensor_name] = _cross_error(
                candidates[name][tensor_name], anchor_tensor, atol=atol
            )
        result["backends"][name] = {"status": "pass", "tensors": tensor_results}
        print(f"{name}: PASS", flush=True)
    del runners, candidates
    torch.cuda.empty_cache()
    return result


def _safe_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        if package == "nvidia-cutlass-dsl":
            return getattr(importlib.import_module("cutlass"), "__version__", None)
        if package == "quack-kernels":
            return getattr(importlib.import_module("quack"), "__version__", None)
        return None


def _release_version(value: str) -> tuple[int, ...]:
    release = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        release.append(int(digits))
    return tuple(release)


def _version_at_least(value: str | None, minimum: str) -> bool:
    if value is None:
        return False
    actual_release = _release_version(value)
    minimum_release = _release_version(minimum)
    width = max(len(actual_release), len(minimum_release))
    return actual_release + (0,) * (width - len(actual_release)) >= minimum_release + (0,) * (
        width - len(minimum_release)
    )


def collect_provenance(
    fa4: Fa4Modules | None,
    magi: MagiModules | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver": driver,
        "torch": torch.__version__,
        "torch_recompile_limit": torch._dynamo.config.recompile_limit,
        "cutlass_dsl": _safe_version("nvidia-cutlass-dsl"),
        "quack": _safe_version("quack-kernels"),
        "flex_commit": _git_output(root, "rev-parse", "HEAD"),
        "flex_dirty": bool(_git_output(root, "status", "--porcelain", "--untracked-files=no")),
        "fa4_repository": FA4_REPOSITORY,
        "fa4_commit": fa4.commit if fa4 is not None else None,
        "fa4_root": str(fa4.root) if fa4 is not None else None,
        "magi_repository": MAGI_REPOSITORY,
        "magi_commit": magi.commit if magi is not None else None,
        "magi_root": str(magi.root) if magi is not None else None,
    }


def _workload_json(workload: Workload) -> dict[str, Any]:
    result = asdict(workload)
    result["dtype"] = str(workload.dtype).removeprefix("torch.")
    return result


def _metric_value(case: dict[str, Any], backend: str, metric: str, field: str) -> float | None:
    backend_result = case["backends"].get(backend, {})
    if backend_result.get("status") != "ok":
        return None
    return backend_result["metrics"].get(metric, {}).get(field)


def add_comparisons(results: dict[str, Any]) -> None:
    priorities = []
    for case in results.get("benchmark", []):
        comparisons = {}
        for phase in PHASE_NAMES:
            flex_ms = _metric_value(case, "flex", phase, "median_ms")
            if flex_ms is None:
                continue
            phase_result = {}
            baselines = []
            for baseline in ("fa4", "magi", "torch"):
                baseline_ms = _metric_value(case, baseline, phase, "median_ms")
                if baseline_ms is not None:
                    phase_result[f"speedup_vs_{baseline}"] = baseline_ms / flex_ms
                    baselines.append(baseline_ms)
            if baselines:
                best = min(baselines)
                phase_result["slowdown_vs_best"] = flex_ms / best - 1.0
                priorities.append(
                    {
                        "mask": case["mask"],
                        "phase": phase,
                        "slowdown_vs_best": flex_ms / best - 1.0,
                    }
                )
            comparisons[phase] = phase_result
        case["comparisons"] = comparisons
        metadata_ms = _metric_value(case, "flex", "metadata", "median_ms")
        combined_ms = _metric_value(case, "flex", "combined", "median_ms")
        if metadata_ms is not None and combined_ms is not None:
            case["flex_plan_fraction_one_step"] = metadata_ms / (metadata_ms + combined_ms)
    results["optimization_priority"] = sorted(
        priorities, key=lambda item: item["slowdown_vs_best"], reverse=True
    )


def write_csv(path: Path, cases: Sequence[dict[str, Any]]) -> None:
    fields = (
        "mask",
        "backend",
        "status",
        "density",
        "nfunc",
        "metadata_ms",
        "forward_ms",
        "forward_tflops",
        "backward_ms",
        "backward_tflops",
        "combined_ms",
        "combined_tflops",
        "metadata_forward_ms",
        "metadata_combined_ms",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for backend, backend_result in case["backends"].items():
                metrics = backend_result.get("metrics", {})
                writer.writerow(
                    {
                        "mask": case["mask"],
                        "backend": backend,
                        "status": backend_result["status"],
                        "density": case["element_density"],
                        "nfunc": case["nfunc"],
                        "metadata_ms": metrics.get("metadata", {}).get("median_ms"),
                        "forward_ms": metrics.get("forward", {}).get("median_ms"),
                        "forward_tflops": metrics.get("forward", {}).get("active_tflops"),
                        "backward_ms": metrics.get("backward", {}).get("median_ms"),
                        "backward_tflops": metrics.get("backward", {}).get("active_tflops"),
                        "combined_ms": metrics.get("combined", {}).get("median_ms"),
                        "combined_tflops": metrics.get("combined", {}).get("active_tflops"),
                        "metadata_forward_ms": metrics.get("metadata_forward", {}).get("median_ms"),
                        "metadata_combined_ms": metrics.get("metadata_combined", {}).get(
                            "median_ms"
                        ),
                        "error": backend_result.get("error"),
                    }
                )


def _format_value(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_readme_tables(results: dict[str, Any]) -> str:
    cases = results.get("benchmark", [])
    provenance = results.get("provenance", {})
    workload = results.get("workload", {})
    protocol = results.get("protocol", {})
    seqlen = workload.get("seqlen")
    seqlen_label = (
        f"{seqlen // 1024}K" if isinstance(seqlen, int) and seqlen % 1024 == 0 else str(seqlen)
    )
    lines = [
        "## Standard Static-Mask Benchmark",
        "",
        "![Static attention mask shapes](docs/assets/static_mask_shapes.png)",
        "",
        "The figure is schematic; benchmark lengths and document counts follow the "
        "definitions below.",
        "",
        f"All values are medians on {provenance.get('gpu', 'B300/SM103')} with BF16, "
        f"B=1, S={seqlen_label}, Hq=Hkv=4, and Dqk=Dv=128.",
        f"Each kernel metric uses {protocol.get('warmup', 5)} warmups and "
        f"{protocol.get('runs', 10)} measured runs; metadata uses "
        f"{protocol.get('metadata_runs', 3)} measured runs. L2 is flushed before every "
        "sample.",
        "FA4 uses native causal for the Causal case and the official Torch BlockMask "
        "mask_mod path for the other masks at commit "
        f"`{provenance.get('fa4_commit', FA4_COMMIT)}`.",
        "Magi backend uses its CUDA Q2K/K2Q CSR planner at commit "
        f"`{provenance.get('magi_commit', MAGI_COMMIT)}`.",
        "Forward, backward, and train timings exclude metadata. Backward graphs are "
        "rebuilt outside the timed region.",
        "TFLOPS use actual visible attention pairs; train means one forward plus one backward.",
        "The forward-table speedup is baseline time divided by Flex time, so values "
        "above 1.0x favor Flex.",
        "",
        "### Performance Comparison",
        "",
        "![Static-mask attention performance on NVIDIA GB300]"
        "(docs/assets/static_mask_benchmark.png)",
        "",
        "The first three panels report active TFLOPS/s; the mask-plan panel reports "
        "construction latency. All bars use the same medians as the tables below.",
        "",
        "### Metadata",
        "",
        "| Mask | Density | Flex blocks (full/partial) | Flex plan (ms) | FA4 plan (ms) | Magi plan (ms) | Torch plan (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        flex_backend = case["backends"].get("flex", {})
        block_stats = flex_backend.get("block_stats") or {}
        block_cell = (
            f"{block_stats.get('full', 0):,}/{block_stats.get('partial', 0):,}"
            if block_stats
            else "N/A"
        )
        lines.append(
            f"| {case['title']} | {case['element_density']:.2%} | {block_cell} | "
            f"{_format_value(_metric_value(case, 'flex', 'metadata', 'median_ms'))} | "
            f"{_format_value(_metric_value(case, 'fa4', 'metadata', 'median_ms'))} | "
            f"{_format_value(_metric_value(case, 'magi', 'metadata', 'median_ms'))} | "
            f"{_format_value(_metric_value(case, 'torch', 'metadata', 'median_ms'))} |"
        )
    lines.extend(
        (
            "",
            "### Forward",
            "",
            "| Mask | Flex ms / TFLOPS | FA4 ms / Flex speedup | Magi ms / Flex speedup | Torch ms / Flex speedup |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for case in cases:
        flex_ms = _metric_value(case, "flex", "forward", "median_ms")
        flex_tf = _metric_value(case, "flex", "forward", "active_tflops")
        fa_ms = _metric_value(case, "fa4", "forward", "median_ms")
        magi_ms = _metric_value(case, "magi", "forward", "median_ms")
        torch_ms = _metric_value(case, "torch", "forward", "median_ms")
        lines.append(
            f"| {case['title']} | {_format_value(flex_ms)} / {_format_value(flex_tf, 1)} | "
            f"{_format_value(fa_ms)} / {_format_value(fa_ms / flex_ms if fa_ms and flex_ms else None, 2)}x | "
            f"{_format_value(magi_ms)} / "
            f"{_format_value(magi_ms / flex_ms if magi_ms and flex_ms else None, 2)}x | "
            f"{_format_value(torch_ms)} / "
            f"{_format_value(torch_ms / flex_ms if torch_ms and flex_ms else None, 2)}x |"
        )
    lines.extend(
        (
            "",
            "### Backward and Training",
            "",
            "| Mask | Flex BWD / train ms | FA4 BWD / train ms | Magi BWD / train ms | Torch BWD / train ms |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for case in cases:
        cells = []
        for backend in BACKEND_NAMES:
            bwd = _metric_value(case, backend, "backward", "median_ms")
            combined = _metric_value(case, backend, "combined", "median_ms")
            cells.append(f"{_format_value(bwd)} / {_format_value(combined)}")
        lines.append(f"| {case['title']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    cases_by_name = {case["mask"]: case for case in cases}
    if set(MASK_NAMES).issubset(cases_by_name):
        local_case = cases_by_name["local"]
        sink_case = cases_by_name["sink_local"]
        dfs_case = cases_by_name["tree_dfs"]
        bfs_case = cases_by_name["tree_bfs"]
        longformer_case = cases_by_name["longformer"]
        local_fwd = _metric_value(local_case, "flex", "forward", "median_ms")
        sink_bwd = _metric_value(sink_case, "flex", "backward", "median_ms")
        local_bwd = _metric_value(local_case, "flex", "backward", "median_ms")
        dfs_fwd = _metric_value(dfs_case, "flex", "forward", "median_ms")
        bfs_fwd = _metric_value(bfs_case, "flex", "forward", "median_ms")
        longformer_fwd = _metric_value(longformer_case, "flex", "forward", "median_ms")
        if all(
            value is not None
            for value in (
                local_fwd,
                sink_bwd,
                local_bwd,
                dfs_fwd,
                bfs_fwd,
                longformer_fwd,
            )
        ):
            lines.extend(
                (
                    "",
                    "### Optimization Signals",
                    "",
                    "- Sink + local has almost the same element density as local, but its "
                    f"Flex backward is {sink_bwd / local_bwd:.2f}x slower. This suggests "
                    "that the transposed high-connectivity sink region is the main backward "
                    "target.",
                    "- DFS and BFS have identical visible-pair counts, while BFS has more "
                    "partial blocks (8,100 versus 4,478) and makes Flex forward "
                    f"{bfs_fwd / dfs_fwd - 1.0:.1%} slower. Plan ordering and locality remain "
                    "material.",
                    "- Longformer has nearly local-mask element density, but global tokens "
                    "expand active tiles from 3,066 to 15,228 and make Flex forward "
                    f"{longformer_fwd / local_fwd:.2f}x slower. Global-row tile amplification "
                    "is the clearest forward optimization target.",
                )
            )
    return "\n".join(lines) + "\n"


def _parse_selection(value: str, allowed: Sequence[str], *, label: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(allowed)
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(selected) - set(allowed))
    if not selected or invalid:
        raise ValueError(f"invalid {label}: {invalid or value}; expected {allowed}")
    return selected


def _validate_environment(backend_names: Sequence[str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the standard benchmark requires CUDA")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError(
            "the standard README benchmark requires B300/SM103; got "
            f"{torch.cuda.get_device_name()} {torch.cuda.get_device_capability()}"
        )
    cutlass_version = _safe_version("nvidia-cutlass-dsl")
    quack_version = _safe_version("quack-kernels")
    uses_flex = "flex" in backend_names
    uses_magi = "magi" in backend_names
    if (uses_flex or uses_magi) and "fa4" in backend_names:
        raise RuntimeError(
            "official FA4 and Flex/Magi require different CUTLASS DSL environments; "
            "run FA4 separately with --backend fa4,torch"
        )
    if uses_flex and (
        not _version_at_least(cutlass_version, "4.5.2") or quack_version != "0.5.0"
    ):
        raise RuntimeError(
            "Flex benchmark requires nvidia-cutlass-dsl>=4.5.2 and "
            "quack-kernels==0.5.0; "
            f"got {cutlass_version} and {quack_version}"
        )
    if uses_magi and (cutlass_version, quack_version) != ("4.5.2", "0.5.0"):
        raise RuntimeError(
            "Magi benchmark requires nvidia-cutlass-dsl==4.5.2 and "
            "quack-kernels==0.5.0; "
            f"got {cutlass_version} and {quack_version}"
        )
    if "fa4" in backend_names and (
        not _version_at_least(cutlass_version, "4.6.2")
        or not _version_at_least(quack_version, "0.5.3")
    ):
        raise RuntimeError(
            "official FA4 benchmark requires nvidia-cutlass-dsl>=4.6.2 and "
            "quack-kernels>=0.5.3; "
            f"got {cutlass_version} and {quack_version}"
        )


def _configure_torch_compile() -> None:
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, TORCH_RECOMPILE_LIMIT
    )
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, TORCH_RECOMPILE_LIMIT
    )


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / timestamp


def _write_result_files(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if results.get("benchmark"):
        write_csv(output_dir / "results.csv", results["benchmark"])
        (output_dir / "readme_tables.md").write_text(
            render_readme_tables(results), encoding="utf-8"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark eight standard static masks across FlexAttention, FA4, "
            "Magi backend, and PyTorch"
        )
    )
    parser.add_argument("--mode", choices=("benchmark", "correctness"), default="benchmark")
    parser.add_argument("--mask", default="all", help="all or a comma-separated mask list")
    parser.add_argument(
        "--backend",
        default=",".join(DEFAULT_BACKEND_NAMES),
        help="comma-separated backend list; run official FA4 in its separate environment",
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="all or a comma-separated subset of forward,backward,combined",
    )
    parser.add_argument("--seqlen", type=int, default=STANDARD_SEQLEN)
    parser.add_argument("--correctness-seqlen", type=int, default=CORRECTNESS_SEQLEN)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--metadata-runs", type=int, default=3)
    parser.add_argument(
        "--fa4-root",
        type=Path,
        default=Path(os.environ.get("FA4_ROOT", DEFAULT_FA4_ROOT)),
    )
    parser.add_argument(
        "--magi-root",
        type=Path,
        default=Path(os.environ.get("MAGI_ROOT", DEFAULT_MAGI_ROOT)),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        masks = _parse_selection(args.mask, MASK_NAMES, label="mask")
        backends = _parse_selection(args.backend, BACKEND_NAMES, label="backend")
        phases = (
            PHASE_NAMES
            if args.phase == "all"
            else _parse_selection(args.phase, PHASE_NAMES, label="phase")
        )
    except ValueError as error:
        parser.error(str(error))
    if args.seqlen <= 0 or args.correctness_seqlen <= 0:
        parser.error("seqlen and correctness-seqlen must be positive")
    if args.warmup < 0 or args.runs <= 0 or args.metadata_runs <= 0:
        parser.error("warmup must be non-negative; runs and metadata-runs must be positive")
    if args.dry_run:
        print(
            f"mode={args.mode} seqlen={args.seqlen} masks={len(masks)} "
            f"backends={','.join(backends)} phases={','.join(phases)}"
        )
        for name in masks:
            spec = make_mask_spec(name, args.seqlen)
            print(
                f"{spec.name}: nfunc={spec.nfunc} density={spec.density:.6%} "
                f"visible_pairs={spec.visible_pairs}"
            )
        return

    _validate_environment(backends)
    _configure_torch_compile()
    fa4 = load_fa4(args.fa4_root) if "fa4" in backends else None
    magi = load_magi(args.magi_root) if "magi" in backends else None
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    results: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "provenance": collect_provenance(fa4, magi),
        "workload": _workload_json(Workload(seqlen=args.seqlen)),
        "protocol": {
            "warmup": args.warmup,
            "runs": args.runs,
            "metadata_runs": args.metadata_runs,
            "l2_flush_before_each_sample": True,
            "backends": list(backends),
            "phases": list(phases),
            "masks": list(masks),
        },
        "benchmark": [],
        "correctness_fp32": [],
        "correctness_full_cross": [],
    }
    print(
        f"gpu={results['provenance']['gpu']} flex={results['provenance']['flex_commit']} "
        f"fa4={results['provenance']['fa4_commit']} "
        f"magi={results['provenance']['magi_commit']} output={output_dir}",
        flush=True,
    )
    if args.mode == "benchmark":
        workload = Workload(seqlen=args.seqlen)
        for name in masks:
            spec = make_mask_spec(name, workload.seqlen)
            case = benchmark_mask(
                workload,
                spec,
                backends,
                phases,
                fa4=fa4,
                magi=magi,
                warmup=args.warmup,
                runs=args.runs,
                metadata_runs=args.metadata_runs,
            )
            results["benchmark"].append(case)
            add_comparisons(results)
            _write_result_files(output_dir, results)
    else:
        reference_seqlen = min(args.correctness_seqlen, args.seqlen)
        reference_workload = Workload(seqlen=reference_seqlen)
        full_workload = Workload(seqlen=args.seqlen)
        for name in masks:
            reference_spec = make_mask_spec(name, reference_seqlen)
            results["correctness_fp32"].append(
                validate_against_fp32(
                    reference_workload,
                    reference_spec,
                    backends,
                    fa4=fa4,
                    magi=magi,
                )
            )
            full_spec = (
                reference_spec
                if reference_seqlen == args.seqlen
                else make_mask_spec(name, args.seqlen)
            )
            results["correctness_full_cross"].append(
                validate_full_cross_backend(
                    full_workload,
                    full_spec,
                    backends,
                    fa4=fa4,
                    magi=magi,
                )
            )
            _write_result_files(output_dir, results)
    add_comparisons(results)
    _write_result_files(output_dir, results)
    print(f"results={output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
