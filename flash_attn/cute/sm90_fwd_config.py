"""Shared SM90 forward consumer configuration for attention and mask planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import Float32
from cutlass.cute.nvgpu import warpgroup


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    num_stages: int = 2


@dataclass(frozen=True)
class _ResolvedSm90FwdConsumerConfig:
    arch: int
    dtype: torch.dtype
    head_dim: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    is_varlen: bool
    pack_gqa: bool
    tile_m: int
    tile_n: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    swap_ab: bool
    physical_subtiles: int
    num_mma_threads: int
    attention_num_threads: int
    num_stages: int
    payload_values_per_thread: int
    payload_valid_words: int
    payload_padded_words: int

    @property
    def block_size(self) -> tuple[int, int]:
        return (self.tile_m, self.tile_n)

    @property
    def topology_planner_compile_key(self) -> tuple:
        qhead_ratio = self.qhead_per_kvhead if self.pack_gqa else 1
        return (
            self.arch,
            self.tile_m,
            self.tile_n,
            self.is_varlen,
            self.pack_gqa,
            qhead_ratio,
        )

    @property
    def payload_planner_compile_key(self) -> tuple:
        return (
            self.topology_planner_compile_key,
            self.dtype,
            self.num_mma_threads,
            self.payload_values_per_thread,
            self.payload_valid_words,
            self.payload_padded_words,
        )


def _tile_size_fwd_sm90(
    head_dim: int,
    head_dim_v: int,
    is_causal: bool,
    is_local: bool,
    sparse_block_size_q: int | None = None,
) -> FwdConfig:
    """Return the native SM90 forward tile configuration."""

    if head_dim <= 64:
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    if head_dim <= 96:
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        return FwdConfig(192, 144, False, True)
    if head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    if head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    return FwdConfig(128, 64 if is_local else 80, True, True)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _native_sm90_fwd_smem_bytes(
    head_dim: int,
    head_dim_v: int,
    config: FwdConfig,
) -> int:
    """Return the safe SM90 forward dynamic shared-memory requirement."""

    tile_hdim = math.ceil(head_dim / 16) * 16
    tile_hdim_v = math.ceil(head_dim_v / 16) * 16
    offset = (2 + 2 * config.num_stages + 2 * config.num_stages) * 8
    fields = (
        config.n_block_size * tile_hdim_v * config.num_stages * 2,
        config.m_block_size * max(tile_hdim, tile_hdim_v) * 2,
        config.n_block_size * tile_hdim * config.num_stages * 2,
        0 if config.mma_pv_is_rs else config.m_block_size * config.n_block_size * 2,
    )
    for size in fields:
        offset = _align_up(offset, 1024) + size
    return _align_up(offset, 1024)


def sm90_native_fwd_can_implement(head_dim: int, head_dim_v: int) -> bool:
    """Match native SM90 public-forward resource and codegen coverage."""

    config = _tile_size_fwd_sm90(head_dim, head_dim_v, True, False)
    tile_hdim_v = math.ceil(head_dim_v / 16) * 16
    # The 3-WG RS+overlap path reserves 160 registers for each MMA thread.
    # A value accumulator wider than 128 exceeds that allocation.
    if head_dim <= 64 and tile_hdim_v > 128:
        return False
    return _native_sm90_fwd_smem_bytes(head_dim, head_dim_v, config) <= 232448


def _resolve_pack_gqa(
    *,
    requested_pack_gqa: bool | None,
    num_q_heads: int,
    num_kv_heads: int,
    hmask: int,
) -> bool:
    """Resolve PackGQA without specializing on the runtime mask-head extent."""

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if hmask not in (1, num_q_heads):
        raise ValueError(f"Hmask must be 1 or Hq ({num_q_heads}); got {hmask}")
    if requested_pack_gqa is True and hmask != 1:
        raise ValueError("pack_gqa=True requires Hmask=1 for arbitrary attention")
    if requested_pack_gqa is not None:
        return requested_pack_gqa
    return num_q_heads > num_kv_heads and hmask == 1


def resolve_sm90_fwd_consumer_config(
    *,
    arch: int,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    num_q_heads: int,
    num_kv_heads: int,
    is_varlen: bool,
    hmask: int,
    pack_gqa: bool | None,
) -> _ResolvedSm90FwdConsumerConfig:
    """Resolve the exact SM90 forward consumer signature used by a packed plan."""

    if arch // 10 != 9:
        raise NotImplementedError("Arbitrary attention currently supports SM90 only")
    if dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("SM90 arbitrary attention supports FP16 and BF16 only")
    if not (8 <= head_dim <= 256 and 8 <= head_dim_v <= 256):
        raise ValueError("SM90 head_dim and head_dim_v must be in [8, 256]")
    alignment = 16 // torch.empty((), dtype=dtype).element_size()
    if head_dim % alignment != 0 or head_dim_v % alignment != 0:
        raise ValueError(f"head_dim and head_dim_v must be divisible by {alignment} for {dtype}")
    if not sm90_native_fwd_can_implement(head_dim, head_dim_v):
        raise NotImplementedError(
            "SM90 arbitrary forward only supports (head_dim, head_dim_v) "
            "signatures implemented by native SM90 CuTe DSL forward; "
            f"got ({head_dim}, {head_dim_v})"
        )

    qhead_per_kvhead = num_q_heads // num_kv_heads
    effective_pack_gqa = _resolve_pack_gqa(
        requested_pack_gqa=pack_gqa,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
    )
    # Match the native SM90 consumer configuration. Arbitrary masking changes
    # only the block traversal and mask payload, not the dimension policy.
    fwd = _tile_size_fwd_sm90(head_dim, head_dim_v, True, False)
    num_mma_threads = 128 * (fwd.m_block_size // 64)
    payload_values_per_thread, remainder = divmod(
        fwd.m_block_size * fwd.n_block_size, num_mma_threads
    )
    if remainder:
        raise AssertionError("QK accumulator values must partition evenly across MMA threads")
    payload_valid_words = math.ceil(payload_values_per_thread / 32)
    # Keep the consumer-native payload compact. The common 1/2/4-word rows are
    # naturally vectorized; uncommon widths avoid carrying unused words.
    payload_padded_words = payload_valid_words

    return _ResolvedSm90FwdConsumerConfig(
        arch=arch,
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        is_varlen=is_varlen,
        pack_gqa=effective_pack_gqa,
        tile_m=fwd.m_block_size,
        tile_n=fwd.n_block_size,
        mma_pv_is_rs=fwd.mma_pv_is_rs,
        intra_wg_overlap=fwd.intra_wg_overlap,
        swap_ab=False,
        physical_subtiles=1,
        num_mma_threads=num_mma_threads,
        attention_num_threads=128 + num_mma_threads,
        num_stages=fwd.num_stages,
        payload_values_per_thread=payload_values_per_thread,
        payload_valid_words=payload_valid_words,
        payload_padded_words=payload_padded_words,
    )


@cute.jit
def make_sm90_fwd_tiled_mma_qk(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
):
    """Build the QK tiled MMA layout shared by the planner and consumer."""

    return sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(tile_m // 64, 1, 1),
        tiler_mn=(64, tile_n),
    )


@cute.jit
def make_sm90_fwd_tiled_mma(
    dtype: type[cutlass.Numeric],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    tile_hdimv: cutlass.Constexpr[int],
    mma_pv_is_rs: cutlass.Constexpr[bool],
):
    """Build the shared QK/PV tiled MMA layouts for planner and consumer."""

    tiled_mma_qk = make_sm90_fwd_tiled_mma_qk(dtype, tile_m, tile_n)
    tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
        dtype,
        dtype,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN,
        Float32,
        atom_layout_mnk=(tile_m // 64, 1, 1),
        tiler_mn=(64, tile_hdimv),
        a_source=(warpgroup.OperandSource.RMEM if mma_pv_is_rs else warpgroup.OperandSource.SMEM),
    )
    return tiled_mma_qk, tiled_mma_pv


__all__ = [
    "FwdConfig",
    "_ResolvedSm90FwdConsumerConfig",
    "_resolve_pack_gqa",
    "_tile_size_fwd_sm90",
    "make_sm90_fwd_tiled_mma",
    "make_sm90_fwd_tiled_mma_qk",
    "resolve_sm90_fwd_consumer_config",
    "sm90_native_fwd_can_implement",
]
