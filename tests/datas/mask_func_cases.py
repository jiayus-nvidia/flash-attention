"""Interval-mask generators shared by correctness tests and benchmarks."""

from __future__ import annotations

import torch

from tests.datas.sequence_cases import AttentionCase


def make_mask_func(case: AttentionCase, device: torch.device | str = "cuda") -> torch.Tensor:
    """Build contiguous sample-local endpoints with shape [Hmask, nfunc, total_q]."""

    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    total_q = sum(case.q_lengths)
    result = torch.empty((case.hmask, case.nfunc, total_q), dtype=torch.int32)
    q_offset = 0
    for q_len, k_len in zip(case.q_lengths, case.k_lengths):
        if case.mask_kind == "random":
            endpoints = torch.randint(
                1,
                k_len + 1,
                (case.hmask, case.nfunc, q_len),
                dtype=torch.int32,
                generator=generator,
            ).sort(dim=1).values
        elif case.mask_kind == "empty":
            endpoints = torch.zeros((case.hmask, case.nfunc, q_len), dtype=torch.int32)
        elif case.mask_kind == "full":
            endpoints = torch.full(
                (case.hmask, case.nfunc, q_len), k_len, dtype=torch.int32
            )
        elif case.mask_kind == "causal":
            end = torch.arange(1, q_len + 1, dtype=torch.int32).clamp(max=k_len)
            endpoints = end.view(1, 1, q_len).expand(case.hmask, case.nfunc, q_len)
        elif case.mask_kind == "tile_boundary":
            end = torch.arange(1, q_len + 1, dtype=torch.int32).clamp(max=k_len)
            begin = torch.clamp(end - 128, min=0)
            endpoints = torch.stack((torch.zeros_like(begin), begin, end), dim=0)
            endpoints = endpoints.unsqueeze(0).expand(case.hmask, 3, q_len)
        elif case.mask_kind == "discontiguous_full":
            endpoints = torch.tensor(
                (64, 256, 384, 512, 640, 768, 896),
                dtype=torch.int32,
            ).view(1, 7, 1)
            endpoints = endpoints.expand(case.hmask, 7, q_len)
        else:
            raise ValueError(f"unsupported mask kind: {case.mask_kind}")
        result[:, :, q_offset : q_offset + q_len] = endpoints
        q_offset += q_len
    return result.to(device=device, non_blocking=True).contiguous()


def make_benchmark_mask_func(
    *,
    mask_kind: str,
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device=device)
    if mask_kind == "causal":
        per_sample = torch.minimum(
            q_idx + 1,
            torch.scalar_tensor(seqlen_k, dtype=torch.int32, device=device),
        )
        return per_sample.repeat(batch_size).view(1, 1, -1).contiguous()
    if mask_kind == "local":
        end = torch.minimum(
            q_idx + 1,
            torch.scalar_tensor(seqlen_k, dtype=torch.int32, device=device),
        )
        begin = torch.clamp(q_idx - 512, min=0)
        zero = torch.zeros_like(begin)
        per_sample = torch.stack((zero, begin, end), dim=0)
        return per_sample.repeat(1, batch_size).view(1, 3, -1).contiguous()
    raise ValueError(f"unsupported benchmark mask kind: {mask_kind}")


__all__ = ["make_benchmark_mask_func", "make_mask_func"]
