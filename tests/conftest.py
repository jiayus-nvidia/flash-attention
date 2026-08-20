"""Pytest configuration for generated FlexAttention cases."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.datas.sequence_cases import RANDOM_CASES, smoke_cases


def pytest_addoption(parser):
    parser.addoption(
        "--full-random-cases",
        action="store_true",
        help="run all 1024 generated correctness cases",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        help="run FlexAttention GPU correctness tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a supported NVIDIA GPU")


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    mode = "varlen" if metafunc.function.__name__ == "test_flex_attn_varlen" else "fixed"
    cases = (
        tuple(case for case in RANDOM_CASES if case.mode == mode)
        if metafunc.config.getoption("--full-random-cases")
        else smoke_cases(mode)
    )
    metafunc.parametrize("case", cases, ids=lambda case: case.id)


def pytest_collection_modifyitems(config, items):
    run_gpu = config.getoption("--run-gpu") and torch.cuda.is_available()
    if not run_gpu:
        skip = pytest.mark.skip(reason="pass --run-gpu on SM90/SM100/SM103 to run GPU tests")
        for item in items:
            if "gpu" in Path(str(item.fspath)).parts:
                item.add_marker(skip)
        return

    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return

    from flex_attn.kernels.sm90.backward_config import sm90_native_bwd_can_implement

    for item in items:
        if "gpu" not in Path(str(item.fspath)).parts:
            continue
        case = getattr(item, "callspec", None)
        case = None if case is None else case.params.get("case")
        if case is None:
            continue
        if not sm90_native_bwd_can_implement(
            case.head_dim,
            case.head_dim_v,
            case.num_q_heads,
            case.num_kv_heads,
        ):
            item.add_marker(pytest.mark.skip(reason="unsupported SM90 training signature"))
