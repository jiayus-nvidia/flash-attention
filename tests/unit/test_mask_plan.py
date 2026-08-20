from collections import Counter
from dataclasses import FrozenInstanceError

import pytest
import torch

from flex_attn.kernels.sm100.bwd.backward_config import (
    resolve_sm100_bwd_consumer_config,
)
from flex_attn.kernels.sm100.fwd.forward_config import (
    resolve_sm100_fwd_consumer_config,
    resolve_sm100_fwd_qstage1_1cta_consumer_config,
    resolve_sm100_fwd_qstage1_2cta_consumer_config,
)
from flex_attn.kernels.sm100.fwd.forward_config_hd256 import (
    resolve_sm100_hd256_fwd_consumer_config,
)
from flex_attn.plan.kernels import BlockSparseTensors, BlockSparseTensorsTorch
from flex_attn.plan.mask_plan import MaskPlanMetadata
from flex_attn.plan.validation import SM100_STANDARD_HEAD_DIMS
from tests.datas.sequence_cases import RANDOM_CASES, smoke_cases


def test_mask_plan_case_matrix():
    assert len(RANDOM_CASES) == 1024
    assert len(smoke_cases("fixed")) + len(smoke_cases("varlen")) == 154
    assert {case.batch_size for case in RANDOM_CASES} == {8}
    assert all(
        len(case.q_lengths) == case.batch_size
        and len(case.k_lengths) == case.batch_size
        for case in RANDOM_CASES
    )
    assert Counter(case.mode for case in RANDOM_CASES) == {"fixed": 512, "varlen": 512}
    assert Counter(case.deterministic for case in RANDOM_CASES) == {False: 512, True: 512}
    assert Counter((case.head_dim, case.head_dim_v) for case in RANDOM_CASES) == {
        (64, 64): 256,
        (128, 128): 256,
        (192, 128): 256,
        (256, 256): 256,
    }
    full_reference_cases = [case for case in RANDOM_CASES if case.full_reference]
    assert len(full_reference_cases) == 1
    assert full_reference_cases[0].mode == "fixed"
    assert full_reference_cases[0].q_bucket == full_reference_cases[0].k_bucket == "1k"
    assert full_reference_cases[0] in smoke_cases("fixed")
    for mode in ("fixed", "varlen"):
        mode_cases = tuple(case for case in RANDOM_CASES if case.mode == mode)
        assert {case.dtype for case in mode_cases} == {"bfloat16", "float16"}
        assert {case.num_kv_heads for case in mode_cases} == {1, 4, 16}
        assert {case.hmask for case in mode_cases} == {1, 16}
        assert {case.nfunc for case in mode_cases} >= {1, 31}
        assert {case.mask_kind for case in mode_cases} >= {
            "discontiguous_full",
            "empty",
            "full",
            "tile_boundary",
        }
        discontiguous_dims = {
            case.head_dim
            for case in mode_cases
            if case.mask_kind == "discontiguous_full"
        }
        assert discontiguous_dims == {128, 192}
    assert "full_block_run_offset" not in BlockSparseTensors._fields
    assert "full_block_runs" not in BlockSparseTensors._fields
    assert "full_block_run_offset" not in BlockSparseTensorsTorch._fields
    assert "full_block_runs" not in BlockSparseTensorsTorch._fields


def test_mask_plan_metadata_is_frozen():
    metadata = MaskPlanMetadata(
        mode="fixed",
        arch=100,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        batch_size=4,
        total_q=512,
        total_k=512,
        max_seqlen_q=128,
        max_seqlen_k=128,
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=128,
        head_dim_v=128,
        hmask=1,
        nfunc=3,
        pack_gqa=True,
        has_backward=True,
    )
    with pytest.raises(FrozenInstanceError):
        metadata.total_q = 0


def test_payload_signature_captures_mma_layout_and_swap_ab():
    config = resolve_sm100_fwd_consumer_config(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=128,
        head_dim_v=128,
        num_q_heads=16,
        num_kv_heads=4,
        is_varlen=False,
        hmask=1,
        pack_gqa=None,
    )
    signature = config.plan_signature
    assert config.q_stage == 2
    assert config.block_size == (256, 128)
    assert signature.mma_atom_layout_id == "tcgen05_f32_ss_qk_cta1_m128n128_major_kk"
    assert signature.swap_ab is False
    assert signature.scheduler_layout_id == "sm100_clc_fwd_work_desc_i32x4_v1"
    assert signature.mma_atom_layout_id in signature.compile_key
    assert signature.swap_ab in signature.compile_key
    assert signature.scheduler_layout_id in signature.compile_key

    qstage1_2cta_config = resolve_sm100_fwd_qstage1_2cta_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=128,
        head_dim_v=128,
        num_q_heads=16,
        num_kv_heads=4,
        is_varlen=True,
        hmask=1,
        pack_gqa=None,
    )
    qstage1_2cta_signature = qstage1_2cta_config.plan_signature
    assert qstage1_2cta_config.q_stage == 1
    assert qstage1_2cta_config.cta_group_size == 2
    assert qstage1_2cta_config.physical_subtiles == 2
    assert qstage1_2cta_config.block_size == (256, 128)
    assert qstage1_2cta_config.pack_gqa is True
    assert qstage1_2cta_signature.kernel_family == "sm100_qstage1_2cta_fwd"
    assert (
        qstage1_2cta_signature.mma_atom_layout_id
        == "tcgen05_f32_ss_qk_cta2_m256n128_major_kk"
    )
    assert qstage1_2cta_signature != signature

    qstage1_1cta_config = resolve_sm100_fwd_qstage1_1cta_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=128,
        head_dim_v=128,
        num_q_heads=16,
        num_kv_heads=4,
        is_varlen=True,
        hmask=1,
        pack_gqa=None,
    )
    qstage1_1cta_signature = qstage1_1cta_config.plan_signature
    assert qstage1_1cta_config.q_stage == 1
    assert qstage1_1cta_config.cta_group_size == 1
    assert qstage1_1cta_config.physical_subtiles == 1
    assert qstage1_1cta_config.block_size == (128, 128)
    assert qstage1_1cta_config.pack_gqa is True
    assert qstage1_1cta_signature.kernel_family == "sm100_qstage1_1cta_fwd"
    assert qstage1_1cta_signature.mma_atom_layout_id.endswith(
        "cta1_m128n128_major_kk"
    )
    assert qstage1_1cta_signature != signature
    assert qstage1_1cta_signature != qstage1_2cta_signature

    for dtype in (torch.float16, torch.bfloat16):
        for head_dim, head_dim_v in SM100_STANDARD_HEAD_DIMS:
            for resolver, expected_q_stage, expected_cta_group_size in (
                (resolve_sm100_fwd_consumer_config, 2, 1),
                (resolve_sm100_fwd_qstage1_1cta_consumer_config, 1, 1),
                (resolve_sm100_fwd_qstage1_2cta_consumer_config, 1, 2),
            ):
                generic_config = resolver(
                    arch=103,
                    dtype=dtype,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    num_q_heads=16,
                    num_kv_heads=4,
                    is_varlen=False,
                    hmask=1,
                    pack_gqa=None,
                )
                assert generic_config.q_stage == expected_q_stage
                assert generic_config.cta_group_size == expected_cta_group_size
            bwd_config = resolve_sm100_bwd_consumer_config(
                arch=103,
                dtype=dtype,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                num_q_heads=16,
                num_kv_heads=4,
                is_varlen=False,
            )
            expected_bwd_cta_group_size = (
                2 if (head_dim, head_dim_v) == (128, 128) else 1
            )
            assert bwd_config.cta_group_size == expected_bwd_cta_group_size


def test_hd256_forward_qstage1_cta_variants():
    config = resolve_sm100_hd256_fwd_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=16,
        num_kv_heads=4,
        is_varlen=True,
        hmask=1,
        pack_gqa=False,
    )
    signature = config.plan_signature
    assert config.q_stage == 1
    assert config.cta_group_size == 1
    assert config.physical_subtiles == 1
    assert config.block_size == (128, 128)
    assert signature.mma_atom_layout_id == "tcgen05_f32_ss_qk_cta1_m128n128_major_kk"
    assert signature.scheduler_layout_id == "sm100_clc_fwd_work_desc_i32x4_v1"

    config_2cta = resolve_sm100_hd256_fwd_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=16,
        num_kv_heads=4,
        is_varlen=True,
        hmask=1,
        pack_gqa=False,
        cta_group_size=2,
    )
    signature_2cta = config_2cta.plan_signature
    assert config_2cta.q_stage == 1
    assert config_2cta.cta_group_size == 2
    assert config_2cta.physical_subtiles == 2
    assert config_2cta.block_size == (256, 128)
    assert signature_2cta.kernel_family == "sm100_hd256_qstage1_2cta_fwd"
    assert signature_2cta.mma_atom_layout_id == "tcgen05_f32_ss_qk_cta2_m256n128_major_kk"
    assert signature_2cta.scheduler_layout_id == "sm100_clc_fwd_work_desc_i32x4_v1"
