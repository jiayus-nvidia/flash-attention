import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from flash_attn_cute import arbitrary_block_sparsity
from flash_attn_cute.arbitrary_block_sparsity import (
    _ArbitraryPlanK2QMaterializeSm100,
    _ArbitraryPlanMaterializeSm100,
    _classify_compile_key,
    _materialize_compile_key,
    _ResolvedSm100BwdTopologyConfig,
    _ResolvedSm100FwdTopologyConfig,
    _ResolvedSm100Hd256DkdvTopologyConfig,
    _ResolvedSm100Hd256DqTopologyConfig,
)
from flash_attn_cute.arbitrary_plan import (
    ArbitraryPlanRuntimeBinding,
    ArbitraryPlanSignature,
    ArbitraryPlanTopology,
    ArbitraryTopologyTensors,
    canonical_blackwell_arch_family,
    validate_arbitrary_attention_plan,
    validate_arbitrary_plan_runtime_binding,
    validate_arbitrary_plan_signature,
)
from flash_attn_cute.cute_dsl_utils import get_aux_tensor_metadata
from flash_attn_cute.sm100_bwd_config import resolve_sm100_bwd_consumer_config
from flash_attn_cute.sm100_fwd_config import resolve_sm100_fwd_consumer_config
from flash_attn_cute.sm100_hd256_2cta_fmha_forward import (
    BlackwellFusedMultiHeadAttentionForward as Sm100Hd256Forward,
)
from flash_attn_cute.sm100_hd256_bwd_config import (
    make_sm100_hd256_dkdv_score_ownership,
    make_sm100_hd256_dq_score_ownership,
    resolve_sm100_hd256_dkdv_consumer_config,
    resolve_sm100_hd256_dq_consumer_config,
)
from flash_attn_cute.sm100_hd256_fwd_config import (
    make_sm100_hd256_fwd_tiled_mma_qk,
    make_sm100_hd256_fwd_tmem_load,
    resolve_sm100_hd256_fwd_consumer_config,
)
from flash_attn_cute.tile_scheduler import SingleTileVarlenScheduler


def _signature() -> ArbitraryPlanSignature:
    return ArbitraryPlanSignature(
        arch_family="sm100",
        direction="forward",
        kernel_family="sm100_generic_fwd",
        tile_m=128,
        tile_n=128,
        q_stage=1,
        cta_group_size=1,
        pack_gqa=False,
        qhead_per_kvhead=1,
        payload_layout_id="sm100_test_v1",
        dq_order_format="none",
        cluster_axis="m",
    )


def _topology_tensors() -> ArbitraryTopologyTensors:
    return ArbitraryTopologyTensors(
        direction="q2k",
        partial_count=object(),
        partial_offset=object(),
        partial_index=object(),
        full_count=None,
        full_offset=None,
        full_index=None,
        cu_total_q_plan_rows=None,
        cu_total_k_plan_rows=None,
        runtime_binding=ArbitraryPlanRuntimeBinding.capture(
            is_varlen=False,
            batch_size=1,
            seqlen_q=1,
            seqlen_k=1,
            total_q=1,
            total_k=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
        ),
    )


def _plan(signature, payload, topology_tensors):
    outer_row_prefix = (
        topology_tensors.cu_total_q_plan_rows
        if topology_tensors.direction == "q2k"
        else topology_tensors.cu_total_k_plan_rows
    )
    return SimpleNamespace(
        plan_signature=signature,
        mask_block_masks=payload,
        topology_tensors=topology_tensors,
        mask_block_cnt=topology_tensors.partial_count,
        mask_block_offset=topology_tensors.partial_offset,
        mask_block_idx=topology_tensors.partial_index,
        full_block_cnt=topology_tensors.full_count,
        full_block_offset=topology_tensors.full_offset,
        full_block_idx=topology_tensors.full_index,
        dq_write_order=topology_tensors.dq_write_order,
        dq_write_order_full=topology_tensors.dq_write_order_full,
        cu_total_m_blocks=outer_row_prefix,
    )


def test_arbitrary_mode_requires_signature_and_payload_together():
    signature = _signature()
    payload = object()
    topology_tensors = _topology_tensors()

    assert (
        validate_arbitrary_attention_plan(
            arbitrary=False,
            block_sparse_tensors=None,
        )
        is None
    )
    with pytest.raises(ValueError, match="create_arbitrary_block_sparse_tensors"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=None,
        )
    assert (
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=_plan(signature, payload, topology_tensors),
        )
        is signature
    )

    with pytest.raises(ValueError, match="plan_signature"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=SimpleNamespace(mask_block_masks=payload),
        )
    with pytest.raises(ValueError, match="mask_block_masks"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=SimpleNamespace(
                plan_signature=signature,
                mask_block_masks=None,
                topology_tensors=topology_tensors,
            ),
        )
    with pytest.raises(ValueError, match="plan_signature"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=SimpleNamespace(
                plan_signature=None,
                mask_block_masks=None,
                topology_tensors=topology_tensors,
            ),
        )
    with pytest.raises(ValueError, match="arbitrary=True"):
        validate_arbitrary_attention_plan(
            arbitrary=False,
            block_sparse_tensors=_plan(signature, payload, topology_tensors),
        )


def test_arbitrary_mode_rejects_topology_drift_and_wrong_direction():
    signature = _signature()
    topology = _topology_tensors()
    plan = _plan(signature, object(), topology)
    plan.mask_block_cnt = object()
    with pytest.raises(ValueError, match="mask_block_cnt"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=plan,
        )

    plan = _plan(signature, object(), topology)
    plan.plan_signature = replace(signature, direction="backward")
    with pytest.raises(ValueError, match="direction"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=plan,
        )

    with pytest.raises(TypeError, match="ArbitraryTopologyTensors"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=SimpleNamespace(
                plan_signature=signature,
                mask_block_masks=object(),
                topology_tensors=object(),
            ),
        )


def test_arbitrary_mode_requires_varlen_q_and_k_row_prefixes_together():
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 9], dtype=torch.int32)
    runtime_binding = ArbitraryPlanRuntimeBinding.capture(
        is_varlen=True,
        batch_size=2,
        seqlen_q=None,
        seqlen_k=None,
        total_q=5,
        total_k=9,
        max_seqlen_q=3,
        max_seqlen_k=5,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
    )
    q_prefix = torch.tensor([0, 1, 2], dtype=torch.int32)
    k_prefix = torch.tensor([0, 1, 2], dtype=torch.int32)
    topology = replace(
        _topology_tensors(),
        runtime_binding=runtime_binding,
        cu_total_q_plan_rows=q_prefix,
        cu_total_k_plan_rows=k_prefix,
    )
    signature = _signature()
    assert (
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=_plan(signature, object(), topology),
        )
        is signature
    )

    for missing_field in ("cu_total_q_plan_rows", "cu_total_k_plan_rows"):
        missing = replace(topology, **{missing_field: None})
        with pytest.raises(ValueError, match="both Q-row and K-row prefixes"):
            validate_arbitrary_attention_plan(
                arbitrary=True,
                block_sparse_tensors=_plan(signature, object(), missing),
            )

    fixed_with_prefixes = replace(
        _topology_tensors(),
        cu_total_q_plan_rows=q_prefix,
        cu_total_k_plan_rows=k_prefix,
    )
    with pytest.raises(ValueError, match="fixed topology must not carry row prefixes"):
        validate_arbitrary_attention_plan(
            arbitrary=True,
            block_sparse_tensors=_plan(signature, object(), fixed_with_prefixes),
        )


def test_architecture_neutral_topology_keeps_physical_factors_explicit():
    topology = ArbitraryPlanTopology(
        tile_m=64,
        tile_n=128,
        q_stage=2,
        cta_group_size=2,
        pack_gqa=True,
        qhead_per_kvhead=4,
    )
    assert topology.physical_subtiles == 4
    assert topology.block_size == (256, 128)
    assert _signature().topology.block_size == (128, 128)

    backward_topology = ArbitraryPlanTopology(
        tile_m=128,
        tile_n=128,
        q_stage=1,
        cta_group_size=2,
        pack_gqa=False,
        qhead_per_kvhead=1,
        cluster_axis="n",
    )
    assert backward_topology.physical_subtiles == 2
    assert backward_topology.block_size == (128, 256)
    assert (
        backward_topology.compile_key
        != replace(backward_topology, cluster_axis="m").compile_key
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tile_m", 128.0),
        ("tile_n", 128.0),
        ("q_stage", 1.0),
        ("cta_group_size", True),
        ("pack_gqa", 0),
        ("qhead_per_kvhead", 1.0),
        ("arch_family", 90),
        ("cluster_axis", 0),
    ],
)
def test_signature_rejects_values_that_only_compare_equal(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(_signature(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("arch_family", "sm110"),
        ("direction", "backward"),
        ("kernel_family", "sm100_hd256_fwd"),
        ("tile_m", 256),
        ("tile_n", 64),
        ("q_stage", 2),
        ("cta_group_size", 2),
        ("pack_gqa", True),
        ("qhead_per_kvhead", 4),
        ("payload_layout_id", "wrong-layout"),
        ("dq_order_format", "rank_only"),
        ("cluster_axis", "n"),
    ],
)
def test_arbitrary_signature_rejects_every_consumer_mismatch(field, value):
    expected = _signature()
    actual = replace(expected, **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_arbitrary_plan_signature(
            actual,
            expected,
            context="test plan",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm100_backward_resolver_d128_grouped_uses_2cta_native_payload(dtype):
    config = resolve_sm100_bwd_consumer_config(
        arch=100,
        dtype=dtype,
        head_dim=128,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=2,
        is_varlen=True,
    )
    assert config.block_size == (128, 256)
    assert config.tile_n == 128
    assert config.sparse_tile_n == 256
    assert config.subtile_factor == 1
    assert config.physical_subtiles == 2
    assert config.cta_group_size == 2
    assert config.cluster_axis == "n"
    assert config.num_mma_threads == 256
    assert config.payload_values_per_thread == 64
    assert config.payload_valid_words == 2
    assert config.payload_padded_words == 4
    assert config.plan_signature.arch_family == "sm100"
    assert config.plan_signature.kernel_family == "sm100_generic_bwd"
    assert config.plan_signature.q_stage == 1
    assert config.plan_signature.cta_group_size == 2
    assert config.plan_signature.cluster_axis == "n"
    assert config.plan_signature.pack_gqa is False
    assert config.plan_signature.qhead_per_kvhead == 2
    assert config.plan_signature.dq_order_format == "rank_only"


@pytest.mark.parametrize("is_varlen", [False, True])
def test_sm100_backward_resolver_d192_preserves_existing_2cta_topology(is_varlen):
    one_cta = resolve_sm100_bwd_consumer_config(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=64,
        head_dim_v=64,
        num_q_heads=4,
        num_kv_heads=4,
        is_varlen=is_varlen,
    )
    config = resolve_sm100_bwd_consumer_config(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=192,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=4,
        is_varlen=is_varlen,
    )
    assert config.tile_m == 128
    assert config.tile_n == 128
    assert config.sparse_tile_m == 128
    assert config.sparse_tile_n == 256
    assert config.block_size == (128, 256)
    assert config.cta_group_size == 2
    assert config.cluster_axis == "n"
    assert config.physical_subtiles == 2
    assert config.num_mma_threads == 256
    assert config.payload_values_per_thread == 64
    assert config.payload_valid_words == 2
    assert config.payload_padded_words == 4
    assert config.plan_signature.cta_group_size == 2
    assert config.plan_signature.cluster_axis == "n"
    assert config.plan_signature.dq_order_format == "rank_only"
    assert config.plan_signature.topology.physical_subtiles == 2
    assert config.plan_signature.topology.block_size == (128, 256)
    assert config.spt is True
    assert config.payload_layout_id != one_cta.payload_layout_id
    assert config.topology_planner_compile_key != one_cta.topology_planner_compile_key
    assert config.planner_compile_key != one_cta.planner_compile_key


@pytest.mark.parametrize(
    "num_kv_heads,qhead_per_kvhead",
    [(2, 2), (1, 4)],
    ids=["gqa_qratio2", "mqa_qratio4"],
)
def test_sm100_backward_d192_grouped_heads_specialize_k2q_compile_key(
    num_kv_heads,
    qhead_per_kvhead,
):
    common = {
        "arch": 100,
        "dtype": torch.bfloat16,
        "head_dim": 192,
        "head_dim_v": 128,
        "num_q_heads": 4,
        "is_varlen": False,
    }
    grouped = resolve_sm100_bwd_consumer_config(
        num_kv_heads=num_kv_heads,
        **common,
    )
    mha = resolve_sm100_bwd_consumer_config(num_kv_heads=4, **common)

    assert grouped.qhead_per_kvhead == qhead_per_kvhead
    assert grouped.plan_signature.pack_gqa is False
    assert grouped.plan_signature.qhead_per_kvhead == qhead_per_kvhead
    assert grouped.block_size == (128, 256)
    assert grouped.cta_group_size == 2
    assert grouped.payload_values_per_thread == 64
    assert grouped.payload_valid_words == 2
    assert grouped.payload_padded_words == 4
    assert grouped.plan_signature != mha.plan_signature
    assert grouped.topology_planner_compile_key != mha.topology_planner_compile_key
    assert grouped.planner_compile_key != mha.planner_compile_key
    assert _classify_compile_key(
        _ResolvedSm100BwdTopologyConfig(grouped)
    ) != _classify_compile_key(_ResolvedSm100BwdTopologyConfig(mha))


def test_sm100_d192_materializer_coordinate_golden_uses_cluster_union_k_tile():
    config = resolve_sm100_bwd_consumer_config(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=192,
        head_dim_v=128,
        num_q_heads=1,
        num_kv_heads=1,
        is_varlen=False,
    )
    materializer = _ArbitraryPlanK2QMaterializeSm100(config)

    # The inherited planner tile is the K256 cluster union. Cluster row 1,
    # CTA rank 1, first physical K row maps to global K = 1 * 256 + 128.
    assert materializer.tile_n == config.sparse_tile_n == 256
    cta1_first_k_in_cluster = config.tile_n
    assert 1 * materializer.tile_n + cta1_first_k_in_cluster == 384

    source = inspect.getsource(_ArbitraryPlanK2QMaterializeSm100._store_payload)
    assert "tiled_mma_sdp.get_slice(cta_rank)" in source
    assert "cute.make_identity_tensor" in source
    assert "(self.tile_n, self.consumer_tile_m)" in source
    assert "local_n_block * Int32(self.tile_n)" in source
    assert "+ Int32(coord[0])" in source


def test_sm100_backward_resolver_supports_sm100_sm103_and_rejects_sm110():
    common = {
        "dtype": torch.bfloat16,
        "head_dim": 192,
        "head_dim_v": 128,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "is_varlen": False,
    }
    configs = [
        resolve_sm100_bwd_consumer_config(arch=arch, **common) for arch in (100, 103)
    ]
    assert [config.arch for config in configs] == [100, 103]
    assert configs[0].plan_signature == configs[1].plan_signature
    assert all(config.cta_group_size == 2 for config in configs)
    assert (
        configs[0].topology_planner_compile_key
        == configs[1].topology_planner_compile_key
    )
    assert _classify_compile_key(
        _ResolvedSm100BwdTopologyConfig(configs[0])
    ) != _classify_compile_key(_ResolvedSm100BwdTopologyConfig(configs[1]))
    fp16 = resolve_sm100_bwd_consumer_config(
        arch=100,
        **{**common, "dtype": torch.float16},
    )
    assert fp16.plan_signature == configs[0].plan_signature
    assert fp16.planner_compile_key != configs[0].planner_compile_key
    with pytest.raises(NotImplementedError, match="SM100/SM103"):
        resolve_sm100_bwd_consumer_config(arch=110, **common)
    with pytest.raises(NotImplementedError, match=r"192, 128"):
        resolve_sm100_bwd_consumer_config(
            arch=100,
            **{**common, "head_dim_v": 96},
        )


def test_builder_does_not_expose_generic_2cta_opt_ins():
    builder_parameters = inspect.signature(
        arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors
    ).parameters
    assert "forward_use_2cta_instrs" not in builder_parameters
    assert "backward_use_2cta_instrs" not in builder_parameters
    assert (
        "use_2cta_instrs"
        not in inspect.signature(resolve_sm100_fwd_consumer_config).parameters
    )
    assert (
        "use_2cta_instrs"
        not in inspect.signature(resolve_sm100_bwd_consumer_config).parameters
    )


def test_sm100_backward_classifier_key_keeps_exact_arch_separate():
    common = {
        "dtype": torch.bfloat16,
        "head_dim": 64,
        "head_dim_v": 64,
        "num_q_heads": 1,
        "num_kv_heads": 1,
        "is_varlen": False,
    }
    sm100 = resolve_sm100_bwd_consumer_config(arch=100, **common)
    # SM100 and SM103 share the payload contract but retain exact-target
    # classifier compilation keys.
    sm103 = resolve_sm100_bwd_consumer_config(arch=103, **common)
    assert sm100.plan_signature == sm103.plan_signature
    assert _classify_compile_key(
        _ResolvedSm100BwdTopologyConfig(sm100)
    ) != _classify_compile_key(_ResolvedSm100BwdTopologyConfig(sm103))


@pytest.mark.parametrize(
    "max_seqlen_q,q_stage,physical_subtiles",
    [
        (128, 1, 1),
        (129, 2, 2),
    ],
)
def test_sm100_forward_resolver_records_consumer_topology(
    max_seqlen_q,
    q_stage,
    physical_subtiles,
):
    config = resolve_sm100_fwd_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=128,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=4,
        is_varlen=True,
        hmask=1,
        pack_gqa=False,
        max_seqlen_q=max_seqlen_q,
    )
    assert config.q_stage == q_stage
    assert config.cta_group_size == 1
    assert config.physical_subtiles == physical_subtiles
    assert config.block_size == (128 * physical_subtiles, 128)
    assert config.payload_values_per_thread == 128
    assert config.payload_valid_words == 4
    assert config.payload_padded_words == 4
    assert config.plan_signature.arch_family == "sm100"
    assert config.plan_signature.kernel_family == "sm100_generic_fwd"
    assert config.plan_signature.q_stage == q_stage
    assert config.plan_signature.cta_group_size == 1


def test_varlen_scheduler_cluster_idx_keeps_cluster_logical_m_block():
    assert "use_cluster_idx" in SingleTileVarlenScheduler.Params.__dataclass_fields__

    create_source = inspect.getsource(SingleTileVarlenScheduler.create)
    assert "params.use_cluster_idx" in create_source
    assert "cute.arch.cluster_idx()[0]" in create_source

    coord_source = inspect.getsource(SingleTileVarlenScheduler._varlen_coord_map)
    assert "if const_expr(params.use_cluster_idx)" in coord_source
    assert "params.cluster_shape_m > 1 and not params.use_cluster_idx" in coord_source

    params = SingleTileVarlenScheduler.Params(
        num_head=1,
        num_batch=3,
        total_q=898,
        num_splits=1,
        max_kvblock_in_l2=1,
        tile_shape_mn=(256, 128),
        cluster_shape_m=2,
        use_cluster_idx=True,
    )
    # Ragged lengths [385, 0, 513] have Q512 logical-row prefix [0, 1, 1, 3].
    # The conservative grid contains four clusters (eight physical CTAs), with
    # the final cluster decoded as invalid padding by the scheduler.
    assert SingleTileVarlenScheduler.get_grid_shape(params)[:2] == (8, 1)


def test_sm100_forward_resolver_keeps_dedicated_hd256_separate():
    with pytest.raises(NotImplementedError, match="generic"):
        resolve_sm100_fwd_consumer_config(
            arch=100,
            dtype=torch.bfloat16,
            head_dim=256,
            head_dim_v=256,
            num_q_heads=1,
            num_kv_heads=1,
            is_varlen=False,
            hmask=1,
            pack_gqa=False,
            max_seqlen_q=128,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm100_hd256_forward_resolver_contract(dtype):
    config = resolve_sm100_hd256_fwd_consumer_config(
        arch=100,
        dtype=dtype,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=8,
        num_kv_heads=2,
        is_varlen=False,
        hmask=8,
        pack_gqa=False,
    )

    assert config.dtype == dtype
    assert config.qhead_per_kvhead == 4
    assert config.pack_gqa is False
    assert config.is_varlen is False
    assert config.kernel_family == "sm100_hd256_fwd"
    assert config.tile_m == 128
    assert config.tile_n == 128
    assert config.q_stage == 1
    assert config.cta_group_size == 2
    assert config.cluster_axis == "m"
    assert config.topology.block_size == config.block_size == (256, 128)
    assert config.topology.physical_subtiles == config.physical_subtiles == 2
    assert config.softmax_threads_per_subtile == 128
    assert config.attention_num_threads == 384
    assert config.payload_values_per_thread == 128
    assert config.payload_valid_words == config.payload_padded_words == 4
    assert (
        config.physical_subtiles,
        config.num_mask_payload_groups,
        config.payload_padded_words,
    ) == (2, 128, 4)

    signature = config.plan_signature
    assert signature.arch_family == "sm100"
    assert signature.direction == "forward"
    assert signature.kernel_family == "sm100_hd256_fwd"
    assert signature.topology == config.topology
    assert signature.dq_order_format == "none"


def test_sm100_hd256_forward_plan_is_portable_but_cubin_keys_are_exact_arch():
    common = {
        "dtype": torch.bfloat16,
        "head_dim": 256,
        "head_dim_v": 256,
        "num_q_heads": 8,
        "num_kv_heads": 2,
        "is_varlen": False,
        "hmask": 1,
        "pack_gqa": False,
    }
    sm100 = resolve_sm100_hd256_fwd_consumer_config(arch=100, **common)
    sm103 = resolve_sm100_hd256_fwd_consumer_config(arch=103, **common)

    # The payload contract is portable across SM100/SM103, while the existing
    # planner wrappers prefix exact arch before caching a compiled CUBIN.
    assert sm100.plan_signature == sm103.plan_signature
    assert sm100.topology_planner_compile_key == sm103.topology_planner_compile_key
    assert sm100.payload_planner_compile_key == sm103.payload_planner_compile_key
    assert _classify_compile_key(sm100) != _classify_compile_key(sm103)
    assert _materialize_compile_key(sm100) != _materialize_compile_key(sm103)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"arch": 101}, "SM100/SM103"),
        ({"arch": 110}, "SM100/SM103"),
        ({"head_dim": 128}, "D=Dv=256"),
        ({"head_dim_v": 128}, "D=Dv=256"),
        ({"dtype": torch.float32}, "FP16 and BF16"),
        ({"pack_gqa": True}, "pack_gqa=False"),
    ],
)
def test_sm100_hd256_forward_resolver_fails_closed(overrides, match):
    common = {
        "arch": 100,
        "dtype": torch.bfloat16,
        "head_dim": 256,
        "head_dim_v": 256,
        "num_q_heads": 8,
        "num_kv_heads": 2,
        "is_varlen": False,
        "hmask": 1,
        "pack_gqa": False,
    }
    with pytest.raises(NotImplementedError, match=match):
        resolve_sm100_hd256_fwd_consumer_config(**{**common, **overrides})


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1], ids=("mha", "gqa", "mqa"))
def test_sm100_hd256_forward_resolver_allows_varlen_grouped_heads(num_kv_heads):
    config = resolve_sm100_hd256_fwd_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=4,
        num_kv_heads=num_kv_heads,
        is_varlen=True,
        hmask=1,
        pack_gqa=False,
    )
    assert config.is_varlen is True
    assert config.qhead_per_kvhead == 4 // num_kv_heads
    assert config.topology_planner_compile_key != (
        resolve_sm100_hd256_fwd_consumer_config(
            arch=103,
            dtype=torch.bfloat16,
            head_dim=256,
            head_dim_v=256,
            num_q_heads=4,
            num_kv_heads=num_kv_heads,
            is_varlen=False,
            hmask=1,
            pack_gqa=False,
        ).topology_planner_compile_key
    )


def test_sm100_hd256_forward_coordinate_helpers_match_2cta_kernel():
    config = resolve_sm100_hd256_fwd_consumer_config(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=1,
        num_kv_heads=1,
        is_varlen=False,
        hmask=1,
        pack_gqa=False,
    )

    mma_source = inspect.getsource(make_sm100_hd256_fwd_tiled_mma_qk)
    assert "tcgen05.CtaGroup.TWO" in mma_source
    assert "(tile_m * _SM100_HD256_FWD_CTA_GROUP_SIZE, tile_n)" in mma_source

    load_source = inspect.getsource(make_sm100_hd256_fwd_tmem_load)
    assert "tcgen05.Ld32x32bOp(tcgen05.Repetition(32))" in load_source
    assert ".get_slice(tidx)" in load_source

    # In the native kernel mma_tile_coord_v is the two-CTA rank used to slice
    # the cooperative MMA.  The two payload planes therefore cover M[0:128]
    # and M[128:256], respectively, in that same rank order.
    kernel_source = inspect.getsource(Sm100Hd256Forward.kernel)
    assert (
        "mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)"
        in kernel_source
    )
    assert "qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)" in kernel_source
    assert tuple(
        (cta_rank, cta_rank * config.tile_m, (cta_rank + 1) * config.tile_m)
        for cta_rank in range(config.cta_group_size)
    ) == ((0, 0, 128), (1, 128, 256))
    coordinate_golden = {
        # (cta_rank, consumer_tidx, word_idx, bit_idx): (q_row, k_col)
        (0, 0, 0, 0): (0, 0),
        (0, 127, 3, 31): (127, 127),
        (1, 0, 0, 0): (128, 0),
        (1, 127, 3, 31): (255, 127),
    }
    for payload_coord, score_coord in coordinate_golden.items():
        cta_rank, consumer_tidx, word_idx, bit_idx = payload_coord
        assert (
            cta_rank * config.tile_m + consumer_tidx,
            word_idx * 32 + bit_idx,
        ) == score_coord


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm100_hd256_backward_resolvers_keep_dq_and_dkdv_payloads_distinct(dtype):
    common = dict(
        arch=103,
        dtype=dtype,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=2,
        num_kv_heads=2,
        is_varlen=False,
        pack_gqa=False,
        use_2cta_instrs=True,
        deterministic=False,
    )
    dq = resolve_sm100_hd256_dq_consumer_config(**common)
    dkdv = resolve_sm100_hd256_dkdv_consumer_config(**common)

    assert dq.block_size == dkdv.block_size == (256, 128)
    assert dq.payload_shape_tail == (2, 128, 4)
    assert dq.payload_values_per_thread == 128
    assert dq.plan_signature.kernel_family == "sm100_hd256_dq"
    assert dq.plan_signature.direction == "backward"
    assert dq.plan_signature.cluster_axis == "m"
    assert dkdv.payload_shape_tail == (4, 256, 1)
    assert dkdv.payload_values_per_thread == 32
    assert dkdv.plan_signature.kernel_family == "sm100_hd256_dkdv"
    assert dkdv.plan_signature.cluster_axis == "n"
    assert dq.plan_signature.dq_order_format == "none"
    assert dkdv.plan_signature.dq_order_format == "none"
    assert dq.payload_layout_id != dkdv.payload_layout_id


def test_sm100_hd256_backward_resolvers_share_family_signature_but_not_cubins():
    common = dict(
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=1,
        num_kv_heads=1,
        is_varlen=False,
        pack_gqa=False,
        use_2cta_instrs=True,
        deterministic=False,
    )
    dq100 = resolve_sm100_hd256_dq_consumer_config(arch=100, **common)
    dq103 = resolve_sm100_hd256_dq_consumer_config(arch=103, **common)
    dkdv100 = resolve_sm100_hd256_dkdv_consumer_config(arch=100, **common)
    dkdv103 = resolve_sm100_hd256_dkdv_consumer_config(arch=103, **common)
    assert dq100.plan_signature == dq103.plan_signature
    assert dkdv100.plan_signature == dkdv103.plan_signature
    assert dq100.planner_compile_key != dq103.planner_compile_key
    assert dkdv100.planner_compile_key != dkdv103.planner_compile_key
    assert _classify_compile_key(
        _ResolvedSm100Hd256DqTopologyConfig(dq100)
    ) != _classify_compile_key(_ResolvedSm100Hd256DqTopologyConfig(dq103))
    assert _classify_compile_key(
        _ResolvedSm100Hd256DkdvTopologyConfig(dkdv100)
    ) != _classify_compile_key(_ResolvedSm100Hd256DkdvTopologyConfig(dkdv103))


@pytest.mark.parametrize("num_kv_heads", [2, 1])
@pytest.mark.parametrize("hmask", [1, 4], ids=("head_broadcast", "head_specific"))
def test_sm100_hd256_backward_resolvers_allow_gqa_mqa(num_kv_heads, hmask):
    common = dict(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=4,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
        is_varlen=False,
        pack_gqa=False,
        use_2cta_instrs=True,
        deterministic=False,
    )
    dq = resolve_sm100_hd256_dq_consumer_config(**common)
    dkdv = resolve_sm100_hd256_dkdv_consumer_config(**common)
    assert dq.qhead_per_kvhead == dkdv.qhead_per_kvhead == 4 // num_kv_heads
    assert dq.hmask == dkdv.hmask == hmask
    assert dq.plan_signature.qhead_per_kvhead == 4 // num_kv_heads
    assert dkdv.plan_signature.qhead_per_kvhead == 4 // num_kv_heads
    assert dq.plan_signature.pack_gqa is False
    assert dkdv.plan_signature.pack_gqa is False


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"arch": 110}, "SM100/SM103"),
        ({"head_dim": 128}, "D=Dv=256"),
        ({"head_dim_v": 128}, "D=Dv=256"),
        ({"hmask": 3}, "Hmask"),
        ({"pack_gqa": True}, "PackGQA"),
        ({"use_2cta_instrs": False}, "2CTA"),
        ({"deterministic": True}, "deterministic"),
    ],
)
def test_sm100_hd256_backward_resolvers_fail_closed(overrides, match):
    common = dict(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=2,
        num_kv_heads=2,
        is_varlen=False,
        pack_gqa=False,
        use_2cta_instrs=True,
        deterministic=False,
    )
    common.update(overrides)
    for resolver in (
        resolve_sm100_hd256_dq_consumer_config,
        resolve_sm100_hd256_dkdv_consumer_config,
    ):
        expected_error = ValueError if "hmask" in overrides else NotImplementedError
        with pytest.raises(expected_error, match=match):
            resolver(**common)


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1], ids=("mha", "gqa", "mqa"))
@pytest.mark.parametrize("hmask", [1, 4], ids=("head_broadcast", "head_specific"))
def test_sm100_hd256_backward_resolvers_allow_varlen_grouped_heads(num_kv_heads, hmask):
    common = dict(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=4,
        num_kv_heads=num_kv_heads,
        hmask=hmask,
        is_varlen=True,
        pack_gqa=False,
        use_2cta_instrs=True,
        deterministic=False,
    )
    dq = resolve_sm100_hd256_dq_consumer_config(**common)
    dkdv = resolve_sm100_hd256_dkdv_consumer_config(**common)
    assert dq.is_varlen is True
    assert dkdv.is_varlen is True
    assert dq.qhead_per_kvhead == dkdv.qhead_per_kvhead == 4 // num_kv_heads
    assert dq.topology_direction == "q2k"
    assert dkdv.topology_direction == "k2q"


def test_sm100_hd256_backward_materializers_use_native_consumer_ownership():
    common = dict(
        arch=100,
        dtype=torch.bfloat16,
        head_dim=256,
        head_dim_v=256,
        num_q_heads=1,
        num_kv_heads=1,
        is_varlen=False,
        pack_gqa=False,
        use_2cta_instrs=True,
        deterministic=False,
    )
    dq = resolve_sm100_hd256_dq_consumer_config(**common)
    dkdv = resolve_sm100_hd256_dkdv_consumer_config(**common)
    dq_materializer = _ArbitraryPlanMaterializeSm100(dq)
    dkdv_materializer = _ArbitraryPlanK2QMaterializeSm100(dkdv)
    assert dq_materializer.tile_m == 256
    assert dkdv_materializer.tile_m == 256
    assert dkdv_materializer.tile_n == 128

    dq_source = inspect.getsource(_ArbitraryPlanMaterializeSm100._store_payload)
    assert "make_sm100_hd256_dq_score_ownership" in dq_source
    assert "Int32(cta_rank)" in dq_source
    dkdv_source = inspect.getsource(_ArbitraryPlanK2QMaterializeSm100._store_payload)
    assert "make_sm100_hd256_dkdv_score_ownership" in dkdv_source
    assert "q_subtile * self.cta_group_size + cta_rank" in dkdv_source
    assert "cta_rank * self.consumer_tile_n" in dkdv_source
    assert "assumed_align=4" in dkdv_source
    assert "_split_hd256_dkdv_wg" in inspect.getsource(
        make_sm100_hd256_dkdv_score_ownership
    )
    assert "partition_D" in inspect.getsource(make_sm100_hd256_dq_score_ownership)


def test_sm100_resolver_matches_dispatch_qstage_when_pack_gqa_is_disabled():
    config = resolve_sm100_fwd_consumer_config(
        arch=103,
        dtype=torch.bfloat16,
        head_dim=128,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=1,
        is_varlen=False,
        hmask=1,
        pack_gqa=False,
        max_seqlen_q=64,
    )
    assert config.pack_gqa is False
    assert config.qhead_per_kvhead == 4
    assert config.q_stage == 2


def test_sm110_signature_is_not_compatible_with_sm100_payloads():
    common = {
        "dtype": torch.bfloat16,
        "head_dim": 128,
        "head_dim_v": 128,
        "num_q_heads": 1,
        "num_kv_heads": 1,
        "is_varlen": False,
        "hmask": 1,
        "pack_gqa": False,
        "max_seqlen_q": 128,
    }
    sm100 = resolve_sm100_fwd_consumer_config(arch=103, **common)
    sm110_cuda12 = resolve_sm100_fwd_consumer_config(arch=101, **common)
    sm110 = resolve_sm100_fwd_consumer_config(arch=110, **common)
    assert sm100.plan_signature.arch_family == "sm100"
    assert sm110_cuda12.plan_signature.arch_family == "sm110"
    assert sm110.plan_signature.arch_family == "sm110"
    assert sm110_cuda12.plan_signature == sm110.plan_signature
    assert (
        sm110_cuda12.topology_planner_compile_key == sm110.topology_planner_compile_key
    )
    assert sm110.q_stage == 1
    with pytest.raises(ValueError, match="arch_family"):
        validate_arbitrary_plan_signature(
            sm100.plan_signature,
            sm110.plan_signature,
            context="cross-family plan",
        )


def test_sm100_planner_cubin_cache_keys_keep_exact_arches_separate():
    common = {
        "dtype": torch.bfloat16,
        "head_dim": 128,
        "head_dim_v": 128,
        "num_q_heads": 1,
        "num_kv_heads": 1,
        "is_varlen": False,
        "hmask": 1,
        "pack_gqa": False,
        "max_seqlen_q": 129,
    }
    sm100a = resolve_sm100_fwd_consumer_config(arch=100, **common)
    sm103a = resolve_sm100_fwd_consumer_config(arch=103, **common)

    # Plans are portable within the family, but compiled CUBINs are not.
    assert sm100a.plan_signature == sm103a.plan_signature
    assert sm100a.topology_planner_compile_key == sm103a.topology_planner_compile_key
    assert _classify_compile_key(
        _ResolvedSm100FwdTopologyConfig(sm100a)
    ) != _classify_compile_key(_ResolvedSm100FwdTopologyConfig(sm103a))
    assert _materialize_compile_key(sm100a) != _materialize_compile_key(sm103a)


@pytest.mark.parametrize(
    "arch,family",
    [(100, "sm100"), (103, "sm100"), (101, "sm110"), (110, "sm110")],
)
def test_canonical_blackwell_arch_family(arch, family):
    assert canonical_blackwell_arch_family(arch) == family


@pytest.mark.parametrize(
    "override,expected",
    [("sm_90a", 90), ("100a", 100), ("sm_103a", 103), ("110f", 110)],
)
def test_fake_plan_builder_arch_honors_override(monkeypatch, override, expected):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setenv("FLASH_ATTENTION_ARCH", override)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    assert (
        arbitrary_block_sparsity._get_plan_builder_arch(torch.device("cuda"))
        == expected
    )


def test_fake_plan_builder_arch_uses_visible_gpu_without_override(monkeypatch):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.delenv("FLASH_ATTENTION_ARCH", raising=False)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: (10, 0),
    )
    assert arbitrary_block_sparsity._get_plan_builder_arch(torch.device("cuda")) == 100


def test_plan_builder_rejects_unvalidated_sm110_before_compile(
    monkeypatch,
):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 101
    )
    q = torch.empty(1, 8, 1, 64, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    func = torch.zeros(1, 1, 8 + 256, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="Thor"):
        arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
            func, q, k, v, pack_gqa=False
        )


def test_fake_sm100_d192_builder_uses_existing_k256_rows_and_cta_payload(monkeypatch):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 100
    )
    for name in (
        "_compile_classify",
        "_compile_materialize",
        "_compile_k2q_count",
        "_compile_k2q_materialize",
    ):
        monkeypatch.setattr(
            arbitrary_block_sparsity, name, lambda *args, **kwargs: None
        )

    q = torch.empty(1, 8, 1, 192, dtype=torch.bfloat16)
    k = torch.empty(1, 257, 1, 192, dtype=torch.bfloat16)
    v = torch.empty(1, 257, 1, 128, dtype=torch.bfloat16)
    func = torch.zeros(1, 1, 8 + 256, dtype=torch.int32)
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )

    bwd_plan = plan.bwd_tensors
    assert bwd_plan is not None
    assert bwd_plan.block_size == (128, 256)
    assert tuple(bwd_plan.mask_block_cnt.shape) == (1, 2)
    assert tuple(bwd_plan.mask_block_masks.shape) == (2, 2, 256, 4)
    assert bwd_plan.plan_signature.cta_group_size == 2
    assert bwd_plan.plan_signature.cluster_axis == "n"


@pytest.mark.parametrize(
    "num_kv_heads,qhead_per_kvhead",
    [(2, 2), (1, 4)],
    ids=["gqa_qratio2", "mqa_qratio4"],
)
@pytest.mark.parametrize("hmask", [1, 4], ids=["hmask_broadcast", "hmask_per_q_head"])
def test_fake_sm100_d192_grouped_heads_builds_1cta_fwd_and_existing_2cta_bwd(
    monkeypatch,
    num_kv_heads,
    qhead_per_kvhead,
    hmask,
):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity,
        "_get_plan_builder_arch",
        lambda device: 100,
    )
    for name in (
        "_compile_classify",
        "_compile_materialize",
        "_compile_k2q_count",
        "_compile_k2q_materialize",
    ):
        monkeypatch.setattr(
            arbitrary_block_sparsity,
            name,
            lambda *args, **kwargs: None,
        )

    q = torch.empty(1, 257, 4, 192, dtype=torch.bfloat16)
    k = torch.empty(1, 257, num_kv_heads, 192, dtype=torch.bfloat16)
    v = torch.empty(1, 257, num_kv_heads, 128, dtype=torch.bfloat16)
    func = torch.zeros(hmask, 1, 257 + 256, dtype=torch.int32)
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )

    assert plan.block_size == (256, 128)
    assert tuple(plan.mask_block_cnt.shape) == (hmask, 2)
    assert tuple(plan.mask_block_masks.shape[1:]) == (2, 128, 4)
    assert plan.plan_signature.pack_gqa is False
    assert plan.plan_signature.qhead_per_kvhead == qhead_per_kvhead
    assert plan.plan_signature.cta_group_size == 1
    bwd_plan = plan.bwd_tensors
    assert bwd_plan is not None
    assert bwd_plan.block_size == (128, 256)
    assert tuple(bwd_plan.mask_block_cnt.shape) == (hmask, 2)
    assert tuple(bwd_plan.mask_block_masks.shape[1:]) == (2, 256, 4)
    assert bwd_plan.plan_signature.pack_gqa is False
    assert bwd_plan.plan_signature.qhead_per_kvhead == qhead_per_kvhead
    assert bwd_plan.plan_signature.cta_group_size == 2


def test_fake_sm100_hd256_builder_uses_q256_rows_and_cta_payload(monkeypatch):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 103
    )
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_compile_classify", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_compile_materialize", lambda *args, **kwargs: None
    )

    q = torch.empty(2, 257, 2, 256, dtype=torch.bfloat16)
    k = torch.empty(2, 129, 2, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    func = torch.zeros(1, 1, q.shape[0] * q.shape[1] + 256, dtype=torch.int32)
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
    )

    assert plan.block_size == (256, 128)
    assert tuple(plan.mask_block_cnt.shape) == (1, 4)
    assert tuple(plan.mask_block_masks.shape[1:]) == (2, 128, 4)
    assert plan.plan_signature.kernel_family == "sm100_hd256_fwd"
    assert plan.plan_signature.cta_group_size == 2
    assert plan.plan_signature.cluster_axis == "m"
    assert plan.topology_tensors.direction == "q2k"
    assert plan.dq_tensors is None


def test_fake_sm100_hd256_backward_builder_emits_three_consumer_views(monkeypatch):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 103
    )
    for name in (
        "_compile_classify",
        "_compile_materialize",
        "_compile_k2q_count",
        "_compile_k2q_materialize",
    ):
        monkeypatch.setattr(
            arbitrary_block_sparsity,
            name,
            lambda *args, **kwargs: None,
        )

    q = torch.empty(2, 257, 2, 256, dtype=torch.bfloat16)
    k = torch.empty(2, 129, 2, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    func = torch.zeros(1, 1, q.shape[0] * q.shape[1] + 256, dtype=torch.int32)
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )

    dq_plan = plan.dq_tensors
    dkdv_plan = plan.bwd_tensors
    assert dq_plan is not None and dkdv_plan is not None
    assert plan.block_size == dq_plan.block_size == dkdv_plan.block_size == (256, 128)
    assert tuple(plan.mask_block_cnt.shape) == (1, 4)
    assert dq_plan.mask_block_cnt is plan.mask_block_cnt
    assert dq_plan.mask_block_idx is plan.mask_block_idx
    assert dq_plan.topology_tensors is plan.topology_tensors
    assert tuple(plan.mask_block_masks.shape[1:]) == (2, 128, 4)
    assert tuple(dq_plan.mask_block_masks.shape[1:]) == (2, 128, 4)
    assert tuple(dkdv_plan.mask_block_cnt.shape) == (1, 4)
    assert tuple(dkdv_plan.mask_block_masks.shape) == (8, 4, 256, 1)
    assert plan.plan_signature.kernel_family == "sm100_hd256_fwd"
    assert dq_plan.plan_signature.kernel_family == "sm100_hd256_dq"
    assert dkdv_plan.plan_signature.kernel_family == "sm100_hd256_dkdv"
    assert dq_plan.dq_write_order is None and dkdv_plan.dq_write_order is None
    assert dkdv_plan.spt is None
    assert (
        plan.topology_tensors.runtime_binding
        is dkdv_plan.topology_tensors.runtime_binding
    )


@pytest.mark.parametrize("num_kv_heads", [2, 1])
def test_fake_sm100_hd256_backward_builder_supports_head_broadcast_gqa_mqa(
    monkeypatch,
    num_kv_heads,
):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 103
    )
    for name in (
        "_compile_classify",
        "_compile_materialize",
        "_compile_k2q_count",
        "_compile_k2q_materialize",
    ):
        monkeypatch.setattr(
            arbitrary_block_sparsity,
            name,
            lambda *args, **kwargs: None,
        )

    num_q_heads = 4
    q = torch.empty(1, 257, num_q_heads, 256, dtype=torch.bfloat16)
    k = torch.empty(1, 129, num_kv_heads, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    func = torch.zeros(1, 1, q.shape[0] * q.shape[1] + 256, dtype=torch.int32)
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )

    qratio = num_q_heads // num_kv_heads
    assert plan.dq_tensors is not None and plan.bwd_tensors is not None
    assert plan.mask_block_cnt.shape[0] == 1
    assert plan.plan_signature.qhead_per_kvhead == qratio
    assert plan.dq_tensors.plan_signature.qhead_per_kvhead == qratio
    assert plan.bwd_tensors.plan_signature.qhead_per_kvhead == qratio
    assert plan.plan_signature.pack_gqa is False
    assert plan.dq_tensors.plan_signature.pack_gqa is False
    assert plan.bwd_tensors.plan_signature.pack_gqa is False


def test_fake_sm100_hd256_backward_builder_supports_head_specific_mqa(monkeypatch):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 103
    )
    q = torch.empty(1, 257, 4, 256, dtype=torch.bfloat16)
    k = torch.empty(1, 129, 1, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    func = torch.zeros(4, 1, q.shape[1] + 256, dtype=torch.int32)
    for name in (
        "_compile_classify",
        "_compile_materialize",
        "_compile_k2q_count",
        "_compile_k2q_materialize",
    ):
        monkeypatch.setattr(
            arbitrary_block_sparsity,
            name,
            lambda *args, **kwargs: None,
        )
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        pack_gqa=False,
        build_backward=True,
    )
    assert plan.dq_tensors is not None and plan.bwd_tensors is not None
    assert plan.mask_block_cnt.shape[0] == q.shape[-2]
    assert plan.dq_tensors.mask_block_cnt.shape[0] == q.shape[-2]
    assert plan.bwd_tensors.mask_block_cnt.shape[0] == q.shape[-2]


def test_fake_sm100_varlen_d192_builder_uses_compact_k256_prefixes(monkeypatch):
    monkeypatch.setattr(arbitrary_block_sparsity, "is_fake_mode", lambda: True)
    monkeypatch.setattr(
        arbitrary_block_sparsity, "_get_plan_builder_arch", lambda device: 100
    )
    for name in (
        "_compile_classify",
        "_compile_materialize",
        "_compile_k2q_count",
        "_compile_k2q_materialize",
    ):
        monkeypatch.setattr(
            arbitrary_block_sparsity, name, lambda *args, **kwargs: None
        )

    q = torch.empty(146, 1, 192, dtype=torch.bfloat16)
    k = torch.empty(386, 1, 192, dtype=torch.bfloat16)
    v = torch.empty(386, 1, 128, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 129, 129, 146], dtype=torch.int32)
    cu_k = torch.tensor([0, 257, 386, 386], dtype=torch.int32)
    func = torch.zeros(1, 1, q.shape[0] + 256, dtype=torch.int32)
    plan = arbitrary_block_sparsity.create_arbitrary_block_sparse_tensors(
        func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=129,
        max_seqlen_k=257,
        pack_gqa=False,
        build_backward=True,
    )

    bwd_plan = plan.bwd_tensors
    assert bwd_plan is not None
    assert bwd_plan.block_size == (128, 256)
    assert tuple(bwd_plan.cu_total_m_blocks.tolist()) == (0, 2, 3, 3)
    topology = bwd_plan.topology_tensors
    assert tuple(topology.cu_total_q_plan_rows.tolist()) == (0, 2, 2, 3)
    assert topology.cu_total_k_plan_rows is bwd_plan.cu_total_m_blocks
    # Fake planning reserves the B*max-K-row upper bound; the compact prefix
    # carries the three runtime-valid rows and prevents zero samples drifting.
    assert tuple(bwd_plan.mask_block_cnt.shape) == (1, 6)
    assert tuple(bwd_plan.mask_block_masks.shape[1:]) == (2, 256, 4)
    assert bwd_plan.plan_signature.cta_group_size == 2


def _fixed_runtime_binding() -> ArbitraryPlanRuntimeBinding:
    return ArbitraryPlanRuntimeBinding.capture(
        is_varlen=False,
        batch_size=2,
        seqlen_q=17,
        seqlen_k=33,
        total_q=34,
        total_k=66,
        max_seqlen_q=17,
        max_seqlen_k=33,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
    )


def _validate_runtime_binding(binding, **overrides):
    runtime = {
        "is_varlen": False,
        "batch_size": 2,
        "seqlen_q": 17,
        "seqlen_k": 33,
        "total_q": 34,
        "total_k": 66,
        "max_seqlen_q": 17,
        "max_seqlen_k": 33,
        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
        "context": "schema test plan",
    }
    runtime.update(overrides)
    return validate_arbitrary_plan_runtime_binding(binding, **runtime)


def test_fixed_runtime_binding_allows_new_values_with_identical_geometry():
    binding = _fixed_runtime_binding()
    assert _validate_runtime_binding(binding) is binding


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 3),
        ("seqlen_q", 18),
        ("seqlen_k", 34),
        ("total_q", 35),
        ("total_k", 67),
        ("max_seqlen_q", 18),
        ("max_seqlen_k", 34),
    ],
)
def test_fixed_runtime_binding_rejects_every_geometry_mismatch(field, value):
    with pytest.raises(ValueError, match=field):
        _validate_runtime_binding(_fixed_runtime_binding(), **{field: value})


def test_varlen_runtime_binding_rejects_equal_aggregate_different_partition():
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 9], dtype=torch.int32)
    binding = ArbitraryPlanRuntimeBinding.capture(
        is_varlen=True,
        batch_size=2,
        seqlen_q=None,
        seqlen_k=None,
        total_q=5,
        total_k=9,
        max_seqlen_q=3,
        max_seqlen_k=5,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
    )
    replacement_q = torch.tensor([0, 1, 5], dtype=torch.int32)
    replacement_k = torch.tensor([0, 5, 9], dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_q provenance mismatch"):
        validate_arbitrary_plan_runtime_binding(
            binding,
            is_varlen=True,
            batch_size=2,
            seqlen_q=None,
            seqlen_k=None,
            total_q=5,
            total_k=9,
            max_seqlen_q=3,
            max_seqlen_k=5,
            cu_seqlens_q=replacement_q,
            cu_seqlens_k=replacement_k,
            context="schema test plan",
        )


def test_varlen_runtime_binding_rejects_in_place_prefix_mutation():
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 9], dtype=torch.int32)
    binding = ArbitraryPlanRuntimeBinding.capture(
        is_varlen=True,
        batch_size=2,
        seqlen_q=None,
        seqlen_k=None,
        total_q=5,
        total_k=9,
        max_seqlen_q=3,
        max_seqlen_k=5,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
    )
    cu_q[1] = 1

    with pytest.raises(ValueError, match="cu_seqlens_q was modified in-place"):
        validate_arbitrary_plan_runtime_binding(
            binding,
            is_varlen=True,
            batch_size=2,
            seqlen_q=None,
            seqlen_k=None,
            total_q=5,
            total_k=9,
            max_seqlen_q=3,
            max_seqlen_k=5,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            context="schema test plan",
        )


def test_aux_metadata_tracks_implicitly_inferred_leading_dimension():
    contiguous = torch.empty((2, 3, 4), dtype=torch.float32)
    middle_contiguous = torch.empty((2, 4, 3), dtype=torch.float32).permute(0, 2, 1)

    assert contiguous.shape == middle_contiguous.shape
    assert contiguous.stride() == (12, 4, 1)
    assert middle_contiguous.stride() == (12, 1, 3)

    contiguous_metadata = get_aux_tensor_metadata([contiguous])
    middle_contiguous_metadata = get_aux_tensor_metadata([middle_contiguous])
    assert contiguous_metadata[0][-1] == 2
    assert middle_contiguous_metadata[0][-1] == 1
    assert contiguous_metadata != middle_contiguous_metadata
