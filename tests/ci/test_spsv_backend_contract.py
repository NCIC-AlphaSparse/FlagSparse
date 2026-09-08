# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static backend contracts for SpSV/SpSM multi-backend integration."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_SOURCE = (
    PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "_common.py"
).read_text(encoding="utf-8")
SPSV_SOURCE = (
    PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "spsv.py"
).read_text(encoding="utf-8")
SPSM_SOURCE = (
    PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "spsm.py"
).read_text(encoding="utf-8")
SPSV_BENCHMARK_SOURCE = (PROJECT_ROOT / "tests" / "test_spsv.py").read_text(
    encoding="utf-8"
)
SPSM_BENCHMARK_SOURCE = (PROJECT_ROOT / "tests" / "test_spsm.py").read_text(
    encoding="utf-8"
)

COMMON_TREE = ast.parse(COMMON_SOURCE)
SPSV_TREE = ast.parse(SPSV_SOURCE)
SPSM_TREE = ast.parse(SPSM_SOURCE)


def _function_source(tree, source, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name!r} not found")


def test_hipsparse_sparse_descriptors_use_create_ref_contract():
    for name in (
        "_hipsparse_create_csr_descriptor",
        "_hipsparse_create_coo_descriptor",
        "_hipsparse_create_csc_descriptor",
    ):
        source = _function_source(COMMON_TREE, COMMON_SOURCE, name)
        assert "spmat_ref" in source
        assert f"hipsparseCreate{name.rsplit('_', 2)[1].capitalize()}" in source

    assert "spmat_ref = spmat.createRef()" in SPSV_SOURCE


def test_spsv_spsm_sources_keep_update_multi_backend_abstraction():
    for source in (SPSV_SOURCE, SPSM_SOURCE):
        assert "torch.cuda.synchronize()" not in source
        assert "torch.cuda.Event" not in source
        assert ".is_cuda" not in source
        assert "_ACCEL.synchronize()" in source
        assert "_is_accel_tensor(" in source


def test_common_launch_tuning_is_backend_scoped_and_reusable():
    for name in (
        "_clip_num_warps_for_backend",
        "_clip_block_tile_for_backend",
        "_backend_launch_overrides",
        "_spmm_rocm_launch_overrides",
        "_spmv_rocm_launch_overrides",
    ):
        assert f"def {name}" in COMMON_SOURCE

    generic = _function_source(COMMON_TREE, COMMON_SOURCE, "_backend_launch_overrides")
    assert 'if info["backend"] != "hip":' in generic
    assert "kind == \"spmm\"" in generic
    assert "kind == \"spmv\"" in generic

    spmm = _function_source(COMMON_TREE, COMMON_SOURCE, "_spmm_rocm_launch_overrides")
    spmv = _function_source(COMMON_TREE, COMMON_SOURCE, "_spmv_rocm_launch_overrides")
    for source in (spmm, spmv):
        assert "if not _is_rocm_runtime():" in source
        assert "_backend_launch_overrides(" in source


def test_spsv_rocm_alg_updates_are_present_but_backend_scoped():
    assert '"FLAGSPARSE_SPSV_ROCM_ENABLE_PERSISTENT_PARALLEL", "1"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_ROCM_ALG3_BLOCK_NNZ", "256"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_ROCM_ALG3_WORKGROUPS_PER_CU", "4"' in SPSV_SOURCE

    normalize = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_normalize_requested_spsv_route"
    )
    assert '"alg3": "csr_nnz_balance" if is_rocm else "csr_roc"' in normalize
    assert "CUDA-only route" in normalize

    launch = _function_source(SPSV_TREE, SPSV_SOURCE, "_spsv_nnz_balance_launch_config")
    assert "if not is_rocm:" in launch
    assert "SPSV_ROCM_ALG3_BLOCK_NNZ" in launch
    assert "_ACCEL.get_device_properties" in launch


def test_spsv_preserves_update_non_rocm_profiles_and_public_sell_api():
    assert "_MACA_SPSV_PROFILES" in SPSV_SOURCE
    assert "_maca_spsv_knob" in SPSV_SOURCE
    assert "_is_mthreads_runtime()" in SPSV_BENCHMARK_SOURCE
    assert "_is_ascend_runtime()" in SPSV_BENCHMARK_SOURCE
    assert "def flagsparse_spsv_sell(" in SPSV_SOURCE

    init_source = (
        PROJECT_ROOT / "src" / "flagsparse" / "__init__.py"
    ).read_text(encoding="utf-8")
    ops_source = (
        PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations" / "__init__.py"
    ).read_text(encoding="utf-8")
    registry_source = (PROJECT_ROOT / "conf" / "operators.yaml").read_text(
        encoding="utf-8"
    )
    for source in (init_source, ops_source, registry_source):
        assert "flagsparse_spsv_sell" in source


def test_spsv_spsm_vendor_selectors_use_common_backend_policy():
    spsv_selector = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_spsv_csr_sparse_ref_backend"
    )
    spsm_selector = _function_source(
        SPSM_TREE, SPSM_SOURCE, "_spsm_csr_sparse_ref_backend"
    )
    for selector in (spsv_selector, spsm_selector):
        assert "_vendor_sparse_library()" in selector
        assert "vendor == \"hipsparse\"" in selector
        assert "_backend_name()" in selector

    assert "def _expected_vendor_sparse_backend" in COMMON_SOURCE
    assert "_expected_vendor_sparse_backend" in SPSV_BENCHMARK_SOURCE
    assert "_expected_vendor_sparse_backend" in SPSM_BENCHMARK_SOURCE
    assert (
        "fs_spsv_impl._is_rocm_runtime() else \"cuSPARSE\""
        not in SPSV_BENCHMARK_SOURCE
    )
    assert (
        "fs_spsm_impl._is_rocm_runtime() else \"cuSPARSE\""
        not in SPSM_BENCHMARK_SOURCE
    )
