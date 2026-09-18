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


def test_hipsparse_spsv_reference_exposes_stage_timing_contract():
    prepare = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_prepare_spsv_csr_ref_hipsparse"
    )
    assert "run_analysis=True" in prepare
    assert "measure_buffer_size=False" in prepare
    assert "buffer_size_ms" in prepare

    benchmark = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_benchmark_spsv_csr_sparse_ref"
    )
    assert '"buffer_size_ms": None' in benchmark
    assert '"analysis_ms": None' in benchmark
    assert '"solve_ms": None' in benchmark
    assert "run_analysis=False" in benchmark
    assert "measure_buffer_size=True" in benchmark

    assert "def _run_spsv_csr_ref_hipsparse_analysis_prepared" in SPSV_SOURCE
    assert "def _benchmark_spsv_hipsparse_stage" in SPSV_SOURCE
    assert "FlagSparse_bufferSize_ms" in SPSV_BENCHMARK_SOURCE
    assert "FlagSparse_solve_ms" in SPSV_BENCHMARK_SOURCE


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
    assert 'kind == "spmm"' in generic
    assert 'kind == "spmv"' in generic

    spmm = _function_source(COMMON_TREE, COMMON_SOURCE, "_spmm_rocm_launch_overrides")
    spmv = _function_source(COMMON_TREE, COMMON_SOURCE, "_spmv_rocm_launch_overrides")
    for source in (spmm, spmv):
        assert "if not _is_rocm_runtime():" in source
        assert "_backend_launch_overrides(" in source


def test_spsv_rocm_alg_updates_are_present_but_backend_scoped():
    assert '"FLAGSPARSE_SPSV_ROCM_ENABLE_PERSISTENT_PARALLEL", "1"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_ROCM_ALG3_BLOCK_NNZ", "256"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_ROCM_ALG3_WORKGROUPS_PER_CU", "4"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_ROCM_TRANS_CW_SERIAL_FALLBACK", "1"' in SPSV_SOURCE
    assert '"FLAGSPARSE_SPSV_CUDA_TRANS_ALG2_WORKGROUPS_PER_CU", "2"' in SPSV_SOURCE
    assert "sell_trans_csc" in SPSV_SOURCE
    assert "_build_spsv_sell_trans_csc_metadata" in SPSV_SOURCE
    assert "_launch_spsv_sell_trans_csc" in SPSV_SOURCE
    assert "_build_spsv_transpose_alg2_gather_metadata" in SPSV_SOURCE
    assert "_build_spsv_transpose_alg2_metadata" in SPSV_SOURCE
    assert "_triton_spsv_csr_transpose_alg2_vector" in SPSV_SOURCE
    assert "_triton_spsv_csr_transpose_alg2_vector_complex" in SPSV_SOURCE

    normalize = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_normalize_requested_spsv_route"
    )
    assert '"alg3": "csr_nnz_balance" if is_rocm else "csr_roc"' in normalize
    assert "CUDA-only route" in normalize
    assert '"transpose_alg2": "transpose_alg2"' in normalize
    assert '"transpose_nnz_balance": "transpose_alg2"' in normalize

    sell_analysis = _function_source(
        SPSV_TREE, SPSV_SOURCE, "flagsparse_spsv_analysis_sell"
    )
    assert "ALG1 keeps the direct SELL scatter queue" in sell_analysis
    assert "ALG2 builds a CSC gather view" in sell_analysis
    assert "compute_dtype = _spsv_effective_compute_dtype(" in sell_analysis

    launch = _function_source(SPSV_TREE, SPSV_SOURCE, "_spsv_nnz_balance_launch_config")
    assert "if not is_rocm:" in launch
    assert "SPSV_ROCM_ALG3_BLOCK_NNZ" in launch
    assert "_ACCEL.get_device_properties" in launch

    nnz_balance = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_triton_spsv_csr_n_lo_nnz_balance_vector"
    )
    nnz_balance_complex = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_triton_spsv_csr_n_lo_nnz_balance_vector_complex"
    )
    for source in (nnz_balance, nnz_balance_complex):
        assert "PERSISTENT=is_rocm and SPSV_ROCM_ENABLE_PERSISTENT_PARALLEL" in source

    worker_count = _function_source(
        SPSV_TREE, SPSV_SOURCE, "_spsv_cu_capped_worker_count"
    )
    assert "SPSV_ROCM_ALG4_WORKER_COUNT" in worker_count
    assert "_ACCEL.get_device_properties" in worker_count

    prepare = _function_source(SPSV_TREE, SPSV_SOURCE, "_prepare_spsv_csr_system")
    assert 'default_solve_kind = "transpose_alg2"' in prepare
    assert "_build_spsv_transpose_alg2_gather_metadata(" in prepare
    assert "_build_spsv_transpose_alg2_metadata(" in prepare

    executor = _function_source(SPSV_TREE, SPSV_SOURCE, "_execute_spsv_csr_plan")
    assert 'elif solve_kind == "transpose_alg2":' in executor
    assert 'trans_alg2_mode == "gather"' in executor
    assert 'trans_alg2_mode == "scatter"' in executor

    buffer_size = _function_source(
        SPSV_TREE, SPSV_SOURCE, "flagsparse_spsv_buffer_size"
    )
    assert (
        'route = "transpose_alg2" if trans_mode in ("T", "C") else "csr_cw"'
        in buffer_size
    )


def test_spsv_preserves_update_non_rocm_profiles_and_public_sell_api():
    assert "_MACA_SPSV_PROFILES" in SPSV_SOURCE
    assert "_maca_spsv_knob" in SPSV_SOURCE
    assert "_is_mthreads_runtime()" in SPSV_BENCHMARK_SOURCE
    assert "_is_ascend_runtime()" in SPSV_BENCHMARK_SOURCE
    assert "def flagsparse_spsv_sell(" in SPSV_SOURCE

    init_source = (PROJECT_ROOT / "src" / "flagsparse" / "__init__.py").read_text(
        encoding="utf-8"
    )
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
        assert 'vendor == "hipsparse"' in selector
        assert "_backend_name()" in selector

    assert "def _expected_vendor_sparse_backend" in COMMON_SOURCE
    assert "_expected_vendor_sparse_backend" in SPSV_BENCHMARK_SOURCE
    assert "_expected_vendor_sparse_backend" in SPSM_BENCHMARK_SOURCE
    assert (
        'fs_spsv_impl._is_rocm_runtime() else "cuSPARSE"' not in SPSV_BENCHMARK_SOURCE
    )
    assert (
        'fs_spsm_impl._is_rocm_runtime() else "cuSPARSE"' not in SPSM_BENCHMARK_SOURCE
    )
