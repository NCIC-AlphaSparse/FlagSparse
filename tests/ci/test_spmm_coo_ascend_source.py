# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Guard the Ascend COO SpMM route against unsupported sparse coalesce."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "src/flagsparse/sparse_operations/spmm_coo.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function_source(name):
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node)


def test_public_ascend_route_bypasses_npu_coalesce():
    source = _function_source("_run_spmm_coo_route")
    dispatch = source.index("if _is_xpu_runtime() or (")
    canonicalize = source.index("_prepare_spmm_coo_canonical_inputs(")

    assert "_is_ascend_runtime() and _use_spmm_coo_ascend_dispatch()" in source
    assert "_spmm_coo_ascend_scatter(" in source
    assert dispatch < canonicalize
    assert "_spmm_coo_compute_dtype(native_data.dtype)" in source


def test_prepared_ascend_route_sorts_without_coalescing():
    source = _function_source("prepare_spmm_coo_route")
    dispatch = source.index("if _use_spmm_coo_ascend_dispatch():")
    coalesce = source.index("_coalesce_coo_entries(")

    assert source.index("_sort_coo_lex_inplace(") < coalesce
    assert dispatch < coalesce


def test_ascend_keeps_float64_and_does_not_promote_float32_compute():
    source = _function_source("_spmm_coo_compute_dtype")
    dispatch = source.index("if _is_ascend_runtime():")
    generic_precision = source.index("if value_dtype == torch.float32:")

    assert dispatch < generic_precision
    assert "return value_dtype" in source


def test_prepared_alg1_uses_the_ascend_scatter_fallback():
    source = _function_source("_run_spmm_coo_alg1_route")
    dispatch = source.index("if _use_spmm_coo_ascend_dispatch():")
    kernel = source.index("_spmm_coo_alg1_process_count_kernel[")

    assert "_spmm_coo_ascend_scatter(" in source
    assert dispatch < kernel
