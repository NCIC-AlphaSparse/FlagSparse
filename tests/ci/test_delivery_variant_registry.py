# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep the Python and C API delivery reports on one variant registry."""

from pathlib import Path

import yaml

from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]

# The delivery list, as handed over in fork/list.xlsx (2026-09-21). Pinned here on
# purpose: the registry is what every summary, table and doc count follows, so a
# variant appearing or disappearing must be a deliberate edit to this test too.
DELIVERY_VARIANT_IDS = [
    "gather_f16_int",
    "gather_f32_int",
    "gather_f64_int",
    "gather_c32_int",
    "gather_c64_int",
    "scatter_f16_int",
    "scatter_f32_int",
    "scatter_f64_int",
    "scatter_c32_int",
    "scatter_c64_int",
    "spmv_csr_f32_int_non",
    "spmv_csr_f64_int_non",
    "spmv_coo_f32_int_non",
    "spmv_coo_f64_int_non",
    "spmm_csr_f32_int_non_non_row",
    "spmm_csr_f64_int_non_non_row",
    "spmm_coo_f32_int_non_non_row",
    "spmm_coo_f64_int_non_non_row",
    "sddmm_csr_f32_int_non_non_row",
    "sddmm_csr_f64_int_non_non_row",
]

# The C API manifest names dtypes by the width of the whole value (c64 = complex of
# two fp32); the registry names the component (c32 = complex64).
CAPI_DTYPE_TO_REGISTRY_TAG = {
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "f64": "f64",
    "c64": "c32",
    "c128": "c64",
}


def test_delivery_registry_is_exactly_the_20_listed_variants():
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    ids = [variant["id"] for variant in variants]
    assert len(ids) == 20 and len(set(ids)) == 20
    assert ids == DELIVERY_VARIANT_IDS


def test_no_undelivered_operator_is_registered():
    """spgemm/spsv/spsm and the complex spmv/spmm variants left the list."""
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    assert {v["operator"] for v in variants} == {
        "gather",
        "scatter",
        "spmv_csr",
        "spmv_coo",
        "spmm_csr",
        "spmm_coo",
        "sddmm_csr",
    }
    for operator in ("spmv_csr", "spmv_coo", "spmm_csr", "spmm_coo", "sddmm_csr"):
        assert {v["dtype"] for v in variants if v["operator"] == operator} == {
            "f32",
            "f64",
        }


def test_capi_manifest_delivers_exactly_the_registered_variants():
    """One registry, two front ends -- down to the dtype, not just the operator.

    capi/conf/operators.yaml says what is delivered with `reporting: delivery`
    (narrowed per dtype by `delivery_dtypes`). The parent-operator check below is
    too coarse to catch a complex spmv variant that is registered on one side and
    not the other, so compare (operator, dtype) pairs.
    """
    registered = {
        (v["operator"], v["dtype"])
        for v in load_delivery_variants(ROOT / "conf" / "operators.yaml")
    }
    manifest = yaml.safe_load(
        (ROOT / "capi" / "conf" / "operators.yaml").read_text(encoding="utf-8")
    )
    delivered = set()
    for op in manifest["operators"]:
        if op.get("status") != "implemented" or op.get("reporting") != "delivery":
            continue
        for dtype in op.get("delivery_dtypes") or op.get("dtypes") or []:
            delivered.add((op["id"], CAPI_DTYPE_TO_REGISTRY_TAG[dtype]))
    assert delivered == registered, {
        "only in the C API manifest": sorted(delivered - registered),
        "only in the registry": sorted(registered - delivered),
    }


def test_both_summary_writers_consume_the_shared_delivery_registry():
    python_runner = (ROOT / "run_flagsparse_pytest.py").read_text(encoding="utf-8")
    capi_writer = (ROOT / "capi" / "tools" / "write_summary.py").read_text(
        encoding="utf-8"
    )
    assert "load_delivery_variants" in python_runner
    assert "_delivery_results" in python_runner
    assert "load_delivery_variants" in capi_writer
    assert "expected_variants" in capi_writer


def test_delivery_only_selects_exactly_the_operators_behind_the_variants():
    """--delivery-only must derive the op list, not be handed one.

    Without it the runner falls back to the yaml's own `ops:` list, which is a
    SUPERSET: it runs operators that contribute no row to the delivery report.
    Nothing is lost that way, but the extra work is invisible until someone
    wonders why a delivery run also benchmarked spmm_bell across 30 matrices.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_runner_under_test", ROOT / "run_flagsparse_pytest.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules["_runner_under_test"] = runner
    spec.loader.exec_module(runner)

    parents = {
        variant["operator"]
        for variant in load_delivery_variants(ROOT / "conf" / "operators.yaml")
    }

    def select(manifest, delivery_only):
        return runner.read_ops(
            project_root=ROOT,
            operators_yaml=manifest,
            op_list=None,
            ops_arg=None,
            stages_arg="all",
            start=None,
            delivery_only=delivery_only,
        )

    selected = select("conf/operators.yaml", True)
    assert set(selected) == parents
    assert len(selected) == len(set(selected))

    # The default is a superset: no variant goes unreported without the flag.
    assert parents <= set(select("conf/operators.yaml", False))

    # The C API manifest spells the same intent with `reporting: delivery`, and
    # the two have to agree -- one registry, two front ends.
    assert set(select("capi/conf/operators.yaml", True)) == parents
