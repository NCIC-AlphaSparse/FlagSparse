# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep the Python registry and the C API manifest on one source of truth."""

from pathlib import Path

import yaml

from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]

# `delivery_variants:` in conf/operators.yaml holds 65 variants: the 20 originally
# delivered to the C API plus 45 more from docs/NEW_OPERATORS_CUSPARSE_12_5.md that
# are implemented and accuracy-tested in Python but not all wrapped by the C API /
# c_fs layer yet. There is no separate "q4" list any more -- a variant's pytest
# cases and benchmark rows are found by its `id`
# (tests/pytest/test_q4_variants_accuracy.py, the `variant` CSV column).
#
# Each variant also carries `capi`: whether the C API has actually implemented that
# exact (operator, dtype, opA) combination -- see `_capi_derived_true` below for the
# precise rule, derived from capi/conf/operators.yaml's `status`/`dtypes`/
# `mixed_dtypes`/`ops` fields. This is independent of that manifest's
# `reporting: delivery` tag, which is the C API's own narrower "what's in its
# default report" scope (checked separately by
# test_the_capi_manifests_own_delivery_scope_has_not_grown below).
#
# All 65 since 2026-10-08: the origin/q4 C API work was ported and every variant
# produced benchmark rows on CUDA (RTX 5090, the 10 delivery matrices). Pinned so
# a regression -- a manifest entry narrowed, an op dropped -- fails here by name.
CAPI_TRUE_COUNT = 65

# The C API manifest names dtypes by the width of the whole value (c64 = complex of
# two fp32); the registry names the component (c32 = complex64).
# A few entries (spvv, spmv_sell, spmm_csc, sddmm_csr's complex) already spell
# complex64 the registry's way, `c32`; the manifest's own convention never uses
# that token, so it can only mean complex64 and passes through unchanged.
CAPI_DTYPE_TO_REGISTRY_TAG = {
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "f64": "f64",
    "i8": "i8",
    "c64": "c32",
    "c128": "c64",
}

# The manifest spells the conjugate transpose like cuSPARSE does; variant ids
# shorten it to `conj`.
OPA_MANIFEST_TO_REGISTRY = {"conj_trans": "conj"}

# The C API's own "what ships in the default report" tag -- unrelated to `capi` in
# the Python registry, see the module docstring.
CAPI_DELIVERY_OPERATORS = {
    "gather",
    "scatter",
    "spmv_csr",
    "spmv_coo",
    "spmm_csr",
    "spmm_coo",
    "sddmm_csr",
}


def _load_capi_manifest() -> dict:
    return yaml.safe_load(
        (ROOT / "capi" / "conf" / "operators.yaml").read_text(encoding="utf-8")
    )


def _split_variant_id(variant_id: str, operator: str) -> tuple[str, str]:
    """``(dtype, opA)`` from a variant id, per the naming rule in
    docs/NEW_OPERATORS_CUSPARSE_12_5.md: ``operator_dtype_int_opA[_opB][_row|_col]``.

    opA defaults to ``"non"`` for operators with no transpose axis at all (gather,
    scatter): nothing follows `_int`, which this also handles for any operator
    whose id happens to end right after `_int`.
    """
    rest = variant_id[len(operator) + 1 :]
    parts = rest.split("_")
    i = parts.index("int")
    dtype = "_".join(parts[:i])
    tail = parts[i + 1 :]
    opA = tail[0] if tail else "non"
    return dtype, opA


def _capi_derived_true(variant: dict, capi_ops: dict) -> tuple[bool, str]:
    """Whether the C API manifest says this (operator, dtype, opA) is implemented.

    Mechanical, not a guess: `status: implemented` plus the variant's dtype (mapped
    through CAPI_DTYPE_TO_REGISTRY_TAG) in that operator's `dtypes:`, plus the
    variant's opA (the sparse-side transpose; everything else -- opB, dense layout
    -- is a stride choice the C API handles uniformly once opA and dtype clear,
    per capi/conf/operators.yaml's own notes on spmm_csr/spmm_coo/sddmm_csr) in that
    operator's `ops:` (default `["non"]` when the operator has no `ops:` field at
    all, i.e. no transpose axis -- gather/scatter).
    """
    entry = capi_ops.get(variant["operator"])
    if entry is None or entry.get("status") != "implemented":
        return False, "no C API entry, or not implemented"
    dtype, opA = _split_variant_id(variant["id"], variant["operator"])
    declared = {CAPI_DTYPE_TO_REGISTRY_TAG.get(d, d) for d in entry.get("dtypes") or []}
    # `mixed_dtypes` names input->output pairs the ordinary per-dtype sweep cannot
    # express (f16f32, i8i32, ...) in the registry's own spelling.
    declared |= set(entry.get("mixed_dtypes") or [])
    if dtype not in declared:
        return False, f"dtype {dtype!r} not in {sorted(declared)}"
    allowed_ops = {
        OPA_MANIFEST_TO_REGISTRY.get(o, o) for o in entry.get("ops") or ["non"]
    }
    if opA not in allowed_ops:
        return False, f"opA={opA!r} not in {sorted(allowed_ops)}"
    return True, "ok"


def test_delivery_registry_has_65_unique_variants():
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    ids = [variant["id"] for variant in variants]
    assert len(ids) == 65 and len(set(ids)) == 65


def test_the_capi_true_subset_is_pinned():
    """`capi: true` disappearing must be a deliberate edit here too."""
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    not_capi = sorted(v["id"] for v in variants if not v["capi"])
    assert not_capi == [], not_capi
    assert sum(v["capi"] for v in variants) == CAPI_TRUE_COUNT


def test_the_capi_field_matches_the_capi_manifests_own_capability_claims():
    """`capi: true`/`false` in the Python registry must agree with what
    capi/conf/operators.yaml itself says is implemented -- both directions: no
    variant claims C API support the manifest does not back, and no variant the
    manifest already supports is left marked `false` (the bug this test replaces
    a looser version of: several variants whose dtype and opA the C API already
    covers had been left `capi: false`).
    """
    capi_ops = {op["id"]: op for op in _load_capi_manifest()["operators"]}
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    mismatches = []
    for variant in variants:
        derived, reason = _capi_derived_true(variant, capi_ops)
        if derived != variant["capi"]:
            mismatches.append((variant["id"], variant["capi"], derived, reason))
    assert not mismatches, mismatches


def test_the_capi_manifests_own_delivery_scope_has_not_grown():
    """spgemm/spsv/spsm and the complex spmv/spmm variants stay out of the C API's
    own default report.

    This checks capi/conf/operators.yaml alone, not the Python registry: `capi:
    true` (above) now also covers operators and dtypes the C API has implemented but
    does not headline (spmv_csc, spgemm_csr's f32/f64, complex spmv/spmm) -- that is
    a wider, different thing from `reporting: delivery`, which is this test's
    concern.
    """
    manifest = _load_capi_manifest()
    delivered = {
        op["id"]: set(op.get("delivery_dtypes") or op.get("dtypes") or [])
        for op in manifest["operators"]
        if op.get("status") == "implemented" and op.get("reporting") == "delivery"
    }
    assert set(delivered) == CAPI_DELIVERY_OPERATORS
    for operator in ("spmv_csr", "spmv_coo", "spmm_csr", "spmm_coo", "sddmm_csr"):
        assert delivered[operator] == {"f32", "f64"}, (operator, delivered[operator])


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

    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    parents = {variant["operator"] for variant in variants}

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

    # The C API manifest spells the same `--delivery-only` intent with its own
    # `reporting: delivery` tag, and the two mechanisms have to agree on what that
    # resolves to -- the same CAPI_DELIVERY_OPERATORS this file already pins above,
    # not the (now broader) `capi: true` operator set.
    assert set(select("capi/conf/operators.yaml", True)) == CAPI_DELIVERY_OPERATORS
