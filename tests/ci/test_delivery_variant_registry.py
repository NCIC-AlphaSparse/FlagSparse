# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep the Python and C API delivery reports on one variant registry."""

from pathlib import Path

from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]


def test_delivery_registry_contains_the_current_40_unique_variant_ids():
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    assert len(variants) == 40
    assert len({variant["id"] for variant in variants}) == 40


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
    SUPERSET: it runs operators that contribute no row to the 40-variant report.
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
