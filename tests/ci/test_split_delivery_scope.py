# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The C API half of a split delivery run only launches what is delivered.

run_flagsparse_split_delivery.py used to call `ctest -R benchmark`, which launches
every benchmark family -- and on MUSA a SpSV case can take the whole GPU context
down, costing every later case its result. It now selects the families that own
a delivery variant: the manifest's own `reporting: delivery` operators plus every
operator the 65-variant registry (conf/operators.yaml) names, which brought
spgemm back (spgemm_csr_f32_int_non_non) and added axpby and spvv. spsm and spsv
are still never launched. Coverage is still checked in the manifest's own
delivery scope, which did not grow.
"""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]
CAPI = ROOT / "capi"

# What the split runner launches: every family that owns a delivery variant.
DELIVERY_FAMILIES = [
    "axpby",
    "gather",
    "scatter",
    "sddmm",
    "spgemm",
    "spmm",
    "spmv",
    "spvv",
]
NOT_DELIVERED_FAMILIES = ["spsm", "spsv"]
# check_manifest.py's `--scope delivery`: the manifest's own `reporting:
# delivery` entries, which the 65-variant registry did not change.
MANIFEST_DELIVERY_FAMILIES = ["gather", "scatter", "sddmm", "spmm", "spmv"]
TAG_OF_FORMAT = {"sparse_vector": "spvec"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


split = _load("_split_delivery_under_test", ROOT / "run_flagsparse_split_delivery.py")
check_manifest = _load(
    "_check_manifest_under_test", CAPI / "tools" / "check_manifest.py"
)


def _ctest_families():
    """The families ctest actually has cases for: `set(FLAGSPARSE_OPS ...)`."""
    text = (CAPI / "ctest" / "CMakeLists.txt").read_text(encoding="utf-8")
    ops = re.search(r"set\(FLAGSPARSE_OPS ([^)]*)\)", text)
    assert ops, "capi/ctest/CMakeLists.txt no longer declares FLAGSPARSE_OPS"
    return ops.group(1).split()


def test_only_the_delivery_benchmark_families_are_selected():
    pattern, families = split.delivery_ctest_regex()
    assert families == DELIVERY_FAMILIES
    for family in DELIVERY_FAMILIES:
        assert re.search(pattern, f"benchmark.{family}"), family
    for family in NOT_DELIVERED_FAMILIES:
        assert not re.search(pattern, f"benchmark.{family}"), family
    # Accuracy and pytest cases are never part of the performance half.
    assert not re.search(pattern, "accuracy.spmv")
    assert not re.search(pattern, "pytest.spmv")


def test_every_selected_family_is_a_real_ctest_family():
    """A renamed family would otherwise select nothing and pass by running nothing."""
    ctest = set(_ctest_families())
    _, families = split.delivery_ctest_regex()
    assert set(families) <= ctest, sorted(set(families) - ctest)
    assert sorted(ctest - set(families)) == NOT_DELIVERED_FAMILIES


def test_the_runner_hands_ctest_the_delivery_pattern(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return 0

    monkeypatch.setattr(split, "run", fake_run)

    class Args:
        benchmark_input = "."
        capi_build_dir = "capi/build"
        strict = False

    split.run_capi_benchmark(Args(), tmp_path / "bench")
    ctest = next(c for c in calls if c[0] == "ctest")
    assert ctest[ctest.index("-R") + 1] == split.delivery_ctest_regex()[0]
    check = next(c for c in calls if "check_manifest.py" in " ".join(map(str, c)))
    assert check[check.index("--scope") + 1] == "delivery"


def _bench_dir(tmp_path, *, with_retained_rows):
    """Benchmark JSON as a delivery-only ctest run writes it."""
    manifest = yaml.safe_load((CAPI / "conf" / "operators.yaml").read_text())
    family_of = {
        op["id"]: check_manifest.re.search(r"benchmark/test_(\w+)\.cpp", t).group(1)
        for op in manifest["operators"]
        if op.get("status") == "implemented"
        for t in op.get("tests") or []
        if "benchmark/test_" in t
    }
    by_family = {}
    for op_id, fmt, dtype in check_manifest.declared_variants(manifest, "delivery"):
        by_family.setdefault(family_of[op_id], []).append(
            {"format": fmt, "dtype": dtype, "status": "ok", "reporting": "delivery"}
        )
    # Every registered variant the C API implements gets a row tagged with its
    # id, in the family that owns its operator -- what gen_variants.py emits
    # and FLAGSPARSE_BENCH_VARIANTS selects.
    for variant in load_delivery_variants():
        if not variant["capi"]:
            continue
        by_family.setdefault(family_of[variant["operator"]], []).append(
            {
                "format": TAG_OF_FORMAT.get(variant["format"], variant["format"]),
                "dtype": variant["dtype"],
                "status": "ok",
                "reporting": "delivery",
                "variant": variant["id"],
            }
        )
    if with_retained_rows:
        # A delivered family also sweeps variants that are not delivered.
        by_family["spmv"] += [
            {"format": "csr", "dtype": "c32", "status": "ok", "reporting": "retained"},
            {"format": "csc", "dtype": "f32", "status": "ok", "reporting": "retained"},
        ]
    for family, rows in by_family.items():
        (tmp_path / f"{family}_benchmark.json").write_text(
            json.dumps({"operator": family, "result": rows}), encoding="utf-8"
        )
    return sorted(by_family)


def _check(tmp_path, scope):
    return subprocess.run(
        [
            sys.executable,
            str(CAPI / "tools" / "check_manifest.py"),
            "--bench-dir",
            str(tmp_path),
            "--scope",
            scope,
        ],
        capture_output=True,
        text=True,
    ).stdout


def test_delivery_scope_accepts_a_run_that_launched_only_delivery_families(tmp_path):
    families = _bench_dir(tmp_path, with_retained_rows=True)
    assert families == DELIVERY_FAMILIES
    out = _check(tmp_path, "delivery")
    assert "NOT MEASURED" not in out and "UNDECLARED ROW" not in out, out
    assert "manifest and rows agree" in out


def test_the_default_scope_still_reports_the_families_that_were_not_run(tmp_path):
    """The delivery scope must not have loosened the full check."""
    _bench_dir(tmp_path, with_retained_rows=True)
    out = _check(tmp_path, "all")
    assert "NOT MEASURED" in out
    assert "spsv_csr" in out


def test_the_delivery_scope_still_flags_a_delivery_variant_with_no_row(tmp_path):
    _bench_dir(tmp_path, with_retained_rows=False)
    path = tmp_path / "spmv_benchmark.json"
    doc = json.loads(path.read_text())
    doc["result"] = [
        r for r in doc["result"] if not (r["format"] == "coo" and r["dtype"] == "f64")
    ]
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = _check(tmp_path, "delivery")
    assert "NOT MEASURED" in out and "spmv_coo" in out and "f64" in out


def test_declared_delivery_variants_are_the_registered_ones():
    manifest = yaml.safe_load((CAPI / "conf" / "operators.yaml").read_text())
    declared = check_manifest.declared_variants(manifest, "delivery")
    assert len(declared) == 20
    assert {op_id for op_id, _fmt, _dtype in declared} == {
        "gather",
        "scatter",
        "spmv_csr",
        "spmv_coo",
        "spmm_csr",
        "spmm_coo",
        "sddmm_csr",
    }


def _drop_tagged(tmp_path, variant_id):
    for path in tmp_path.glob("*_benchmark.json"):
        doc = json.loads(path.read_text())
        doc["result"] = [r for r in doc["result"] if r.get("variant") != variant_id]
        path.write_text(json.dumps(doc), encoding="utf-8")


def test_the_delivery_scope_flags_a_registered_variant_with_no_tagged_row(tmp_path):
    _bench_dir(tmp_path, with_retained_rows=False)
    _drop_tagged(tmp_path, "spmm_csr_f32_int_trans_non_row")
    out = _check(tmp_path, "delivery")
    assert "NOT MEASURED    spmm_csr_f32_int_trans_non_row" in out, out


def test_a_tag_outside_the_registry_is_flagged(tmp_path):
    _bench_dir(tmp_path, with_retained_rows=False)
    path = tmp_path / "spmv_benchmark.json"
    doc = json.loads(path.read_text())
    doc["result"].append(
        {
            "format": "csr",
            "dtype": "f32",
            "status": "ok",
            "reporting": "delivery",
            "variant": "spmv_csr_f32_int_bogus",
        }
    )
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = _check(tmp_path, "delivery")
    assert "UNKNOWN VARIANT spmv_csr_f32_int_bogus" in out, out
