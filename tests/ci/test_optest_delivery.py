# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cover optest's delivery selection, projection and exit status without a GPU."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from tools import delivery_variants
from tools.optest_delivery import _write_delivery_summary, run_delivery

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def optest_runtime(monkeypatch):
    operations = ModuleType("flagsparse.sparse_operations")
    operations._common = SimpleNamespace(
        _backend_name=lambda: "iluvatar",
        _accel_fallback_reason=lambda: None,
        _accel_device_type=lambda: "cuda",
        _accel_module=lambda: SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, operations.__name__, operations)


@pytest.mark.parametrize("enabled", [False, True])
def test_optest_forwards_scaled_baseline_to_children(
    monkeypatch, enabled, optest_runtime
):
    import run_flagsparse_pytest as runner
    from tools import run_optest

    monkeypatch.setenv("FLAGOS_GPU_VENDOR", "iluvatar")
    monkeypatch.setenv("FLAGSPARSE_BACKEND", "iluvatar")
    monkeypatch.delenv("FLAGSPARSE_OPTEST_SCALED_BASELINE", raising=False)
    monkeypatch.setattr(runner, "_SCALED_BASELINE", None)
    monkeypatch.setattr(runner, "_BIV_BASELINE_POLICY", False)

    def main():
        config = json.loads(os.environ["FLAGSPARSE_OPTEST_SCALED_BASELINE"])
        if enabled:
            assert config == {
                "reference": str(ROOT / "conf/h800_reference.json"),
                "card": "iluvatar-biv150",
                "reference_peaks": "h800-sxm",
            }
        else:
            assert config is None
        assert sys.argv[1:] == ["--gpus", "0"]
        return 7

    monkeypatch.setitem(
        sys.modules,
        "flagos_optest",
        SimpleNamespace(
            gpu_check=SimpleNamespace(),
            run_tests=SimpleNamespace(_KNOWN_LIBS=[], main=main),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_optest", *(["--h800-reference"] if enabled else []), "--gpus", "0"],
    )
    assert run_optest.main() == 7
    assert "FLAGSPARSE_OPTEST_SCALED_BASELINE" not in os.environ


@pytest.mark.parametrize("enabled", [False, True])
def test_optest_benchmark_scales_before_recording(
    tmp_path, monkeypatch, enabled, optest_runtime
):
    import run_flagsparse_pytest as runner
    from benchmark import test_sparse_perf as benchmark

    reference = tmp_path / "reference" / "spmv_csr"
    reference.mkdir(parents=True)
    (reference / "performance.csv").write_text(
        "matrix,dtype,op,status,cusparse_ms,triton_ms\n"
        "wave.mtx,float32,non,PASS,2,0.1\n"
        "cage12.mtx,float32,non,PASS,3,0.2\n"
    )
    config = {
        "reference": str(reference.parent),
        "card": "iluvatar-biv150",
        "reference_peaks": "h800-sxm",
    }
    monkeypatch.setenv(
        "FLAGSPARSE_OPTEST_SCALED_BASELINE", json.dumps(config if enabled else None)
    )
    monkeypatch.setenv(
        "FLAGSPARSE_OPTEST_BENCHMARK_INPUT", str(ROOT / "tests/data/wave.mtx")
    )
    monkeypatch.setattr(benchmark, "DELIVERY_DIR", str(tmp_path))
    original = [
        {
            "matrix": "wave.mtx",
            "dtype": "float32",
            "op": "non",
            "status": "PASS",
            "triton_ms": "1",
        },
        {
            "matrix": "cage12.mtx",
            "dtype": "float32",
            "op": "non",
            "status": "PASS",
            "triton_ms": "1",
            "cusparse_ms": "4",
            "triton_speedup_vs_cusparse": "4",
        },
    ]

    def performance(**kwargs):
        assert runner._SCALED_BASELINE == (config if enabled else None)
        csv_path = tmp_path / "performance.csv"
        rows, _ = runner._apply_scaled_baseline("spmv_csr", csv_path, original)
        runner.write_benchmark_json_from_csv(
            "spmv_csr", csv_path, tmp_path / "performance_result.json", rows=rows
        )
        return {"status": "Passed", "records": rows}

    monkeypatch.setattr(benchmark, "run_performance", performance)
    recorded = []
    benchmark.test_core_performance("spmv_csr", tmp_path, recorded.extend, monkeypatch)
    missing, local = recorded[0]["result"]
    assert local["speedup"] == 4
    assert local["latency_base"] == 4
    assert local["latency"] == 1
    if enabled:
        assert missing["latency"] == 1
        assert missing["latency_base"] == pytest.approx(2 * 3050 / 1150)
        assert missing["speedup"] == pytest.approx(2 * 3050 / 1150)
        assert missing["h800_scaled_card"] == "iluvatar-biv150"
    else:
        assert missing["latency"] is None
        assert missing["speedup"] is None
    phase = json.loads((tmp_path / "spmv_csr.json").read_text())
    data = runner._flaggems_perf_data(phase["records"])["float32"]
    assert data.get("baseline") == ("partly_h800_scaled" if enabled else None)


@pytest.mark.parametrize(
    "backend,delivery", [("iluvatar", True), ("iluvatar", False), ("cuda", True)]
)
def test_optest_accuracy_keeps_user_filter_and_iluvatar_profile(
    monkeypatch, backend, delivery
):
    from tests import conftest as accuracy_config

    monkeypatch.setenv("FLAGSPARSE_BACKEND", backend)
    monkeypatch.delenv("FLAGSPARSE_OPTEST_DELIVERY_DIR", raising=False)
    if delivery:
        monkeypatch.setenv("FLAGSPARSE_OPTEST_DELIVERY_DIR", "/tmp/optest-test")
    config = SimpleNamespace(
        option=SimpleNamespace(keyword="requested_case"),
        getoption=lambda option: "normal" if option == "--mode" else False,
    )
    accuracy_config.pytest_configure(config)
    if backend == "iluvatar" and delivery:
        from _pytest.mark.expression import Expression

        selection = Expression.compile(config.option.keyword)
        assert selection.evaluate(lambda name: name in "requested_case-float32")
        assert selection.evaluate(lambda name: name in "requested_case-test_q4_variant")
        assert not selection.evaluate(lambda name: name in "requested_case-float64")
        assert not selection.evaluate(lambda name: name in "float32")
    else:
        assert config.option.keyword == "requested_case"


def test_optest_delivery_schedules_once_per_parent():
    catalog = yaml.safe_load((ROOT / "conf/operators.yaml").read_text())["ops"]
    variants = delivery_variants.load_delivery_variants()
    seen = []
    optest = SimpleNamespace(ROOT=ROOT, CFG=SimpleNamespace())
    optest.get_ops_from_inventory = lambda: catalog

    def select():
        entries = optest.get_ops_from_inventory()
        assert [entry["id"] for entry in entries] == [v["id"] for v in variants]
        assert all(entry["description"] and entry["stages"] for entry in entries)
        return [entry["id"] for entry in entries]

    def main():
        seen.extend(optest.get_ops_to_test())
        inputs = Path(os.environ["FLAGSPARSE_OPTEST_BENCHMARK_INPUT"])
        assert {p.stem for p in inputs.glob("*.mtx")} == set(
            delivery_variants.load_delivery_matrices()
        )
        return 3

    optest.get_ops_to_test = select
    optest.main = main
    assert run_delivery(optest, None) == 3
    assert seen == list(dict.fromkeys(v["operator"] for v in variants))
    assert optest.get_ops_to_test is select


@pytest.mark.parametrize("missing", [False, True])
def test_optest_delivery_keeps_classic_and_tagged_variants_distinct(tmp_path, missing):
    op = "spmv_csr"
    classic, transposed = "spmv_csr_f32_int_non", "spmv_csr_f32_int_trans"
    ids = {classic, transposed}
    variants = [v for v in delivery_variants.load_delivery_variants() if v["id"] in ids]
    accuracy_path = tmp_path / "accuracy_result.json"
    accuracy_path.write_text(
        json.dumps(
            {
                "test_spmv_csr_accuracy.py::test_values[float32-int32-non]": {
                    "params": {
                        "dtype": "torch.float32",
                        "index_dtype": "torch.int32",
                        "op": "non",
                    },
                    "result": "passed",
                },
                f"test_q4_variants_accuracy.py::test_variant[{transposed}-small]": {
                    "params": {"name": transposed},
                    "result": "passed",
                },
            }
        )
    )
    row = {"dtype": "float32", "latency_base": "2", "latency": "1", "speedup": "2"}
    rows = [
        dict(row, op="non"),
        dict(row, dtype="float64", op="non", latency_base="9", speedup="9"),
    ]
    if not missing:
        rows.append(
            dict(row, variant=transposed, op="trans", latency="4", speedup="0.5")
        )
    (tmp_path / f"{op}.json").write_text(
        json.dumps({"phase": "performance", "status": "Passed", "records": rows})
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "result": {
                    op: {
                        "accuracy": {"data_file": accuracy_path.name},
                        "performance": {},
                    }
                }
            }
        )
    )
    optest = SimpleNamespace(
        CFG=SimpleNamespace(output_dir=tmp_path), count_op_statuses=lambda *_: {}
    )
    assert _write_delivery_summary(optest, variants, tmp_path) == int(missing)
    results = json.loads(summary_path.read_text())["result"]
    assert set(results) == {classic, transposed}
    assert results[classic]["accuracy"]["passed"] == 1
    classic_data = results[classic]["performance"]["data"]
    assert set(classic_data) == {"fp32"}
    assert classic_data["fp32"]["speedup"] == 2
    assert results[transposed]["accuracy"]["passed"] == 1
    performance = results[transposed]["performance"]
    if missing:
        assert performance["status"] == "NotFound"
        assert not performance["data"]
    else:
        assert performance["data"]["fp32"]["speedup"] == 0.5
