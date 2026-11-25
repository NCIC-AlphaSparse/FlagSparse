# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cover optest's delivery selection, projection and exit status without a GPU."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from tools import delivery_variants, optest_delivery
from tools.optest_delivery import _write_delivery_summary, run_delivery

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "op,flavor,as_directory,uses_matrix_timeout",
    [
        ("spmm_csr", "performance", True, True),
        ("sddmm_csr", "performance", True, True),
        ("spmm_csr", "accuracy", True, False),
        ("spmm_csr", "performance", False, False),
        ("spgemm_csr", "performance", True, False),
    ],
)
def test_optest_ascend_benchmark_timeout(
    tmp_path, monkeypatch, op, flavor, as_directory, uses_matrix_timeout
):
    for name in ("first.mtx", "second.mtx", "ignore.txt"):
        (tmp_path / name).touch()
    input_path = tmp_path if as_directory else tmp_path / "first.mtx"
    monkeypatch.setenv("FLAGSPARSE_OPTEST_BENCHMARK_INPUT", str(input_path))

    def run_cmd(operator, command, **kwargs):
        assert operator == op
        assert command == "benchmark command"
        return kwargs

    optest = SimpleNamespace(run_cmd=run_cmd)
    env = {"ASCEND_RT_VISIBLE_DEVICES": "7"}
    with optest_delivery._ascend_benchmark_timeout_adapter(optest):
        result = optest.run_cmd(
            op, "benchmark command", flavor=flavor, env=env, timeout=3600
        )
    assert optest.run_cmd is run_cmd
    assert env == {"ASCEND_RT_VISIBLE_DEVICES": "7"}
    if uses_matrix_timeout:
        assert result["env"]["FLAGSPARSE_OPTEST_MATRIX_TIMEOUT"] == "3600"
        assert result["timeout"] > 2 * 3600
    else:
        assert result["env"] == env
        assert result["timeout"] == 3600


def test_optest_delivery_reports_parent_timeout(tmp_path):
    variant = next(
        variant
        for variant in delivery_variants.load_delivery_variants()
        if variant["operator"] == "spmm_csr"
    )
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "spmm_csr": {
                        "accuracy": {},
                        "performance": {"status": "Timeout", "exit_code": -100},
                    }
                }
            }
        )
    )
    optest = SimpleNamespace(
        CFG=SimpleNamespace(output_dir=tmp_path), count_op_statuses=lambda *_: {}
    )
    assert _write_delivery_summary(optest, [variant], tmp_path) == 1
    performance = json.loads(path.read_text())["result"][variant["id"]]["performance"]
    assert performance["status"] == "Timeout"
    assert performance["data"] == {}


@pytest.mark.parametrize(
    "phase_status,returncode,expected_status,expected_code",
    [
        ("Passed", 0, "Passed", 0),
        ("TIMEOUT", -9, "Timeout", -100),
        ("Passed", -100, "Timeout", -100),
        ("FAIL", 1, "Error", 1),
    ],
)
def test_optest_ascend_accuracy_adapter(
    tmp_path, monkeypatch, phase_status, returncode, expected_status, expected_code
):
    def native_accuracy(**kwargs):
        assert kwargs["gpu_id"] == 6
        assert kwargs["op"] == kwargs["marker"] == "spmv_csr"
        assert kwargs["timeout"] == 3600
        if phase_status == "Passed":
            (kwargs["op_dir"] / "accuracy_result.json").write_text("{}")
        return {"status": phase_status, "returncode": returncode, "duration": 2.0}

    def parse_accuracy(path):
        assert path.is_file()
        return {"status": "Passed", "passed": 1}

    def generic_accuracy(gpu_id, op):
        return gpu_id, op

    optest = SimpleNamespace(
        ROOT=ROOT,
        CFG=SimpleNamespace(output_dir=tmp_path, quick=True, timeout=3600),
        TIMEOUT=-100,
        op_marker=lambda op: op,
        parse_accuracy_data=parse_accuracy,
        run_accuracy_q=generic_accuracy,
    )
    monkeypatch.setattr(optest_delivery, "run_accuracy", native_accuracy)
    with optest_delivery._ascend_accuracy_adapter(optest):
        assert optest.run_accuracy_q(6, "axpby") == (6, "axpby")
        result = optest.run_accuracy_q(6, "spmv_csr")
    assert optest.run_accuracy_q is generic_accuracy
    assert result["status"] == expected_status
    assert result["exit_code"] == expected_code
    assert result["duration"] == 2.0
    assert result["data_file"] == (
        "spmv_csr/accuracy_result.json" if expected_status == "Passed" else None
    )


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process groups")
def test_optest_outer_timeout_stops_matrix(tmp_path):
    import fcntl

    lock_path = tmp_path / "matrix.lock"
    started = tmp_path / "matrix.pid"
    matrix = """
import fcntl
import os
import signal
import sys
from pathlib import Path

with open(sys.argv[1], "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    pid = Path(sys.argv[2])
    pid.with_suffix(".tmp").write_text(str(os.getpid()))
    pid.with_suffix(".tmp").replace(pid)
    signal.pause()
"""
    wrapper = """
import os
import sys
from pathlib import Path

from run_flagsparse_pytest import run_subprocess

run_subprocess([sys.executable, "-c", *sys.argv[1:]], project_root=Path.cwd(),
               env=os.environ, timeout=30)
"""
    env = dict(
        os.environ,
        FLAGSPARSE_OPTEST_DELIVERY_DIR=str(tmp_path),
        FLAGSPARSE_OPTEST_MATRIX_TIMEOUT="30",
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", wrapper, matrix, str(lock_path), str(started)],
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not started.exists() and proc.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert started.exists(), "matrix subprocess did not start"
        os.killpg(proc.pid, signal.SIGTERM)
        assert proc.wait(timeout=5) == -signal.SIGTERM
        with lock_path.open("r+") as lock:
            deadline = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        pytest.fail("matrix subprocess survived the outer termination")
                    time.sleep(0.02)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        if started.exists():
            matrix_pid = int(started.read_text())
            try:
                os.killpg(matrix_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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
