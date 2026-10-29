"""Ascend delivery preparation is separate from repeated FP64 run timing."""

import csv
import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bench():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    path = Path(__file__).resolve().parents[2] / "benchmark/benchmark_ascend.py"
    spec = importlib.util.spec_from_file_location("_ascend_prepare_timing", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_prepare_runs_once_before_timed_calls(monkeypatch, bench, fmt):
    events = []
    arrays = (object(), object(), object())
    shape = (3, 4)
    dense = {"value": 1}
    prepared = object()

    def prepare(*args):
        assert args == (*arrays, shape)
        events.append("prepare")
        return prepared

    def run(plan, operand):
        assert plan is prepared and operand is dense
        events.append("run")
        return operand["value"]

    fs = SimpleNamespace(
        **{f"prepare_spmm_{fmt}_route": prepare, f"flagsparse_spmm_{fmt}_run": run}
    )
    torch = SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: events.append("synchronize"))
    )
    clock = itertools.chain((10.0, 10.025), itertools.count(20.0, 0.001))
    monkeypatch.setattr(
        bench, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    call, prepare_ms = bench._prepare_fp64_spmm(torch, fs, fmt, arrays, shape, dense)
    assert events == ["synchronize", "prepare", "synchronize"]
    assert prepare_ms == pytest.approx(25.0)
    timing = bench._bench(torch, call, 5, 15)
    assert events.count("prepare") == 1
    assert events.count("run") == 20
    assert timing["mean_ms"] == pytest.approx(1.0)
    dense["value"] = 7
    assert call() == 7


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_delivery_csv_exports_prepare_only_when_measured(
    monkeypatch, tmp_path, bench, fmt, dtype
):
    record = dict(
        op=f"spmm_{fmt}",
        dtype=dtype,
        matrix="edge",
        flagsparse={"mean_ms": 2.0},
        pytorch={"mean_ms": 4.0},
        scipy_max_abs_error={"flagsparse": 0.0},
        status="PASS",
    )
    if dtype == "float64":
        record.update(prepare_ms=3.5, timed="prepared")
    monkeypatch.setattr(bench, "run", lambda *args, **kwargs: [record])
    output = tmp_path / "summary.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_ascend.py",
            "--op",
            f"spmm_{fmt}",
            "--dtypes",
            dtype,
            "--csv-summary",
            str(output),
        ],
    )
    bench.main()
    with output.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert ("prepare_ms" in reader.fieldnames) == (dtype == "float64")
    assert rows[0]["triton_ms"] == "2.0"
    if dtype == "float64":
        assert rows[0]["prepare_ms"] == "3.5"
        assert rows[0]["timed"] == "prepared"
