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
    # Carries the metadata the entry reads to size its output buffer, plus a
    # mutable value so `run` can prove the operand is re-read every call.
    dense = SimpleNamespace(shape=(4, 2), dtype="f64", device="npu:0", value=1)
    prepared = object()
    buffers = []

    def prepare(*args):
        assert args == (*arrays, shape)
        events.append("prepare")
        return prepared

    def run(plan, operand, out=None):
        assert plan is prepared and operand is dense
        events.append("run")
        buffers.append(out)
        return operand.value

    fs = SimpleNamespace(
        **{f"prepare_spmm_{fmt}_route": prepare, f"flagsparse_spmm_{fmt}_run": run}
    )

    def empty(size, dtype=None, device=None):
        events.append("empty")
        return SimpleNamespace(size=size, dtype=dtype, device=device)

    torch = SimpleNamespace(
        empty=empty,
        npu=SimpleNamespace(synchronize=lambda: events.append("synchronize")),
    )
    clock = itertools.chain((10.0, 10.025), itertools.count(20.0, 0.001))
    monkeypatch.setattr(
        bench, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    call, prepare_ms = bench._prepare_fp64_spmm(torch, fs, fmt, arrays, shape, dense)
    # The output buffer is allocated after prepare is timed, so it changes
    # neither the event order nor prepare_ms.
    assert events == ["synchronize", "prepare", "synchronize", "empty"]
    assert prepare_ms == pytest.approx(25.0)
    timing = bench._bench(torch, call, 5, 15)
    assert events.count("prepare") == 1
    assert events.count("run") == 20
    assert events.count("empty") == 1
    assert timing["mean_ms"] == pytest.approx(1.0)
    # Every call writes into the SAME buffer, sized from the dense operand. The
    # CANN baseline binds its output once, and this operator's run time depends
    # on where its C sits, so drawing a fresh one per call on our side alone
    # would make the ratio an allocator lottery.
    assert len(set(id(buffer) for buffer in buffers)) == 1
    assert buffers[0].size == (shape[0], dense.shape[1])
    assert (buffers[0].dtype, buffers[0].device) == (dense.dtype, dense.device)
    dense.value = 7
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
