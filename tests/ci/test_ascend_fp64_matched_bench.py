"""CPU checks for matched allocation timing and CANN output descriptor lifetime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def probe():
    pytest.importorskip("scipy")
    path = Path(__file__).resolve().parents[2] / "tools/ascend_fp64_matched_bench.py"
    spec = importlib.util.spec_from_file_location("matched_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_statistic(probe):
    result = probe.timing([9, 8, 7, 6, 5, 19, 18, 17, 16, 15, 29, 28, 27, 26, 25])
    assert result["mean_ms"] == 17
    assert result["group_min_median_ms"] == 15


@pytest.mark.parametrize(
    "correct,matched,expected",
    [
        (True, True, "PASS"),
        (False, True, "INCORRECT"),
        (True, False, "OUTPUT_ADDRESS_MISMATCH"),
    ],
)
def test_speedup_requires_accuracy_and_exact_address(probe, correct, matched, expected):
    samples = []
    for iteration in range(3):
        for side in ("A", "B"):
            samples.append(
                dict(
                    warmup=False,
                    side=side,
                    ms=1 if side == "A" else 2,
                    output=dict(pointer=128 if matched or side == "A" else 256),
                )
            )
    result = probe.summarize(samples, {"A": 0.0, "B": 0.0}, correct)
    assert result["status"] == expected
    assert result["speedup"] == (2 if expected == "PASS" else None)


@pytest.mark.parametrize("status", [0, 3])
def test_cann_fresh_output_cleanup(monkeypatch, probe, status):
    events = []
    stream = SimpleNamespace(
        npu_stream=1, synchronize=lambda: events.append("synchronize")
    )
    monkeypatch.setattr(
        probe.torch,
        "npu",
        SimpleNamespace(current_stream=lambda device: stream),
        raising=False,
    )
    acl = SimpleNamespace(
        create_tensor_desc=lambda *args: 101,
        create_data_buffer=lambda *args: 102,
        destroy_data_buffer=lambda handle: events.append(("buffer", handle)),
        destroy_tensor_desc=lambda handle: events.append(("descriptor", handle)),
        get_recent_err_msg=lambda: "injected",
    )
    plan = SimpleNamespace(
        output=torch.empty((2, 3), dtype=torch.float64),
        acl=acl,
        execute=lambda *args: status,
        input_desc=None,
        input_buffers=None,
        attribute=None,
    )
    if status:
        with pytest.raises(RuntimeError, match="returned 3"):
            probe.cann_fresh_output(plan)
    else:
        first = probe.cann_fresh_output(plan)
        second = probe.cann_fresh_output(plan)
        assert first.data_ptr() != second.data_ptr()
        assert first.shape == plan.output.shape and first.dtype == torch.float64
    assert events[-3:] == ["synchronize", ("buffer", 102), ("descriptor", 101)]
