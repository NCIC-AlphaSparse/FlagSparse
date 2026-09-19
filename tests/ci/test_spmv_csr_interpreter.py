"""Opt-in CPU execution of real Triton kernels and GPU-plan code via its interpreter.

Run in a fresh process with TRITON_INTERPRET=1. This is not hardware validation.
"""

import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "requires an isolated TRITON_INTERPRET=1 process", allow_module_level=True
    )

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

SOURCE = Path(__file__).resolve().parents[2] / "src/flagsparse/sparse_operations"


def close_tolerances(dtype):
    module = importlib.import_module("tests.pytest.accuracy_utils")
    return module.close_tolerances(dtype)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kernels = load_module("_csr_interpreter_kernels", "_spmv_csr_kernels.py")
policy = load_module("_csr_interpreter_policy", "_spmv_csr_config.py")


@pytest.mark.parametrize("alg", policy.NEW_ALGORITHMS)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
@pytest.mark.parametrize(
    "indices,indptr",
    [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "lengths",
    [
        [],
        [0] * 19,
        [0, 1, 7, 8, 9, 31, 32] * 3,
        [0, 33, 1024, 1, 1025, 32, 2048, 2049, 7],
    ],
)
def test_interpreted_csr_algorithms(alg, dtype, indices, indptr, lengths):
    torch.manual_seed(71)
    ptr = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0).to(indptr)
    nnz = int(ptr[-1])
    col = (torch.arange(nnz) % 4099).to(indices)
    data, x = torch.randn(nnz, dtype=dtype), torch.randn(4099, dtype=dtype)
    prepared = SimpleNamespace(
        n_rows=len(lengths), data=data, kernel_indices=col, kernel_indptr=ptr
    )
    caps = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True)
    config, _ = policy.resolve_config(alg, caps)
    plan = (
        kernels.build_plan(prepared, config, alg == "row_adaptive_split")
        if lengths and alg in ("row_split_reduce", "row_adaptive_split")
        else None
    )
    out = torch.full((len(lengths),), float("nan"), dtype=dtype)
    kernels.compute(prepared, x, out, alg, config, plan)
    reference_dtype = torch.complex128 if data.is_complex() else torch.float64
    expected = torch.tensor(
        [
            sum(
                data[int(ptr[r]) : int(ptr[r + 1])].to(reference_dtype)
                * x[col[int(ptr[r]) : int(ptr[r + 1])].long()].to(reference_dtype)
            )
            for r in range(len(lengths))
        ],
        dtype=dtype,
    )
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("alg", ["row_split_reduce", "row_adaptive_split"])
@pytest.mark.parametrize("width", [32, 64])
def test_interpreted_multilevel_reduction(alg, width):
    counts = [0, 1, 4, 8, 9, 8 * 17 + 1, 2]
    ptr = torch.tensor([0] + counts).cumsum(0)
    nnz = int(ptr[-1])
    col = torch.arange(nnz) % 101
    data = torch.tensor(
        ([1e6, 1, -1e6, -1] * (nnz // 4)) + [1] * (nnz % 4), dtype=torch.float32
    )
    prepared = SimpleNamespace(
        n_rows=len(counts), data=data, kernel_indices=col, kernel_indptr=ptr
    )
    caps = policy.BackendCaps(
        "cuda" if width == 32 else "rocm", "test", "test", width, 1024, True, True, True
    )
    config, _ = policy.resolve_config(
        alg,
        caps,
        {
            "short_row_threshold": 4,
            "split_row_threshold": 8,
            "row_split_reduce": {"segment_nnz": 8, "reduce_block_size": 2},
        },
    )
    plans = []
    for sign in (1, -1):
        plan = kernels.build_plan(prepared, config, alg == "row_adaptive_split")
        plans.append(plan)
        assert len(plan["levels"]) == 5
        out = torch.full((len(counts),), float("nan"))
        kernels.compute(
            prepared, torch.full((101,), float(sign)), out, alg, config, plan
        )
        expected = torch.tensor(
            [
                data[ptr[r] : ptr[r + 1]].double().sum() * sign
                for r in range(len(counts))
            ],
            dtype=torch.float32,
        )
        rtol, atol = close_tolerances(torch.float32)
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
    assert plans[0]["starts"].data_ptr() != plans[1]["starts"].data_ptr()


@pytest.mark.parametrize("alg", policy.NEW_ALGORITHMS)
def test_interpreted_public_api_and_fallback(alg, monkeypatch):
    """Run real public orchestration on CPU tensors; only device/events are mocked."""
    import importlib

    mod = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
    actual_kernels = importlib.import_module(
        "flagsparse.sparse_operations._spmv_csr_kernels"
    )
    caps = mod._csr_config.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True)

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 7.0

    monkeypatch.setattr(
        mod, "_is_accel_tensor", lambda tensor: tensor.device.type == "cpu"
    )
    monkeypatch.setattr(mod, "_spmv_device_context", lambda device: nullcontext())
    monkeypatch.setattr(mod, "_spmv_backend_caps", lambda device: caps)
    monkeypatch.setattr(
        mod, "_ACCEL", SimpleNamespace(Event=Event, synchronize=lambda: None)
    )
    data = torch.tensor([1.0, -1.0, 3.0, 2.0, 1.0])
    col = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
    ptr = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    x = torch.tensor([2.0, 5.0, 3.0])
    config = {"row_vector": {"block_nnz": 64}}
    prepared = mod.prepare_spmv_csr(data, col, ptr, (3, 3), alg=alg, config=config)
    out = torch.empty(3)
    y, ms, meta = mod.flagsparse_spmv_csr_run(
        prepared, x, out=out, timing=True, return_time=True, return_meta=True
    )
    assert y is out
    torch.testing.assert_close(y, torch.tensor([-3.0, 0.0, 18.0]))
    assert ms == meta["gpu_ms"] + meta["process_cpu_ms"] == 7.0
    assert meta["compute_dtype"] == "float64"
    inherited = mod.flagsparse_spmv_csr(prepared=prepared, x=x)
    torch.testing.assert_close(y, inherited)
    with pytest.raises(ValueError, match="overlap"):
        mod.flagsparse_spmv_csr_run(prepared, x, out=x)
    with pytest.raises(ValueError):
        mod.flagsparse_spmv_csr(prepared=prepared, x=x, use_opt=False)
    original = actual_kernels.compute

    def reject_int64(route, *args, **kwargs):
        if route.kernel_indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 kernel indices")
        return original(route, *args, **kwargs)

    monkeypatch.setattr(actual_kernels, "compute", reject_int64)
    retried, meta = mod.flagsparse_spmv_csr_run(prepared, x, return_meta=True)
    torch.testing.assert_close(retried, y)
    assert meta["index_fallback_applied"] and meta["alg_resolved"] == alg
    assert meta["config"] == prepared.config


@pytest.mark.parametrize("alg", policy.NEW_ALGORITHMS)
@pytest.mark.parametrize("m", [0, 7])
def test_interpreted_zero_columns(alg, m):
    prepared = SimpleNamespace(
        n_rows=m,
        data=torch.empty(0),
        kernel_indices=torch.empty(0, dtype=torch.int64),
        kernel_indptr=torch.zeros(m + 1, dtype=torch.int32),
    )
    caps = policy.BackendCaps("rocm", "gfx90a", "hip", 64, 1024, True, True, True)
    config, _ = policy.resolve_config(alg, caps)
    plan = (
        kernels.build_plan(prepared, config, alg == "row_adaptive_split")
        if m and alg in ("row_split_reduce", "row_adaptive_split")
        else None
    )
    out = torch.full((m,), float("nan"))
    kernels.compute(prepared, torch.empty(0), out, alg, config, plan)
    torch.testing.assert_close(out, torch.zeros(m))
