# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import json
import os
from pathlib import Path

import pytest
import torch

from flagsparse import flagsparse_spmv_csr
from flagsparse.sparse_operations._spmv_csr_benchmark import golden_csr
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import SPMV_MN_SHAPES

spmv_mod = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
pytestmark = pytest.mark.skipif(
    not accelerator_available(), reason=ACCELERATOR_REQUIRED
)


def _value_dtype_cases():
    cases = [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]
    return [(name, dtype) for name, dtype in cases if dtype is not None]


def _random_dense(shape, dtype, device):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _tol(dtype):
    # fp32 / fp16 / bf16 and complex64 accumulate in fp32-precision components and
    # carry standard fp32 SpMV error (order-dependent), so they use a realistic
    # fp32 tolerance rather than the fp64-then-cast accuracy of the former
    # baseline. fp64 and complex128 accumulate in fp64 and keep the strict shared
    # tolerance.
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.complex64):
        return 1e-3, 1e-3
    return close_tolerances(dtype)


def _random_csr_mn(M, N, dtype, index_dtype, device):
    """Random CSR matrix: kernel inputs on ``device``, dense oracle copy on CPU.

    Construction runs on CPU because ``torch.where`` has no muDNN kernel for float64
    or complex on Moore Threads, so building the matrix on the accelerator failed
    before the operator under test was ever reached.  See ``golden_device()``.
    """
    golden = golden_device()
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=golden) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_dense((M, N), dtype, golden),
        torch.zeros((), dtype=dtype, device=golden),
    )
    rows, cols = torch.nonzero(mask, as_tuple=True)
    data = dense[rows, cols].contiguous()
    row_counts = torch.bincount(rows, minlength=M)
    indptr = torch.zeros(M + 1, dtype=torch.int64, device=golden)
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return (
        data.to(device),
        cols.to(index_dtype).contiguous().to(device),
        indptr.to(index_dtype).to(device),
        dense,
    )


def _make_x(length, dtype, device):
    return _random_dense((length,), dtype, device)


def _op_transposes(op):
    return op in ("trans", "conj")


def _apply_dense_op(dense, op):
    if op == "non":
        return dense
    if op == "trans":
        return dense.t()
    if op == "conj":
        return dense.conj().t()
    raise ValueError(f"unsupported op: {op}")


def _assert_close(actual, expected, dtype):
    """Compare on the golden device; ``expected`` never leaves CPU."""
    rtol, atol = _tol(dtype)
    ref_dtype = _reference_dtype(dtype)
    golden = golden_device()
    assert torch.allclose(
        actual.to(device=golden, dtype=ref_dtype),
        expected.to(device=golden, dtype=ref_dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.spmv_csr
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize(
    "name,dtype", _value_dtype_cases(), ids=[c[0] for c in _value_dtype_cases()]
)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmv_csr_matches_dense_reference(M, N, name, dtype, index_dtype, op):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csr_mn(M, N, dtype, index_dtype, device)
    transpose = _op_transposes(op)
    x_len = M if transpose else N
    x = _make_x(x_len, dtype, golden_device())
    ref_dtype = _reference_dtype(dtype)
    ref_mat = _apply_dense_op(dense, op)
    ref = (ref_mat.to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
    out = flagsparse_spmv_csr(
        data,
        indices,
        indptr,
        x.to(device),
        shape=(M, N),
        op=op,
        index_fallback_policy="auto",
    )
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_csr
def test_spmv_csr_prepared_transpose_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csr_mn(
        8, 10, torch.float32, torch.int32, device
    )
    prepared = spmv_mod.prepare_spmv_csr(data, indices, indptr, (8, 10), transpose=True)
    x = torch.randn(8, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="does not match prepared.transpose"):
        flagsparse_spmv_csr(x=x, prepared=prepared, transpose=False)


@pytest.mark.spmv_csr
def test_spmv_csr_prepared_op_mismatch_rejected():
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csr_mn(
        8, 10, torch.complex64, torch.int32, device
    )
    prepared = spmv_mod.prepare_spmv_csr(data, indices, indptr, (8, 10), op="conj")
    x = _make_x(8, torch.complex64, device)
    with pytest.raises(ValueError, match="does not match prepared.op"):
        flagsparse_spmv_csr(x=x, prepared=prepared, op="trans")


@pytest.mark.spmv_csr
def test_spmv_csr_int64_auto_fallback_to_int32(monkeypatch):
    device = accelerator_device()
    data, indices, indptr, dense = _random_csr_mn(
        12, 9, torch.float32, torch.int64, device
    )
    x = torch.randn(9, dtype=torch.float32, device=golden_device())
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    state = {"forced_once": False}
    original = spmv_mod._triton_spmv_csr_impl_prepared

    def fail_int64_once(prepared, x_in, out=None):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("unsupported int64 kernel indices")
        return original(prepared, x_in, out=out)

    monkeypatch.setattr(spmv_mod, "_triton_spmv_csr_impl_prepared", fail_int64_once)
    out = flagsparse_spmv_csr(
        data,
        indices,
        indptr,
        x.to(device),
        shape=(12, 9),
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(
        out.to(device=golden_device(), dtype=torch.float64), ref, rtol=rtol, atol=atol
    )


@pytest.mark.spmv_csr
def test_spmv_csr_int64_strict_no_fallback(monkeypatch):
    device = accelerator_device()
    data, indices, indptr, _dense = _random_csr_mn(
        12, 9, torch.float32, torch.int64, device
    )
    x = torch.randn(9, dtype=torch.float32, device=device)

    def fail_int64(prepared, x_in, out=None):
        if prepared.kernel_indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 kernel indices")
        return spmv_mod._triton_spmv_csr_impl_prepared(prepared, x_in, out=out)

    monkeypatch.setattr(spmv_mod, "_triton_spmv_csr_impl_prepared", fail_int64)
    with pytest.raises(RuntimeError, match="unsupported int64 kernel indices"):
        flagsparse_spmv_csr(
            data,
            indices,
            indptr,
            x,
            shape=(12, 9),
            index_fallback_policy="strict",
        )


@pytest.mark.spmv_csr
def test_spmv_csr_int64_auto_does_not_fallback_when_index_exceeds_int32(monkeypatch):
    device = accelerator_device()
    limit = spmv_mod._INDEX_LIMIT_INT32
    data = torch.ones(1, dtype=torch.float32, device=device)
    prepared = spmv_mod.PreparedCsrSpmv(
        data=data,
        kernel_indices=torch.tensor([limit + 1], dtype=torch.int64, device=device),
        kernel_indptr=torch.tensor([0, 1], dtype=torch.int64, device=device),
        shape=(1, limit + 2),
        n_rows=1,
        n_cols=limit + 2,
        block_nnz=256,
        max_segments=1,
        max_row_nnz=1,
        opt_buckets=[],
        transpose=False,
        index_fallback_policy="auto",
    )

    def fail_launch(_prepared, _x, use_opt=False, opt_buckets=None):
        raise RuntimeError("unsupported native int64 kernel indices")

    monkeypatch.setattr(spmv_mod, "_run_spmv_prepared", fail_launch)
    x = torch.empty(0, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="int32 fallback is unsafe"):
        spmv_mod._run_spmv_prepared_with_fallback(prepared, x, use_opt=False)


NEW_ALGORITHMS = spmv_mod.SPMV_CSR_NEW_ALGORITHMS
REGRESSIONS = json.loads(
    (Path(__file__).resolve().parents[1] / "data/spmv_csr_regressions.json").read_text()
)


def _native_case(lengths, dtype, col_dtype, ptr_dtype, n=4099):
    device = accelerator_device()
    counts = torch.tensor(lengths, dtype=torch.int64)
    ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    nnz = int(ptr[-1])
    torch.manual_seed(71)
    data = torch.randn(nnz, dtype=dtype, device=device)
    col = (torch.arange(nnz, dtype=torch.int64) % n).to(device=device, dtype=col_dtype)
    ptr = ptr.to(device=device, dtype=ptr_dtype)
    x = torch.randn(n, dtype=dtype, device=device)
    return data, col, ptr, x, (len(lengths), n)


def _new_prepared(data, col, ptr, shape, alg, **kwargs):
    if spmv_mod._backend_name() not in ("cuda", "rocm"):
        pytest.skip("new CSR algorithms have no verified profile for this backend")
    return spmv_mod.prepare_spmv_csr(data, col, ptr, shape, alg=alg, **kwargs)


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", NEW_ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spmv_csr_default_segment_multilevel(alg, dtype):
    data, col, ptr, x, shape = _native_case(
        [1024 * 257 + 1], dtype, torch.int64, torch.int32
    )
    prepared = _new_prepared(data, col, ptr, shape, alg)
    result = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        result.cpu(), golden_csr(data, col, ptr, x, shape), rtol=rtol, atol=atol
    )


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", NEW_ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "col_dtype,ptr_dtype",
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
def test_spmv_csr_new_boundaries(alg, dtype, col_dtype, ptr_dtype, lengths):
    data, col, ptr, x, shape = _native_case(lengths, dtype, col_dtype, ptr_dtype)
    prepared = _new_prepared(data, col, ptr, shape, alg)
    out = torch.full((shape[0],), float("nan"), device=data.device, dtype=dtype)
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(
        prepared, x, out=out, return_meta=True, timing=True
    )
    assert actual is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape), rtol=rtol, atol=atol
    )
    assert meta["ms"] == meta["gpu_ms"] + meta["process_cpu_ms"]
    assert meta["compute_dtype"] == "float64"
    assert meta["alg_resolved"] == alg
    assert not prepared.opt_buckets


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", ["row_split_reduce", "row_adaptive_split"])
@pytest.mark.parametrize("dtype", spmv_mod.SUPPORTED_SPMV_VALUE_DTYPES)
def test_spmv_csr_multilevel_and_rebuild(alg, dtype, monkeypatch):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels

    config = {
        "short_row_threshold": 4,
        "split_row_threshold": 8,
        "row_split_reduce": {"segment_nnz": 8, "reduce_block_size": 2},
    }
    data, col, ptr, x, shape = _native_case(
        [0, 1, 4, 8, 9, 8 * 17 + 1, 2], dtype, torch.int64, torch.int32
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, config=config)
    plans = []
    original = kernels.build_plan

    def record_plan(*args):
        plan = original(*args)
        plans.append(plan)
        return plan

    monkeypatch.setattr(kernels, "build_plan", record_plan)
    for vector in (x, x * -0.75):
        result = spmv_mod.flagsparse_spmv_csr_run(prepared, vector)
        rtol, atol = close_tolerances(dtype)
        torch.testing.assert_close(
            result.cpu(),
            golden_csr(data, col, ptr, vector, shape),
            rtol=rtol,
            atol=atol,
        )
    assert len(plans) == 2 and plans[0] is not plans[1]
    assert len(plans[0]["levels"]) == 5


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", NEW_ALGORITHMS)
def test_spmv_csr_cancellation(alg):
    data, col, ptr, x, shape = _native_case(
        [1025, 0, 1031], torch.float32, torch.int64, torch.int64
    )
    data.copy_(
        torch.tensor(
            [1e6, 1, -1e6, -1] * (data.numel() // 4) + [1] * (data.numel() % 4),
            device=data.device,
        )
    )
    x.fill_(1)
    prepared = _new_prepared(data, col, ptr, shape, alg)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(data.dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape), rtol=rtol, atol=atol
    )


@pytest.mark.spmv_csr
def test_spmv_csr_new_out_and_prepared_validation():
    data, col, ptr, x, shape = _native_case(
        [1, 2, 3, 0], torch.float32, torch.int32, torch.int64, n=4
    )
    prepared = _new_prepared(data, col, ptr, shape, "row_vector")
    for out in (
        x,
        data[:4],
        torch.empty(3, device=x.device),
        torch.empty(4, device=x.device, dtype=torch.float64),
    ):
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out)
    for args in (
        {"op": "trans"},
        {"alg": "row_tile"},
        {"config": {"row_vector": {"block_nnz": 64}}},
    ):
        with pytest.raises(ValueError):
            spmv_mod.flagsparse_spmv_csr_run(prepared, x, **args)
    transposed = spmv_mod.prepare_spmv_csr(
        data, col, ptr, shape, alg="row_vector", op="trans"
    )
    assert transposed.shape == shape
    assert transposed.kernel_indptr is ptr


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", NEW_ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_csr_new_fallback_preserves_selection(alg, dtype, op, monkeypatch):
    from flagsparse.sparse_operations import _spmv_csr_kernels as kernels

    data, col, ptr, x, shape = _native_case([0, 3, 65], dtype, torch.int64, torch.int64)
    if op != "non":
        x = _make_x(shape[0], dtype, data.device)
    config = {"row_vector": {"block_nnz": 64}}
    prepared = _new_prepared(data, col, ptr, shape, alg, config=config, op=op)
    original = kernels.compute

    def fail_i64(route, vector, output, algorithm, cfg, plan):
        if route.kernel_indices.dtype == torch.int64:
            raise RuntimeError("unsupported int64 kernel indices")
        assert algorithm == alg and cfg == prepared.config
        return original(route, vector, output, algorithm, cfg, plan)

    monkeypatch.setattr(kernels, "compute", fail_i64)
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(prepared, x, return_meta=True)
    rtol, atol = close_tolerances(data.dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape, op), rtol=rtol, atol=atol
    )
    assert meta["index_fallback_applied"] and meta["alg_resolved"] == alg
    assert meta["indices_dtype"] == "int32" and meta["input_indices_dtype"] == "int64"
    assert prepared.op == spmv_mod._normalize_spmv_op(op)
    prepared.index_fallback_policy = "strict"
    with pytest.raises(RuntimeError, match="unsupported int64"):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x)


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", spmv_mod.SPMV_CSR_SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("dtype", spmv_mod.SUPPORTED_SPMV_VALUE_DTYPES)
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("matrix_name", REGRESSIONS["matrices"])
def test_spmv_csr_external_matrix_regressions(alg, dtype, op, matrix_name):
    """Set FLAGSPARSE_SPMV_CSR_MTX_DIR to run every supplied matrix, including FP32 failures."""
    directory = os.environ.get("FLAGSPARSE_SPMV_CSR_MTX_DIR")
    if not directory:
        pytest.skip("external matrix regression directory not configured")
    from tests.mtx_fast import load_csr

    paths = list(Path(directory).rglob(matrix_name))
    assert (
        len(paths) == 1
    ), f"expected exactly one regression input {matrix_name} under {directory}"
    path = paths[0]
    data, col, ptr, shape = load_csr(path, dtype=dtype, device=accelerator_device())
    torch.manual_seed(2026)
    assert not col.numel() or int(col.max()) <= 2147483647
    col, ptr = col.to(torch.int32), ptr.to(torch.int64)
    x = torch.randn(
        shape[1] if op == "non" else shape[0], dtype=dtype, device=data.device
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(),
        golden_csr(data, col, ptr, x, shape, op),
        rtol=rtol,
        atol=atol,
        msg=str(path),
    )


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", spmv_mod.SPMV_CSR_SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("dtype", spmv_mod.SUPPORTED_SPMV_VALUE_DTYPES)
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize(
    "col_dtype,ptr_dtype",
    [
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    ],
)
def test_spmv_csr_full_dtype_op_surface(alg, dtype, op, col_dtype, ptr_dtype):
    if alg == "legacy_bucket_vector" and col_dtype == torch.int64:
        pytest.skip("legacy bucket requires int32 column indices")
    data, col, ptr, _, shape = _native_case(
        [3, 0, 1, 5], dtype, col_dtype, ptr_dtype, n=7
    )
    # Unsorted, duplicate column IDs and an explicit zero are retained by transpose.
    col.copy_(
        torch.tensor([5, 1, 5, 2, 6, 0, 3, 0, 1], device=col.device, dtype=col_dtype)
    )
    data[0] = 0
    x = _make_x(shape[1] if op == "non" else shape[0], dtype, data.device)
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    assert prepared.shape == shape and prepared.kernel_indptr is ptr
    output_size = shape[0] if op == "non" else shape[1]
    out = torch.full((output_size,), float("nan"), dtype=dtype, device=data.device)
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(
        prepared, x, out=out, return_meta=True
    )
    assert actual is out
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape, op), rtol=rtol, atol=atol
    )
    expected_compute = str(dtype).removeprefix("torch.")
    if dtype in (torch.float16, torch.bfloat16):
        expected_compute = "float32"
    elif dtype == torch.float32 and alg in (*NEW_ALGORITHMS, "legacy_rowpar"):
        expected_compute = "float64"
    assert meta["compute_dtype"] == expected_compute
    assert meta["transpose_strategy"] == (
        "none" if op == "non" else "per_run_csr_rebuild"
    )
    assert meta["alg_resolved"] == alg
    with pytest.raises(ValueError):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x, out=out[:-1])
    with pytest.raises(ValueError):
        spmv_mod.flagsparse_spmv_csr_run(prepared, x[:-1])


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", spmv_mod.SPMV_CSR_SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmv_csr_transpose_rebuilt_inside_each_run(alg, op, monkeypatch):
    data, col, ptr, _, shape = _native_case(
        [1] * 2051, torch.complex64, torch.int32, torch.int64, n=3
    )
    col.zero_()  # Short original rows become one long output row and two empty rows.
    data.fill_(1 + 0.5j)
    x = torch.full((shape[0],), 1 + 0.25j, dtype=data.dtype, device=data.device)
    rebuilds = []
    original = spmv_mod._transpose_csr_for_spmv

    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        rebuilds.append(value)
        return value

    monkeypatch.setattr(spmv_mod, "_transpose_csr_for_spmv", record)
    config = (
        {"row_split_reduce": {"segment_nnz": 8, "reduce_block_size": 2}}
        if alg in NEW_ALGORITHMS
        else None
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op, config=config)
    assert not rebuilds
    saved_data = data.clone()
    actual, meta = spmv_mod.flagsparse_spmv_csr_run(
        prepared, x, timing=True, return_meta=True
    )
    assert len(rebuilds) == 2  # Complete run and separate phase diagnostic.
    second, plain = spmv_mod.flagsparse_spmv_csr_run(prepared, -x, return_meta=True)
    assert len(rebuilds) == 3
    assert rebuilds[0][2].data_ptr() != rebuilds[1][2].data_ptr()
    assert prepared.shape == shape and prepared.data is data
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)
    rtol, atol = close_tolerances(data.dtype)
    ref = golden_csr(data, col, ptr, x, shape, op)
    torch.testing.assert_close(actual.cpu(), ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(second.cpu(), -ref, rtol=rtol, atol=atol)
    for timing_meta in (plain, meta):
        assert (
            timing_meta["ms"] == timing_meta["gpu_ms"] + timing_meta["process_cpu_ms"]
        )
    assert meta["process_gpu_ms"] >= 0 and meta["compute_ms"] >= 0


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", spmv_mod.SPMV_CSR_SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("shape", [(0, 7), (5, 0), (0, 0), (5, 7)])
def test_spmv_csr_empty_dtype_ops(alg, dtype, op, shape):
    device = accelerator_device()
    data = torch.empty(0, dtype=dtype, device=device)
    col = torch.empty(0, dtype=torch.int32, device=device)
    ptr = torch.zeros(shape[0] + 1, dtype=torch.int64, device=device)
    x = torch.ones(shape[1] if op == "non" else shape[0], dtype=dtype, device=device)
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    assert actual.shape == (shape[0] if op == "non" else shape[1],)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.spmv_csr
@pytest.mark.parametrize("alg", spmv_mod.SPMV_CSR_SUPPORTED_ALGORITHMS)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_csr_complex_cancellation(alg, dtype, op):
    data, col, ptr, _, shape = _native_case(
        [1025, 0, 1031], dtype, torch.int32, torch.int64, n=3
    )
    col.zero_()
    pattern = torch.tensor(
        [10000 + 5000j, 1 - 2j, -10000 - 5000j, -1 + 2j],
        dtype=dtype,
        device=data.device,
    )
    data.copy_(pattern.repeat((data.numel() + 3) // 4)[: data.numel()])
    x = torch.full(
        (shape[1] if op == "non" else shape[0],),
        1 + 0.5j,
        dtype=dtype,
        device=data.device,
    )
    prepared = _new_prepared(data, col, ptr, shape, alg, op=op)
    actual = spmv_mod.flagsparse_spmv_csr_run(prepared, x)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(
        actual.cpu(), golden_csr(data, col, ptr, x, shape, op), rtol=rtol, atol=atol
    )
