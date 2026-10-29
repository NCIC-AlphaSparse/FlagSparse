"""Real-device regressions for native full-precision prepared SpMM."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _ascend_fp64_spmm as native
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


pytestmark = pytest.mark.skipif(
    not (_is_ascend_runtime() and accelerator_available()),
    reason="real Ascend runtime required",
)


def _plan(fmt, op="non", empty=False):
    rows = torch.tensor([1, 1, 2, 2, 4, 4], dtype=torch.int32)
    columns = torch.tensor([0, 1, 2, 2, 0, 3], dtype=torch.int32)
    values = torch.tensor([1 + 2**-40, -1, 2, -0.25, 3, -2], dtype=torch.float64)
    if empty:
        rows, columns, values = rows[:0], columns[:0], values[:0]
    matrix = torch.zeros((5, 4), dtype=torch.float64)
    matrix.index_put_((rows.long(), columns.long()), values, accumulate=True)
    if fmt == "coo":
        order = torch.arange(values.numel() - 1, -1, -1)
        arrays = values[order], rows[order], columns[order]
    else:
        ptr = torch.cat((torch.zeros(1, dtype=torch.int64), torch.bincount(rows.long(), minlength=5).cumsum(0)))
        arrays = values, columns, ptr.to(torch.int32)
    prepared = getattr(fs, f"prepare_spmm_{fmt}_route")(
        *(tensor.to(accelerator_device()) for tensor in arrays), matrix.shape, op=op
    )
    run = getattr(fs, f"flagsparse_spmm_{fmt}_run")
    return prepared, run, matrix if op == "non" else matrix.T


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("width", [1, 7, 32, 64])
def test_native_fp64_spmm_matches_cpu_and_keeps_owned_outputs(monkeypatch, fmt, op, width):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, run, matrix = _plan(fmt, op)
    generator = torch.Generator().manual_seed(882)
    rhs = torch.randn((matrix.shape[1], width), generator=generator, dtype=torch.float64)
    dense = rhs.to(accelerator_device())
    first, meta = run(prepared, dense, return_meta=True)
    assert meta["implementation"] == "cann_sparse_tensor_dense_matmul_fp64"
    assert first.dtype == torch.float64
    torch.testing.assert_close(first.cpu(), matrix @ rhs, rtol=1e-12, atol=1e-12)
    dense.copy_((rhs * -2 + 0.125).to(dense.device))
    second = run(prepared, dense)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first.cpu(), matrix @ rhs, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(second.cpu(), matrix @ (rhs * -2 + 0.125), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_native_fp64_does_not_round_inputs_to_float32(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, run, matrix = _plan(fmt)
    rhs = torch.ones((4, 1), dtype=torch.float64, device=accelerator_device())
    actual = run(prepared, rhs).cpu()
    assert actual[1, 0].item() == 2**-40
    torch.testing.assert_close(actual, matrix @ rhs.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_native_fp64_handles_strided_rhs_and_changing_width(monkeypatch, fmt, layout):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, run, matrix = _plan(fmt)
    for width in (3, 8, 1, 0):
        storage = torch.arange(4 * width * 2, dtype=torch.float64).reshape(4, width * 2)
        rhs = storage[:, ::2]
        dense = storage.to(accelerator_device())[:, ::2]
        actual = run(prepared, dense, dense_layout=layout)
        torch.testing.assert_close(actual.cpu(), matrix @ rhs, rtol=1e-12, atol=1e-12)
        if layout == "col" and width > 1:
            assert actual.stride(0) == 1


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_native_fp64_empty_sparse_matrix_uses_zero_result(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, run, matrix = _plan(fmt, empty=True)
    result = run(prepared, torch.ones((4, 3), dtype=torch.float64, device=accelerator_device()))
    assert torch.equal(result.cpu(), torch.zeros((5, 3), dtype=torch.float64))


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_native_fp64_failure_falls_back_without_poisoning_plan(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "auto")
    prepared, run, matrix = _plan(fmt)
    assert prepared.ascend_fp64_plan is not None

    def fail(self, dense):
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(native._NativeFP64Spmm, "run", fail)
    dense = torch.ones((4, 2), dtype=torch.float64, device=accelerator_device())
    with pytest.warns(RuntimeWarning, match="using scatter"):
        result = run(prepared, dense)
    assert prepared.ascend_fp64_plan is None
    torch.testing.assert_close(result.cpu(), matrix @ dense.cpu(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("use_out", [False, True])
def test_public_fp64_uses_native_and_honors_out(monkeypatch, fmt, op, use_out):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, unused, matrix = _plan(fmt)
    effective = matrix if op == "non" else matrix.T
    rhs = torch.ones((effective.shape[1], 3), dtype=torch.float64, device=accelerator_device())
    if fmt == "coo":
        order = torch.arange(prepared.nnz - 1, -1, -1, device=rhs.device)
        values = prepared.data.view(torch.int32).reshape(-1, 2).index_select(0, order).reshape(-1).view(torch.float64)
        arrays = values, prepared.row.index_select(0, order), prepared.col.index_select(0, order)
    else:
        arrays = prepared.data, prepared.kernel_indices, prepared.kernel_indptr
    target = torch.empty((effective.shape[0], 3), dtype=torch.float64, device=rhs.device) if use_out else None
    actual, meta = getattr(fs, f"flagsparse_spmm_{fmt}")(
        *arrays, rhs, matrix.shape, op=op, out=target, return_meta=True
    )
    assert meta["implementation"] == "cann_sparse_tensor_dense_matmul_fp64"
    if use_out:
        assert actual is target
    torch.testing.assert_close(actual.cpu(), effective @ rhs.cpu(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_native_fp64_reads_live_sparse_values(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, run, matrix = _plan(fmt)
    dense = torch.ones((4, 2), dtype=torch.float64, device=accelerator_device())
    original = run(prepared, dense)
    prepared.data.copy_((prepared.data.cpu() * 2).to(dense.device))
    actual = run(prepared, dense)
    torch.testing.assert_close(actual.cpu(), 2 * matrix @ dense.cpu(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(original.cpu(), matrix @ dense.cpu(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_public_fp64_accepts_int64_indices(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, unused, matrix = _plan(fmt)
    if fmt == "coo":
        arrays = prepared.data, prepared.row.long(), prepared.col.long()
    else:
        arrays = prepared.data, prepared.kernel_indices.long(), prepared.kernel_indptr.long()
    rhs = torch.ones((4, 2), dtype=torch.float64, device=accelerator_device())
    result = getattr(fs, f"flagsparse_spmm_{fmt}")(*arrays, rhs, matrix.shape)
    torch.testing.assert_close(result.cpu(), matrix @ rhs.cpu(), rtol=1e-12, atol=1e-12)


def test_public_sorted_coo_fp64_does_not_sort_again(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, unused, matrix = _plan("coo")
    rhs = torch.ones((4, 2), dtype=torch.float64, device=accelerator_device())

    def fail(*args, **kwargs):
        pytest.fail("sorted COO unexpectedly ran argsort")

    monkeypatch.setattr(native.torch, "argsort", fail)
    result = fs.flagsparse_spmm_coo(prepared.data, prepared.row, prepared.col, rhs, matrix.shape)
    torch.testing.assert_close(result.cpu(), matrix @ rhs.cpu(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_public_fp64_handles_noncontiguous_sparse_values(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, unused, matrix = _plan(fmt)
    storage = torch.empty((prepared.nnz, 2), dtype=torch.float64, device=prepared.data.device)
    storage[:, 0].copy_(prepared.data)
    values = storage[:, 0]
    assert not values.is_contiguous()
    if fmt == "coo":
        arrays = values, prepared.row, prepared.col
    else:
        arrays = values, prepared.kernel_indices, prepared.kernel_indptr
    rhs = torch.ones((4, 3), dtype=torch.float64, device=values.device)
    result = getattr(fs, f"flagsparse_spmm_{fmt}")(*arrays, rhs, matrix.shape)
    torch.testing.assert_close(result.cpu(), matrix @ rhs.cpu(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_public_fp64_out_can_overlap_dense_storage(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, unused, matrix = _plan(fmt)
    storage = torch.arange(15, dtype=torch.float64, device=prepared.data.device).reshape(5, 3)
    rhs = storage[:4]
    golden = matrix @ rhs.cpu()
    arrays = ((prepared.data, prepared.row, prepared.col) if fmt == "coo" else
              (prepared.data, prepared.kernel_indices, prepared.kernel_indptr))
    result = getattr(fs, f"flagsparse_spmm_{fmt}")(*arrays, rhs, matrix.shape, out=storage)
    assert result is storage
    torch.testing.assert_close(result.cpu(), golden, rtol=1e-12, atol=1e-12)


def test_public_coo_fp64_out_cast_is_not_bound_as_double_buffer(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "native")
    prepared, unused, matrix = _plan("coo")
    rhs = torch.ones((4, 3), dtype=torch.float64, device=prepared.data.device)
    output = torch.empty((5, 3), dtype=torch.float32, device=rhs.device)
    result = fs.flagsparse_spmm_coo(prepared.data, prepared.row, prepared.col, rhs, matrix.shape, out=output)
    assert result is output
    torch.testing.assert_close(result.cpu(), (matrix @ rhs.cpu()).float(), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
def test_fp64_grad_enabled_inputs_keep_the_differentiable_route(monkeypatch, fmt):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "auto")
    prepared, run, matrix = _plan(fmt)
    rhs = torch.ones((4, 2), dtype=torch.float64, device=prepared.data.device, requires_grad=True)
    with torch.enable_grad():
        assert native.run_fp64_spmm(prepared, rhs, "row") is None
        assert native.run_fp64_spmm_once(prepared.data, torch.zeros(prepared.nnz, dtype=torch.int32, device=rhs.device),
                                        torch.zeros(prepared.nnz, dtype=torch.int32, device=rhs.device),
                                        rhs, matrix.shape) is None
