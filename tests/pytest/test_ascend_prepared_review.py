"""Ascend review regressions: graph output lifetime and prepared scalars."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import mixed_spmx
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = pytest.mark.skipif(
    not (_is_ascend_runtime() and accelerator_available()),
    reason="real Ascend runtime required",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.complex64])
@pytest.mark.parametrize(
    "alpha,beta", [(2.0, None), (None, 0.5), (2.0, 0.5), (None, None)]
)
def test_prepared_axpby_call_time_scalars(dtype, alpha, beta):
    dev = accelerator_device()
    values = torch.tensor([1, -2, 3], dtype=dtype, device=dev)
    indices = torch.tensor([0, 2, 4], dtype=torch.int32, device=dev)
    y = torch.arange(1, 7, dtype=torch.float32).to(dtype).to(dev)
    prepared = fs.prepare_axpby(values, indices, y, alpha=1.0, beta=0.25)
    expected = y.cpu() * (0.25 if beta is None else beta)
    expected[indices.cpu().long()] += values.cpu() * (1.0 if alpha is None else alpha)
    result = fs.flagsparse_axpby(y=y, prepared=prepared, alpha=alpha, beta=beta)
    assert result is y
    torch.testing.assert_close(result.cpu(), expected)


def _spmm_plan(fmt, dtype, mixed):
    dev = accelerator_device()
    rows = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int32, device=dev)
    cols = torch.tensor([0, 2, 1, 0, 2], dtype=torch.int32, device=dev)
    ptr = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device=dev)
    values_cpu = torch.tensor([1, -2, 3, 4, 5], dtype=dtype)
    if dtype.is_complex:
        values_cpu += torch.tensor([2, 1, -3, 4, -2], dtype=dtype) * 1j
    values = values_cpu.to(dev)
    dense_cpu = torch.zeros(3, 3, dtype=dtype)
    dense_cpu[rows.cpu().long(), cols.cpu().long()] = values_cpu
    if mixed:
        prepare = getattr(mixed_spmx, f"prepare_spmm_{fmt}_mixed")
        run_impl = getattr(mixed_spmx, f"spmm_{fmt}_mixed_prepared")
        args = (values, rows, cols) if fmt == "coo" else (values, cols, ptr)
        prepared = prepare(*args, (3, 3))
        run = lambda rhs: run_impl(prepared, rhs)
        attr = "runner"
    else:
        prepare = getattr(fs, f"prepare_spmm_{fmt}_route")
        run_impl = getattr(fs, f"flagsparse_spmm_{fmt}_run")
        args = (values, rows, cols) if fmt == "coo" else (values, cols, ptr)
        prepared = prepare(*args, (3, 3))
        run = lambda rhs: run_impl(prepared, rhs)
        attr = "ascend_runner"
    if fmt == "csc":
        dense_cpu = dense_cpu.T
    return prepared, run, attr, dense_cpu


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize(
    "dtype,mixed",
    [(torch.float32, False), (torch.complex64, False), (torch.float16, True)],
)
def test_spmm_graph_results_own_storage(fmt, dtype, mixed):
    prepared, run, attr, matrix = _spmm_plan(fmt, dtype, mixed)
    rhs_cpu = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).to(dtype)
    rhs = rhs_cpu.to(accelerator_device())
    for _ in range(3):
        run(rhs)
    runner = getattr(prepared, attr)
    assert callable(runner), f"{fmt} {dtype} silently fell back to eager"
    raw_rhs = rhs.to(prepared.acc_dtype) if mixed else rhs
    if fmt == "csc" and dtype.is_complex:
        raw_rhs = torch.view_as_real(rhs)
    raw_first = runner(raw_rhs)
    raw_saved = raw_first.clone()
    raw_second = runner(raw_rhs * -2)
    print(
        f"GRAPH_OUTPUT_REUSE {fmt} {dtype} mixed={mixed} same_ptr={raw_first.data_ptr() == raw_second.data_ptr()} overwritten={not torch.equal(raw_first.cpu(), raw_saved.cpu())}"
    )
    first = run(rhs)
    saved = first.clone()
    second = run(rhs * -2)
    assert torch.equal(first.cpu(), saved.cpu())
    assert first.data_ptr() != second.data_ptr()
    reference = matrix.to(
        torch.complex128 if dtype.is_complex else torch.float64
    ) @ rhs_cpu.to(torch.complex128 if dtype.is_complex else torch.float64)
    torch.testing.assert_close(first.cpu(), reference.to(first.dtype))
    torch.testing.assert_close(second.cpu(), (-2 * reference).to(second.dtype))


def test_sell_graph_results_own_storage():
    dev = accelerator_device()
    values = torch.arange(1, 65, dtype=torch.float32).to(torch.complex64).to(dev)
    cols = (torch.arange(64, dtype=torch.int32) % 8).to(dev)
    offsets = torch.tensor([0, 64], dtype=torch.int32, device=dev)
    prepared = fs.prepare_spmv_sell(
        values, cols, offsets, (32, 8), slice_size=32, route="call"
    )
    rhs = torch.arange(1, 9, dtype=torch.float32).to(torch.complex64).to(dev)
    for _ in range(3):
        fs.flagsparse_spmv_sell(prepared=prepared, x=rhs)
    assert callable(prepared.ascend_runner), "SELL silently fell back to eager"
    raw_first = prepared.ascend_runner(torch.view_as_real(rhs))
    raw_saved = raw_first.clone()
    raw_second = prepared.ascend_runner(torch.view_as_real(rhs) * -2)
    print(
        f"GRAPH_OUTPUT_REUSE sell complex64 same_ptr={raw_first.data_ptr() == raw_second.data_ptr()} overwritten={not torch.equal(raw_first.cpu(), raw_saved.cpu())}"
    )
    first = fs.flagsparse_spmv_sell(prepared=prepared, x=rhs)
    saved = first.clone()
    second = fs.flagsparse_spmv_sell(prepared=prepared, x=rhs * -2)
    assert torch.equal(first.cpu(), saved.cpu())
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(second.cpu(), first.cpu() * -2)


def test_graph_plans_preserve_global_dynamo_limits():
    original_limit = torch._dynamo.config.cache_size_limit
    rhs = torch.ones(3, 4, device=accelerator_device())
    for _ in range(12):
        prepared, run, attr, _matrix = _spmm_plan("csc", torch.float32, False)
        run(rhs)
        assert callable(getattr(prepared, attr))
    assert torch._dynamo.config.cache_size_limit == original_limit


def test_transposed_csr_reports_materialized_runner():
    from flagsparse.sparse_operations._common import _ascend_execution_meta

    device = accelerator_device()
    values = torch.tensor([1.0, 2.0, 3.0], device=device)
    cols = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int32, device=device)
    prepared = fs.prepare_spmm_csr_route(values, cols, ptr, (3, 3), op="trans")
    _, meta = fs.flagsparse_spmm_csr_run(
        prepared, torch.ones(3, 4, device=device), return_meta=True
    )
    assert callable(prepared.materialized_route.ascend_runner)
    assert meta["execution_path"] == "graph"
    assert _ascend_execution_meta(prepared)["execution_path"] == "graph"


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("mixed", [False, True])
def test_graph_failure_is_warned_once_and_visible(monkeypatch, fmt, mixed):
    from flagsparse.sparse_operations import spmm_coo, spmm_csc, spmm_csr
    from flagsparse.sparse_operations._common import _ascend_execution_meta

    module = (
        mixed_spmx
        if mixed
        else {"csr": spmm_csr, "coo": spmm_coo, "csc": spmm_csc}[fmt]
    )

    def fail_compile(function):
        raise RuntimeError("injected graph failure")

    monkeypatch.setattr(module, "_ascend_compile", fail_compile)
    prepared, run, attr, matrix = _spmm_plan(
        fmt, torch.float16 if mixed else torch.float32, mixed
    )
    rhs = torch.ones(
        3,
        4,
        dtype=torch.float16 if mixed else torch.float32,
        device=accelerator_device(),
    )
    with pytest.warns(RuntimeWarning, match="injected graph failure") as observed:
        first = run(rhs)
        second = run(rhs * -2)
    assert len(observed) == 1
    assert _ascend_execution_meta(prepared, attr)["execution_path"] == "eager"
    expected = matrix.float() @ rhs.cpu().float()
    torch.testing.assert_close(first.cpu(), expected.to(first.dtype))
    torch.testing.assert_close(second.cpu(), (-2 * expected).to(second.dtype))
