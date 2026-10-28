"""CPU regression coverage for the Ascend prepared CSC SpMV path."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import spmv_csc  # noqa: E402


@pytest.fixture
def ascend_csc(monkeypatch):
    monkeypatch.setattr(spmv_csc, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(spmv_csc, "_SPMV_CSC_CUDA", False)
    monkeypatch.setattr(
        spmv_csc, "_is_accel_tensor", lambda tensor: tensor.device.type == "cpu"
    )
    monkeypatch.setattr(spmv_csc, "_spmv_rocm_launch_overrides", lambda **kwargs: None)
    monkeypatch.setattr(
        spmv_csc,
        "_get_device_backend_info",
        lambda device: {"backend": "ascend", "device_warp_size": 32},
    )
    monkeypatch.setattr(spmv_csc, "_ACCEL", SimpleNamespace(synchronize=lambda: None))
    return spmv_csc


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
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_prepared_csc_operations(ascend_csc, dtype, op, index_dtype):
    values = torch.tensor([2, -1, 3, 4, -2, 1], dtype=dtype)
    if dtype.is_complex:
        values = values + torch.tensor([1j, -2j, 3j, 1j, -1j, 2j], dtype=dtype)
    indices = torch.tensor([0, 2, 1, 1, 3, 0], dtype=index_dtype)
    indptr = torch.tensor([0, 2, 2, 4, 5, 6], dtype=index_dtype)
    x = torch.arange(1, 5 if op != "non" else 6, dtype=torch.float32).to(dtype)
    if dtype.is_complex:
        x = x + 0.5j
    plan = ascend_csc.prepare_spmv_csc(values, indices, indptr, (4, 5), op=op, x=x)
    cols = torch.tensor([0, 0, 2, 2, 3, 4])
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    dense = torch.zeros((4, 5), dtype=compute_dtype)
    dense.index_put_((indices.long(), cols), values.to(compute_dtype), accumulate=True)
    matrix = dense if op == "non" else (dense.conj().t() if op == "conj" else dense.t())
    expected = (matrix @ x.to(compute_dtype)).to(dtype)
    out = torch.empty_like(expected)
    result, elapsed, meta = ascend_csc.flagsparse_spmv_csc(
        prepared=plan, x=x, out=out, return_time=True, return_meta=True
    )
    assert result is out
    assert elapsed >= 0
    assert meta["launch_backend"] == "ascend"
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_prepared_csc_cache_observes_mutations(ascend_csc, dtype):
    values = torch.tensor([2, 3, 4], dtype=dtype)
    indices = torch.tensor([0, 1, 0], dtype=torch.int32)
    indptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    x = torch.tensor([1, 2], dtype=dtype)
    plan = ascend_csc.prepare_spmv_csc(values, indices, indptr, (2, 2), x=x)
    cached_state = plan.ascend_state
    torch.testing.assert_close(
        ascend_csc.flagsparse_spmv_csc(prepared=plan, x=x),
        torch.tensor([10, 3], dtype=values.dtype),
    )
    assert plan.ascend_state is cached_state
    assert not hasattr(ascend_csc, "_ASCEND_CSC_INDEX_CACHE")
    assert not hasattr(ascend_csc, "_ASCEND_CSC_VALUE_CACHE")
    values[0] = 5
    torch.testing.assert_close(
        ascend_csc.flagsparse_spmv_csc(prepared=plan, x=x),
        torch.tensor([13, 3], dtype=values.dtype),
    )
    x[0] = 3
    torch.testing.assert_close(
        ascend_csc.flagsparse_spmv_csc(prepared=plan, x=x),
        torch.tensor([23, 9], dtype=values.dtype),
    )
    indices[2] = 1
    indptr[1] = 1
    torch.testing.assert_close(
        ascend_csc.flagsparse_spmv_csc(prepared=plan, x=x),
        torch.tensor([15, 14], dtype=values.dtype),
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("nnz", [0, 3])
def test_prepared_csc_half_float32_output(ascend_csc, op, nnz):
    values = torch.tensor([2048, 1, -2048][:nnz], dtype=torch.float16)
    indices = torch.zeros(nnz, dtype=torch.int32)
    indptr = torch.tensor([0, nnz], dtype=torch.int32)
    x = torch.ones(1, dtype=values.dtype)
    plan = ascend_csc.prepare_spmv_csc(values, indices, indptr, (1, 1), op=op, x=x)
    result = ascend_csc.flagsparse_spmv_csc(prepared=plan, x=x, out_dtype=torch.float32)
    torch.testing.assert_close(result, torch.tensor([float(bool(nnz))]))
    with pytest.raises(TypeError, match="writes"):
        ascend_csc.flagsparse_spmv_csc(prepared=plan, x=x, out_dtype=torch.int32)


def test_half_preparation_remains_ascend_only(ascend_csc, monkeypatch):
    monkeypatch.setattr(ascend_csc, "_is_ascend_runtime", lambda: False)
    with pytest.raises(TypeError, match="CSC SpMV supports"):
        ascend_csc.prepare_spmv_csc(
            torch.ones(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            (1, 1),
        )


def test_prepare_validates_optional_rhs(ascend_csc):
    with pytest.raises(ValueError, match="x length"):
        ascend_csc.prepare_spmv_csc(
            torch.ones(1, dtype=torch.float16),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            (1, 1),
            x=torch.ones(2, dtype=torch.float16),
        )


def test_prepared_csc_releases_owned_conversions(ascend_csc):
    import gc
    import weakref

    values = torch.tensor([2, 3], dtype=torch.float16)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    rhs = torch.ones(2, dtype=values.dtype)
    plan = ascend_csc.prepare_spmv_csc(values, indices, indptr, (2, 2), x=rhs)
    value_ref = weakref.ref(plan.ascend_state["values"])
    rhs_ref = weakref.ref(plan.ascend_state["rhs"][1])
    assert value_ref() is not None
    del plan
    gc.collect()
    assert value_ref() is None
    assert rhs_ref() is None


def test_csc_preparation_does_not_publish_global_topology(ascend_csc, monkeypatch):
    from flagsparse.sparse_operations import _common

    monkeypatch.setattr(_common, "_CSR_ROW_IDS_CACHE", {})
    values = torch.tensor([2, 3], dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    plan = ascend_csc.prepare_spmv_csc(values, indices, indptr, (2, 2))
    assert plan.ascend_state is not None
    assert not _common._CSR_ROW_IDS_CACHE


def test_ascend_direct_row_cache_has_a_byte_limit(monkeypatch):
    from flagsparse.sparse_operations import _common

    monkeypatch.setattr(_common, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(_common, "_CSR_ROW_IDS_CACHE", {})
    monkeypatch.setattr(_common, "_ASCEND_ROW_IDS_CACHE_MAX_BYTES", 48)
    first = torch.tensor([0, 4], dtype=torch.int32)
    second = torch.tensor([0, 4], dtype=torch.int32)
    oversized = torch.tensor([0, 8], dtype=torch.int32)
    _common._csr_row_ids_torch(first, 1)
    _common._csr_row_ids_torch(second, 1)
    assert id(first) not in _common._CSR_ROW_IDS_CACHE
    assert id(second) in _common._CSR_ROW_IDS_CACHE
    _common._csr_row_ids_torch(oversized, 1)
    assert id(oversized) not in _common._CSR_ROW_IDS_CACHE
    assert (
        sum(
            entry[3].numel() * entry[3].element_size()
            for entry in _common._CSR_ROW_IDS_CACHE.values()
        )
        <= 48
    )
