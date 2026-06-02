import pytest
import torch

from flagsparse import flagsparse_gather, flagsparse_scatter
from flagsparse.sparse_operations import gather_scatter as gather_scatter_ops

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import FLOAT_DTYPES, GATHER_SCATTER_SHAPES


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
INDEX_DTYPES = [torch.int32, torch.int64]
INDEX_DTYPE_IDS = ["int32", "int64"]
RESET_OUTPUT_CASES = [True, False]
RESET_OUTPUT_IDS = ["reset", "inplace"]

GATHER_DTYPE_CASES = [
    ("float", torch.float32),
    ("double", torch.float64),
    ("half", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("complex64", torch.complex64),
    ("complex128", torch.complex128),
]
GATHER_DTYPE_IDS = [name for name, _ in GATHER_DTYPE_CASES]


def _scatter_dtype_cases():
    cases = [(str(dtype).replace("torch.", ""), dtype) for dtype in FLOAT_DTYPES]
    cases.append(("complex64", torch.complex64))
    cases.append(("complex128", torch.complex128))
    return cases


SCATTER_DTYPE_CASES = _scatter_dtype_cases()
SCATTER_DTYPE_IDS = [name for name, _ in SCATTER_DTYPE_CASES]


def _skip_unavailable_dtype(dtype_name, dtype):
    if dtype is None:
        pytest.skip(f"{dtype_name} dtype is unavailable in this torch build")
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")


def _build_random_values(size, dtype, device):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(size, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(size, dtype=torch.float32, device=device)
        imag = torch.randn(size, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(size, dtype=torch.float64, device=device)
        imag = torch.randn(size, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported dtype in test: {dtype}")


@pytest.mark.gather
@pytest.mark.parametrize("dense_size, nnz", GATHER_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype_name,dtype", GATHER_DTYPE_CASES, ids=GATHER_DTYPE_IDS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
def test_gather_matches_indexing(dense_size, nnz, dtype_name, dtype, index_dtype):
    _skip_unavailable_dtype(dtype_name, dtype)
    device = torch.device("cuda")
    nnz = min(nnz, dense_size)
    dense = _build_random_values(dense_size, dtype, device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    ref = dense[indices.to(torch.int64)]
    got = flagsparse_gather(dense, indices)
    assert torch.equal(ref, got)


@pytest.mark.gather
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
def test_gather_complex128_matches_indexing(index_dtype):
    device = torch.device("cuda")
    dense_size = 4096
    nnz = 1024
    real = torch.randn(dense_size, dtype=torch.float64, device=device)
    imag = torch.randn(dense_size, dtype=torch.float64, device=device)
    dense = torch.complex(real, imag)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    ref = dense.index_select(0, indices.to(torch.int64))
    got = flagsparse_gather(dense, indices)
    rtol, atol = close_tolerances(torch.complex128)
    assert torch.allclose(got, ref, atol=atol, rtol=rtol)


@pytest.mark.scatter
@pytest.mark.parametrize("dense_size, nnz", GATHER_SCATTER_SHAPES)
@pytest.mark.parametrize("dtype_name,dtype", SCATTER_DTYPE_CASES, ids=SCATTER_DTYPE_IDS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=INDEX_DTYPE_IDS)
@pytest.mark.parametrize("reset_output", RESET_OUTPUT_CASES, ids=RESET_OUTPUT_IDS)
def test_scatter_matches_index_copy(
    dense_size, nnz, dtype_name, dtype, index_dtype, reset_output
):
    _skip_unavailable_dtype(dtype_name, dtype)
    device = torch.device("cuda")
    nnz = min(nnz, dense_size)
    vals = _build_random_values(nnz, dtype, device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    dense_ref = _build_random_values(dense_size, dtype, device)
    dense = dense_ref.clone()
    if reset_output:
        dense_ref.zero_()
    dense_ref.index_copy_(0, indices.to(torch.int64), vals)
    flagsparse_scatter(
        dense,
        indices,
        vals,
        reset_output=reset_output,
        dtype_policy="auto",
    )
    assert torch.equal(dense, dense_ref)


@pytest.mark.scatter
def test_scatter_int64_auto_fallback_to_int32(monkeypatch):
    device = torch.device("cuda")
    dense_size = 257
    nnz = 129
    vals = torch.randn(nnz, dtype=torch.float32, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int64)
    dense = torch.randn(dense_size, dtype=torch.float32, device=device)
    dense_ref = dense.clone()
    dense_ref.zero_()
    dense_ref.index_copy_(0, indices.to(torch.int64), vals)

    original_launch = gather_scatter_ops._launch_triton_scatter_kernel
    state = {"forced_once": False}

    def fake_launch(dense_values, sparse_values, kernel_indices, nnz, block_size=1024):
        if kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original_launch(
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            block_size=block_size,
        )

    monkeypatch.setattr(gather_scatter_ops, "_launch_triton_scatter_kernel", fake_launch)

    flagsparse_scatter(
        dense,
        indices,
        vals,
        reset_output=True,
        dtype_policy="auto",
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    assert torch.equal(dense, dense_ref)


@pytest.mark.scatter
def test_scatter_int64_strict_no_fallback(monkeypatch):
    device = torch.device("cuda")
    dense_size = 257
    nnz = 129
    vals = torch.randn(nnz, dtype=torch.float32, device=device)
    indices = torch.randperm(dense_size, device=device)[:nnz].to(torch.int64)
    dense = torch.randn(dense_size, dtype=torch.float32, device=device)

    original_launch = gather_scatter_ops._launch_triton_scatter_kernel

    def fake_launch(dense_values, sparse_values, kernel_indices, nnz, block_size=1024):
        if kernel_indices.dtype == torch.int64:
            raise RuntimeError("forced int64 launch failure")
        return original_launch(
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            block_size=block_size,
        )

    monkeypatch.setattr(gather_scatter_ops, "_launch_triton_scatter_kernel", fake_launch)

    with pytest.raises(RuntimeError, match="Triton scatter failed for index dtype"):
        flagsparse_scatter(
            dense,
            indices,
            vals,
            reset_output=True,
            dtype_policy="auto",
            index_fallback_policy="strict",
        )
