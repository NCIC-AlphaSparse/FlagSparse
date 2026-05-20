"""CPU-only regression checks for pure runtime policy helpers."""

import os

import pytest

if os.environ.get("FLAGSPARSE_TRITON_SMOKE") != "1":
    pytest.skip(
        "triton smoke is opt-in and excluded from CPU-only CI", allow_module_level=True
    )

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import gather_scatter as gather_scatter_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_coo as spmv_coo_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csr as spmv_csr_ops  # noqa: E402
from flagsparse.sparse_operations import _common  # noqa: E402


def test_scatter_dtype_policy_fallback_is_explicit():
    value_dtype, fell_back, reason = _common._resolve_scatter_value_dtype(
        "complex32", dtype_policy="auto"
    )
    if _common._torch_complex32_dtype() is None:
        assert value_dtype == torch.complex64
        assert fell_back is True
        assert reason is not None
    else:
        assert value_dtype == _common._torch_complex32_dtype()
        assert fell_back is False
        assert reason is None


def test_scatter_dtype_policy_strict_rejects_missing_complex32():
    if _common._torch_complex32_dtype() is not None:
        pytest.skip("torch has native complex32 in this environment")
    with pytest.raises(TypeError, match="complex32 is unavailable"):
        _common._resolve_scatter_value_dtype("complex32", dtype_policy="strict")


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_index_fallback_policy_normalization(policy):
    assert spmv_csr_ops._normalize_spmv_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_coo_index_fallback_policy_normalization(policy):
    assert spmv_coo_ops._normalize_spmv_coo_index_fallback_policy(policy) == policy


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmv_coo_op_normalization(op, expected):
    assert spmv_coo_ops._normalize_spmv_coo_op(op) == expected


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_csr_op_transpose_contract(op):
    if op == "non":
        assert (
            spmv_csr_ops._spmv_op_transposes(spmv_csr_ops._normalize_spmv_op(op))
            is False
        )
    else:
        assert (
            spmv_csr_ops._spmv_op_transposes(spmv_csr_ops._normalize_spmv_op(op))
            is True
        )


def test_scatter_policy_validator_rejects_unknown_policy():
    with pytest.raises(
        ValueError, match="index_fallback_policy must be 'auto' or 'strict'"
    ):
        gather_scatter_ops._triton_scatter_impl(
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.int64),
            1,
            index_fallback_policy="invalid",
        )
