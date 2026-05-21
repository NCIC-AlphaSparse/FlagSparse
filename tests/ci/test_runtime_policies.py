"""CPU-only regression checks for pure runtime policy helpers."""

import os

import pytest

if os.environ.get("FLAGSPARSE_TRITON_SMOKE") != "1":
    pytest.skip(
        "triton smoke is opt-in and excluded from CPU-only CI", allow_module_level=True
    )

torch = pytest.importorskip("torch")

# fmt: off
# isort: off
from flagsparse.sparse_operations import _common  # noqa: E402
from flagsparse.sparse_operations import (  # noqa: E402
    gather_scatter as gather_scatter_ops,
)
from flagsparse.sparse_operations import spmv_coo as spmv_coo_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csr as spmv_csr_ops  # noqa: E402
# isort: on
# fmt: on


def test_scatter_dtype_policy_complex64_resolves_directly():
    value_dtype, fell_back, reason = _common._resolve_scatter_value_dtype(
        "complex64", dtype_policy="auto"
    )
    assert value_dtype == torch.complex64
    assert fell_back is False
    assert reason is None


def test_scatter_dtype_policy_complex128_resolves_directly():
    value_dtype, fell_back, reason = _common._resolve_scatter_value_dtype(
        "complex128", dtype_policy="strict"
    )
    assert value_dtype == torch.complex128
    assert fell_back is False
    assert reason is None


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
