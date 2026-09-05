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
from flagsparse.sparse_operations import spmv_bsr as spmv_bsr_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csc as spmv_csc_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csr as spmv_csr_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_bell as spmm_bell_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_bsr as spmm_bsr_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_csc as spmm_csc_ops  # noqa: E402
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


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_csc_index_fallback_policy_normalization(policy):
    assert spmv_csc_ops._normalize_spmv_csc_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmv_bsr_index_fallback_policy_normalization(policy):
    assert spmv_bsr_ops._normalize_spmv_bsr_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmm_bsr_index_fallback_policy_normalization(policy):
    assert spmm_bsr_ops._normalize_spmm_bsr_index_fallback_policy(policy) == policy


@pytest.mark.parametrize("policy", ["auto", "strict"])
def test_spmm_csc_index_fallback_policy_normalization(policy):
    assert spmm_csc_ops._normalize_spmm_csc_index_fallback_policy(policy) == policy


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


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmv_csc_op_normalization(op, expected):
    assert spmv_csc_ops._normalize_spmv_csc_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmv_bsr_op_normalization(op, expected):
    assert spmv_bsr_ops._normalize_spmv_bsr_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmm_bsr_op_normalization(op, expected):
    assert spmm_bsr_ops._normalize_spmm_bsr_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmm_csc_op_normalization(op, expected):
    assert spmm_csc_ops._normalize_spmm_csc_op(op) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (None, 0),
        ("non", 0),
        ("trans", 1),
        ("conj", 2),
    ],
)
def test_spmm_bell_op_normalization(op, expected):
    assert spmm_bell_ops._normalize_spmm_bell_op(op) == expected


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


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_csc_op_transpose_contract(op):
    if op == "non":
        assert (
            spmv_csc_ops._spmv_csc_op_transposes(
                spmv_csc_ops._normalize_spmv_csc_op(op)
            )
            is False
        )
    else:
        assert (
            spmv_csc_ops._spmv_csc_op_transposes(
                spmv_csc_ops._normalize_spmv_csc_op(op)
            )
            is True
        )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_bsr_op_transpose_contract(op):
    if op == "non":
        assert (
            spmv_bsr_ops._spmv_bsr_op_transposes(
                spmv_bsr_ops._normalize_spmv_bsr_op(op)
            )
            is False
        )
    else:
        assert (
            spmv_bsr_ops._spmv_bsr_op_transposes(
                spmv_bsr_ops._normalize_spmv_bsr_op(op)
            )
            is True
        )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmv_bsr_supported_ops_accepted_by_policy(op):
    assert (
        spmv_bsr_ops._ensure_spmv_bsr_supported_op(
            spmv_bsr_ops._normalize_spmv_bsr_op(op)
        )
        is None
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_bsr_supported_ops_accepted_by_policy(op):
    assert (
        spmm_bsr_ops._ensure_spmm_bsr_supported_op(
            spmm_bsr_ops._normalize_spmm_bsr_op(op)
        )
        is None
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_csc_op_transpose_contract(op):
    if op == "non":
        assert (
            spmm_csc_ops._spmm_csc_op_transposes(
                spmm_csc_ops._normalize_spmm_csc_op(op)
            )
            is False
        )
    else:
        assert (
            spmm_csc_ops._spmm_csc_op_transposes(
                spmm_csc_ops._normalize_spmm_csc_op(op)
            )
            is True
        )


@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmm_bell_op_transpose_contract(op):
    assert (
        spmm_bell_ops._spmm_bell_op_transposes(
            spmm_bell_ops._normalize_spmm_bell_op(op)
        )
        is True
    )


def test_spmm_bell_non_op_transpose_contract():
    assert (
        spmm_bell_ops._spmm_bell_op_transposes(
            spmm_bell_ops._normalize_spmm_bell_op("non")
        )
        is False
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_csc_supported_ops_accepted_by_policy(op):
    assert (
        spmm_csc_ops._ensure_spmm_csc_supported_op(
            spmm_csc_ops._normalize_spmm_csc_op(op)
        )
        is None
    )


def test_spmm_bell_supported_non_op_accepted_by_policy():
    assert (
        spmm_bell_ops._ensure_spmm_bell_supported_op(
            spmm_bell_ops._normalize_spmm_bell_op("non")
        )
        is None
    )


@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmm_bell_reserved_ops_rejected_by_policy(op):
    with pytest.raises(ValueError, match="reserved"):
        spmm_bell_ops._ensure_spmm_bell_supported_op(
            spmm_bell_ops._normalize_spmm_bell_op(op)
        )


@pytest.mark.parametrize("alg", ["auto", "base", "spmm_bsr_base"])
def test_spmm_bsr_algorithm_normalization(alg):
    normalized = spmm_bsr_ops._normalize_spmm_bsr_alg(alg)
    assert normalized in ("auto", "spmm_bsr_base")


@pytest.mark.parametrize("alg", ["auto", "base", "spmm_csc_base"])
def test_spmm_csc_algorithm_normalization(alg):
    normalized = spmm_csc_ops._normalize_spmm_csc_alg(alg)
    assert normalized in ("auto", "spmm_csc_base")


@pytest.mark.parametrize("alg", ["auto", "base", "spmm_bell_base"])
def test_spmm_bell_algorithm_normalization(alg):
    normalized = spmm_bell_ops._normalize_spmm_bell_alg(alg)
    assert normalized in ("auto", "spmm_bell_base")


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
