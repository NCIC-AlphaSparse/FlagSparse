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

"""Accuracy coverage for lower NON_TRANS SELL SpSV."""

import pytest
import torch

from flagsparse import (
    flagsparse_spsv_analysis_sell,
    flagsparse_spsv_create_workspace,
    flagsparse_spsv_solve_sell,
)
from tests.pytest.param_shapes import CORE_DTYPES, CORE_DTYPE_IDS, SPSV_N


pytestmark = [
    pytest.mark.spsv_sell,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _sell_matches(actual, expected, dtype):
    if actual.shape != expected.shape:
        return False
    if not bool(torch.isfinite(actual).all().item()) or not bool(
        torch.isfinite(expected).all().item()
    ):
        return False
    error = float(
        torch.max(torch.abs(torch.abs(actual) - torch.abs(expected))).item()
    )
    scale = float(torch.max(torch.abs(expected)).item())
    if scale == 0.0:
        return error == 0.0
    tolerance = 1e-6 if dtype in (torch.float32, torch.complex64) else 1e-12
    return error / scale <= tolerance


def _lower_triangular_sell(
    n,
    dtype,
    index_dtype,
    slice_size,
    device,
    *,
    unit_diagonal=False,
    store_diagonal=True,
):
    complex_values = dtype in (torch.complex64, torch.complex128)
    dense = torch.zeros((n, n), dtype=dtype)
    row_values = []
    row_columns = []
    for row in range(n):
        columns = list(range(max(0, row - 3), row + 1))
        if unit_diagonal and not store_diagonal:
            columns.remove(row)
        if row % 2:
            columns.reverse()  # Exercise the documented unsorted-index path.
        values = []
        for col in columns:
            if row == col:
                value = (
                    complex(3.0 + 0.05 * row, 0.75 + 0.01 * row)
                    if complex_values
                    else 3.0 + 0.05 * row
                )
            else:
                real = 0.01 * ((row + col) % 7 + 1)
                value = complex(real, -0.5 * real) if complex_values else real
            dense[row, col] = value
            values.append(value)
        row_columns.append(columns)
        row_values.append(values)
        if unit_diagonal:
            # UNIT semantics ignore an explicitly stored diagonal value and
            # also allow the diagonal entry to be absent from sparse storage.
            dense[row, row] = 1

    n_slices = (n + slice_size - 1) // slice_size
    widths = []
    for slice_id in range(n_slices):
        row0 = slice_id * slice_size
        row1 = min(row0 + slice_size, n)
        widths.append(max(len(row_columns[row]) for row in range(row0, row1)))
    offsets_host = [0]
    for width in widths:
        offsets_host.append(offsets_host[-1] + width * slice_size)

    sell_values = torch.zeros(offsets_host[-1], dtype=dtype, device=device)
    sell_columns = torch.full(
        (offsets_host[-1],), -1, dtype=index_dtype, device=device
    )
    for slice_id in range(n_slices):
        row0 = slice_id * slice_size
        row1 = min(row0 + slice_size, n)
        for row in range(row0, row1):
            for slot, (col, value) in enumerate(
                zip(row_columns[row], row_values[row])
            ):
                offset = offsets_host[slice_id] + slot * slice_size + row - row0
                sell_values[offset] = value
                sell_columns[offset] = col

    offsets = torch.tensor(offsets_host, dtype=index_dtype, device=device)
    return sell_values, sell_columns, offsets, dense.to(device)


def _reference_solve(dense, b, dtype):
    if dtype == torch.float32:
        reference_dtype = torch.float64
    elif dtype == torch.complex64:
        reference_dtype = torch.complex128
    else:
        reference_dtype = dtype
    return torch.linalg.solve_triangular(
        dense.to(reference_dtype),
        b.to(reference_dtype).unsqueeze(1),
        upper=False,
    ).squeeze(1).to(dtype)


@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", CORE_DTYPES, ids=CORE_DTYPE_IDS)
@pytest.mark.parametrize(
    "index_dtype", (torch.int32, torch.int64), ids=("int32", "int64")
)
@pytest.mark.parametrize("slice_size", (8, 32), ids=("slice8", "slice32"))
@pytest.mark.parametrize("alg_num", (1, 2), ids=("alg1", "alg2"))
def test_spsv_sell_matches_dense_and_supports_inplace(
    n, dtype, index_dtype, slice_size, alg_num
):
    device = torch.device("cuda")
    values, columns, offsets, dense = _lower_triangular_sell(
        n, dtype, index_dtype, slice_size, device
    )
    if dtype in (torch.complex64, torch.complex128):
        real_dtype = values.real.dtype
        b = torch.complex(
            torch.linspace(0.25, 1.25, n, dtype=real_dtype, device=device),
            torch.linspace(-0.5, 0.5, n, dtype=real_dtype, device=device),
        )
    else:
        b = torch.linspace(0.25, 1.25, n, dtype=dtype, device=device)
    expected = _reference_solve(dense, b, dtype)

    descr = flagsparse_spsv_analysis_sell(
        values,
        columns,
        offsets,
        (n, n),
        slice_size=slice_size,
        alg_num=alg_num,
    )
    workspace = flagsparse_spsv_create_workspace(descr)
    result = flagsparse_spsv_solve_sell(descr, b, workspace=workspace)
    inplace = b.clone()
    flagsparse_spsv_solve_sell(
        descr,
        inplace,
        out=inplace,
        workspace=workspace,
    )

    assert _sell_matches(result, expected, dtype)
    assert _sell_matches(inplace, expected, dtype)


@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", CORE_DTYPES, ids=CORE_DTYPE_IDS)
@pytest.mark.parametrize(
    "index_dtype", (torch.int32, torch.int64), ids=("int32", "int64")
)
@pytest.mark.parametrize("alg_num", (1, 2), ids=("alg1", "alg2"))
@pytest.mark.parametrize(
    "store_diagonal", (True, False), ids=("stored_diag_ignored", "missing_diag")
)
def test_spsv_sell_unit_diagonal(
    n, dtype, index_dtype, alg_num, store_diagonal
):
    device = torch.device("cuda")
    values, columns, offsets, dense = _lower_triangular_sell(
        n,
        dtype,
        index_dtype,
        8,
        device,
        unit_diagonal=True,
        store_diagonal=store_diagonal,
    )
    if dtype in (torch.complex64, torch.complex128):
        real_dtype = values.real.dtype
        b = torch.complex(
            torch.linspace(0.25, 1.25, n, dtype=real_dtype, device=device),
            torch.linspace(-0.5, 0.5, n, dtype=real_dtype, device=device),
        )
    else:
        b = torch.linspace(0.25, 1.25, n, dtype=dtype, device=device)
    expected = _reference_solve(dense, b, dtype)

    descr = flagsparse_spsv_analysis_sell(
        values,
        columns,
        offsets,
        (n, n),
        slice_size=8,
        alg_num=alg_num,
        unit_diagonal=True,
    )
    assert descr.unit_diagonal is True
    assert descr.diag_type == "unit"
    workspace = flagsparse_spsv_create_workspace(descr)
    result = flagsparse_spsv_solve_sell(descr, b, workspace=workspace)
    inplace = b.clone()
    flagsparse_spsv_solve_sell(
        descr,
        inplace,
        out=inplace,
        workspace=workspace,
    )

    assert _sell_matches(result, expected, dtype)
    assert _sell_matches(inplace, expected, dtype)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("alg_num", (1, 2), ids=("alg1", "alg2"))
def test_spsv_sell_non_unit_zero_diagonal_is_not_silently_repaired(
    dtype, alg_num
):
    """A singular NON_UNIT row must keep its IEEE non-finite solve result."""

    device = torch.device("cuda")
    slice_size = 8
    values = torch.zeros(slice_size, dtype=dtype, device=device)
    columns = torch.full(
        (slice_size,), -1, dtype=torch.int32, device=device
    )
    columns[0] = 0
    offsets = torch.tensor([0, slice_size], dtype=torch.int32, device=device)
    b = torch.ones(1, dtype=dtype, device=device)

    descr = flagsparse_spsv_analysis_sell(
        values,
        columns,
        offsets,
        (1, 1),
        slice_size=slice_size,
        alg_num=alg_num,
    )
    result = flagsparse_spsv_solve_sell(
        descr,
        b,
        workspace=flagsparse_spsv_create_workspace(descr),
    )

    assert not torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("columns", "match"),
    (
        ([-1] * 8, "exactly one diagonal"),
        ([0] * 8, "exactly one diagonal"),
        ([-1] * 8 + [0] + [-1] * 7, "padding .* trailing"),
    ),
    ids=("missing_diagonal", "duplicate_diagonal", "middle_padding"),
)
def test_spsv_sell_non_unit_rejects_malformed_structure(columns, match):
    device = torch.device("cuda")
    values = torch.ones(len(columns), dtype=torch.float32, device=device)
    col_indices = torch.tensor(columns, dtype=torch.int32, device=device)
    offsets = torch.tensor(
        [0, len(columns)], dtype=torch.int32, device=device
    )

    with pytest.raises(ValueError, match=match):
        flagsparse_spsv_analysis_sell(
            values,
            col_indices,
            offsets,
            (1, 1),
            slice_size=8,
        )


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float64, torch.complex64, torch.complex128)
)
@pytest.mark.parametrize("alg_num", (1, 2), ids=("alg1", "alg2"))
@pytest.mark.parametrize("nan_in_diagonal", (False, True))
def test_spsv_sell_nonfinite_input_is_not_silently_repaired(
    dtype, alg_num, nan_in_diagonal
):
    device = torch.device("cuda")
    slice_size = 8
    values = torch.zeros(slice_size, dtype=dtype, device=device)
    values[0] = float("nan") if nan_in_diagonal else 1
    columns = torch.full(
        (slice_size,), -1, dtype=torch.int32, device=device
    )
    columns[0] = 0
    offsets = torch.tensor([0, slice_size], dtype=torch.int32, device=device)
    b = torch.full(
        (1,),
        1 if nan_in_diagonal else float("nan"),
        dtype=dtype,
        device=device,
    )

    descr = flagsparse_spsv_analysis_sell(
        values,
        columns,
        offsets,
        (1, 1),
        slice_size=slice_size,
        alg_num=alg_num,
    )
    result = flagsparse_spsv_solve_sell(
        descr,
        b,
        workspace=flagsparse_spsv_create_workspace(descr),
    )

    assert torch.isnan(result).all()
