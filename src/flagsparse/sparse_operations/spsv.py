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

"""Sparse triangular solve (SpSV) for CSR, COO, and SELL matrices."""

from ._common import *

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
import time
import triton
import triton.language as tl

SUPPORTED_SPSV_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SUPPORTED_SPSV_INDEX_DTYPES = (torch.int32, torch.int64)
# Public support-matrix aliases for the standalone SELL lifecycle API.
SUPPORTED_SPSV_SELL_VALUE_DTYPES = SUPPORTED_SPSV_VALUE_DTYPES
SUPPORTED_SPSV_SELL_INDEX_DTYPES = SUPPORTED_SPSV_INDEX_DTYPES
SPSV_NON_TRANS_SUPPORTED_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.int32),
    (torch.complex128, torch.int32),
    (torch.float32, torch.int64),
    (torch.float64, torch.int64),
    (torch.complex64, torch.int64),
    (torch.complex128, torch.int64),
)
SPSV_TRANS_SUPPORTED_COMBOS = (
    (torch.float32, torch.int32),
    (torch.float64, torch.int32),
    (torch.complex64, torch.int32),
    (torch.complex128, torch.int32),
    (torch.float32, torch.int64),
    (torch.float64, torch.int64),
    (torch.complex64, torch.int64),
    (torch.complex128, torch.int64),
)
SPSV_SELL_ALG1 = 1
SPSV_SELL_ALG2 = 2


def _normalize_spsv_sell_alg_num(alg_num):
    try:
        alg_num = int(alg_num)
    except (TypeError, ValueError) as exc:
        raise ValueError("SELL SpSV alg_num must be 1 or 2") from exc
    if alg_num not in (SPSV_SELL_ALG1, SPSV_SELL_ALG2):
        raise ValueError("SELL SpSV alg_num must be 1 or 2")
    return alg_num


def _resolve_spsv_sell_alg2_worker_count(n_slices, requested=None):
    n_slices = int(n_slices)
    if n_slices <= 0:
        return 1
    if requested is not None:
        requested = int(requested)
        if requested <= 0:
            raise ValueError("SELL SpSV ALG2 worker count must be positive")
        return min(requested, n_slices)

    # A SELL worker already exposes up to slice_size rows.  Keep the automatic
    # grid conservative to limit ready-flag polling contention; callers can
    # still override it explicitly for architecture-specific experiments.
    target = 64
    return _snap_cw_worker_count(min(target, n_slices), n_slices)


def _spsv_env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


SPSV_PROMOTE_FP32_TO_FP64 = _spsv_env_flag("FLAGSPARSE_SPSV_PROMOTE_FP32_TO_FP64", "0")
SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64", "0"
)
SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128", "0"
)
# SELL TRANS keeps the input/output dtype unchanged, but can use a wider
# residual accumulator for high-fan-in columns.  This is deliberately a
# separate switch from the full compute-dtype promotion flags above.
SPSV_SELL_TRANS_FP64_RESIDUAL = _spsv_env_flag(
    "FLAGSPARSE_SPSV_SELL_TRANS_FP64_RESIDUAL", "0"
)
_SPSV_CSR_PREPROCESS_CACHE = OrderedDict()
_SPSV_CSR_PREPROCESS_CACHE_SIZE = 8
_SPSV_SELL_TRANS_ANALYSIS_CACHE = OrderedDict()
_SPSV_SELL_TRANS_ANALYSIS_CACHE_SIZE = 4


@dataclass
class FlagSparseSpSVDescr:
    """Host-side analysis handle for Triton SpSV.

    This is the Triton/Python equivalent of the CUDA-side SpSV descriptor:
    it stores the analyzed matrix metadata, the selected solve route, and the
    workspace layout needed by the current implementation.
    """

    format: str
    canonical_format: str
    shape: tuple
    lower: bool
    unit_diagonal: bool
    fill_mode: str
    diag_type: str
    matrix_type: str
    index_base: int
    transpose_mode: str
    value_dtype: torch.dtype
    compute_dtype: torch.dtype
    index_dtype: torch.dtype
    solve_kind: str
    route_name: str
    storage_view: str
    buffer_size: int
    workspace_layout: tuple
    data: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    indptr: torch.Tensor = field(repr=False)
    solve_plan: dict = field(repr=False)


@dataclass
class FlagSparseSpSVWorkspace:
    """Caller-owned workspace object for Triton SpSV host APIs."""

    buffer_size: int
    layout: tuple
    device: torch.device
    buffers: dict = field(default_factory=dict, repr=False)
    prepared_solve_kind: str = ""
    prepared_signature: tuple | None = None


@dataclass
class FlagSparseSpSVHandle:
    """Host-side execution handle for Triton SpSV."""

    device: torch.device
    stream: object = None


@dataclass
class FlagSparseSpMatDescr:
    """Sparse matrix descriptor mirroring the CUDA SpMat inputs."""

    format: str
    shape: tuple
    values: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    indptr_or_col: torch.Tensor = field(repr=False)
    lower: bool = True
    unit_diagonal: bool = False
    diag_type: str = "non_unit"
    fill_mode: str = "lower"
    matrix_type: str = "triangular"
    index_base: int = 0


@dataclass
class FlagSparseDnVecDescr:
    """Dense vector descriptor mirroring the CUDA DnVec inputs."""

    values: torch.Tensor = field(repr=False)


def flagsparse_create_spsv_handle(device=None, stream=None):
    if device is None:
        device = torch.device("cuda")
    return FlagSparseSpSVHandle(device=torch.device(device), stream=stream)


def flagsparse_create_dnvec(values):
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim != 1:
        raise ValueError("DnVec values must be 1D")
    return FlagSparseDnVecDescr(values=values)


def flagsparse_create_spmat_csr(
    values,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    matrix_type="triangular",
    index_base=0,
):
    return FlagSparseSpMatDescr(
        format="csr",
        shape=(int(shape[0]), int(shape[1])),
        values=values,
        indices=indices,
        indptr_or_col=indptr,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        diag_type="unit" if unit_diagonal else "non_unit",
        fill_mode="lower" if lower else "upper",
        matrix_type=str(matrix_type),
        index_base=int(index_base),
    )


def flagsparse_create_spmat_coo(
    values,
    row,
    col,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    matrix_type="triangular",
    index_base=0,
):
    return FlagSparseSpMatDescr(
        format="coo",
        shape=(int(shape[0]), int(shape[1])),
        values=values,
        indices=row,
        indptr_or_col=col,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        diag_type="unit" if unit_diagonal else "non_unit",
        fill_mode="lower" if lower else "upper",
        matrix_type=str(matrix_type),
        index_base=int(index_base),
    )


def _clear_spsv_csr_preprocess_cache():
    _SPSV_CSR_PREPROCESS_CACHE.clear()


def _as_strided_contiguous(tensor):
    if tensor is None:
        return None
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    if tensor.layout != torch.strided:
        out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        out.copy_(tensor)
        return out
    return tensor.contiguous()


def _complex_interleaved_view(tensor):
    tensor_strided = _as_strided_contiguous(tensor)
    return torch.view_as_real(tensor_strided).reshape(-1).contiguous()


def _attach_spsv_complex_plan_views(plan):
    kernel_data = plan.get("kernel_data")
    if kernel_data is None or not torch.is_complex(kernel_data):
        return plan
    plan["kernel_data_ri"] = _complex_interleaved_view(kernel_data)
    return plan


def _validate_spsv_non_trans_combo(data_dtype, index_dtype, fmt_name):
    """Validate NON_TRANS support matrix and keep error messages explicit."""
    if (data_dtype, index_dtype) in SPSV_NON_TRANS_SUPPORTED_COMBOS:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports NON_TRANS combinations: "
        "(float32, int32/int64), (float64, int32/int64), "
        "(complex64, int32/int64), (complex128, int32/int64)"
    )


def _validate_spsv_trans_combo(data_dtype, index_dtype, fmt_name):
    if (data_dtype, index_dtype) in SPSV_TRANS_SUPPORTED_COMBOS:
        return
    raise TypeError(
        f"{fmt_name} SpSV currently supports TRANS/CONJ combinations: "
        "(float32, int32/int64), (float64, int32/int64), "
        "(complex64, int32/int64), (complex128, int32/int64)"
    )


def _normalize_spsv_transpose_mode(transpose):
    if isinstance(transpose, bool):
        return "T" if transpose else "N"
    token = str(transpose).strip().upper()
    if token in ("N", "NON", "NON_TRANS"):
        return "N"
    if token in ("T", "TRANS"):
        return "T"
    if token in ("C", "H", "CONJ", "CONJ_TRANS", "CONJUGATE_TRANSPOSE"):
        return "C"
    raise ValueError(
        "transpose must be bool or one of: "
        "N/NON/NON_TRANS, T/TRANS, C/H/CONJ/CONJ_TRANS/CONJUGATE_TRANSPOSE"
    )


def _prepare_spsv_inputs(data, indices, indptr, b, shape):
    """Validate and normalize inputs for sparse solve A x = b with CSR A."""
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, b)):
        raise TypeError("data, indices, indptr, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, b)):
        raise ValueError("data, indices, indptr, b must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")

    if data.dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")

    indices64 = indices.to(torch.int64).contiguous()
    indptr64 = indptr.to(torch.int64).contiguous()
    if indices64.numel() > 0 and int(indices64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(indices64.max().item())} exceeds Triton int32 kernel range"
        )
    if indptr64.numel() > 0:
        if int(indptr64[0].item()) != 0:
            raise ValueError("indptr[0] must be 0")
        if int(indptr64[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if bool(torch.any(indptr64[1:] < indptr64[:-1]).item()):
            raise ValueError("indptr must be non-decreasing")
    if indices64.numel() > 0:
        if bool(torch.any(indices64 < 0).item()):
            raise IndexError("indices must be non-negative")
        max_idx = int(indices64.max().item())
        if max_idx >= n_cols:
            raise IndexError(f"indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        indices.dtype,
        indices64,
        indptr64,
        b.contiguous(),
        n_rows,
        n_cols,
    )


def _prepare_spsv_sell_matrix_inputs(
    values,
    col_indices,
    slice_offsets,
    shape,
    slice_size,
    *,
    unit_diagonal=False,
    skip_validated_checks=False,
):
    """Validate the reusable sparse-matrix part of a SELL SpSV problem."""

    tensors = (values, col_indices, slice_offsets)
    if not all(torch.is_tensor(t) for t in tensors):
        raise TypeError("SELL SpSV matrix inputs must be torch.Tensor")
    if any(not t.is_cuda or t.ndim != 1 for t in tensors):
        raise ValueError("SELL SpSV matrix inputs must be 1D CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("SELL SpSV matrix inputs must use one CUDA device")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError("SELL SpSV requires a square matrix")
    slice_size = int(slice_size)
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    n_slices = (n_rows + slice_size - 1) // slice_size
    if slice_offsets.numel() != n_slices + 1:
        raise ValueError("invalid slice_offsets length")
    if values.numel() != col_indices.numel():
        raise ValueError("invalid SELL values or columns length")
    if values.dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "SELL values must be float32, float64, complex64, or complex128"
        )
    if (
        col_indices.dtype not in SUPPORTED_SPSV_INDEX_DTYPES
        or slice_offsets.dtype != col_indices.dtype
    ):
        raise TypeError("SELL columns and offsets must share int32 or int64 dtype")
    offsets = slice_offsets.contiguous()
    cols = col_indices.contiguous()
    if skip_validated_checks:
        return _as_strided_contiguous(values), cols, offsets, n_rows, slice_size

    slice_lengths = offsets[1:] - offsets[:-1]
    if (
        int(offsets[0].item()) != 0
        or int(offsets[-1].item()) != values.numel()
        or bool(torch.any(slice_lengths < 0).item())
        or bool(torch.any(slice_lengths % slice_size != 0).item())
    ):
        raise ValueError("invalid SELL slice offsets")
    if cols.numel() > 0:
        if bool(torch.any(cols < -1).item()):
            raise IndexError("SELL padding must use column index -1")
        if bool(torch.any(cols >= n_cols).item()):
            raise IndexError("SELL column index is out of range")
    _validate_spsv_sell_structure(
        cols,
        offsets,
        n_rows,
        slice_size,
        unit_diagonal=bool(unit_diagonal),
    )

    return _as_strided_contiguous(values), cols, offsets, n_rows, slice_size


def _validate_spsv_sell_structure(
    cols,
    offsets,
    n_rows,
    slice_size,
    *,
    unit_diagonal,
):
    """Reject malformed lower SELL data before a dependency kernel runs."""

    if n_rows == 0:
        return

    slice_lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)
    n_entries = int(cols.numel())
    slice_ids = torch.repeat_interleave(
        torch.arange(offsets.numel() - 1, device=cols.device, dtype=torch.int64),
        slice_lengths,
        output_size=n_entries,
    )
    slice_starts = torch.repeat_interleave(
        offsets[:-1].to(torch.int64),
        slice_lengths,
        output_size=n_entries,
    )
    entry_offsets = torch.arange(
        n_entries, device=cols.device, dtype=torch.int64
    )
    local_offsets = entry_offsets - slice_starts
    rows = slice_ids * slice_size + local_offsets.remainder(slice_size)
    slots = local_offsets // slice_size
    row_mask = rows < n_rows
    rows = rows[row_mask]
    slots = slots[row_mask]
    row_cols = cols.to(torch.int64)[row_mask]

    # ALG2 completes a row at its first padding entry.  Enforcing trailing
    # padding keeps that behavior identical to ALG1 for every accepted matrix.
    padding = row_cols == -1
    sentinel = n_entries + 1
    first_padding = torch.full(
        (n_rows,), sentinel, dtype=torch.int64, device=cols.device
    )
    first_padding.scatter_reduce_(
        0,
        rows,
        torch.where(padding, slots, torch.full_like(slots, sentinel)),
        reduce="amin",
        include_self=True,
    )
    last_value = torch.full(
        (n_rows,), -1, dtype=torch.int64, device=cols.device
    )
    last_value.scatter_reduce_(
        0,
        rows,
        torch.where(~padding, slots, torch.full_like(slots, -1)),
        reduce="amax",
        include_self=True,
    )
    if bool(torch.any(last_value > first_padding).item()):
        raise ValueError("SELL padding (-1) must be trailing within each row")

    if not unit_diagonal:
        diagonal = row_cols == rows
        diagonal_count = torch.zeros(
            n_rows, dtype=torch.int32, device=cols.device
        )
        diagonal_count.scatter_add_(0, rows, diagonal.to(torch.int32))
        if bool(torch.any(diagonal_count != 1).item()):
            raise ValueError(
                "NON_UNIT SELL SpSV requires exactly one diagonal entry per row"
            )


def _spsv_diag_eps_for_dtype(value_dtype):
    return 1e-12 if value_dtype in (torch.float64, torch.complex128) else 1e-6


def _tensor_cache_token(tensor):
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        int(tensor.numel()),
        storage_ptr,
        int(getattr(tensor, "_version", 0)),
    )


def _spsv_sell_trans_analysis_signature(
    values,
    col_indices,
    slice_offsets,
    shape,
    slice_size,
    unit_diagonal,
):
    if not all(torch.is_tensor(t) for t in (values, col_indices, slice_offsets)):
        return None
    return (
        "sell_trans",
        tuple(int(v) for v in shape),
        int(slice_size),
        bool(unit_diagonal),
        str(values.device),
        str(values.dtype),
        int(values.numel()),
        _tensor_cache_token(col_indices),
        _tensor_cache_token(slice_offsets),
    )


def _spsv_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsv_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _normalize_spsv_format(fmt):
    token = str(fmt).strip().lower()
    if token not in ("csr", "coo"):
        raise ValueError("format must be 'csr' or 'coo'")
    return token


def _normalize_spsv_storage_view(storage_view):
    if storage_view is None:
        return "csr_as_csc"
    token = str(storage_view).strip().lower()
    aliases = {
        "csr_as_csc": "csr_as_csc",
        "csc_view": "csr_as_csc",
        "reuse_csr_storage": "csr_as_csc",
    }
    if token not in aliases:
        raise ValueError(
            "storage_view must be one of: csr_as_csc, csc_view, reuse_csr_storage"
        )
    return aliases[token]


def _resolve_spsv_stream(handle, stream, device):
    resolved = stream
    if handle is not None:
        if not isinstance(handle, FlagSparseSpSVHandle):
            raise TypeError("handle must be a FlagSparseSpSVHandle or None")
        if torch.device(handle.device) != torch.device(device):
            raise ValueError("handle device must match the solve device")
        if resolved is None:
            resolved = handle.stream
    return resolved


def _coerce_spsv_alpha(alpha, dtype, device):
    if torch.is_tensor(alpha):
        alpha_tensor = alpha.to(device=device, dtype=dtype).reshape(-1)
        if alpha_tensor.numel() != 1:
            raise ValueError("alpha must be a scalar tensor")
        return alpha_tensor.reshape(())
    return torch.tensor(alpha, device=device, dtype=dtype)


def _spsv_alpha_is_identity(alpha):
    if torch.is_tensor(alpha):
        alpha_flat = alpha.reshape(-1)
        if alpha_flat.numel() != 1:
            raise ValueError("alpha must be a scalar tensor")
        return bool((alpha_flat == 1).all().item())
    return alpha == 1


def _workspace_entry(name, numel, dtype):
    return {
        "name": str(name),
        "numel": int(numel),
        "dtype": dtype,
        "bytes": int(numel) * int(torch.empty((), dtype=dtype).element_size()),
    }


def _workspace_size_bytes(layout):
    return int(sum(int(entry["bytes"]) for entry in layout))


def _spsv_effective_compute_dtype(value_dtype, trans_mode, compute_dtype=None):
    if compute_dtype is not None:
        if compute_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
            raise TypeError(
                "compute_dtype must be one of: float32, float64, complex64, complex128"
            )
        return compute_dtype
    if (
        value_dtype == torch.complex64
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128
    ):
        return torch.complex128
    if value_dtype == torch.float32 and SPSV_PROMOTE_FP32_TO_FP64:
        return torch.float64
    if (
        value_dtype == torch.float32
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64
    ):
        return torch.float64
    return value_dtype


def _spsv_sell_trans_residual_dtype(value_dtype):
    """Return the internal residual dtype without changing user I/O dtype."""

    if not SPSV_SELL_TRANS_FP64_RESIDUAL:
        return value_dtype
    if value_dtype == torch.float32:
        return torch.float64
    if value_dtype == torch.complex64:
        return torch.complex128
    return value_dtype


def _build_spsv_workspace_layout(n_rows, solve_kind, value_dtype=None):
    n_rows = int(n_rows)
    if solve_kind in ("csr_cw", "sell_alg1", "sell_alg2"):
        return (
            _workspace_entry("ready", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    if solve_kind == "sell_trans":
        if value_dtype is None:
            raise ValueError("value_dtype is required for SELL TRANS workspace sizing")
        return (
            _workspace_entry(
                "residual",
                n_rows,
                _spsv_sell_trans_residual_dtype(value_dtype),
            ),
            _workspace_entry("indegree", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    if solve_kind == "csr_roc":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
        )
    if solve_kind == "csr_smblk":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
        )
    if solve_kind == "csr_cw_levelschd":
        return (
            _workspace_entry("ready", n_rows, torch.int32),
        )
    if solve_kind == "csr_nnz_balance":
        if value_dtype is None:
            raise ValueError("value_dtype is required for csr_nnz_balance workspace sizing")
        return (
            _workspace_entry("tmp_sum", n_rows, value_dtype),
            _workspace_entry("ready", n_rows, torch.int32),
            _workspace_entry("indegree", n_rows, torch.int32),
        )
    if solve_kind == "transpose_cw":
        if value_dtype is None:
            raise ValueError("value_dtype is required for transpose_cw workspace sizing")
        return (
            _workspace_entry("residual", n_rows, value_dtype),
            _workspace_entry("indegree", n_rows, torch.int32),
            _workspace_entry("row_counter", 1, torch.int32),
        )
    raise ValueError(f"Unsupported SpSV solve kind for workspace sizing: {solve_kind}")


def _clone_spsv_plan(plan):
    cloned = dict(plan)
    matrix_stats = plan.get("matrix_stats")
    if matrix_stats is not None:
        cloned["matrix_stats"] = dict(matrix_stats)
    return cloned


def _alloc_spsv_workspace_buffers(layout, device):
    buffers = {}
    for entry in layout:
        buffers[entry["name"]] = torch.empty(
            int(entry["numel"]), dtype=entry["dtype"], device=device
        )
    return buffers


def _resolve_spsv_workspace(workspace, layout, device):
    if workspace is None:
        return _alloc_spsv_workspace_buffers(layout, device)
    if not isinstance(workspace, FlagSparseSpSVWorkspace):
        raise TypeError("workspace must be a FlagSparseSpSVWorkspace or None")
    if torch.device(workspace.device) != torch.device(device):
        raise ValueError("workspace device must match the solve device")
    if int(workspace.buffer_size) < _workspace_size_bytes(layout):
        raise ValueError("workspace buffer is smaller than the required SpSV size")
    required = {entry["name"]: entry for entry in layout}
    for name, entry in required.items():
        buf = workspace.buffers.get(name)
        if buf is None:
            workspace.buffers[name] = torch.empty(
                int(entry["numel"]), dtype=entry["dtype"], device=device
            )
            continue
        if buf.device != torch.device(device):
            raise ValueError(f"workspace buffer {name!r} is on the wrong device")
        if buf.dtype != entry["dtype"] or int(buf.numel()) < int(entry["numel"]):
            workspace.buffers[name] = torch.empty(
                int(entry["numel"]), dtype=entry["dtype"], device=device
            )
    return workspace.buffers


def _transpose_cw_preprocess_signature(
    solve_plan, n_rows, unit_diagonal, block_nnz_use, max_segments_use
):
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    return (
        "transpose_cw",
        int(n_rows),
        bool(solve_plan["lower_eff"]),
        bool(unit_diagonal),
        int(block_nnz_use),
        int(max_segments_use),
        _tensor_cache_token(kernel_indices32),
        _tensor_cache_token(kernel_indptr64),
    )


def flagsparse_spsv_buffer_size(
    shape,
    value_dtype,
    *,
    format="csr",
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    alpha=None,
    handle=None,
    vecX=None,
    vecY=None,
    storage_view="csr_as_csc",
):
    """Return the caller-managed workspace size for the current Triton SpSV route.

    This is the Triton host-side equivalent of the CUDA bufferSize query.
    The returned byte count matches the scratch buffers used by the current
    Triton implementation, rather than the raw CUDA ABI layout.
    """

    _normalize_spsv_format(format)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSV expects a square matrix, got shape={shape}")
    if value_dtype not in SUPPORTED_SPSV_VALUE_DTYPES:
        raise TypeError(
            "value_dtype must be one of: float32, float64, complex64, complex128"
        )
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    storage_view = _normalize_spsv_storage_view(storage_view)
    compute_dtype = _spsv_effective_compute_dtype(
        value_dtype, trans_mode, compute_dtype=compute_dtype
    )
    route = _normalize_requested_spsv_route(solve_kind, trans_mode)
    if route is None:
        route = "transpose_cw" if trans_mode in ("T", "C") else "csr_cw"
    if trans_mode in ("T", "C") and storage_view != "csr_as_csc":
        raise ValueError("TRANS/CONJ SpSV only supports storage_view='csr_as_csc'")
    layout = _build_spsv_workspace_layout(n_rows, route, value_dtype=compute_dtype)
    return _workspace_size_bytes(layout)


def flagsparse_spsv_create_workspace(descr, device=None):
    """Allocate a caller-owned SpSV workspace object from an analysis descriptor."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if device is None:
        device = descr.data.device
    device = torch.device(device)
    buffers = _alloc_spsv_workspace_buffers(descr.workspace_layout, device)
    return FlagSparseSpSVWorkspace(
        buffer_size=int(descr.buffer_size),
        layout=tuple(descr.workspace_layout),
        device=device,
        buffers=buffers,
    )


def _csr_preprocess_cache_key(
    data, indices, indptr, shape, lower, trans_mode, unit_diagonal, requested_route=None, storage_view="csr_as_csc"
):
    return (
        "csr_preprocess",
        trans_mode,
        bool(lower),
        bool(unit_diagonal),
        str(requested_route),
        str(storage_view),
        int(shape[0]),
        int(shape[1]),
        _tensor_cache_token(data),
        _tensor_cache_token(indices),
        _tensor_cache_token(indptr),
    )


def _normalize_requested_spsv_route(solve_kind, trans_mode):
    if solve_kind is None:
        return None
    token = str(solve_kind).strip().lower()
    aliases = {
        "csr_cw": "csr_cw",
        "csr_roc": "csr_roc",
        "roc": "csr_roc",
        "alg3": "csr_roc",
        "csr_smblk": "csr_smblk",
        "smblk": "csr_smblk",
        "alg4": "csr_smblk",
        "csr_levelschd": "csr_cw_levelschd",
        "csr_cw_levelschd": "csr_cw_levelschd",
        "levelschd": "csr_cw_levelschd",
        "level_sched": "csr_cw_levelschd",
        "alg2": "csr_cw_levelschd",
        "csr_nnz_balance": "csr_nnz_balance",
        "nnz_balance": "csr_nnz_balance",
        "alg8": "csr_nnz_balance",
        "cw": "csr_cw" if trans_mode == "N" else "transpose_cw",
        "transpose_cw": "transpose_cw",
        "csc_cw": "transpose_cw",
    }
    route = aliases.get(token)
    if route is None:
        raise ValueError(
            "solve_kind must be one of: csr_cw, csr_roc, csr_smblk, csr_cw_levelschd, csr_nnz_balance, transpose_cw"
        )
    if trans_mode in ("T", "C") and route != "transpose_cw":
        raise ValueError("TRANS/CONJ SpSV only supports solve_kind='transpose_cw'")
    if trans_mode == "N" and route == "transpose_cw":
        raise ValueError("NON_TRANS SpSV cannot use solve_kind='transpose_cw'")
    return route


@triton.jit
def _spsv_csc_preprocess_kernel(
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_rows:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        row = tl.load(indices_ptr + offsets, mask=mask, other=0)
        if LOWER:
            dep_mask = mask & (row > col if UNIT_DIAG else row >= col)
        else:
            dep_mask = mask & (row < col if UNIT_DIAG else row <= col)
        tl.atomic_add(indegree_ptr + row, 1, mask=dep_mask)


def _sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=True):
    if data.numel() == 0:
        return data, indices64, indptr64
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    key = row_ids * max(1, n_cols)
    if lower:
        key = key + indices64
    else:
        key = key + (n_cols - 1 - indices64)
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order], indices64[order], indptr64


def _spsv_csr_row_length_summary(indptr64, n_rows):
    if indptr64.numel() <= 1 or int(n_rows) <= 0:
        return torch.empty(0, dtype=torch.int64, device=indptr64.device), 0.0, 0
    row_lengths = (indptr64[1:] - indptr64[:-1]).to(torch.int64)
    avg_nnz_per_row = float(row_lengths.to(torch.float32).mean().item())
    max_nnz_per_row = int(row_lengths.max().item()) if row_lengths.numel() > 0 else 0
    return row_lengths, avg_nnz_per_row, max_nnz_per_row


def _csr_rows_are_sorted(indices64, indptr64, n_rows, lower=True):
    if indices64.numel() <= 1 or int(n_rows) <= 0:
        return True
    row_lengths, _, max_nnz_per_row = _spsv_csr_row_length_summary(indptr64, n_rows)
    if row_lengths.numel() == 0 or max_nnz_per_row <= 1:
        return True
    same_row = torch.ones(
        indices64.numel() - 1,
        dtype=torch.bool,
        device=indices64.device,
    )
    row_ends = indptr64[1:-1].to(torch.int64) - 1
    row_ends = row_ends[(row_ends >= 0) & (row_ends < same_row.numel())]
    if row_ends.numel() > 0:
        same_row[row_ends] = False
    if not bool(torch.any(same_row).item()):
        return True
    if lower:
        ordered = indices64[1:] >= indices64[:-1]
    else:
        ordered = indices64[1:] <= indices64[:-1]
    return bool(torch.all(ordered | (~same_row)).item())


def _maybe_sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=True):
    if _csr_rows_are_sorted(indices64, indptr64, n_rows, lower=lower):
        return data, indices64, indptr64
    return _sort_csr_rows(data, indices64, indptr64, n_rows, n_cols, lower=lower)


def _cw_rhs_bucket(n_rhs):
    if n_rhs <= 1:
        return 1
    if n_rhs <= 2:
        return 2
    if n_rhs <= 4:
        return 4
    if n_rhs <= 8:
        return 8
    if n_rhs <= 16:
        return 16
    return 32


def _snap_cw_worker_count(target, n_rows):
    if n_rows <= 0:
        return 1
    target = max(1, min(int(target), int(n_rows)))
    snapped = 1
    tier = 1
    while tier < target and tier < 4096:
        tier *= 2
        if tier <= target:
            snapped = tier
    return int(max(1, min(snapped, int(n_rows))))


def _cw_worker_count(n_rows, max_frontier, avg_nnz_per_row, n_rhs):
    if n_rows <= 0:
        return 1
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    if rhs_bucket == 1:
        target = min(n_rows, 32)
    else:
        target = max(32, min(n_rows, 512))
    if max_frontier > 0:
        target = min(target, max(4, min(n_rows, max_frontier * 2)))
    if avg_nnz_per_row > 8192:
        target = max(4, target // 8)
    elif avg_nnz_per_row > 4096:
        target = max(4, target // 4)
    elif avg_nnz_per_row > 2048:
        target = max(4, target // 2)
    elif avg_nnz_per_row > 1024:
        target = max(8, (target * 2) // 3)
    elif avg_nnz_per_row > 512:
        target = max(8, (target * 3) // 5)
    if rhs_bucket >= 16:
        target = max(4, target // 4)
    elif rhs_bucket >= 8:
        target = max(4, target // 2)
    elif rhs_bucket >= 4:
        target = max(4, (target * 3) // 4)
    return _snap_cw_worker_count(target, n_rows)

def _resolve_cw_worker_count(n_rows, matrix_stats, n_rhs, cached_worker_count=None):
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    max_frontier = int(matrix_stats.get("max_frontier", n_rows))
    avg_frontier = float(matrix_stats.get("avg_frontier", float(max_frontier)))
    frontier_ratio = float(matrix_stats.get("frontier_ratio", 1.0 if n_rows > 0 else 0.0))
    num_levels = int(matrix_stats.get("num_levels", 0))
    avg_nnz_per_row = float(matrix_stats.get("avg_nnz_per_row", 0.0))
    if cached_worker_count is not None and rhs_bucket == 1:
        target = int(max(1, min(int(cached_worker_count), int(max(n_rows, 1)))))
    else:
        target = _cw_worker_count(
            n_rows,
            max_frontier,
            avg_nnz_per_row,
            rhs_bucket,
        )
    if frontier_ratio < 0.01 or avg_frontier < 4.0:
        target = min(target, max(1, min(n_rows, 4)))
    elif frontier_ratio < 0.02 or avg_frontier < 8.0:
        target = min(target, max(1, min(n_rows, 8)))
    elif frontier_ratio < 0.05 or avg_frontier < 16.0:
        target = min(target, max(2, min(n_rows, 16)))
    if num_levels > max(1024, n_rows // 2):
        target = max(1, target // 2)
    if avg_nnz_per_row > 2048:
        target = max(1, target // 2)
    return _snap_cw_worker_count(
        target,
        n_rows,
    )

@triton.jit
def _spsv_csr_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    BLOCK_RHS: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    row = tl.atomic_add(row_counter_ptr, 1)
    while row < n_rows:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        diag = tl.load(diag_ptr + row)
        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)

        for rhs_base in range(0, n_rhs, BLOCK_RHS):
            rhs_offsets = rhs_base + tl.arange(0, BLOCK_RHS)
            rhs_mask = rhs_offsets < n_rhs
            acc = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                nnz_offsets = idx + tl.arange(0, BLOCK_NNZ)
                nnz_mask = nnz_offsets < end
                a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
                col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)
                if LOWER:
                    dep_mask = nnz_mask & (col < row)
                else:
                    dep_mask = nnz_mask & (col > row)

                for k in range(BLOCK_NNZ):
                    if dep_mask[k]:
                        dep_col = col[k]
                        while tl.load(ready_ptr + dep_col) == 0:
                            pass
                        x_ptrs = x_ptr + dep_col * stride_x0 + rhs_offsets
                        x_vals = tl.load(x_ptrs, mask=rhs_mask, other=0.0)
                        acc += a[k] * x_vals

            rhs_ptrs = b_ptr + row * stride_b0 + rhs_offsets
            rhs = tl.load(rhs_ptrs, mask=rhs_mask, other=0.0)
            x_row = (rhs - acc) / diag_safe
            x_row = tl.where(x_row == x_row, x_row, 0.0)
            out_ptrs = x_ptr + row * stride_x0 + rhs_offsets
            tl.store(out_ptrs, x_row, mask=rhs_mask)

        tl.debug_barrier()
        tl.store(ready_ptr + row, 1)
        row = tl.atomic_add(row_counter_ptr, 1)

def _build_spsv_cw_matrix_stats(
    indptr64,
    n_rows,
    *,
    avg_nnz_per_row=None,
    max_nnz_per_row=None,
):
    if avg_nnz_per_row is None or max_nnz_per_row is None:
        _, avg_nnz_per_row, max_nnz_per_row = _spsv_csr_row_length_summary(indptr64, n_rows)
    return {
        "num_levels": 0,
        "max_frontier": int(n_rows),
        "avg_frontier": float(n_rows),
        "frontier_ratio": 1.0 if n_rows > 0 else 0.0,
        "avg_nnz_per_row": avg_nnz_per_row,
        "max_nnz_per_row": max_nnz_per_row,
        "n_rows": int(n_rows),
    }


def _build_spsv_nnz_balance_launch_order(indptr64, n_rows, *, lower):
    n_rows = int(n_rows)
    device = indptr64.device
    total_nnz = int(indptr64[-1].item()) if indptr64.numel() > 0 else 0
    if total_nnz <= 0 or n_rows <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if lower:
        return torch.arange(total_nnz, dtype=torch.int32, device=device)

    indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
    launch_order = []
    for row in range(n_rows - 1, -1, -1):
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        launch_order.extend(range(start, end))
    return torch.tensor(launch_order, dtype=torch.int32, device=device)


def _supports_spsv_advanced_nontrans_routes(trans_mode, lower, unit_diagonal, value_dtype):
    return (
        trans_mode == "N"
        and (not bool(unit_diagonal))
        and value_dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
    )


def _choose_spsv_nontrans_auto_route(
    n_rows,
    matrix_stats,
    *,
    lower,
    unit_diagonal,
    value_dtype,
):
    """Heuristic route picker for NON_TRANS triangular solves.

    This is an analysis-time auto selector, not a runtime autotuner.  The goal
    is to keep default routing predictable while still steering obviously
    serialized systems and wide-frontier systems onto more suitable kernels.
    """
    if bool(unit_diagonal):
        return "csr_cw"
    n_rows = int(n_rows)
    if n_rows <= 0:
        return "csr_cw"

    # Upper NON sweeps consistently favor ALG4 (csr_smblk), so keep that as the
    # unconditional AUTO route on the upper side.
    if not lower:
        if value_dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            return "csr_smblk"
        return "csr_cw"

    # Keep lower NON AUTO predictable as well: default to ALG4 (csr_smblk) for
    # supported dtypes instead of sending many SuiteSparse-derived cases through
    # heuristic detours such as level scheduling.
    if value_dtype in (torch.float32, torch.float64):
        return "csr_smblk"

    if value_dtype in (torch.complex64, torch.complex128):
        return "csr_smblk"
    return "csr_cw"


@triton.jit
def _spsv_levelschd_analysis_kernel(
    indices_ptr,
    indptr_ptr,
    levels_ptr,
    ready_ptr,
    indegree_ptr,
    n_rows,
    BLOCK_ROWS: tl.constexpr,
    UNIT_DIAGONAL: tl.constexpr,
):
    first_row = tl.program_id(0) * BLOCK_ROWS
    local_rows = tl.arange(0, BLOCK_ROWS)
    local_levels = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    for local_row in range(BLOCK_ROWS):
        row = first_row + local_row
        if row < n_rows:
            start = tl.load(indptr_ptr + row)
            end = tl.load(indptr_ptr + row + 1)
            ptr = start
            max_level = tl.zeros((), dtype=tl.int32)
            degree = tl.zeros((), dtype=tl.int32)
            row_done = 0
            while row_done == 0:
                if ptr >= end:
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    if col < first_row:
                        dep_ready = _load_ready_flag_i32(ready_ptr, col)
                        while dep_ready == 0:
                            dep_ready = _load_ready_flag_i32(ready_ptr, col)
                        dep_level = tl.atomic_add(levels_ptr + col, 0)
                        max_level = tl.maximum(max_level, dep_level)
                        degree += 1
                        ptr += 1
                    elif col < row:
                        local_idx = col - first_row
                        dep_level = tl.sum(
                            tl.where(local_rows == local_idx, local_levels, 0),
                            axis=0,
                        )
                        max_level = tl.maximum(max_level, dep_level)
                        degree += 1
                        ptr += 1
                    else:
                        if (not UNIT_DIAGONAL) and (col == row):
                            degree += 1
                        row_done = 1
            row_level = max_level + 1
            _publish_i32_once(levels_ptr, row, row_level)
            tl.store(indegree_ptr + row, degree)
            local_levels = tl.where(local_rows == local_row, row_level, local_levels)
            _publish_ready_flag_i32(ready_ptr, row)


def _build_spsv_level_schedule_metadata_lower_gpu(
    indices64, indptr64, n_rows, *, unit_diagonal, minimal=False
):
    n_rows = int(n_rows)
    device = indices64.device
    base_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    empty_meta = {
        "row_map32": torch.empty(0, dtype=torch.int32, device=device),
        "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
        "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
        "matrix_stats": {
            **base_stats,
            "num_levels": 0,
            "max_frontier": 0,
            "avg_frontier": 0.0,
            "frontier_ratio": 0.0,
        },
    }
    if n_rows == 0:
        return empty_meta

    indices32 = indices64.to(torch.int32).contiguous()
    levels32 = torch.zeros(n_rows, dtype=torch.int32, device=device)
    ready32 = torch.zeros(n_rows, dtype=torch.int32, device=device)
    indegree32 = torch.empty(n_rows, dtype=torch.int32, device=device)
    _spsv_levelschd_analysis_kernel[(triton.cdiv(n_rows, 8),)](
        indices32,
        indptr64,
        levels32,
        ready32,
        indegree32,
        n_rows,
        BLOCK_ROWS=8,
        UNIT_DIAGONAL=bool(unit_diagonal),
        num_warps=1,
    )

    # Stable GPU sort reproduces the row_map stage after roc-style level analysis.

    try:
        row_map64 = torch.argsort(levels32.to(torch.int64), stable=True)
    except TypeError:
        row_map64 = torch.argsort(levels32.to(torch.int64))
    row_map32 = row_map64.to(torch.int32).contiguous()
    sorted_levels32 = levels32.index_select(0, row_map64)
    if sorted_levels32.numel() > 0:
        _, frontier_counts64 = torch.unique_consecutive(sorted_levels32, return_counts=True)
        level_ptr32 = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(frontier_counts64.to(torch.int32), dim=0),
            ]
        )
        num_levels = int(frontier_counts64.numel())
        max_frontier = int(frontier_counts64.max().item())
        avg_frontier = float(frontier_counts64.to(torch.float32).mean().item())
    else:
        level_ptr32 = torch.zeros(1, dtype=torch.int32, device=device)
        num_levels = 0
        max_frontier = 0
        avg_frontier = 0.0

    row_lengths64 = indptr64[1:] - indptr64[:-1]
    csr_row_idx32 = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int32),
        row_lengths64.to(torch.int64),
    ).contiguous()
    matrix_stats = {
        **base_stats,
        "num_levels": int(num_levels),
        "max_frontier": int(max_frontier),
        "avg_frontier": float(avg_frontier),
        "frontier_ratio": (float(max_frontier) / float(n_rows)) if n_rows > 0 else 0.0,
    }
    if minimal:
        return {
            "row_map32": row_map32,
            "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
            "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
            "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
            "matrix_stats": matrix_stats,
        }
    return {
        "row_map32": row_map32,
        "level_ptr32": level_ptr32,
        "indegree_init32": indegree32,
        "csr_row_idx32": csr_row_idx32,
        "matrix_stats": matrix_stats,
    }


@triton.jit
def _spsv_nnz_balance_preprocess_kernel(
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    row_idx_ptr,
    n_rows,
    WARP_SIZE: tl.constexpr,
    UNIT_DIAGONAL: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lane = tl.arange(0, WARP_SIZE)
    ptr = start + lane
    degree = tl.zeros((WARP_SIZE,), dtype=tl.int32)
    active = ptr < end
    while tl.sum(active.to(tl.int32), axis=0) > 0:
        cols = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
        if UNIT_DIAGONAL:
            valid = active & (cols < row)
        else:
            valid = active & (cols <= row)
        tl.store(row_idx_ptr + ptr, row, mask=valid)
        degree += valid.to(tl.int32)
        ptr = ptr + WARP_SIZE
        active = valid & (ptr < end)
    tl.store(indegree_ptr + row, tl.sum(degree, axis=0))


def _build_spsv_nnz_balance_metadata(indices64, indptr64, n_rows, *, lower, unit_diagonal):
    n_rows = int(n_rows)
    device = indices64.device
    base_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    empty_meta = {
        "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
        "launch_order32": torch.empty(0, dtype=torch.int32, device=device),
        "matrix_stats": base_stats,
    }
    if n_rows == 0:
        return empty_meta
    if indices64.is_cuda:
        if not lower:
            # Upper-triangular preprocessing reuses the generic host path for now.
            indices_cpu = indices64.to("cpu", non_blocking=False).tolist()
            indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
            indegree_init = [0] * n_rows
            row_idx = [0] * int(indices64.numel())
            for row in range(n_rows):
                start = int(indptr_cpu[row])
                end = int(indptr_cpu[row + 1])
                degree = 0
                for ptr in range(start, end):
                    col = int(indices_cpu[ptr])
                    if col > row:
                        row_idx[ptr] = row
                        degree += 1
                        continue
                    if (not unit_diagonal) and col == row:
                        row_idx[ptr] = row
                        degree += 1
                    break
                indegree_init[row] = degree
            return {
                "indegree_init32": torch.tensor(indegree_init, dtype=torch.int32, device=device),
                "csr_row_idx32": torch.tensor(row_idx, dtype=torch.int32, device=device),
                "launch_order32": _build_spsv_nnz_balance_launch_order(
                    indptr64, n_rows, lower=lower
                ),
                "matrix_stats": base_stats,
            }
        indices32 = indices64.to(torch.int32).contiguous()
        indegree32 = torch.zeros(n_rows, dtype=torch.int32, device=device)
        row_idx32 = torch.zeros(indices32.numel(), dtype=torch.int32, device=device)
        _spsv_nnz_balance_preprocess_kernel[(n_rows,)](
            indices32,
            indptr64,
            indegree32,
            row_idx32,
            n_rows,
            WARP_SIZE=32,
            UNIT_DIAGONAL=bool(unit_diagonal),
            num_warps=1,
        )
        return {
            "indegree_init32": indegree32,
            "csr_row_idx32": row_idx32,
            "launch_order32": _build_spsv_nnz_balance_launch_order(
                indptr64, n_rows, lower=lower
            ),
            "matrix_stats": base_stats,
        }

    indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
    indices_cpu = indices64.to("cpu", non_blocking=False).tolist()
    indegree_init = [0] * n_rows
    row_idx = [0] * int(indices64.numel())
    for row in range(n_rows):
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        degree = 0
        for ptr in range(start, end):
            col = int(indices_cpu[ptr])
            if lower:
                if col < row:
                    row_idx[ptr] = row
                    degree += 1
                    continue
                if (not unit_diagonal) and col == row:
                    row_idx[ptr] = row
                    degree += 1
                break
            else:
                if col > row:
                    row_idx[ptr] = row
                    degree += 1
                    continue
                if (not unit_diagonal) and col == row:
                    row_idx[ptr] = row
                    degree += 1
                break
        indegree_init[row] = degree
    return {
        "indegree_init32": torch.tensor(indegree_init, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.tensor(row_idx, dtype=torch.int32, device=device),
        "launch_order32": _build_spsv_nnz_balance_launch_order(
            indptr64, n_rows, lower=lower
        ),
        "matrix_stats": base_stats,
    }


def _build_spsv_level_schedule_metadata(
    indices64, indptr64, n_rows, *, lower, unit_diagonal, minimal=False
):
    n_rows = int(n_rows)
    device = indices64.device
    base_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    empty_meta = {
        "row_map32": torch.empty(0, dtype=torch.int32, device=device),
        "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
        "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
        "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
        "matrix_stats": {
            **base_stats,
            "num_levels": 0,
            "max_frontier": 0,
            "avg_frontier": 0.0,
            "frontier_ratio": 0.0,
        },
    }
    if n_rows == 0:
        return empty_meta

    if lower and indices64.is_cuda:
        return _build_spsv_level_schedule_metadata_lower_gpu(
            indices64,
            indptr64,
            n_rows,
            unit_diagonal=unit_diagonal,
            minimal=minimal,
        )

    indptr_cpu = indptr64.to("cpu", non_blocking=False).tolist()
    indices_cpu = indices64.to("cpu", non_blocking=False).tolist()
    levels = [0] * n_rows
    indegree_init = [0] * n_rows
    level_buckets = {}

    row_iter = range(n_rows) if lower else range(n_rows - 1, -1, -1)
    for row in row_iter:
        start = int(indptr_cpu[row])
        end = int(indptr_cpu[row + 1])
        deps = []
        degree = 0
        for ptr in range(start, end):
            col = int(indices_cpu[ptr])
            if lower:
                if unit_diagonal:
                    if col < row:
                        deps.append(col)
                        degree += 1
                    else:
                        break
                else:
                    if col < row:
                        deps.append(col)
                        degree += 1
                        continue
                    if col == row:
                        degree += 1
                    break
            else:
                if unit_diagonal:
                    if col > row:
                        deps.append(col)
                        degree += 1
                        continue
                    break
                else:
                    if col > row:
                        deps.append(col)
                        degree += 1
                        continue
                    if col == row:
                        degree += 1
                    break
        indegree_init[row] = degree
        row_level = 1
        if deps:
            row_level = max(levels[col] for col in deps) + 1
        levels[row] = row_level
        level_buckets.setdefault(row_level, []).append(row)

    row_map = []
    level_ptr = [0]
    frontier_sizes = []
    for level_id in sorted(level_buckets):
        rows = level_buckets[level_id]
        frontier_sizes.append(len(rows))
        row_map.extend(rows)
        level_ptr.append(len(row_map))

    row_lengths64 = indptr64[1:] - indptr64[:-1]
    csr_row_idx32 = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int32),
        row_lengths64.to(torch.int64),
    ).contiguous()
    num_levels = len(frontier_sizes)
    max_frontier = max(frontier_sizes) if frontier_sizes else 0
    avg_frontier = (float(sum(frontier_sizes)) / float(num_levels)) if frontier_sizes else 0.0
    matrix_stats = {
        **base_stats,
        "num_levels": int(num_levels),
        "max_frontier": int(max_frontier),
        "avg_frontier": float(avg_frontier),
        "frontier_ratio": (float(max_frontier) / float(n_rows)) if n_rows > 0 else 0.0,
    }
    if minimal:
        return {
            "row_map32": torch.tensor(row_map, dtype=torch.int32, device=device),
            "level_ptr32": torch.zeros(1, dtype=torch.int32, device=device),
            "indegree_init32": torch.empty(0, dtype=torch.int32, device=device),
            "csr_row_idx32": torch.empty(0, dtype=torch.int32, device=device),
            "matrix_stats": matrix_stats,
        }
    return {
        "row_map32": torch.tensor(row_map, dtype=torch.int32, device=device),
        "level_ptr32": torch.tensor(level_ptr, dtype=torch.int32, device=device),
        "indegree_init32": torch.tensor(indegree_init, dtype=torch.int32, device=device),
        "csr_row_idx32": csr_row_idx32,
        "matrix_stats": matrix_stats,
    }


def _prepare_spsv_csr_system(
    data,
    indices64,
    indptr64,
    n_rows,
    n_cols,
    lower,
    trans_mode,
    unit_diagonal,
    requested_solve_kind=None,
    storage_view="csr_as_csc",
):
    if trans_mode == "N":
        data, indices64, indptr64 = _maybe_sort_csr_rows(
            data, indices64, indptr64, n_rows, n_cols, lower=lower
        )
        requested_route = _normalize_requested_spsv_route(requested_solve_kind, trans_mode)
        _, avg_nnz_per_row, max_nnz_per_row = _spsv_csr_row_length_summary(indptr64, n_rows)
        base_stats = _build_spsv_cw_matrix_stats(
            indptr64,
            n_rows,
            avg_nnz_per_row=avg_nnz_per_row,
            max_nnz_per_row=max_nnz_per_row,
        )
        default_block_nnz, default_max_segments = _auto_spsv_launch_config(
            indptr64,
            max_nnz_per_row=max_nnz_per_row,
        )
        if lower:
            nontrans_variant = "csr_u_lo_cw" if unit_diagonal else "csr_n_lo_cw"
        else:
            nontrans_variant = "csr_u_up_cw" if unit_diagonal else "csr_n_up_cw"
        level_meta = None
        nnz_meta = None
        auto_route = None
        auto_matrix_stats = base_stats
        if requested_route is None and _supports_spsv_advanced_nontrans_routes(
            "N", lower, unit_diagonal, data.dtype
        ):
            auto_route = _choose_spsv_nontrans_auto_route(
                n_rows,
                base_stats,
                lower=lower,
                unit_diagonal=unit_diagonal,
                value_dtype=data.dtype,
            )
            if auto_route in ("csr_cw_levelschd", "csr_nnz_balance"):
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
                auto_matrix_stats = level_meta["matrix_stats"]
            if auto_route == "csr_nnz_balance":
                nnz_meta = _build_spsv_nnz_balance_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )

        effective_route = requested_route if requested_route is not None else auto_route

        if effective_route == "csr_cw":
            default_solve_kind = "csr_cw"
            matrix_stats = base_stats
            supported_solve_kinds = ("csr_cw",)
        elif effective_route == "csr_roc":
            if bool(unit_diagonal):
                raise ValueError("solve_kind='csr_roc' currently supports non-unit diagonal only")
            if level_meta is None:
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    minimal=True,
                )
            elif int(level_meta["level_ptr32"].numel()) > 1 or int(level_meta["indegree_init32"].numel()) > 0:
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    minimal=True,
                )
            matrix_stats = level_meta["matrix_stats"]
            default_solve_kind = "csr_roc"
            supported_solve_kinds = ("csr_roc",)
        elif effective_route == "csr_smblk":
            if bool(unit_diagonal):
                raise ValueError("solve_kind='csr_smblk' currently supports non-unit diagonal only")
            matrix_stats = auto_matrix_stats if requested_route is None else base_stats
            default_solve_kind = "csr_smblk"
            supported_solve_kinds = ("csr_smblk",)
        elif effective_route == "csr_cw_levelschd":
            if level_meta is None:
                level_meta = _build_spsv_level_schedule_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
            matrix_stats = level_meta["matrix_stats"]
            default_solve_kind = "csr_cw_levelschd"
            supported_solve_kinds = ("csr_cw_levelschd",)
        elif effective_route == "csr_nnz_balance":
            if nnz_meta is None:
                nnz_meta = _build_spsv_nnz_balance_metadata(
                    indices64,
                    indptr64,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                )
            matrix_stats = auto_matrix_stats if requested_route is None else nnz_meta["matrix_stats"]
            default_solve_kind = "csr_nnz_balance"
            supported_solve_kinds = ("csr_nnz_balance",)
        else:
            if not _supports_spsv_advanced_nontrans_routes(
                "N", lower, unit_diagonal, data.dtype
            ):
                matrix_stats = base_stats
                default_solve_kind = "csr_cw"
                supported_solve_kinds = ("csr_cw",)
            else:
                matrix_stats = auto_matrix_stats
                default_solve_kind = auto_route if auto_route is not None else "csr_smblk"
                supported_solve_kinds = (default_solve_kind,)
        route_name = nontrans_variant
        if default_solve_kind == "csr_roc":
            route_name = "csr_n_lo_roc" if lower else "csr_n_up_roc"
        elif default_solve_kind == "csr_smblk":
            route_name = "csr_n_lo_smblk" if lower else "csr_n_up_smblk"
        elif default_solve_kind == "csr_cw_levelschd":
            route_name = "csr_n_lo_cw_levelschd" if lower else "csr_n_up_cw_levelschd"
        elif default_solve_kind == "csr_nnz_balance":
            route_name = "csr_n_lo_nnz_balance" if lower else "csr_n_up_nnz_balance"
        cw_plan = {
            "solve_kind": default_solve_kind,
            "default_solve_kind": default_solve_kind,
            "supported_solve_kinds": tuple(supported_solve_kinds),
            "nontrans_variant": nontrans_variant,
            "kernel_data": data,
            "kernel_indices32": indices64.to(torch.int32),
            "kernel_indptr64": indptr64,
            "lower_eff": lower,
            "default_block_nnz": default_block_nnz,
            "default_max_segments": default_max_segments,
            "storage_view": "csr",
            "cw_worker_count": _cw_worker_count(
                n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
            ),
            "matrix_stats": matrix_stats,
            "route_name": route_name,
            "level_row_map32": (
                level_meta["row_map32"]
                if level_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
            "level_ptr32": (
                level_meta["level_ptr32"]
                if level_meta is not None
                else torch.zeros(1, dtype=torch.int32, device=data.device)
            ),
            "nnz_balance_indegree32": (
                nnz_meta["indegree_init32"]
                if nnz_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
            "nnz_balance_row_idx32": (
                nnz_meta["csr_row_idx32"]
                if nnz_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
            "nnz_balance_launch_order32": (
                nnz_meta["launch_order32"]
                if nnz_meta is not None
                else torch.empty(0, dtype=torch.int32, device=data.device)
            ),
        }
        _attach_spsv_complex_plan_views(cw_plan)
        return cw_plan

    lower_eff = not lower
    storage_view = _normalize_spsv_storage_view(storage_view)
    if storage_view != "csr_as_csc":
        raise ValueError("TRANS/CONJ SpSV only supports storage_view='csr_as_csc'")
    matrix_stats = _build_spsv_cw_matrix_stats(indptr64, n_rows)
    default_block_nnz, default_max_segments = _choose_transpose_family_launch_config(
        indptr64
    )
    cw_plan = {
        "solve_kind": "transpose_cw",
        "default_solve_kind": "transpose_cw",
        "supported_solve_kinds": ("transpose_cw",),
        "kernel_data": data,
        "kernel_indices32": indices64.to(torch.int32),
        "kernel_indptr64": indptr64,
        "lower_eff": lower_eff,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "cw_worker_count": _cw_worker_count(
            n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
        ),
        "matrix_stats": matrix_stats,
        "storage_view": storage_view,
        "route_name": "transpose_cw",
    }
    _attach_spsv_complex_plan_views(cw_plan)
    return cw_plan


def _resolve_spsv_csr_runtime(
    data,
    indices,
    indptr,
    b,
    shape,
    lower,
    transpose,
    unit_diagonal=False,
    requested_solve_kind=None,
    storage_view="csr_as_csc",
):
    input_data = data
    input_indices = indices
    input_indptr = indptr
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    data, input_index_dtype, indices, indptr, b, n_rows, n_cols = _prepare_spsv_inputs(
        data, indices, indptr, b, shape
    )
    original_output_dtype = None
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "CSR")
    else:
        _validate_spsv_trans_combo(data.dtype, input_index_dtype, "CSR")

    preprocess_key = _csr_preprocess_cache_key(
        input_data,
        input_indices,
        input_indptr,
        (n_rows, n_cols),
        lower,
        trans_mode,
        unit_diagonal,
        _normalize_requested_spsv_route(requested_solve_kind, trans_mode),
        _normalize_spsv_storage_view(storage_view),
    )
    cached = _spsv_cache_get(_SPSV_CSR_PREPROCESS_CACHE, preprocess_key)
    if cached is None:
        cached = _prepare_spsv_csr_system(
            data,
            indices,
            indptr,
            n_rows,
            n_cols,
            lower,
            trans_mode,
            unit_diagonal,
            requested_solve_kind=requested_solve_kind,
            storage_view=storage_view,
        )
        _spsv_cache_put(
            _SPSV_CSR_PREPROCESS_CACHE,
            preprocess_key,
            cached,
            _SPSV_CSR_PREPROCESS_CACHE_SIZE,
        )
    return (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        n_cols,
        cached,
    )


def _select_spsv_runtime_plan(solve_plan, trans_mode, requested_solve_kind=None):
    requested_route = _normalize_requested_spsv_route(requested_solve_kind, trans_mode)
    routed = _clone_spsv_plan(solve_plan)
    if requested_route is None:
        requested_route = str(
            solve_plan.get("default_solve_kind", solve_plan.get("solve_kind", "csr_cw"))
        )
    supported = tuple(solve_plan.get("supported_solve_kinds", (solve_plan.get("solve_kind"),)))
    if requested_route not in supported:
        raise ValueError(
            f"solve_kind={requested_route!r} is not available for this SpSV problem; "
            f"supported routes: {', '.join(str(route) for route in supported if route)}"
        )
    routed["solve_kind"] = requested_route
    if requested_route == "csr_cw":
        routed["route_name"] = str(solve_plan.get("nontrans_variant", requested_route))
    elif requested_route == "csr_roc":
        routed["route_name"] = "csr_n_lo_roc" if bool(routed.get("lower_eff", True)) else "csr_n_up_roc"
    elif requested_route == "csr_smblk":
        routed["route_name"] = "csr_n_lo_smblk" if bool(routed.get("lower_eff", True)) else "csr_n_up_smblk"
    elif requested_route == "csr_cw_levelschd":
        routed["route_name"] = (
            "csr_n_lo_cw_levelschd" if bool(routed.get("lower_eff", True)) else "csr_n_up_cw_levelschd"
        )
    elif requested_route == "csr_nnz_balance":
        routed["route_name"] = (
            "csr_n_lo_nnz_balance" if bool(routed.get("lower_eff", True)) else "csr_n_up_nnz_balance"
        )
    else:
        routed["route_name"] = requested_route
    return routed


@triton.jit
def _publish_ready_flag_i32(flag_ptr, idx):
    """Publish a ready flag through an atomic write-like operation."""

    tl.atomic_add(flag_ptr + idx, 1)


@triton.jit
def _load_ready_flag_i32(flag_ptr, idx):
    """Mirror the original volatile/atomic polling pattern more closely."""

    return tl.atomic_add(flag_ptr + idx, 0)


@triton.jit
def _publish_i32_once(slot_ptr, idx, value):
    """Publish a single int32 payload via an atomic write-like update."""

    tl.atomic_add(slot_ptr + idx, value)


@triton.jit
def _load_scalar_fp32(ptr, idx):
    return tl.atomic_add(ptr + idx, 0.0)


@triton.jit
def _load_scalar_fp64(ptr, idx):
    return tl.atomic_add(ptr + idx, 0.0)


@triton.jit
def _complex_atomic_add_interleaved(ptr_ri, idx, delta_re, delta_im, mask):
    """Complex atomicAdd equivalent for interleaved real/imag buffers."""

    tl.atomic_add(ptr_ri + idx * 2, delta_re, mask=mask)
    tl.atomic_add(ptr_ri + idx * 2 + 1, delta_im, mask=mask)


@triton.jit
def _propagate_real(residual_ptr, idx, delta, mask):
    """Publish a real contribution into shared residual state."""

    tl.atomic_add(residual_ptr + idx, delta, mask=mask)


@triton.jit
def _propagate_then_release_real(residual_ptr, indegree_ptr, idx, delta, mask):
    """Approximate 'write contribution then decrement dependency count'."""

    _propagate_real(residual_ptr, idx, delta, mask)
    tl.atomic_add(indegree_ptr + idx, -1, mask=mask)


@triton.jit
def _propagate_complex(residual_ri_ptr, idx, delta_re, delta_im, mask):
    """Publish a complex contribution into shared residual state."""

    _complex_atomic_add_interleaved(residual_ri_ptr, idx, delta_re, delta_im, mask)


@triton.jit
def _propagate_then_release_complex(residual_ri_ptr, indegree_ptr, idx, delta_re, delta_im, mask):
    """Complex propagation + dependency release for transpose-style solve."""

    _propagate_complex(residual_ri_ptr, idx, delta_re, delta_im, mask)
    tl.atomic_add(indegree_ptr + idx, -1, mask=mask)


@triton.jit
def _spsv_csr_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        ptr = start
        if USE_FP64_ACC:
            rhs = tl.load(b_ptr + row).to(tl.float64)
            tmp_sum = tl.zeros((), dtype=tl.float64)
        else:
            rhs = tl.load(b_ptr + row).to(tl.float32)
            tmp_sum = tl.zeros((), dtype=tl.float32)
        row_done = 0
        while row_done == 0:
            if UNIT_DIAG:
                if ptr >= end:
                    x_row = rhs - tmp_sum
                    x_row = tl.where(x_row == x_row, x_row, 0.0)
                    tl.store(x_ptr + row, x_row)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    stop_at_diag = (col >= row) if LOWER else (col <= row)
                    if stop_at_diag:
                        x_row = rhs - tmp_sum
                        x_row = tl.where(x_row == x_row, x_row, 0.0)
                        tl.store(x_ptr + row, x_row)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(data_ptr + ptr).to(tl.float64)
                            y_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(data_ptr + ptr).to(tl.float32)
                            y_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * y_dep
                        ptr += 1
            else:
                if ptr >= end:
                    x_row = rhs * 0
                    tl.store(x_ptr + row, x_row)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    if col == row:
                        if USE_FP64_ACC:
                            diag = tl.load(data_ptr + ptr).to(tl.float64)
                        else:
                            diag = tl.load(data_ptr + ptr).to(tl.float32)
                        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
                        x_row = (rhs - tmp_sum) / diag_safe
                        x_row = tl.where(x_row == x_row, x_row, 0.0)
                        tl.store(x_ptr + row, x_row)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(data_ptr + ptr).to(tl.float64)
                            y_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(data_ptr + ptr).to(tl.float32)
                            y_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * y_dep
                        ptr += 1
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    lane2 = tl.arange(0, 2)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        rhs_re = tl.load(b_ri_ptr + row * 2)
        rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
            tmp_re = tl.zeros((), dtype=tl.float64)
            tmp_im = tl.zeros((), dtype=tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)
            tmp_re = tl.zeros((), dtype=tl.float32)
            tmp_im = tl.zeros((), dtype=tl.float32)
        ptr = start
        row_done = 0
        while row_done == 0:
            if UNIT_DIAG:
                if ptr >= end:
                    x_re_out = rhs_re - tmp_re
                    x_im_out = rhs_im - tmp_im
                    x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                    x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                    out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                    tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    stop_at_diag = (col >= row) if LOWER else (col <= row)
                    if stop_at_diag:
                        x_re_out = rhs_re - tmp_re
                        x_im_out = rhs_im - tmp_im
                        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        a_re = tl.load(data_ri_ptr + ptr * 2)
                        a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                        else:
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                        tmp_re += a_re * x_re - a_im * x_im
                        tmp_im += a_re * x_im + a_im * x_re
                        ptr += 1
            else:
                if ptr >= end:
                    out_vals = tl.where(lane2 == 0, rhs_re * 0, rhs_im * 0)
                    tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                    row_done = 1
                else:
                    col = tl.load(indices_ptr + ptr)
                    if col == row:
                        diag_re = tl.load(data_ri_ptr + ptr * 2)
                        diag_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            diag_re = diag_re.to(tl.float64)
                            diag_im = diag_im.to(tl.float64)
                        else:
                            diag_re = diag_re.to(tl.float32)
                            diag_im = diag_im.to(tl.float32)
                        num_re = rhs_re - tmp_re
                        num_im = rhs_im - tmp_im
                        den = diag_re * diag_re + diag_im * diag_im
                        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                        x_re_out = (num_re * diag_re + num_im * diag_im) / den_safe
                        x_im_out = (num_im * diag_re - num_re * diag_im) / den_safe
                        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
                        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)
                        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
                        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)
                        row_done = 1
                    else:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        a_re = tl.load(data_ri_ptr + ptr * 2)
                        a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                        if USE_FP64_ACC:
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                        else:
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                        tmp_re += a_re * x_re - a_im * x_im
                        tmp_im += a_re * x_im + a_im * x_re
                        ptr += 1
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_sell_trans_analysis_kernel(
    col_indices_ptr,
    slice_offsets_ptr,
    indegree_init_ptr,
    diag_offsets_ptr,
    row_widths_ptr,
    n_rows,
    SLICE_SIZE: tl.constexpr,
):
    """Build immutable transpose metadata from column-major SELL."""

    row = tl.program_id(0)
    row_i64 = row.to(tl.int64)
    slice_id = row // SLICE_SIZE
    row_in_slice = row_i64 - slice_id.to(tl.int64) * SLICE_SIZE
    # Normalize analysis indexing to int64 for both supported SELL index
    # dtypes.  This keeps every loop-carried index independent of pointer dtype.
    slice_start = tl.load(slice_offsets_ptr + slice_id).to(tl.int64)
    slice_end = tl.load(slice_offsets_ptr + slice_id + 1).to(tl.int64)
    width = (slice_end - slice_start) // SLICE_SIZE
    zero_i64 = tl.zeros((), dtype=tl.int64)
    # NON_UNIT validation guarantees that the diagonal replaces this value;
    # UNIT does not read the recorded diagonal offset.
    diag_offset = zero_i64
    row_width = width
    slot = zero_i64
    while slot < width:
        offset = slice_start + slot * SLICE_SIZE + row_in_slice
        col = tl.load(col_indices_ptr + offset)
        diag_offset = tl.where(col.to(tl.int64) == row_i64, offset, diag_offset)
        row_width = tl.where((col < 0) & (row_width == width), slot, row_width)
        # For a lower A, row i of A^T depends on every A row j>i that
        # contains column i.  Padding and the diagonal are not dependencies.
        dependency = (col >= 0) & (col.to(tl.int64) < row_i64)
        tl.atomic_add(indegree_init_ptr + col, 1, mask=dependency)
        slot += 1
    tl.store(diag_offsets_ptr + row, diag_offset)
    tl.store(row_widths_ptr + row, row_width.to(tl.int32))


@triton.jit
def _spsv_sell_trans_init_kernel(
    b_ptr,
    residual_ptr,
    indegree_init_ptr,
    indegree_ptr,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
    VALUE_ELEMENTS_PER_ROW: tl.constexpr,
    RESIDUAL_FP64: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value_mask = offsets < n_rows * VALUE_ELEMENTS_PER_ROW
    row_mask = offsets < n_rows
    rhs = tl.load(b_ptr + offsets, mask=value_mask, other=0.0)
    if RESIDUAL_FP64:
        rhs = rhs.to(tl.float64)
    tl.store(
        residual_ptr + offsets,
        rhs,
        mask=value_mask,
    )
    tl.store(
        indegree_ptr + offsets,
        tl.load(indegree_init_ptr + offsets, mask=row_mask, other=0),
        mask=row_mask,
    )


@triton.jit
def _spsv_sell_trans_kernel(
    values_ptr,
    col_indices_ptr,
    slice_offsets_ptr,
    diag_offsets_ptr,
    row_widths_ptr,
    residual_ptr,
    x_ptr,
    indegree_ptr,
    block_counter_ptr,
    n_rows,
    n_blocks,
    SLICE_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    """Slice-cooperative transpose SELL solver."""

    lanes = tl.arange(0, BLOCK_ROWS)
    zero_i32 = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    logical_block = tl.atomic_add(block_counter_ptr, 1, sem="relaxed")
    while logical_block < n_blocks:
        first_row = n_rows - (logical_block + 1) * BLOCK_ROWS
        raw_row = first_row + (BLOCK_ROWS - 1 - lanes)
        valid_row = (raw_row >= 0) & (raw_row < n_rows)
        row = tl.maximum(raw_row, 0)
        slice_id = row // SLICE_SIZE
        row_in_slice = row - slice_id * SLICE_SIZE
        slice_start = tl.load(slice_offsets_ptr + slice_id, mask=valid_row, other=0)
        row_width = tl.load(row_widths_ptr + row, mask=valid_row, other=0)
        pending = valid_row

        while tl.sum(pending.to(tl.int32), axis=0) != 0:
            ready_hint = tl.load(
                indegree_ptr + row,
                mask=pending,
                other=1,
                cache_modifier=".cv",
                volatile=True,
            )
            candidate = pending & (ready_hint == 0)
            ready_value = tl.atomic_add(
                indegree_ptr + row,
                zero_i32,
                mask=candidate,
                sem="acquire",
            )
            ready = candidate & (ready_value == 0)
            if USE_FP64_ACC:
                rhs = tl.load(residual_ptr + row, mask=ready, other=0.0).to(tl.float64)
            else:
                rhs = tl.load(residual_ptr + row, mask=ready, other=0.0).to(tl.float32)
            if UNIT_DIAG:
                diag = rhs * 0.0 + 1.0
            else:
                diag_offset = tl.load(diag_offsets_ptr + row, mask=ready, other=0)
                diag = tl.load(values_ptr + diag_offset, mask=ready, other=1.0)
                if USE_FP64_ACC:
                    diag = diag.to(tl.float64)
                else:
                    diag = diag.to(tl.float32)

            x_row = rhs / diag
            tl.store(x_ptr + row, x_row, mask=ready)

            slot = zero_i32
            while tl.sum((ready & (slot < row_width)).to(tl.int32), axis=0) != 0:
                active = ready & (slot < row_width)
                offset = slice_start + slot * SLICE_SIZE + row_in_slice
                col = tl.load(col_indices_ptr + offset, mask=active, other=-1)
                dependency = active & (col >= 0) & (col < row)
                a = tl.load(values_ptr + offset, mask=dependency, other=0.0)
                if USE_FP64_ACC:
                    a = a.to(tl.float64)
                    x_value = x_row.to(tl.float64)
                else:
                    a = a.to(tl.float32)
                    x_value = x_row.to(tl.float32)
                tl.atomic_add(
                    residual_ptr + col,
                    -a * x_value,
                    mask=dependency,
                )
                tl.atomic_add(
                    indegree_ptr + col,
                    -1,
                    mask=dependency,
                    sem="release",
                )
                slot += 1
            pending = pending & ~ready

        logical_block = tl.atomic_add(block_counter_ptr, 1, sem="relaxed")


@triton.jit
def _spsv_sell_trans_kernel_complex(
    values_ri_ptr,
    col_indices_ptr,
    slice_offsets_ptr,
    diag_offsets_ptr,
    row_widths_ptr,
    residual_ri_ptr,
    x_ri_ptr,
    indegree_ptr,
    block_counter_ptr,
    n_rows,
    n_blocks,
    SLICE_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    CONJ_TRANS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    """Complex slice-cooperative transpose SELL solver."""

    lanes = tl.arange(0, BLOCK_ROWS)
    zero_i32 = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    logical_block = tl.atomic_add(block_counter_ptr, 1, sem="relaxed")
    while logical_block < n_blocks:
        first_row = n_rows - (logical_block + 1) * BLOCK_ROWS
        raw_row = first_row + (BLOCK_ROWS - 1 - lanes)
        valid_row = (raw_row >= 0) & (raw_row < n_rows)
        row = tl.maximum(raw_row, 0)
        slice_id = row // SLICE_SIZE
        row_in_slice = row - slice_id * SLICE_SIZE
        slice_start = tl.load(slice_offsets_ptr + slice_id, mask=valid_row, other=0)
        row_width = tl.load(row_widths_ptr + row, mask=valid_row, other=0)
        pending = valid_row

        while tl.sum(pending.to(tl.int32), axis=0) != 0:
            ready_hint = tl.load(
                indegree_ptr + row,
                mask=pending,
                other=1,
                cache_modifier=".cv",
                volatile=True,
            )
            candidate = pending & (ready_hint == 0)
            ready_value = tl.atomic_add(
                indegree_ptr + row,
                zero_i32,
                mask=candidate,
                sem="acquire",
            )
            ready = candidate & (ready_value == 0)
            rhs_re = tl.load(residual_ri_ptr + row * 2, mask=ready, other=0.0)
            rhs_im = tl.load(residual_ri_ptr + row * 2 + 1, mask=ready, other=0.0)
            if USE_FP64_ACC:
                rhs_re = rhs_re.to(tl.float64)
                rhs_im = rhs_im.to(tl.float64)
            else:
                rhs_re = rhs_re.to(tl.float32)
                rhs_im = rhs_im.to(tl.float32)
            if UNIT_DIAG:
                diag_re = rhs_re * 0.0 + 1.0
                diag_im = rhs_im * 0.0
            else:
                diag_offset = tl.load(diag_offsets_ptr + row, mask=ready, other=0)
                diag_re = tl.load(
                    values_ri_ptr + diag_offset * 2,
                    mask=ready,
                    other=1.0,
                )
                diag_im = tl.load(
                    values_ri_ptr + diag_offset * 2 + 1,
                    mask=ready,
                    other=0.0,
                )
                if CONJ_TRANS:
                    diag_im = -diag_im
                if USE_FP64_ACC:
                    diag_re = diag_re.to(tl.float64)
                    diag_im = diag_im.to(tl.float64)
                else:
                    diag_re = diag_re.to(tl.float32)
                    diag_im = diag_im.to(tl.float32)

            den = diag_re * diag_re + diag_im * diag_im
            x_re = (rhs_re * diag_re + rhs_im * diag_im) / den
            x_im = (rhs_im * diag_re - rhs_re * diag_im) / den
            tl.store(x_ri_ptr + row * 2, x_re, mask=ready)
            tl.store(x_ri_ptr + row * 2 + 1, x_im, mask=ready)

            slot = zero_i32
            while tl.sum((ready & (slot < row_width)).to(tl.int32), axis=0) != 0:
                active = ready & (slot < row_width)
                offset = slice_start + slot * SLICE_SIZE + row_in_slice
                col = tl.load(col_indices_ptr + offset, mask=active, other=-1)
                dependency = active & (col >= 0) & (col < row)
                a_re = tl.load(
                    values_ri_ptr + offset * 2,
                    mask=dependency,
                    other=0.0,
                )
                a_im = tl.load(
                    values_ri_ptr + offset * 2 + 1,
                    mask=dependency,
                    other=0.0,
                )
                if CONJ_TRANS:
                    a_im = -a_im
                if USE_FP64_ACC:
                    a_re = a_re.to(tl.float64)
                    a_im = a_im.to(tl.float64)
                    x_re_acc = x_re.to(tl.float64)
                    x_im_acc = x_im.to(tl.float64)
                else:
                    a_re = a_re.to(tl.float32)
                    a_im = a_im.to(tl.float32)
                    x_re_acc = x_re.to(tl.float32)
                    x_im_acc = x_im.to(tl.float32)
                delta_re = -(a_re * x_re_acc - a_im * x_im_acc)
                delta_im = -(a_re * x_im_acc + a_im * x_re_acc)
                tl.atomic_add(
                    residual_ri_ptr + col * 2,
                    delta_re,
                    mask=dependency,
                )
                tl.atomic_add(
                    residual_ri_ptr + col * 2 + 1,
                    delta_im,
                    mask=dependency,
                )
                tl.atomic_add(
                    indegree_ptr + col,
                    -1,
                    mask=dependency,
                    sem="release",
                )
                slot += 1
            pending = pending & ~ready

        logical_block = tl.atomic_add(block_counter_ptr, 1, sem="relaxed")


@triton.jit
def _spsv_sell_cw_kernel_alg1(
    values_ptr,
    col_indices_ptr,
    slice_offsets_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    SLICE_SIZE: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    """ALG1: original persistent scalar-row dependency solver."""

    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = logical_row
        slice_id = row // SLICE_SIZE
        row_in_slice = row - slice_id * SLICE_SIZE
        slice_start = tl.load(slice_offsets_ptr + slice_id)
        slice_end = tl.load(slice_offsets_ptr + slice_id + 1)
        width = (slice_end - slice_start) // SLICE_SIZE
        if USE_FP64_ACC:
            rhs = tl.load(b_ptr + row).to(tl.float64)
            tmp_sum = tl.zeros((), dtype=tl.float64)
            diag = tl.zeros((), dtype=tl.float64)
        else:
            rhs = tl.load(b_ptr + row).to(tl.float32)
            tmp_sum = tl.zeros((), dtype=tl.float32)
            diag = tl.zeros((), dtype=tl.float32)
        if UNIT_DIAG:
            diag += 1.0
        slot = 0
        while slot < width:
            offset = slice_start + slot * SLICE_SIZE + row_in_slice
            col = tl.load(col_indices_ptr + offset)
            valid = (col >= 0) & (col < n_rows)
            if valid:
                if col == row:
                    if not UNIT_DIAG:
                        if USE_FP64_ACC:
                            diag = tl.load(values_ptr + offset).to(tl.float64)
                        else:
                            diag = tl.load(values_ptr + offset).to(tl.float32)
                else:
                    is_dependency = col < row
                    if is_dependency:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        if USE_FP64_ACC:
                            a = tl.load(values_ptr + offset).to(tl.float64)
                            x_dep = tl.load(x_ptr + col).to(tl.float64)
                        else:
                            a = tl.load(values_ptr + offset).to(tl.float32)
                            x_dep = tl.load(x_ptr + col).to(tl.float32)
                        tmp_sum += a * x_dep
            slot += 1
        x_row = (rhs - tmp_sum) / diag
        tl.store(x_ptr + row, x_row)
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_sell_cw_kernel_alg1_complex(
    values_ri_ptr,
    col_indices_ptr,
    slice_offsets_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    SLICE_SIZE: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    """Complex ALG1 using the interleaved real/imaginary storage convention."""

    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = logical_row
        slice_id = row // SLICE_SIZE
        row_in_slice = row - slice_id * SLICE_SIZE
        slice_start = tl.load(slice_offsets_ptr + slice_id)
        slice_end = tl.load(slice_offsets_ptr + slice_id + 1)
        width = (slice_end - slice_start) // SLICE_SIZE
        rhs_re = tl.load(b_ri_ptr + row * 2)
        rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
            tmp_sum_re = tl.zeros((), dtype=tl.float64)
            tmp_sum_im = tl.zeros((), dtype=tl.float64)
            diag_re = tl.zeros((), dtype=tl.float64)
            diag_im = tl.zeros((), dtype=tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)
            tmp_sum_re = tl.zeros((), dtype=tl.float32)
            tmp_sum_im = tl.zeros((), dtype=tl.float32)
            diag_re = tl.zeros((), dtype=tl.float32)
            diag_im = tl.zeros((), dtype=tl.float32)
        if UNIT_DIAG:
            diag_re += 1.0
        slot = 0
        while slot < width:
            offset = slice_start + slot * SLICE_SIZE + row_in_slice
            col = tl.load(col_indices_ptr + offset)
            valid = (col >= 0) & (col < n_rows)
            if valid:
                if col == row:
                    if not UNIT_DIAG:
                        diag_re = tl.load(values_ri_ptr + offset * 2)
                        diag_im = tl.load(values_ri_ptr + offset * 2 + 1)
                        if USE_FP64_ACC:
                            diag_re = diag_re.to(tl.float64)
                            diag_im = diag_im.to(tl.float64)
                        else:
                            diag_re = diag_re.to(tl.float32)
                            diag_im = diag_im.to(tl.float32)
                else:
                    is_dependency = col < row
                    if is_dependency:
                        dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        while dep_ready != 1:
                            dep_ready = tl.atomic_add(ready_ptr + col, 0)
                        a_re = tl.load(values_ri_ptr + offset * 2)
                        a_im = tl.load(values_ri_ptr + offset * 2 + 1)
                        x_re = tl.load(x_ri_ptr + col * 2)
                        x_im = tl.load(x_ri_ptr + col * 2 + 1)
                        if USE_FP64_ACC:
                            a_re = a_re.to(tl.float64)
                            a_im = a_im.to(tl.float64)
                            x_re = x_re.to(tl.float64)
                            x_im = x_im.to(tl.float64)
                        else:
                            a_re = a_re.to(tl.float32)
                            a_im = a_im.to(tl.float32)
                            x_re = x_re.to(tl.float32)
                            x_im = x_im.to(tl.float32)
                        tmp_sum_re += a_re * x_re - a_im * x_im
                        tmp_sum_im += a_re * x_im + a_im * x_re
            slot += 1

        num_re = rhs_re - tmp_sum_re
        num_im = rhs_im - tmp_sum_im
        den = diag_re * diag_re + diag_im * diag_im
        out_re = (num_re * diag_re + num_im * diag_im) / den
        out_im = (num_im * diag_re - num_re * diag_im) / den
        tl.store(x_ri_ptr + row * 2, out_re)
        tl.store(x_ri_ptr + row * 2 + 1, out_im)
        _publish_ready_flag_i32(ready_ptr, row)
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_sell_slice_kernel_alg2(
    values_ptr,
    col_indices_ptr,
    slice_offsets_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    n_slices,
    SLICE_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    """ALG2: persistent slice-cooperative SELL dependency solver.

    Each lane owns one row in a SELL slice.  Advancing all rows through their
    current SELL slot keeps column/index loads coalesced, while the per-lane
    slot state lets independent rows continue instead of waiting for the most
    serialized row in the slice.  Claiming slices from one ascending counter
    keeps the active dependency window contiguous even when not every program
    can reside concurrently.
    """

    lanes = tl.arange(0, BLOCK_ROWS)
    zero_i32 = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    logical_slice = tl.atomic_add(row_counter_ptr, 1, sem="relaxed")
    while logical_slice < n_slices:
        slice_id = logical_slice
        slice_start = tl.load(slice_offsets_ptr + slice_id)
        slice_end = tl.load(slice_offsets_ptr + slice_id + 1)
        width = (slice_end - slice_start) // SLICE_SIZE
        row = slice_id * SLICE_SIZE + lanes
        valid_row = (lanes < SLICE_SIZE) & (row < n_rows)
        rhs = tl.load(b_ptr + row, mask=valid_row, other=0.0)
        if USE_FP64_ACC:
            rhs = rhs.to(tl.float64)
            tmp_sum = tl.zeros((BLOCK_ROWS,), dtype=tl.float64)
            diag = tl.zeros((BLOCK_ROWS,), dtype=tl.float64)
        else:
            rhs = rhs.to(tl.float32)
            tmp_sum = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
            diag = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        slot = zero_i32
        if UNIT_DIAG:
            diag += 1.0
        done = ~valid_row
        while tl.sum((~done).to(tl.int32), axis=0) != 0:
            in_bounds = slot < width
            active = (~done) & in_bounds
            offset = slice_start + slot * SLICE_SIZE + lanes
            col = tl.load(col_indices_ptr + offset, mask=active, other=-1)

            dependency = active & (col >= 0) & (col < row)
            # Poll with volatile non-caching loads first.  Deep dependency
            # chains otherwise issue thousands of contending atomic RMWs for
            # flags that are still zero.  One acquire atomic is retained after
            # the hint turns nonzero to order the dependent x load.
            ready_hint = tl.load(
                ready_ptr + col,
                mask=dependency,
                other=0,
                cache_modifier=".cv",
                volatile=True,
            )
            acquire_mask = dependency & (ready_hint != 0)
            dep_ready = tl.atomic_add(
                ready_ptr + col,
                zero_i32,
                mask=acquire_mask,
                sem="acquire",
            )
            consume = acquire_mask & (dep_ready != 0)
            value = tl.load(values_ptr + offset, mask=consume, other=0.0)
            # The ready atomic is the acquire point for the preceding x store;
            # the dependent value itself does not need another floating-point
            # atomic read.
            x_dep = tl.load(
                x_ptr + col,
                mask=consume,
                other=0.0,
                cache_modifier=".cv",
            )
            if USE_FP64_ACC:
                value = value.to(tl.float64)
                x_dep = x_dep.to(tl.float64)
            else:
                value = value.to(tl.float32)
                x_dep = x_dep.to(tl.float32)
            tmp_sum += tl.where(consume, value * x_dep, 0.0)

            diagonal = active & (col == row)
            if not UNIT_DIAG:
                diag_value = tl.load(
                    values_ptr + offset, mask=diagonal, other=0.0
                )
                if USE_FP64_ACC:
                    diag_value = diag_value.to(tl.float64)
                else:
                    diag_value = diag_value.to(tl.float32)
                diag = tl.where(diagonal, diag_value, diag)

            # Padding marks the end of this row even when another row in the
            # same slice is much longer.  Finish immediately instead of
            # scanning all remaining padded slots up to the slice width.
            padding = active & (col < 0)
            skip = active & ((col < 0) | (col > row))
            slot = slot + (consume | diagonal | skip).to(tl.int32)
            finished = (~done) & (padding | (slot >= width))
            x_row = (rhs - tmp_sum) / diag
            tl.store(
                x_ptr + row,
                x_row,
                mask=finished,
                cache_modifier=".wt",
            )
            # Release publication follows the x store; dependent lanes use an
            # acquire atomic before reading x.
            tl.atomic_add(ready_ptr + row, 1, mask=finished, sem="release")
            done = done | finished

        logical_slice = tl.atomic_add(row_counter_ptr, 1, sem="relaxed")


@triton.jit
def _spsv_sell_slice_kernel_alg2_complex(
    values_ri_ptr,
    col_indices_ptr,
    slice_offsets_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    n_slices,
    SLICE_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    """Complex slice-cooperative SELL solver with coalesced component loads."""

    lanes = tl.arange(0, BLOCK_ROWS)
    zero_i32 = tl.zeros((BLOCK_ROWS,), dtype=tl.int32)
    logical_slice = tl.atomic_add(row_counter_ptr, 1, sem="relaxed")
    while logical_slice < n_slices:
        slice_id = logical_slice
        slice_start = tl.load(slice_offsets_ptr + slice_id)
        slice_end = tl.load(slice_offsets_ptr + slice_id + 1)
        width = (slice_end - slice_start) // SLICE_SIZE
        row = slice_id * SLICE_SIZE + lanes
        valid_row = (lanes < SLICE_SIZE) & (row < n_rows)
        rhs_re = tl.load(b_ri_ptr + row * 2, mask=valid_row, other=0.0)
        rhs_im = tl.load(b_ri_ptr + row * 2 + 1, mask=valid_row, other=0.0)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
            tmp_sum_re = tl.zeros((BLOCK_ROWS,), dtype=tl.float64)
            tmp_sum_im = tl.zeros((BLOCK_ROWS,), dtype=tl.float64)
            diag_re = tl.zeros((BLOCK_ROWS,), dtype=tl.float64)
            diag_im = tl.zeros((BLOCK_ROWS,), dtype=tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)
            tmp_sum_re = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
            tmp_sum_im = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
            diag_re = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
            diag_im = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        slot = zero_i32
        if UNIT_DIAG:
            diag_re += 1.0
        done = ~valid_row
        while tl.sum((~done).to(tl.int32), axis=0) != 0:
            in_bounds = slot < width
            active = (~done) & in_bounds
            offset = slice_start + slot * SLICE_SIZE + lanes
            col = tl.load(col_indices_ptr + offset, mask=active, other=-1)

            dependency = active & (col >= 0) & (col < row)
            ready_hint = tl.load(
                ready_ptr + col,
                mask=dependency,
                other=0,
                cache_modifier=".cv",
                volatile=True,
            )
            acquire_mask = dependency & (ready_hint != 0)
            dep_ready = tl.atomic_add(
                ready_ptr + col,
                zero_i32,
                mask=acquire_mask,
                sem="acquire",
            )
            consume = acquire_mask & (dep_ready != 0)
            value_re = tl.load(
                values_ri_ptr + offset * 2, mask=consume, other=0.0
            )
            value_im = tl.load(
                values_ri_ptr + offset * 2 + 1, mask=consume, other=0.0
            )
            x_re = tl.load(
                x_ri_ptr + col * 2,
                mask=consume,
                other=0.0,
                cache_modifier=".cv",
            )
            x_im = tl.load(
                x_ri_ptr + col * 2 + 1,
                mask=consume,
                other=0.0,
                cache_modifier=".cv",
            )
            if USE_FP64_ACC:
                value_re = value_re.to(tl.float64)
                value_im = value_im.to(tl.float64)
                x_re = x_re.to(tl.float64)
                x_im = x_im.to(tl.float64)
            else:
                value_re = value_re.to(tl.float32)
                value_im = value_im.to(tl.float32)
                x_re = x_re.to(tl.float32)
                x_im = x_im.to(tl.float32)
            product_re = value_re * x_re - value_im * x_im
            product_im = value_re * x_im + value_im * x_re
            tmp_sum_re += tl.where(consume, product_re, 0.0)
            tmp_sum_im += tl.where(consume, product_im, 0.0)

            diagonal = active & (col == row)
            if not UNIT_DIAG:
                diag_value_re = tl.load(
                    values_ri_ptr + offset * 2, mask=diagonal, other=0.0
                )
                diag_value_im = tl.load(
                    values_ri_ptr + offset * 2 + 1,
                    mask=diagonal,
                    other=0.0,
                )
                if USE_FP64_ACC:
                    diag_value_re = diag_value_re.to(tl.float64)
                    diag_value_im = diag_value_im.to(tl.float64)
                else:
                    diag_value_re = diag_value_re.to(tl.float32)
                    diag_value_im = diag_value_im.to(tl.float32)
                diag_re = tl.where(diagonal, diag_value_re, diag_re)
                diag_im = tl.where(diagonal, diag_value_im, diag_im)

            padding = active & (col < 0)
            skip = active & ((col < 0) | (col > row))
            slot = slot + (consume | diagonal | skip).to(tl.int32)
            finished = (~done) & (padding | (slot >= width))
            num_re = rhs_re - tmp_sum_re
            num_im = rhs_im - tmp_sum_im
            den = diag_re * diag_re + diag_im * diag_im
            out_re = (num_re * diag_re + num_im * diag_im) / den
            out_im = (num_im * diag_re - num_re * diag_im) / den
            tl.store(
                x_ri_ptr + row * 2,
                out_re,
                mask=finished,
                cache_modifier=".wt",
            )
            tl.store(
                x_ri_ptr + row * 2 + 1,
                out_im,
                mask=finished,
                cache_modifier=".wt",
            )
            tl.atomic_add(ready_ptr + row, 1, mask=finished, sem="release")
            done = done | finished

        logical_slice = tl.atomic_add(row_counter_ptr, 1, sem="relaxed")


@triton.jit
def _spsv_csr_transpose_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    residual_ptr,
    x_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        ready_value = 0 if UNIT_DIAG else 1
        dep_ready = tl.atomic_add(indegree_ptr + row, 0)
        while dep_ready != ready_value:
            dep_ready = tl.atomic_add(indegree_ptr + row, 0)

        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        rhs = tl.load(residual_ptr + row)
        if UNIT_DIAG:
            diag = rhs * 0 + 1.0
        else:
            diag = rhs * 0
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                offsets = idx + tl.arange(0, BLOCK_NNZ)
                mask = offsets < end
                a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
                dep_row = tl.load(indices_ptr + offsets, mask=mask, other=0)
                is_diag = dep_row == row
                diag = diag + tl.sum(tl.where(mask & is_diag, a, 0.0), axis=0)
        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
        x_row = rhs / diag_safe
        x_row = tl.where(x_row == x_row, x_row, 0.0)
        tl.store(x_ptr + row, x_row)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            if LOWER:
                target_mask = mask & (col > row)
            else:
                target_mask = mask & (col < row)
            _propagate_then_release_real(
                residual_ptr, indegree_ptr, col, -a * x_row, target_mask
            )
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_transpose_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    indegree_ptr,
    residual_ri_ptr,
    x_ri_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    CONJ_TRANS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.atomic_add(row_counter_ptr, 1)
    lane2 = tl.arange(0, 2)
    while logical_row < n_rows:
        row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
        ready_value = 0 if UNIT_DIAG else 1
        dep_ready = tl.atomic_add(indegree_ptr + row, 0)
        while dep_ready != ready_value:
            dep_ready = tl.atomic_add(indegree_ptr + row, 0)
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)

        rhs_re = tl.load(residual_ri_ptr + row * 2)
        rhs_im = tl.load(residual_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)

        if UNIT_DIAG:
            diag_re = rhs_re * 0 + 1.0
            diag_im = rhs_im * 0
        else:
            if USE_FP64_ACC:
                diag_re = tl.zeros((), dtype=tl.float64)
                diag_im = tl.zeros((), dtype=tl.float64)
            else:
                diag_re = tl.zeros((), dtype=tl.float32)
                diag_im = tl.zeros((), dtype=tl.float32)
            for seg in range(MAX_SEGMENTS):
                idx = start + seg * BLOCK_NNZ
                offsets = idx + tl.arange(0, BLOCK_NNZ)
                mask = offsets < end
                dep_row = tl.load(indices_ptr + offsets, mask=mask, other=0)
                a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
                a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
                if CONJ_TRANS:
                    a_im = -a_im
                if USE_FP64_ACC:
                    a_re = a_re.to(tl.float64)
                    a_im = a_im.to(tl.float64)
                else:
                    a_re = a_re.to(tl.float32)
                    a_im = a_im.to(tl.float32)
                is_diag = dep_row == row
                diag_re = diag_re + tl.sum(tl.where(mask & is_diag, a_re, 0.0), axis=0)
                diag_im = diag_im + tl.sum(tl.where(mask & is_diag, a_im, 0.0), axis=0)

        den = diag_re * diag_re + diag_im * diag_im
        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
        x_re_out = (rhs_re * diag_re + rhs_im * diag_im) / den_safe
        x_im_out = (rhs_im * diag_re - rhs_re * diag_im) / den_safe
        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

        out_vals = tl.where(lane2 == 0, x_re_out, x_im_out)
        tl.store(x_ri_ptr + row * 2 + lane2, out_vals)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
            a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
            if CONJ_TRANS:
                a_im = -a_im
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
            if LOWER:
                target_mask = mask & (col > row)
            else:
                target_mask = mask & (col < row)
            prod_re = a_re * x_re_out - a_im * x_im_out
            prod_im = a_re * x_im_out + a_im * x_re_out
            _propagate_then_release_complex(
                residual_ri_ptr,
                indegree_ptr,
                col,
                -prod_re,
                -prod_im,
                target_mask,
            )
        logical_row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_roc_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    n_rows,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lanes = tl.arange(0, WARP_SIZE)
    ptr = start + lanes
    if USE_FP64_ACC:
        rhs = tl.load(b_ptr + row).to(tl.float64)
        local_sum = tl.where(lanes == 0, rhs, 0.0).to(tl.float64)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float64)
    else:
        rhs = tl.load(b_ptr + row).to(tl.float32)
        local_sum = tl.where(lanes == 0, rhs, 0.0).to(tl.float32)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float32)

    loop_done = 0
    while loop_done == 0:
        active = ptr < end
        col = tl.load(indices_ptr + ptr, mask=active, other=row)
        dep_mask = active & (col < row if LOWER else col > row)
        if tl.sum(dep_mask.to(tl.int32), axis=0) == 0:
            loop_done = 1
        else:
            dep_ready = tl.atomic_add(
                ready_ptr + col,
                tl.zeros((WARP_SIZE,), dtype=tl.int32),
                mask=dep_mask,
            )
            advance_mask = dep_mask & (dep_ready != 0)
            a = tl.load(data_ptr + ptr, mask=advance_mask, other=0.0)
            if USE_FP64_ACC:
                a = a.to(tl.float64)
                y_dep = tl.atomic_add(x_ptr + col, zero_vec, mask=advance_mask).to(tl.float64)
            else:
                a = a.to(tl.float32)
                y_dep = tl.atomic_add(x_ptr + col, zero_vec, mask=advance_mask).to(tl.float32)
            local_sum += tl.where(advance_mask, -a * y_dep, 0.0)
            ptr = ptr + tl.where(advance_mask, WARP_SIZE, 0)

    active = ptr < end
    col = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
    diag_mask = active & (col == row)
    diag = tl.load(data_ptr + ptr, mask=diag_mask, other=0.0)
    if USE_FP64_ACC:
        diag = diag.to(tl.float64)
    else:
        diag = diag.to(tl.float32)
    diag_val = tl.sum(diag, axis=0)
    diag_safe = tl.where(tl.abs(diag_val) < DIAG_EPS, 1.0, diag_val)
    out = tl.sum(local_sum, axis=0) / diag_safe
    out = tl.where(out == out, out, 0.0)
    if USE_FP64_ACC:
        tl.atomic_add(x_ptr + row, out.to(tl.float64))
    else:
        tl.atomic_add(x_ptr + row, out.to(tl.float32))
    _publish_ready_flag_i32(ready_ptr, row)


@triton.jit
def _spsv_csr_roc_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    n_rows,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lanes = tl.arange(0, WARP_SIZE)
    ptr = start + lanes

    rhs_re = tl.load(b_ri_ptr + row * 2)
    rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        local_sum_re = tl.where(lanes == 0, rhs_re, 0.0).to(tl.float64)
        local_sum_im = tl.where(lanes == 0, rhs_im, 0.0).to(tl.float64)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        local_sum_re = tl.where(lanes == 0, rhs_re, 0.0).to(tl.float32)
        local_sum_im = tl.where(lanes == 0, rhs_im, 0.0).to(tl.float32)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float32)

    loop_done = 0
    while loop_done == 0:
        active = ptr < end
        col = tl.load(indices_ptr + ptr, mask=active, other=row)
        dep_mask = active & (col < row if LOWER else col > row)
        if tl.sum(dep_mask.to(tl.int32), axis=0) == 0:
            loop_done = 1
        else:
            dep_ready = tl.atomic_add(
                ready_ptr + col,
                tl.zeros((WARP_SIZE,), dtype=tl.int32),
                mask=dep_mask,
            )
            advance_mask = dep_mask & (dep_ready != 0)
            a_re = tl.load(data_ri_ptr + ptr * 2, mask=advance_mask, other=0.0)
            a_im = tl.load(data_ri_ptr + ptr * 2 + 1, mask=advance_mask, other=0.0)
            x_re = tl.atomic_add(x_ri_ptr + col * 2, zero_vec, mask=advance_mask)
            x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, zero_vec, mask=advance_mask)
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
                x_re = x_re.to(tl.float64)
                x_im = x_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
                x_re = x_re.to(tl.float32)
                x_im = x_im.to(tl.float32)
            prod_re = a_re * x_re - a_im * x_im
            prod_im = a_re * x_im + a_im * x_re
            local_sum_re += tl.where(advance_mask, -prod_re, 0.0)
            local_sum_im += tl.where(advance_mask, -prod_im, 0.0)
            ptr = ptr + tl.where(advance_mask, WARP_SIZE, 0)

    active = ptr < end
    col = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
    diag_mask = active & (col == row)
    diag_re = tl.load(data_ri_ptr + ptr * 2, mask=diag_mask, other=0.0)
    diag_im = tl.load(data_ri_ptr + ptr * 2 + 1, mask=diag_mask, other=0.0)
    if USE_FP64_ACC:
        diag_re = diag_re.to(tl.float64)
        diag_im = diag_im.to(tl.float64)
    else:
        diag_re = diag_re.to(tl.float32)
        diag_im = diag_im.to(tl.float32)
    diag_re = tl.sum(diag_re, axis=0)
    diag_im = tl.sum(diag_im, axis=0)
    sum_re = tl.sum(local_sum_re, axis=0)
    sum_im = tl.sum(local_sum_im, axis=0)
    den = diag_re * diag_re + diag_im * diag_im
    den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
    out_re = (sum_re * diag_re + sum_im * diag_im) / den_safe
    out_im = (sum_im * diag_re - sum_re * diag_im) / den_safe
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)
    tl.atomic_add(x_ri_ptr + row * 2, out_re)
    tl.atomic_add(x_ri_ptr + row * 2 + 1, out_im)
    _publish_ready_flag_i32(ready_ptr, row)


@triton.jit
def _spsv_csr_smblk_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lanes = tl.arange(0, WARP_SIZE)
    ptr = start + lanes
    if USE_FP64_ACC:
        rhs = tl.load(b_ptr + row).to(tl.float64)
        local_sum = tl.where(lanes == 0, rhs, 0.0).to(tl.float64)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float64)
    else:
        rhs = tl.load(b_ptr + row).to(tl.float32)
        local_sum = tl.where(lanes == 0, rhs, 0.0).to(tl.float32)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float32)

    loop_done = 0
    while loop_done == 0:
        active = ptr < end
        col = tl.load(indices_ptr + ptr, mask=active, other=row)
        dep_mask = active & (col < row if LOWER else col > row)
        if tl.sum(dep_mask.to(tl.int32), axis=0) == 0:
            loop_done = 1
        else:
            dep_ready = tl.atomic_add(
                ready_ptr + col,
                tl.zeros((WARP_SIZE,), dtype=tl.int32),
                mask=dep_mask,
            )
            advance_mask = dep_mask & (dep_ready != 0)
            a = tl.load(data_ptr + ptr, mask=advance_mask, other=0.0)
            if USE_FP64_ACC:
                a = a.to(tl.float64)
                y_dep = tl.atomic_add(x_ptr + col, zero_vec, mask=advance_mask).to(
                    tl.float64
                )
            else:
                a = a.to(tl.float32)
                y_dep = tl.atomic_add(x_ptr + col, zero_vec, mask=advance_mask).to(
                    tl.float32
                )
            local_sum += tl.where(advance_mask, -a * y_dep, 0.0)
            ptr = ptr + tl.where(advance_mask, WARP_SIZE, 0)

    active = ptr < end
    col = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
    diag_mask = active & (col == row)
    diag = tl.load(data_ptr + ptr, mask=diag_mask, other=0.0)
    if USE_FP64_ACC:
        diag = diag.to(tl.float64)
    else:
        diag = diag.to(tl.float32)
    diag_val = tl.sum(diag, axis=0)
    diag_safe = tl.where(tl.abs(diag_val) < DIAG_EPS, 1.0, diag_val)
    out = tl.sum(local_sum, axis=0) / diag_safe
    out = tl.where(out == out, out, 0.0)
    tl.store(x_ptr + row, out)
    _publish_ready_flag_i32(ready_ptr, row)


@triton.jit
def _spsv_csr_smblk_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    n_rows,
    LOWER: tl.constexpr,
    REVERSE_ORDER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
    WARP_SIZE: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.where(REVERSE_ORDER, n_rows - 1 - logical_row, logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lanes = tl.arange(0, WARP_SIZE)
    ptr = start + lanes

    rhs_re = tl.load(b_ri_ptr + row * 2)
    rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        local_sum_re = tl.where(lanes == 0, rhs_re, 0.0).to(tl.float64)
        local_sum_im = tl.where(lanes == 0, rhs_im, 0.0).to(tl.float64)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        local_sum_re = tl.where(lanes == 0, rhs_re, 0.0).to(tl.float32)
        local_sum_im = tl.where(lanes == 0, rhs_im, 0.0).to(tl.float32)
        zero_vec = tl.zeros((WARP_SIZE,), dtype=tl.float32)

    loop_done = 0
    while loop_done == 0:
        active = ptr < end
        col = tl.load(indices_ptr + ptr, mask=active, other=row)
        dep_mask = active & (col < row if LOWER else col > row)
        if tl.sum(dep_mask.to(tl.int32), axis=0) == 0:
            loop_done = 1
        else:
            dep_ready = tl.atomic_add(
                ready_ptr + col,
                tl.zeros((WARP_SIZE,), dtype=tl.int32),
                mask=dep_mask,
            )
            advance_mask = dep_mask & (dep_ready != 0)
            a_re = tl.load(data_ri_ptr + ptr * 2, mask=advance_mask, other=0.0)
            a_im = tl.load(data_ri_ptr + ptr * 2 + 1, mask=advance_mask, other=0.0)
            x_re = tl.atomic_add(x_ri_ptr + col * 2, zero_vec, mask=advance_mask)
            x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, zero_vec, mask=advance_mask)
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
                x_re = x_re.to(tl.float64)
                x_im = x_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
                x_re = x_re.to(tl.float32)
                x_im = x_im.to(tl.float32)
            prod_re = a_re * x_re - a_im * x_im
            prod_im = a_re * x_im + a_im * x_re
            local_sum_re += tl.where(advance_mask, -prod_re, 0.0)
            local_sum_im += tl.where(advance_mask, -prod_im, 0.0)
            ptr = ptr + tl.where(advance_mask, WARP_SIZE, 0)

    active = ptr < end
    col = tl.load(indices_ptr + ptr, mask=active, other=row + 1)
    diag_mask = active & (col == row)
    diag_re = tl.load(data_ri_ptr + ptr * 2, mask=diag_mask, other=0.0)
    diag_im = tl.load(data_ri_ptr + ptr * 2 + 1, mask=diag_mask, other=0.0)
    if USE_FP64_ACC:
        diag_re = diag_re.to(tl.float64)
        diag_im = diag_im.to(tl.float64)
    else:
        diag_re = diag_re.to(tl.float32)
        diag_im = diag_im.to(tl.float32)
    diag_re = tl.sum(diag_re, axis=0)
    diag_im = tl.sum(diag_im, axis=0)
    sum_re = tl.sum(local_sum_re, axis=0)
    sum_im = tl.sum(local_sum_im, axis=0)
    den = diag_re * diag_re + diag_im * diag_im
    den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
    out_re = (sum_re * diag_re + sum_im * diag_im) / den_safe
    out_im = (sum_im * diag_re - sum_re * diag_im) / den_safe
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)
    tl.store(x_ri_ptr + row * 2, out_re)
    tl.store(x_ri_ptr + row * 2 + 1, out_im)
    _publish_ready_flag_i32(ready_ptr, row)


@triton.jit
def _spsv_csr_cw_levelschd_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ptr,
    x_ptr,
    ready_ptr,
    n_rows,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    ptr = start
    if USE_FP64_ACC:
        rhs = tl.load(b_ptr + row).to(tl.float64)
        tmp_sum = tl.zeros((), dtype=tl.float64)
    else:
        rhs = tl.load(b_ptr + row).to(tl.float32)
        tmp_sum = tl.zeros((), dtype=tl.float32)
    row_done = 0
    while row_done == 0:
        if ptr >= end:
            x_row = rhs * 0
            if USE_FP64_ACC:
                tl.atomic_add(x_ptr + row, x_row.to(tl.float64))
            else:
                tl.atomic_add(x_ptr + row, x_row.to(tl.float32))
            row_done = 1
        else:
            col = tl.load(indices_ptr + ptr)
            if col == row:
                if USE_FP64_ACC:
                    diag = tl.load(data_ptr + ptr).to(tl.float64)
                else:
                    diag = tl.load(data_ptr + ptr).to(tl.float32)
                diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
                x_row = (rhs - tmp_sum) / diag_safe
                x_row = tl.where(x_row == x_row, x_row, 0.0)
                if USE_FP64_ACC:
                    tl.atomic_add(x_ptr + row, x_row.to(tl.float64))
                else:
                    tl.atomic_add(x_ptr + row, x_row.to(tl.float32))
                row_done = 1
            else:
                dep_ready = _load_ready_flag_i32(ready_ptr, col)
                while dep_ready != 1:
                    dep_ready = _load_ready_flag_i32(ready_ptr, col)
                if USE_FP64_ACC:
                    a = tl.load(data_ptr + ptr).to(tl.float64)
                    y_dep = _load_scalar_fp64(x_ptr, col).to(tl.float64)
                else:
                    a = tl.load(data_ptr + ptr).to(tl.float32)
                    y_dep = _load_scalar_fp32(x_ptr, col).to(tl.float32)
                tmp_sum += a * y_dep
                ptr += 1
    _publish_ready_flag_i32(ready_ptr, row)


@triton.jit
def _spsv_csr_cw_levelschd_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    row_map_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    n_rows,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    logical_row = tl.program_id(0)
    if logical_row >= n_rows:
        return
    row = tl.load(row_map_ptr + logical_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    ptr = start
    rhs_re = tl.load(b_ri_ptr + row * 2)
    rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        tmp_sum_re = tl.zeros((), dtype=tl.float64)
        tmp_sum_im = tl.zeros((), dtype=tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        tmp_sum_re = tl.zeros((), dtype=tl.float32)
        tmp_sum_im = tl.zeros((), dtype=tl.float32)
    row_done = 0
    while row_done == 0:
        if ptr >= end:
            zero = rhs_re * 0
            tl.atomic_add(x_ri_ptr + row * 2, zero)
            tl.atomic_add(x_ri_ptr + row * 2 + 1, zero)
            row_done = 1
        else:
            col = tl.load(indices_ptr + ptr)
            if col == row:
                diag_re = tl.load(data_ri_ptr + ptr * 2)
                diag_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                if USE_FP64_ACC:
                    diag_re = diag_re.to(tl.float64)
                    diag_im = diag_im.to(tl.float64)
                else:
                    diag_re = diag_re.to(tl.float32)
                    diag_im = diag_im.to(tl.float32)
                sum_re = rhs_re - tmp_sum_re
                sum_im = rhs_im - tmp_sum_im
                den = diag_re * diag_re + diag_im * diag_im
                den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                out_re = (sum_re * diag_re + sum_im * diag_im) / den_safe
                out_im = (sum_im * diag_re - sum_re * diag_im) / den_safe
                out_re = tl.where(out_re == out_re, out_re, 0.0)
                out_im = tl.where(out_im == out_im, out_im, 0.0)
                tl.atomic_add(x_ri_ptr + row * 2, out_re)
                tl.atomic_add(x_ri_ptr + row * 2 + 1, out_im)
                row_done = 1
            else:
                dep_ready = _load_ready_flag_i32(ready_ptr, col)
                while dep_ready != 1:
                    dep_ready = _load_ready_flag_i32(ready_ptr, col)
                a_re = tl.load(data_ri_ptr + ptr * 2)
                a_im = tl.load(data_ri_ptr + ptr * 2 + 1)
                x_re = tl.atomic_add(x_ri_ptr + col * 2, 0.0)
                x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, 0.0)
                if USE_FP64_ACC:
                    a_re = a_re.to(tl.float64)
                    a_im = a_im.to(tl.float64)
                    x_re = x_re.to(tl.float64)
                    x_im = x_im.to(tl.float64)
                else:
                    a_re = a_re.to(tl.float32)
                    a_im = a_im.to(tl.float32)
                    x_re = x_re.to(tl.float32)
                    x_im = x_im.to(tl.float32)
                tmp_sum_re += a_re * x_re - a_im * x_im
                tmp_sum_im += a_re * x_im + a_im * x_re
                ptr += 1
    _publish_ready_flag_i32(ready_ptr, row)


@triton.jit
def _spsv_csr_nnz_balance_kernel(
    launch_order_ptr,
    row_idx_ptr,
    col_idx_ptr,
    val_ptr,
    b_ptr,
    x_ptr,
    tmp_sum_ptr,
    ready_ptr,
    indegree_ptr,
    nnz,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    val_id = tl.program_id(0)
    if val_id >= nnz:
        return
    entry_id = tl.load(launch_order_ptr + val_id)
    row = tl.load(row_idx_ptr + entry_id)
    col = tl.load(col_idx_ptr + entry_id)
    if LOWER:
        if row < col:
            return
    else:
        if row > col:
            return
    if USE_FP64_ACC:
        a = tl.load(val_ptr + entry_id).to(tl.float64)
    else:
        a = tl.load(val_ptr + entry_id).to(tl.float32)
    done = 0
    while done == 0:
        if row != col:
            dep_ready = _load_ready_flag_i32(ready_ptr, col)
            if dep_ready == 1:
                if USE_FP64_ACC:
                    dep_x = _load_scalar_fp64(x_ptr, col).to(tl.float64)
                else:
                    dep_x = _load_scalar_fp32(x_ptr, col).to(tl.float32)
                tl.atomic_add(tmp_sum_ptr + row, dep_x * a)
                tl.atomic_add(indegree_ptr + row, -1)
                done = 1
        else:
            diag_degree = tl.atomic_add(indegree_ptr + row, 0)
            if diag_degree == 1:
                if USE_FP64_ACC:
                    rhs = tl.load(b_ptr + row).to(tl.float64)
                    sum_val = tl.atomic_add(tmp_sum_ptr + row, 0.0).to(tl.float64)
                else:
                    rhs = tl.load(b_ptr + row).to(tl.float32)
                    sum_val = tl.atomic_add(tmp_sum_ptr + row, 0.0).to(tl.float32)
                diag_safe = tl.where(tl.abs(a) < DIAG_EPS, 1.0, a)
                out = (rhs - sum_val) / diag_safe
                out = tl.where(out == out, out, 0.0)
                tl.store(x_ptr + row, out)
                _publish_ready_flag_i32(ready_ptr, row)
                done = 1


@triton.jit
def _spsv_csr_nnz_balance_kernel_complex(
    launch_order_ptr,
    row_idx_ptr,
    col_idx_ptr,
    val_ri_ptr,
    b_ri_ptr,
    x_ri_ptr,
    tmp_sum_ri_ptr,
    ready_ptr,
    indegree_ptr,
    nnz,
    LOWER: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    val_id = tl.program_id(0)
    if val_id >= nnz:
        return
    entry_id = tl.load(launch_order_ptr + val_id)
    row = tl.load(row_idx_ptr + entry_id)
    col = tl.load(col_idx_ptr + entry_id)
    if LOWER:
        if row < col:
            return
    else:
        if row > col:
            return
    val_re = tl.load(val_ri_ptr + entry_id * 2)
    val_im = tl.load(val_ri_ptr + entry_id * 2 + 1)
    if USE_FP64_ACC:
        val_re = val_re.to(tl.float64)
        val_im = val_im.to(tl.float64)
    else:
        val_re = val_re.to(tl.float32)
        val_im = val_im.to(tl.float32)
    done = 0
    while done == 0:
        if row != col:
            dep_ready = _load_ready_flag_i32(ready_ptr, col)
            if dep_ready == 1:
                dep_x_re = tl.atomic_add(x_ri_ptr + col * 2, 0.0)
                dep_x_im = tl.atomic_add(x_ri_ptr + col * 2 + 1, 0.0)
                if USE_FP64_ACC:
                    dep_x_re = dep_x_re.to(tl.float64)
                    dep_x_im = dep_x_im.to(tl.float64)
                else:
                    dep_x_re = dep_x_re.to(tl.float32)
                    dep_x_im = dep_x_im.to(tl.float32)
                prod_re = dep_x_re * val_re - dep_x_im * val_im
                prod_im = dep_x_re * val_im + dep_x_im * val_re
                tl.atomic_add(tmp_sum_ri_ptr + row * 2, prod_re)
                tl.atomic_add(tmp_sum_ri_ptr + row * 2 + 1, prod_im)
                tl.atomic_add(indegree_ptr + row, -1)
                done = 1
        if row == col:
            diag_degree = tl.atomic_add(indegree_ptr + row, 0)
            if diag_degree == 1:
                rhs_re = tl.load(b_ri_ptr + row * 2)
                rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
                sum_re = tl.atomic_add(tmp_sum_ri_ptr + row * 2, 0.0)
                sum_im = tl.atomic_add(tmp_sum_ri_ptr + row * 2 + 1, 0.0)
                if USE_FP64_ACC:
                    rhs_re = rhs_re.to(tl.float64)
                    rhs_im = rhs_im.to(tl.float64)
                    sum_re = sum_re.to(tl.float64)
                    sum_im = sum_im.to(tl.float64)
                else:
                    rhs_re = rhs_re.to(tl.float32)
                    rhs_im = rhs_im.to(tl.float32)
                    sum_re = sum_re.to(tl.float32)
                    sum_im = sum_im.to(tl.float32)
                num_re = rhs_re - sum_re
                num_im = rhs_im - sum_im
                den = val_re * val_re + val_im * val_im
                den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
                out_re = (num_re * val_re + num_im * val_im) / den_safe
                out_im = (num_im * val_re - num_re * val_im) / den_safe
                out_re = tl.where(out_re == out_re, out_re, 0.0)
                out_im = tl.where(out_im == out_im, out_im, 0.0)
                tl.store(x_ri_ptr + row * 2, out_re)
                tl.store(x_ri_ptr + row * 2 + 1, out_im)
                _publish_ready_flag_i32(ready_ptr, row)
                done = 1


def _auto_spsv_launch_config(indptr, block_nnz=None, max_segments=None, *, max_nnz_per_row=None):
    if max_nnz_per_row is None:
        if indptr.numel() <= 1:
            max_nnz_per_row = 0
        else:
            max_nnz_per_row = int((indptr[1:] - indptr[:-1]).max().item())

    auto_block = block_nnz is None
    if block_nnz is None:
        if max_nnz_per_row <= 64:
            block_nnz_use = 64
        elif max_nnz_per_row <= 256:
            block_nnz_use = 128
        elif max_nnz_per_row <= 1024:
            block_nnz_use = 256
        elif max_nnz_per_row <= 4096:
            block_nnz_use = 512
        elif max_nnz_per_row <= 16384:
            block_nnz_use = 1024
        else:
            block_nnz_use = 2048
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be a positive integer")

    required_segments = max(
        (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
    )
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max(
                    (max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1
                )
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be a positive integer")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )

    return block_nnz_use, max_segments_use


def _triton_spsv_csr_cw_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    worker_count=None,
    matrix_stats=None,
    ready_in=None,
    row_counter_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return x
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    use_fp64_acc = data.dtype == torch.float64
    grid = (worker_count,)
    _spsv_csr_cw_kernel[grid](
        data,
        indices,
        indptr,
        b_vec,
        x,
        ready,
        row_counter,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_cw_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    unit_diagonal=False,
    diag_eps=1e-12,
    worker_count=None,
    matrix_stats=None,
    data_ri_in=None,
    ready_in=None,
    row_counter_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return x

    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_cw_kernel_complex[grid](
        data_ri,
        indices,
        indptr,
        b_ri,
        x_ri,
        ready,
        row_counter,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    return x


def _launch_spsv_sell(
    values,
    col_indices,
    slice_offsets,
    b_vec,
    n_rows,
    *,
    slice_size,
    out,
    ready=None,
    row_counter=None,
    alg_num=1,
    alg2_worker_count=None,
    unit_diagonal=False,
    values_ri_in=None,
):
    alg_num = _normalize_spsv_sell_alg_num(alg_num)
    if ready is None or row_counter is None:
        raise ValueError("SELL SpSV ALG1/ALG2 require ready/counter workspace")
    ready.zero_()
    row_counter.zero_()
    if n_rows == 0:
        return out
    is_complex = torch.is_complex(values)
    use_fp64_acc = values.dtype in (torch.float64, torch.complex128)
    if is_complex:
        values_ri = (
            values_ri_in
            if values_ri_in is not None
            else _complex_interleaved_view(values)
        )
        b_ri = torch.view_as_real(_as_strided_contiguous(b_vec)).reshape(-1)
        out_ri = torch.view_as_real(out).reshape(-1)
    if alg_num == SPSV_SELL_ALG1:
        worker_count = _snap_cw_worker_count(min(n_rows, 32), n_rows)
        if is_complex:
            _spsv_sell_cw_kernel_alg1_complex[(int(worker_count),)](
                values_ri,
                col_indices,
                slice_offsets,
                b_ri,
                out_ri,
                ready,
                row_counter,
                n_rows,
                SLICE_SIZE=int(slice_size),
                UNIT_DIAG=bool(unit_diagonal),
                USE_FP64_ACC=use_fp64_acc,
            )
        else:
            _spsv_sell_cw_kernel_alg1[(int(worker_count),)](
                values,
                col_indices,
                slice_offsets,
                b_vec,
                out,
                ready,
                row_counter,
                n_rows,
                SLICE_SIZE=int(slice_size),
                UNIT_DIAG=bool(unit_diagonal),
                USE_FP64_ACC=use_fp64_acc,
            )
        return out

    n_slices = (int(n_rows) + int(slice_size) - 1) // int(slice_size)
    block_rows = 1 << (int(slice_size) - 1).bit_length()
    # One warp/program advances SELL slices from a contiguous dynamic frontier.
    # The worker cap limits ready polling contention; benchmarks may override
    # it explicitly.
    worker_count = _resolve_spsv_sell_alg2_worker_count(
        n_slices, requested=alg2_worker_count
    )
    if is_complex:
        _spsv_sell_slice_kernel_alg2_complex[(int(worker_count),)](
            values_ri,
            col_indices,
            slice_offsets,
            b_ri,
            out_ri,
            ready,
            row_counter,
            n_rows,
            n_slices,
            SLICE_SIZE=int(slice_size),
            BLOCK_ROWS=int(block_rows),
            UNIT_DIAG=bool(unit_diagonal),
            USE_FP64_ACC=use_fp64_acc,
            num_warps=1,
        )
    else:
        _spsv_sell_slice_kernel_alg2[(int(worker_count),)](
            values,
            col_indices,
            slice_offsets,
            b_vec,
            out,
            ready,
            row_counter,
            n_rows,
            n_slices,
            SLICE_SIZE=int(slice_size),
            BLOCK_ROWS=int(block_rows),
            UNIT_DIAG=bool(unit_diagonal),
            USE_FP64_ACC=use_fp64_acc,
            num_warps=1,
        )
    return out


def _launch_spsv_sell_trans(
    values,
    col_indices,
    slice_offsets,
    indegree_init,
    diag_offsets,
    row_widths,
    b_vec,
    n_rows,
    *,
    slice_size,
    out,
    residual,
    indegree,
    row_counter,
    unit_diagonal=False,
    conj_trans=False,
    values_ri_in=None,
):
    """Launch the dedicated SELL transpose dependency-propagation route."""

    row_counter.zero_()
    if n_rows == 0:
        return out

    is_complex = torch.is_complex(values)
    if is_complex:
        b_init = torch.view_as_real(b_vec).reshape(-1)
        residual_init = torch.view_as_real(residual).reshape(-1)
        value_elements_per_row = 2
    else:
        b_init = b_vec
        residual_init = residual
        value_elements_per_row = 1
    init_block_size = 256
    _spsv_sell_trans_init_kernel[
        (
            triton.cdiv(
                int(n_rows) * value_elements_per_row,
                init_block_size,
            ),
        )
    ](
        b_init,
        residual_init,
        indegree_init,
        indegree,
        n_rows,
        BLOCK_SIZE=init_block_size,
        VALUE_ELEMENTS_PER_ROW=value_elements_per_row,
        RESIDUAL_FP64=bool(residual.dtype in (torch.float64, torch.complex128)),
        num_warps=4,
    )
    # The residual buffer controls the accumulation dtype.  For float32 and
    # complex64 input, only this internal buffer is widened; x remains in the
    # original user-visible dtype.
    use_fp64_acc = residual.dtype in (torch.float64, torch.complex128)
    if is_complex:
        values_ri = (
            values_ri_in
            if values_ri_in is not None
            else _complex_interleaved_view(values)
        )
        residual_ri = torch.view_as_real(residual).reshape(-1)
        out_ri = torch.view_as_real(out).reshape(-1)

    block_rows = 1 << (int(slice_size) - 1).bit_length()
    n_blocks = (int(n_rows) + block_rows - 1) // block_rows
    worker_count = _snap_cw_worker_count(min(512, n_blocks), n_blocks)

    if is_complex:
        _spsv_sell_trans_kernel_complex[(int(worker_count),)](
            values_ri,
            col_indices,
            slice_offsets,
            diag_offsets,
            row_widths,
            residual_ri,
            out_ri,
            indegree,
            row_counter,
            n_rows,
            int(n_blocks),
            SLICE_SIZE=int(slice_size),
            BLOCK_ROWS=int(block_rows),
            UNIT_DIAG=bool(unit_diagonal),
            CONJ_TRANS=bool(conj_trans),
            USE_FP64_ACC=use_fp64_acc,
            num_warps=1,
        )
    else:
        _spsv_sell_trans_kernel[(int(worker_count),)](
            values,
            col_indices,
            slice_offsets,
            diag_offsets,
            row_widths,
            residual,
            out,
            indegree,
            row_counter,
            n_rows,
            int(n_blocks),
            SLICE_SIZE=int(slice_size),
            BLOCK_ROWS=int(block_rows),
            UNIT_DIAG=bool(unit_diagonal),
            USE_FP64_ACC=use_fp64_acc,
            num_warps=1,
        )
    return out


def _triton_spsv_csr_u_lo_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=True, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_lo_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=True, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_up_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=False, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_up_cw_vector(*args, **kwargs):
    return _triton_spsv_csr_cw_vector(*args, lower=False, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_lo_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=True, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_lo_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=True, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_u_up_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=False, unit_diagonal=True, **kwargs)


def _triton_spsv_csr_n_up_cw_vector_complex(*args, **kwargs):
    return _triton_spsv_csr_cw_vector_complex(*args, lower=False, unit_diagonal=False, **kwargs)


def _triton_spsv_csr_n_lo_roc_vector(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    use_fp64_acc = data.dtype == torch.float64
    _spsv_csr_roc_kernel[(n_rows,)](
        data,
        indices,
        indptr,
        row_map,
        b_vec,
        x,
        ready,
        n_rows,
        LOWER=lower,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        WARP_SIZE=32,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_roc_vector_complex(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    _spsv_csr_roc_kernel_complex[(n_rows,)](
        data_ri,
        indices,
        indptr,
        row_map,
        b_ri,
        x_ri,
        ready,
        n_rows,
        LOWER=lower,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        WARP_SIZE=32,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_smblk_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    use_fp64_acc = data.dtype == torch.float64
    _spsv_csr_smblk_kernel[(n_rows,)](
        data,
        indices,
        indptr,
        b_vec,
        x,
        ready,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        WARP_SIZE=32,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_smblk_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(
        n_rows, dtype=torch.int32, device=b_vec.device
    )
    ready.zero_()
    if n_rows == 0:
        return x
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    _spsv_csr_smblk_kernel_complex[(n_rows,)](
        data_ri,
        indices,
        indptr,
        b_ri,
        x_ri,
        ready,
        n_rows,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        WARP_SIZE=32,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_cw_levelschd_vector(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    ready.zero_()
    if n_rows == 0:
        return x
    use_fp64_acc = data.dtype == torch.float64
    grid = (n_rows,)
    _spsv_csr_cw_levelschd_kernel[grid](
        data,
        indices,
        indptr,
        row_map,
        b_vec,
        x,
        ready,
        n_rows,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_cw_levelschd_vector_complex(
    data,
    indices,
    indptr,
    row_map,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    ready_in=None,
):
    x = torch.zeros_like(b_vec)
    ready = (
        ready_in
        if ready_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    ready.zero_()
    if n_rows == 0:
        return x
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    grid = (n_rows,)
    _spsv_csr_cw_levelschd_kernel_complex[grid](
        data_ri,
        indices,
        indptr,
        row_map,
        b_ri,
        x_ri,
        ready,
        n_rows,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_nnz_balance_vector(
    data,
    indices,
    launch_order,
    row_idx,
    indegree_init,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    tmp_sum_in=None,
    ready_in=None,
    indegree_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    tmp_sum = tmp_sum_in if tmp_sum_in is not None else torch.zeros_like(b_vec)
    ready = ready_in if ready_in is not None else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.empty(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    tmp_sum.zero_()
    ready.zero_()
    indegree.copy_(indegree_init)
    use_fp64_acc = data.dtype == torch.float64
    grid = (int(data.numel()),)
    _spsv_csr_nnz_balance_kernel[grid](
        launch_order,
        row_idx,
        indices,
        data,
        b_vec,
        x,
        tmp_sum,
        ready,
        indegree,
        int(data.numel()),
        LOWER=lower,
        USE_FP64_ACC=use_fp64_acc,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_n_lo_nnz_balance_vector_complex(
    data,
    indices,
    launch_order,
    row_idx,
    indegree_init,
    b_vec,
    n_rows,
    *,
    lower=True,
    diag_eps=1e-12,
    data_ri_in=None,
    tmp_sum_in=None,
    ready_in=None,
    indegree_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    tmp_sum = tmp_sum_in if tmp_sum_in is not None else torch.zeros_like(b_vec)
    ready = (
        ready_in
        if ready_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.empty(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    tmp_sum.zero_()
    ready.zero_()
    indegree.copy_(indegree_init)
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()
    tmp_sum_ri = torch.view_as_real(tmp_sum.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    grid = (int(data.numel()),)
    _spsv_csr_nnz_balance_kernel_complex[grid](
        launch_order,
        row_idx,
        indices,
        data_ri,
        b_ri,
        x_ri,
        tmp_sum_ri,
        ready,
        indegree,
        int(data.numel()),
        LOWER=lower,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
        num_warps=1,
    )
    return x


def _triton_spsv_csr_transpose_cw_vector(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
    residual_in=None,
    indegree_in=None,
    row_counter_in=None,
    preprocessed=False,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    residual = residual_in if residual_in is not None else b_vec.clone()
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    residual.copy_(b_vec)
    row_counter.zero_()
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    if not preprocessed:
        _run_spsv_csc_preprocess(
            indices,
            indptr,
            indegree,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz_use=block_nnz_use,
            max_segments_use=max_segments_use,
        )
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_transpose_cw_kernel[grid](
        data,
        indices,
        indptr,
        indegree,
        residual,
        x,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_transpose_cw_vector_complex(
    data,
    indices,
    indptr,
    b_vec,
    n_rows,
    lower=True,
    unit_diagonal=False,
    conjugate=False,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
    data_ri_in=None,
    residual_in=None,
    indegree_in=None,
    row_counter_in=None,
    preprocessed=False,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    residual_work = (
        residual_in if residual_in is not None else b_vec.contiguous().clone()
    )
    indegree = (
        indegree_in
        if indegree_in is not None
        else torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    )
    row_counter = (
        row_counter_in
        if row_counter_in is not None
        else torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    )
    residual_work.copy_(b_vec.contiguous())
    row_counter.zero_()
    if not preprocessed:
        _run_spsv_csc_preprocess(
            indices,
            indptr,
            indegree,
            n_rows,
            lower=lower,
            unit_diagonal=unit_diagonal,
            block_nnz_use=block_nnz_use,
            max_segments_use=max_segments_use,
        )
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    residual_ri = torch.view_as_real(residual_work).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    if component_dtype == torch.float16:
        x_ri_work = torch.zeros((n_rows, 2), dtype=torch.float32, device=b_vec.device)
        x_ri = x_ri_work.reshape(-1).contiguous()
    else:
        x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_transpose_cw_kernel_complex[grid](
        data_ri,
        indices,
        indptr,
        indegree,
        residual_ri,
        x_ri,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        REVERSE_ORDER=not lower,
        UNIT_DIAG=unit_diagonal,
        CONJ_TRANS=conjugate,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    if component_dtype == torch.float16:
        return torch.view_as_complex(x_ri_work.contiguous())
    return x


def _choose_transpose_family_launch_config(indptr, block_nnz=None, max_segments=None):
    if block_nnz is not None or max_segments is not None:
        return _auto_spsv_launch_config(indptr, block_nnz=block_nnz, max_segments=max_segments)

    if indptr.numel() <= 1:
        return 32, 1
    max_nnz_per_row = int((indptr[1:] - indptr[:-1]).max().item())
    for cand in (32, 64, 128, 256, 512, 1024):
        req = max((max_nnz_per_row + cand - 1) // cand, 1)
        if req <= 2048:
            return cand, req
    cand = 2048
    req = max((max_nnz_per_row + cand - 1) // cand, 1)
    return cand, req


def _run_spsv_csc_preprocess(
    indices,
    indptr,
    indegree,
    n_rows,
    *,
    lower,
    unit_diagonal,
    block_nnz_use,
    max_segments_use,
):
    indegree.zero_()
    if n_rows == 0:
        return indegree
    grid = (n_rows,)
    _spsv_csc_preprocess_kernel[grid](
        indices,
        indptr,
        indegree,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        UNIT_DIAG=unit_diagonal,
    )
    return indegree


def _prepare_spsv_coo_inputs(data, row, col, b, shape):
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, b)):
        raise ValueError("data, row, col, b must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if row.numel() != data.numel() or col.numel() != data.numel():
        raise ValueError("data, row, col must have the same length")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")

    if data.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise TypeError(
            "data dtype must be one of: float32, float64, complex64, complex128"
        )
    if b.dtype != data.dtype:
        raise TypeError("b dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_SPSV_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    input_index_dtype = (
        torch.int64
        if row.dtype == torch.int64 or col.dtype == torch.int64
        else torch.int32
    )
    row64 = row.to(torch.int64).contiguous()
    col64 = col.to(torch.int64).contiguous()
    if col64.numel() > 0 and int(col64.max().item()) > _INDEX_LIMIT_INT32:
        raise ValueError(
            f"int64 index value {int(col64.max().item())} exceeds Triton int32 kernel range"
        )
    if row64.numel() > 0:
        if bool(torch.any(row64 < 0).item()):
            raise IndexError("row indices must be non-negative")
        if bool(torch.any(col64 < 0).item()):
            raise IndexError("col indices must be non-negative")
        max_row = int(row64.max().item())
        max_col = int(col64.max().item())
        if max_row >= n_rows:
            raise IndexError(f"row indices out of range for n_rows={n_rows}")
        if max_col >= n_cols:
            raise IndexError(f"col indices out of range for n_cols={n_cols}")

    return (
        data.contiguous(),
        input_index_dtype,
        row64,
        col64,
        b.contiguous(),
        n_rows,
        n_cols,
    )


def _build_coo_row_ptr(row_sorted, n_rows):
    row_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=row_sorted.device)
    if row_sorted.numel() > 0:
        nnz_per_row = torch.bincount(row_sorted, minlength=n_rows)
        row_ptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return row_ptr


def _coo_order_for_spsv(data, row64, col64):
    if data.numel() == 0:
        return data, row64, col64
    key = row64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order], row64[order], col64[order]


def _coo2csr_for_spsv(data, row64, col64, n_rows, assume_ordered=False):
    nnz = data.numel()
    if nnz == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        indices = torch.empty(0, dtype=torch.int64, device=data.device)
        return data, indices, indptr

    if not assume_ordered:
        data, row64, col64 = _coo_order_for_spsv(data, row64, col64)

    indptr = _build_coo_row_ptr(row64, n_rows)
    indices = col64.to(torch.int64).contiguous()
    return data.contiguous(), indices, indptr


def _analyze_spsv_csr_descriptor(
    data,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
    format_name="csr",
    clear_cache=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    n_rows = int(shape[0])
    dummy_b = torch.empty(n_rows, dtype=data.dtype, device=data.device)
    (
        matrix_data,
        _dummy_b,
        _original_output_dtype,
        trans_mode,
        n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        dummy_b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        requested_solve_kind=solve_kind,
        storage_view=storage_view,
    )
    input_index_dtype = indices.dtype
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    compute_dtype = _spsv_effective_compute_dtype(
        matrix_data.dtype, trans_mode, compute_dtype=compute_dtype
    )
    layout = _build_spsv_workspace_layout(
        n_rows, solve_plan["solve_kind"], value_dtype=compute_dtype
    )
    if workspace is not None:
        _resolve_spsv_workspace(workspace, layout, matrix_data.device)
    return FlagSparseSpSVDescr(
        format=_normalize_spsv_format(format_name),
        canonical_format="csr",
        shape=(int(shape[0]), int(shape[1])),
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
        fill_mode="lower" if lower else "upper",
        diag_type="unit" if unit_diagonal else "non_unit",
        matrix_type="triangular",
        index_base=0,
        transpose_mode=trans_mode,
        value_dtype=matrix_data.dtype,
        compute_dtype=compute_dtype,
        index_dtype=input_index_dtype,
        solve_kind=solve_plan["solve_kind"],
        route_name=str(solve_plan.get("route_name", solve_plan["solve_kind"])),
        storage_view=str(solve_plan.get("storage_view", "csr")),
        buffer_size=_workspace_size_bytes(layout),
        workspace_layout=layout,
        data=matrix_data,
        indices=indices.contiguous(),
        indptr=indptr.contiguous(),
        solve_plan=_clone_spsv_plan(solve_plan),
    )


def flagsparse_spsv_analysis_csr(
    data,
    indices,
    indptr,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
    clear_cache=False,
):
    """Analyze a CSR SpSV problem and return a reusable Triton descriptor."""

    return _analyze_spsv_csr_descriptor(
        data,
        indices,
        indptr,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        handle=handle,
        workspace=workspace,
        storage_view=storage_view,
        format_name="csr",
        clear_cache=clear_cache,
    )


def _analyze_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    (
        _data,
        _b,
        _original_output_dtype,
        trans_mode,
        _n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        requested_solve_kind=solve_kind,
    )
    _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    if return_time:
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0


def flagsparse_spsv_analysis_coo(
    data,
    row,
    col,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    solve_kind=None,
    compute_dtype=None,
    handle=None,
    workspace=None,
    storage_view="csr_as_csc",
):
    """Analyze a COO SpSV problem by canonicalizing COO into CSR first."""

    dummy_b = torch.empty(int(shape[0]), dtype=data.dtype, device=data.device)
    data, _input_index_dtype, row64, col64, _b, n_rows, _n_cols = _prepare_spsv_coo_inputs(
        data, row, col, dummy_b, shape
    )
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, row.dtype, "COO")
    else:
        _validate_spsv_trans_combo(data.dtype, row.dtype, "COO")
    data_csr, indices_csr, indptr_csr = _coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    return _analyze_spsv_csr_descriptor(
        data_csr,
        indices_csr,
        indptr_csr,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        handle=handle,
        workspace=workspace,
        storage_view=storage_view,
        format_name="coo",
        clear_cache=False,
    )


def _execute_spsv_csr_plan(
    data,
    b,
    solve_plan,
    trans_mode,
    n_rows,
    *,
    alpha=1,
    unit_diagonal=False,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    original_output_dtype=None,
    compute_dtype=None,
    handle=None,
    stream=None,
):
    solve_plan = _clone_spsv_plan(solve_plan)
    solve_kind = solve_plan["solve_kind"]
    kernel_data = solve_plan["kernel_data"]
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    default_block_nnz = solve_plan["default_block_nnz"]
    default_max_segments = solve_plan["default_max_segments"]
    cw_worker_count = solve_plan.get("cw_worker_count")
    nontrans_variant = solve_plan.get("nontrans_variant", "csr_n_lo_cw")
    lower_eff = solve_plan["lower_eff"]
    matrix_stats = solve_plan.get("matrix_stats", {})
    level_row_map32 = solve_plan.get("level_row_map32")
    nnz_balance_row_idx32 = solve_plan.get("nnz_balance_row_idx32")
    nnz_balance_indegree32 = solve_plan.get("nnz_balance_indegree32")
    nnz_balance_launch_order32 = solve_plan.get("nnz_balance_launch_order32")
    kernel_indices = kernel_indices32
    kernel_indptr = kernel_indptr64
    compute_dtype = _spsv_effective_compute_dtype(
        data.dtype, trans_mode, compute_dtype=compute_dtype
    )
    data_in = kernel_data
    b_in = b
    if compute_dtype != data.dtype:
        data_in = kernel_data.to(compute_dtype)
        b_in = b.to(compute_dtype)
    alpha_in = _coerce_spsv_alpha(alpha, compute_dtype, b.device)
    if not _spsv_alpha_is_identity(alpha):
        b_in = b_in * alpha_in
    solve_stream = _resolve_spsv_stream(handle, stream, b.device)

    if solve_kind == "transpose_cw":
        if block_nnz is None and max_segments is None:
            block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        else:
            block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
                kernel_indptr, block_nnz=block_nnz, max_segments=max_segments
            )
        vec_real = _triton_spsv_csr_transpose_cw_vector
        vec_complex = _triton_spsv_csr_transpose_cw_vector_complex
    elif solve_kind == "csr_cw":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        nontrans_real_wrappers = {
            "csr_u_lo_cw": _triton_spsv_csr_u_lo_cw_vector,
            "csr_n_lo_cw": _triton_spsv_csr_n_lo_cw_vector,
            "csr_u_up_cw": _triton_spsv_csr_u_up_cw_vector,
            "csr_n_up_cw": _triton_spsv_csr_n_up_cw_vector,
        }
        nontrans_complex_wrappers = {
            "csr_u_lo_cw": _triton_spsv_csr_u_lo_cw_vector_complex,
            "csr_n_lo_cw": _triton_spsv_csr_n_lo_cw_vector_complex,
            "csr_u_up_cw": _triton_spsv_csr_u_up_cw_vector_complex,
            "csr_n_up_cw": _triton_spsv_csr_n_up_cw_vector_complex,
        }
        vec_real = nontrans_real_wrappers[nontrans_variant]
        vec_complex = nontrans_complex_wrappers[nontrans_variant]
    elif solve_kind == "csr_roc":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_roc_vector
        vec_complex = _triton_spsv_csr_n_lo_roc_vector_complex
    elif solve_kind == "csr_smblk":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_smblk_vector
        vec_complex = _triton_spsv_csr_n_lo_smblk_vector_complex
    elif solve_kind == "csr_cw_levelschd":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_cw_levelschd_vector
        vec_complex = _triton_spsv_csr_n_lo_cw_levelschd_vector_complex
    elif solve_kind == "csr_nnz_balance":
        block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        vec_real = _triton_spsv_csr_n_lo_nnz_balance_vector
        vec_complex = _triton_spsv_csr_n_lo_nnz_balance_vector_complex
    else:
        raise RuntimeError(f"unexpected SpSV solve kind: {solve_kind}")
    diag_eps = _spsv_diag_eps_for_dtype(compute_dtype)

    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    worker_count_use = cw_worker_count
    matrix_stats_use = dict(matrix_stats)
    if solve_kind in ("csr_cw", "transpose_cw"):
        worker_count_use = _resolve_cw_worker_count(
            n_rows,
            matrix_stats_use,
            1,
            cached_worker_count=cw_worker_count,
        )
    complex_kernel_data_ri = None
    if torch.is_complex(data_in):
        if compute_dtype == solve_plan["kernel_data"].dtype:
            complex_kernel_data_ri = solve_plan.get("kernel_data_ri")
        if complex_kernel_data_ri is None:
            complex_kernel_data_ri = _complex_interleaved_view(data_in)
    workspace_buffers = _resolve_spsv_workspace(
        workspace,
        _build_spsv_workspace_layout(n_rows, solve_kind, value_dtype=compute_dtype),
        b.device,
    )
    ready_buf = workspace_buffers.get("ready")
    tmp_sum_buf = workspace_buffers.get("tmp_sum")
    residual_buf = workspace_buffers.get("residual")
    indegree_buf = workspace_buffers.get("indegree")
    row_counter_buf = workspace_buffers.get("row_counter")
    transpose_preprocessed = False
    if solve_kind == "csr_nnz_balance":
        if tmp_sum_buf is None or ready_buf is None or indegree_buf is None:
            raise RuntimeError("csr_nnz_balance workspace is missing required buffers")
        tmp_sum_buf.zero_()
        ready_buf.zero_()
        indegree_buf.copy_(nnz_balance_indegree32)
    if solve_kind == "transpose_cw":
        if residual_buf is None or indegree_buf is None or row_counter_buf is None:
            raise RuntimeError("transpose_cw workspace is missing required buffers")
        transpose_sig = _transpose_cw_preprocess_signature(
            solve_plan,
            n_rows,
            unit_diagonal,
            block_nnz_use,
            max_segments_use,
        )
        preprocess_stream_ctx = (
            torch.cuda.stream(solve_stream)
            if solve_stream is not None
            else nullcontext()
        )
        with preprocess_stream_ctx:
            _run_spsv_csc_preprocess(
                kernel_indices,
                kernel_indptr,
                indegree_buf,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
            )
        transpose_preprocessed = True
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = "transpose_cw"
            workspace.prepared_signature = transpose_sig
    stream_ctx = (
        torch.cuda.stream(solve_stream)
        if solve_stream is not None
        else nullcontext()
    )
    with stream_ctx:
        if torch.is_complex(data_in):
            if vec_complex is None:
                raise ValueError(f"solve_kind={solve_kind!r} currently supports real dtypes only")
            if solve_kind == "transpose_cw":
                x = vec_complex(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                conjugate=(trans_mode == "C"),
                block_nnz=block_nnz,
                max_segments=max_segments,
                diag_eps=diag_eps,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                data_ri_in=complex_kernel_data_ri,
                residual_in=residual_buf,
                indegree_in=indegree_buf,
                row_counter_in=row_counter_buf,
                preprocessed=transpose_preprocessed,
                )
            else:
                if solve_kind == "csr_roc":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    level_row_map32,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    )
                elif solve_kind == "csr_smblk":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    )
                elif solve_kind == "csr_cw_levelschd":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    level_row_map32,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    )
                elif solve_kind == "csr_nnz_balance":
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    nnz_balance_launch_order32,
                    nnz_balance_row_idx32,
                    nnz_balance_indegree32,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    diag_eps=diag_eps,
                    data_ri_in=complex_kernel_data_ri,
                    tmp_sum_in=tmp_sum_buf,
                    ready_in=ready_buf,
                    indegree_in=indegree_buf,
                    )
                else:
                    x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    b_in,
                    n_rows,
                    diag_eps=diag_eps,
                    worker_count=worker_count_use,
                    matrix_stats=matrix_stats_use,
                    data_ri_in=complex_kernel_data_ri,
                    ready_in=ready_buf,
                    row_counter_in=row_counter_buf,
                    )
        else:
            if solve_kind == "transpose_cw":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                unit_diagonal=unit_diagonal,
                block_nnz=block_nnz,
                max_segments=max_segments,
                diag_eps=diag_eps,
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                residual_in=residual_buf,
                indegree_in=indegree_buf,
                row_counter_in=row_counter_buf,
                preprocessed=transpose_preprocessed,
                )
            elif solve_kind == "csr_roc":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                level_row_map32,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                ready_in=ready_buf,
                )
            elif solve_kind == "csr_smblk":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                ready_in=ready_buf,
                )
            elif solve_kind == "csr_cw_levelschd":
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                level_row_map32,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                ready_in=ready_buf,
                )
            elif solve_kind == "csr_nnz_balance":
                x = vec_real(
                data_in,
                kernel_indices,
                nnz_balance_launch_order32,
                nnz_balance_row_idx32,
                nnz_balance_indegree32,
                b_in,
                n_rows,
                lower=lower_eff,
                diag_eps=diag_eps,
                tmp_sum_in=tmp_sum_buf,
                ready_in=ready_buf,
                indegree_in=indegree_buf,
                )
            else:
                x = vec_real(
                data_in,
                kernel_indices,
                kernel_indptr,
                b_in,
                n_rows,
                diag_eps=diag_eps,
                worker_count=worker_count_use,
                matrix_stats=matrix_stats_use,
                ready_in=ready_buf,
                row_counter_in=row_counter_buf,
                )
    target_dtype = original_output_dtype if original_output_dtype is not None else data.dtype
    if x.dtype != target_dtype:
        x = x.to(target_dtype)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out

    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsv_solve_csr(
    descr,
    b,
    *,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    handle=None,
    stream=None,
):
    """Solve a previously analyzed CSR SpSV problem."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if descr.canonical_format != "csr":
        raise ValueError("descr must reference a CSR-canonicalized SpSV analysis")
    if not torch.is_tensor(b):
        raise TypeError("b must be a torch.Tensor")
    if not b.is_cuda:
        raise ValueError("b must be a CUDA tensor")
    if b.ndim != 1:
        raise ValueError("b must be a 1D dense vector (DnVec)")
    if int(b.numel()) != int(descr.shape[0]):
        raise ValueError(f"b length must equal n_rows={descr.shape[0]}")
    if b.dtype != descr.value_dtype:
        raise TypeError("b dtype must match the analyzed matrix dtype")
    return _execute_spsv_csr_plan(
        descr.data,
        b.contiguous(),
        descr.solve_plan,
        descr.transpose_mode,
        int(descr.shape[0]),
        alpha=alpha,
        unit_diagonal=descr.unit_diagonal,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        original_output_dtype=descr.value_dtype,
        compute_dtype=compute_dtype if compute_dtype is not None else descr.compute_dtype,
        handle=handle,
        stream=stream,
    )


def flagsparse_spsv_solve_coo(
    descr,
    b,
    *,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    workspace=None,
    handle=None,
    stream=None,
):
    """Solve a previously analyzed COO SpSV problem via its CSR canonical form."""

    return flagsparse_spsv_solve_csr(
        descr,
        b,
        alpha=alpha,
        compute_dtype=compute_dtype,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        handle=handle,
        stream=stream,
    )


def _materialize_spsv_workspace_state(descr, workspace=None):
    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    buffers = _resolve_spsv_workspace(
        workspace, descr.workspace_layout, descr.data.device
    )
    solve_kind = descr.solve_kind
    preprocess_sig = None
    if solve_kind in {
        "csr_cw",
        "csr_roc",
        "csr_smblk",
        "csr_cw_levelschd",
        "csr_nnz_balance",
    }:
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = ""
            workspace.prepared_signature = None
    elif solve_kind == "transpose_cw":
        residual = buffers.get("residual")
        indegree = buffers.get("indegree")
        row_counter = buffers.get("row_counter")
        block_nnz_use = int(descr.solve_plan["default_block_nnz"])
        max_segments_use = int(descr.solve_plan["default_max_segments"])
        preprocess_sig = _transpose_cw_preprocess_signature(
            descr.solve_plan,
            int(descr.shape[0]),
            bool(descr.unit_diagonal),
            block_nnz_use,
            max_segments_use,
        )
        if residual is not None:
            residual.zero_()
        if indegree is not None:
            _run_spsv_csc_preprocess(
                descr.solve_plan["kernel_indices32"],
                descr.solve_plan["kernel_indptr64"],
                indegree,
                int(descr.shape[0]),
                lower=bool(descr.solve_plan["lower_eff"]),
                unit_diagonal=bool(descr.unit_diagonal),
                block_nnz_use=block_nnz_use,
                max_segments_use=max_segments_use,
            )
        if row_counter is not None:
            row_counter.zero_()
        if isinstance(workspace, FlagSparseSpSVWorkspace):
            workspace.prepared_solve_kind = "transpose_cw"
            workspace.prepared_signature = preprocess_sig
    else:
        raise RuntimeError(f"unexpected SpSV solve kind: {solve_kind}")
    if workspace is None:
        return FlagSparseSpSVWorkspace(
            buffer_size=int(descr.buffer_size),
            layout=tuple(descr.workspace_layout),
            device=descr.data.device,
            buffers=buffers,
            prepared_solve_kind=(
                "transpose_cw" if solve_kind == "transpose_cw" else ""
            ),
            prepared_signature=preprocess_sig,
        )
    return workspace


def flagsparse_spsv_preprocess_csr(descr, *, workspace=None):
    """Materialize caller-managed workspace for a CSR SpSV descriptor."""

    return _materialize_spsv_workspace_state(descr, workspace=workspace)


def flagsparse_spsv_preprocess_coo(descr, *, workspace=None):
    """Materialize caller-managed workspace for a COO SpSV descriptor."""

    return _materialize_spsv_workspace_state(descr, workspace=workspace)


def flagsparse_spsv_buffer_size_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    storage_view="csr_as_csc",
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    return flagsparse_spsv_buffer_size(
        matA.shape,
        matA.values.dtype,
        format=matA.format,
        transpose=opA,
        solve_kind=solve_kind,
        compute_dtype=compute_dtype,
        alpha=alpha,
        handle=handle,
        vecX=vecX,
        vecY=vecY,
        storage_view=storage_view,
    )


def flagsparse_spsv_analysis_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    workspace=None,
    storage_view="csr_as_csc",
    clear_cache=False,
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    if matA.format == "csr":
        return flagsparse_spsv_analysis_csr(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            solve_kind=solve_kind,
            compute_dtype=compute_dtype,
            handle=handle,
            workspace=workspace,
            storage_view=storage_view,
            clear_cache=clear_cache,
        )
    if matA.format == "coo":
        return flagsparse_spsv_analysis_coo(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            solve_kind=solve_kind,
            compute_dtype=compute_dtype,
            handle=handle,
            workspace=workspace,
            storage_view=storage_view,
        )
    raise ValueError("matA.format must be 'csr' or 'coo'")


def flagsparse_spsv_solve_ex(
    handle,
    opA,
    alpha,
    matA,
    vecX,
    vecY=None,
    descr=None,
    *,
    compute_dtype=None,
    solve_kind=None,
    workspace=None,
    stream=None,
    storage_view="csr_as_csc",
    block_nnz=None,
    max_segments=None,
    return_time=False,
):
    if not isinstance(matA, FlagSparseSpMatDescr):
        raise TypeError("matA must be a FlagSparseSpMatDescr")
    if not isinstance(vecX, FlagSparseDnVecDescr):
        raise TypeError("vecX must be a FlagSparseDnVecDescr")
    out_tensor = None if vecY is None else vecY.values
    if matA.format == "csr":
        return flagsparse_spsv_csr(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            vecX.values,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out_tensor,
            return_time=return_time,
            descr=descr,
            workspace=workspace,
            solve_kind=solve_kind,
            handle=handle,
            stream=stream,
            storage_view=storage_view,
        )
    if matA.format == "coo":
        return flagsparse_spsv_coo(
            matA.values,
            matA.indices,
            matA.indptr_or_col,
            vecX.values,
            matA.shape,
            lower=matA.lower,
            unit_diagonal=matA.unit_diagonal,
            transpose=opA,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out_tensor,
            return_time=return_time,
            descr=descr,
            workspace=workspace,
            solve_kind=solve_kind,
            handle=handle,
            stream=stream,
            storage_view=storage_view,
        )
    raise ValueError("matA.format must be 'csr' or 'coo'")


def flagsparse_spsv_analysis_sell(
    values,
    col_indices,
    slice_offsets,
    shape,
    *,
    slice_size,
    alg_num=None,
    alg2_worker_count=None,
    unit_diagonal=False,
    transpose=False,
    workspace=None,
):
    """Analyze a lower SELL SpSV problem and return a descriptor.

    When ``unit_diagonal`` is true, stored diagonal entries are ignored and
    missing diagonal entries are interpreted as one, matching SpSV descriptor
    semantics used by cuSPARSE.  ``transpose`` accepts the same N/T/C forms as
    the CSR API; T solves A^T x=b and C solves A^H x=b.
    """

    unit_diagonal = bool(unit_diagonal)
    trans_mode = _normalize_spsv_transpose_mode(transpose)
    trans_analysis_key = None
    trans_analysis = None
    if trans_mode != "N":
        trans_analysis_key = _spsv_sell_trans_analysis_signature(
            values,
            col_indices,
            slice_offsets,
            shape,
            slice_size,
            unit_diagonal,
        )
        if trans_analysis_key is not None:
            trans_analysis = _spsv_cache_get(
                _SPSV_SELL_TRANS_ANALYSIS_CACHE,
                trans_analysis_key,
            )
    values, cols, offsets, n_rows, slice_size = (
        _prepare_spsv_sell_matrix_inputs(
            values,
            col_indices,
            slice_offsets,
            shape,
            slice_size,
            unit_diagonal=unit_diagonal,
            skip_validated_checks=trans_analysis is not None,
        )
    )
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(values.dtype, cols.dtype, "SELL")
    else:
        _validate_spsv_trans_combo(values.dtype, cols.dtype, "SELL")
    if trans_mode == "N":
        alg_num = (
            SPSV_SELL_ALG1
            if alg_num is None
            else _normalize_spsv_sell_alg_num(alg_num)
        )
        n_slices = (n_rows + slice_size - 1) // slice_size
        if alg_num == SPSV_SELL_ALG1:
            if alg2_worker_count is not None:
                raise ValueError("alg2_worker_count is only valid for SELL SpSV ALG2")
            resolved_workers = None
        else:
            resolved_workers = _resolve_spsv_sell_alg2_worker_count(
                n_slices, requested=alg2_worker_count
            )
        solve_kind = f"sell_alg{alg_num}"
    else:
        if alg_num is not None:
            raise ValueError("SELL TRANS/CONJ do not accept alg_num")
        if alg2_worker_count is not None:
            raise ValueError("SELL TRANS/CONJ do not accept worker-count tuning")
        if trans_analysis is None:
            indegree_init = torch.zeros(
                n_rows,
                dtype=torch.int32,
                device=values.device,
            )
            diag_offsets = torch.empty(
                n_rows,
                dtype=torch.int64,
                device=values.device,
            )
            row_widths = torch.empty(
                n_rows,
                dtype=torch.int32,
                device=values.device,
            )
            if n_rows > 0:
                _spsv_sell_trans_analysis_kernel[(n_rows,)](
                    cols,
                    offsets,
                    indegree_init,
                    diag_offsets,
                    row_widths,
                    n_rows,
                    SLICE_SIZE=int(slice_size),
                    num_warps=1,
                )
            trans_analysis = {
                "indegree_init": indegree_init,
                "diag_offsets": diag_offsets,
                "row_widths": row_widths,
            }
            if trans_analysis_key is not None:
                _spsv_cache_put(
                    _SPSV_SELL_TRANS_ANALYSIS_CACHE,
                    trans_analysis_key,
                    trans_analysis,
                    _SPSV_SELL_TRANS_ANALYSIS_CACHE_SIZE,
                )
        solve_kind = "sell_trans"
    layout = _build_spsv_workspace_layout(
        n_rows, solve_kind, value_dtype=values.dtype
    )
    if workspace is not None:
        _resolve_spsv_workspace(workspace, layout, values.device)
    solve_plan = {
        "solve_kind": solve_kind,
        "kernel_data": values,
        "slice_size": int(slice_size),
    }
    if trans_mode == "N":
        solve_plan["alg_num"] = int(alg_num)
        solve_plan["alg2_worker_count"] = resolved_workers
    else:
        solve_plan["indegree_init"] = trans_analysis["indegree_init"]
        solve_plan["diag_offsets"] = trans_analysis["diag_offsets"]
        solve_plan["row_widths"] = trans_analysis["row_widths"]
    _attach_spsv_complex_plan_views(solve_plan)
    return FlagSparseSpSVDescr(
        format="sell",
        canonical_format="sell",
        shape=(n_rows, n_rows),
        lower=True,
        unit_diagonal=unit_diagonal,
        fill_mode="lower",
        diag_type="unit" if unit_diagonal else "non_unit",
        matrix_type="triangular",
        index_base=0,
        transpose_mode=trans_mode,
        value_dtype=values.dtype,
        compute_dtype=values.dtype,
        index_dtype=cols.dtype,
        solve_kind=solve_kind,
        route_name=solve_kind,
        storage_view="sell",
        buffer_size=_workspace_size_bytes(layout),
        workspace_layout=layout,
        data=values,
        indices=cols,
        indptr=offsets,
        solve_plan=solve_plan,
    )


def flagsparse_spsv_solve_sell(
    descr,
    b,
    *,
    out=None,
    workspace=None,
):
    """Solve a previously analyzed lower SELL SpSV problem."""

    if not isinstance(descr, FlagSparseSpSVDescr):
        raise TypeError("descr must be a FlagSparseSpSVDescr")
    if descr.canonical_format != "sell":
        raise ValueError("descr must reference a SELL SpSV analysis")
    if not torch.is_tensor(b):
        raise TypeError("b must be a torch.Tensor")
    if not b.is_cuda or b.ndim != 1:
        raise ValueError("b must be a 1D CUDA tensor")
    if b.device != descr.data.device:
        raise ValueError("b device must match the analyzed SELL matrix")
    if b.dtype != descr.value_dtype:
        raise TypeError("b dtype must match the analyzed SELL matrix")
    n_rows = int(descr.shape[0])
    if int(b.numel()) != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")
    if out is None:
        out = torch.empty_like(b)
    elif (
        not torch.is_tensor(out)
        or not out.is_cuda
        or out.ndim != 1
        or not out.is_contiguous()
        or out.device != b.device
        or out.dtype != b.dtype
        or int(out.numel()) != n_rows
    ):
        raise ValueError("out must be a matching contiguous 1D CUDA tensor")
    if out.is_conj() or out.is_neg():
        raise ValueError("out must not be an unresolved conjugate/negative view")
    buffers = _resolve_spsv_workspace(
        workspace, descr.workspace_layout, descr.data.device
    )
    plan = descr.solve_plan
    if descr.transpose_mode in ("T", "C"):
        return _launch_spsv_sell_trans(
            descr.data,
            descr.indices,
            descr.indptr,
            plan["indegree_init"],
            plan["diag_offsets"],
            plan["row_widths"],
            _as_strided_contiguous(b),
            n_rows,
            slice_size=int(plan["slice_size"]),
            out=out,
            residual=buffers.get("residual"),
            indegree=buffers.get("indegree"),
            row_counter=buffers.get("row_counter"),
            unit_diagonal=bool(descr.unit_diagonal),
            conj_trans=descr.transpose_mode == "C",
            values_ri_in=plan.get("kernel_data_ri"),
        )
    return _launch_spsv_sell(
        descr.data,
        descr.indices,
        descr.indptr,
        _as_strided_contiguous(b),
        n_rows,
        slice_size=int(plan["slice_size"]),
        out=out,
        ready=buffers.get("ready"),
        row_counter=buffers.get("row_counter"),
        alg_num=int(plan["alg_num"]),
        alg2_worker_count=plan.get("alg2_worker_count"),
        unit_diagonal=bool(descr.unit_diagonal),
        values_ri_in=plan.get("kernel_data_ri"),
    )


def flagsparse_spsv_sell(
    values,
    col_indices,
    slice_offsets,
    b,
    shape,
    *,
    slice_size,
    alg_num=None,
    alg2_worker_count=None,
    unit_diagonal=False,
    transpose=False,
    out=None,
    workspace=None,
):
    """Analyze and solve a lower SELL SpSV problem."""

    descr = flagsparse_spsv_analysis_sell(
        values,
        col_indices,
        slice_offsets,
        shape,
        slice_size=slice_size,
        alg_num=alg_num,
        alg2_worker_count=alg2_worker_count,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        workspace=workspace,
    )
    return flagsparse_spsv_solve_sell(descr, b, out=out, workspace=workspace)


def flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    descr=None,
    workspace=None,
    solve_kind=None,
    handle=None,
    stream=None,
    storage_view="csr_as_csc",
):
    """Sparse triangular solve using Triton CSR CW kernels.

    Current support matrix:
    - NON_TRANS: float32/float64/complex64/complex128 with int32/int64 indices
    - TRANS/CONJ: float32/float64/complex64/complex128 with int32/int64 indices
    """
    if descr is not None:
        if not isinstance(descr, FlagSparseSpSVDescr):
            raise TypeError("descr must be a FlagSparseSpSVDescr or None")
        return flagsparse_spsv_solve_csr(
            descr,
            b,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            workspace=workspace,
            handle=handle,
            stream=stream,
        )
    (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        _n_cols,
        solve_plan,
    ) = _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
        requested_solve_kind=solve_kind,
        storage_view=storage_view,
    )
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, trans_mode, requested_solve_kind=solve_kind
    )
    return _execute_spsv_csr_plan(
        data,
        b,
        solve_plan,
        trans_mode,
        n_rows,
        alpha=alpha,
        unit_diagonal=unit_diagonal,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        original_output_dtype=original_output_dtype,
        compute_dtype=compute_dtype,
        handle=handle,
        stream=stream,
    )

def flagsparse_spsv_coo(
    data,
    row,
    col,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    alpha=1,
    compute_dtype=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
    descr=None,
    workspace=None,
    solve_kind=None,
    handle=None,
    stream=None,
    storage_view="csr_as_csc",
):
    """COO SpSV by canonicalizing COO into CSR, then reusing CSR SpSV."""
    if descr is not None:
        if not isinstance(descr, FlagSparseSpSVDescr):
            raise TypeError("descr must be a FlagSparseSpSVDescr or None")
        return flagsparse_spsv_solve_coo(
            descr,
            b,
            alpha=alpha,
            compute_dtype=compute_dtype,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            workspace=workspace,
            handle=handle,
            stream=stream,
        )
    data, input_index_dtype, row64, col64, b, n_rows, n_cols = _prepare_spsv_coo_inputs(
        data, row, col, b, shape
    )
    if n_rows != n_cols:
        raise ValueError(f"A must be square, got shape={shape}")

    trans_mode = _normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        _validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "COO")
    else:
        _validate_spsv_trans_combo(data.dtype, input_index_dtype, "COO")
    data_csr, indices_csr, indptr_csr = _coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    return flagsparse_spsv_csr(
        data_csr,
        indices_csr,
        indptr_csr,
        b,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        alpha=alpha,
        compute_dtype=compute_dtype,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        workspace=workspace,
        solve_kind=solve_kind,
        handle=handle,
        stream=stream,
        storage_view=storage_view,
    )
