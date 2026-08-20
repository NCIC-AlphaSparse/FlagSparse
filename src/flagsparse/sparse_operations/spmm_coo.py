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

"""Native COO SpMM kernels, route helpers, and internal benchmark entry points."""

import ctypes
from dataclasses import dataclass

from . import _common as _common_mod
from ._common import *
from .spmm_csr import (
    SUPPORTED_SPMM_VALUE_DTYPES,
    _select_spmm_alg1_warp_and_factor,
    _spmm_coo_reference_tolerance,
    _spmm_relative_threshold,
    _spmm_validation_metrics,
)

SPMM_COO_OP_NON = 0
SPMM_COO_OP_TRANS = 1
SPMM_COO_OP_CONJ_TRANS = 2
SPMM_COO_OP_NAMES = {
    SPMM_COO_OP_NON: "non",
    SPMM_COO_OP_TRANS: "trans",
    SPMM_COO_OP_CONJ_TRANS: "conj",
}
_SPMM_COO_OP_NAME_TO_CODE = {name: code for code, name in SPMM_COO_OP_NAMES.items()}

HipPointer = _common_mod.HipPointer


def _normalize_spmm_coo_op(op=None, transpose=False):
    if op is None:
        return SPMM_COO_OP_TRANS if bool(transpose) else SPMM_COO_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMM_COO_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMM_COO_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMM_COO_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmm_coo_op_to_name(op):
    return SPMM_COO_OP_NAMES[_normalize_spmm_coo_op(op)]


def _spmm_coo_op_transposes(op):
    return _normalize_spmm_coo_op(op) in (SPMM_COO_OP_TRANS, SPMM_COO_OP_CONJ_TRANS)


def _materialize_spmm_coo_op(data, row, col, shape, op_code):
    if op_code == SPMM_COO_OP_NON:
        return data, row, col, shape
    data_op = data
    if op_code == SPMM_COO_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
        data_op = data.conj()
        if hasattr(data_op, "resolve_conj"):
            data_op = data_op.resolve_conj()
    return data_op, col, row, (int(shape[1]), int(shape[0]))


def _normalize_dense_layout(layout):
    token = "row" if layout is None else str(layout).strip().lower()
    if token in ("auto", "default"):
        return "row"
    if token in ("row", "row_major", "row-major", "c", "c_order"):
        return "row"
    if token in (
        "col",
        "column",
        "col_major",
        "column_major",
        "col-major",
        "column-major",
        "f",
        "fortran",
    ):
        return "col"
    raise ValueError("dense_layout must be one of: auto, row, col")


def _is_col_major_2d(tensor):
    return (
        torch.is_tensor(tensor)
        and tensor.ndim == 2
        and tensor.stride(0) == 1
        and tensor.stride(1) >= max(1, int(tensor.shape[0]))
    )


def _dense_layout_name(tensor):
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        return "unknown"
    if tensor.is_contiguous():
        return "row"
    if _is_col_major_2d(tensor):
        return "col"
    return "strided"


def _empty_dense_layout(shape, dtype, device, layout):
    layout = _normalize_dense_layout(layout)
    rows, cols = int(shape[0]), int(shape[1])
    if layout == "col":
        return torch.empty_strided(
            (rows, cols),
            (1, max(1, rows)),
            dtype=dtype,
            device=device,
        )
    return torch.empty((rows, cols), dtype=dtype, device=device)


def _zeros_dense_layout(shape, dtype, device, layout):
    out = _empty_dense_layout(shape, dtype, device, layout)
    out.zero_()
    return out


def _materialize_dense_layout(tensor, layout):
    layout = _normalize_dense_layout(layout)
    if tensor.ndim != 2:
        raise ValueError("dense layout materialization expects a 2D tensor")
    if layout == "row":
        return tensor.contiguous()
    if _is_col_major_2d(tensor):
        return tensor
    out = _empty_dense_layout(tensor.shape, tensor.dtype, tensor.device, layout)
    out.copy_(tensor)
    return out


def _spmm_coo_compute_dtype(value_dtype):
    if _is_complex_dtype(value_dtype):
        return torch.complex128 if value_dtype == torch.complex64 else value_dtype
    if value_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if value_dtype == torch.float32:
        return torch.float64
    return value_dtype


def _sort_coo_lex_inplace(data, row, col, n_cols):
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    if data.numel() == 0:
        return data.contiguous(), row64, col64
    key = row64 * max(1, int(n_cols)) + col64
    order = torch.argsort(key)
    return (
        data[order].contiguous(),
        row64[order].contiguous(),
        col64[order].contiguous(),
    )


def _coalesce_coo_entries(data, row, col, shape):
    """Merge duplicate (row, col) by summing values (PyTorch COO coalesce)."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        z = torch.empty(0, dtype=torch.int64, device=data.device)
        return data.contiguous(), z, z.clone()
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    coo = torch.sparse_coo_tensor(
        torch.stack([row64, col64]),
        data,
        size=(n_rows, n_cols),
        device=data.device,
        dtype=data.dtype,
    ).coalesce()
    idx = coo.indices()
    return coo.values().contiguous(), idx[0].contiguous(), idx[1].contiguous()


def _build_torch_sparse_coo(data, row, col, shape):
    """Coalesced CUDA COO tensor for ``torch.sparse.mm`` (indices int64)."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        empty_idx = torch.empty((2, 0), dtype=torch.int64, device=data.device)
        return torch.sparse_coo_tensor(
            empty_idx,
            data,
            size=(n_rows, n_cols),
            device=data.device,
            dtype=data.dtype,
        )
    row_i = row.to(torch.int64)
    col_i = col.to(torch.int64)
    indices = torch.stack([row_i, col_i])
    return torch.sparse_coo_tensor(
        indices,
        data,
        size=(n_rows, n_cols),
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _build_random_coo(n_rows, n_cols, nnz, value_dtype, index_dtype, device):
    nnz = int(nnz)
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if int(n_rows) < 0 or int(n_cols) < 0:
        raise ValueError("matrix dimensions must be non-negative")
    if index_dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("index_dtype must be torch.int32 or torch.int64")
    if value_dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError("value_dtype is not supported by COO SpMM")

    data = _build_random_dense(nnz, value_dtype, device)
    if nnz == 0:
        row = torch.empty((0,), dtype=index_dtype, device=device)
        col = torch.empty((0,), dtype=index_dtype, device=device)
        return data, row, col
    if n_rows == 0 or n_cols == 0:
        raise ValueError("nnz must be 0 when either matrix dimension is zero")
    row = torch.randint(0, int(n_rows), (nnz,), dtype=index_dtype, device=device)
    col = torch.randint(0, int(n_cols), (nnz,), dtype=index_dtype, device=device)
    return data, row, col


def _prepare_spmm_coo_canonical_prepared(
    data,
    row,
    col,
    B,
    n_rows,
    n_cols,
    n_dense_cols,
    dense_layout="row",
):
    dense_layout = _normalize_dense_layout(dense_layout)
    output_dtype = data.dtype
    compute_dtype = _spmm_coo_compute_dtype(output_dtype)
    data_compute = data if compute_dtype == output_dtype else data.to(compute_dtype)
    B_compute = B if compute_dtype == output_dtype else B.to(compute_dtype)
    B_compute = _materialize_dense_layout(B_compute, dense_layout)
    canonical_data, canonical_row, canonical_col = _coalesce_coo_entries(
        data_compute,
        row,
        col,
        (n_rows, n_cols),
    )
    canonical_data, canonical_row, canonical_col = _sort_coo_lex_inplace(
        canonical_data,
        canonical_row,
        canonical_col,
        n_cols,
    )
    return (
        canonical_data,
        canonical_row,
        canonical_col,
        B_compute,
        n_rows,
        n_cols,
        n_dense_cols,
        output_dtype,
        compute_dtype,
    )


def _prepare_spmm_coo_canonical_inputs(data, row, col, B, shape, dense_layout="row"):
    data, kernel_row, kernel_col, B, n_rows, n_cols, n_dense_cols = (
        _prepare_spmm_coo_inputs(data, row, col, B, shape, dense_layout=dense_layout)
    )
    return _prepare_spmm_coo_canonical_prepared(
        data,
        kernel_row,
        kernel_col,
        B,
        n_rows,
        n_cols,
        n_dense_cols,
        dense_layout=dense_layout,
    )


def _spmm_coo_sparse_ref_backend(value_dtype, index_dtype):
    if _is_rocm_runtime():
        reason = _hipsparse_spmm_coo_skip_reason(value_dtype, index_dtype)
        if reason is None:
            return "hipsparse", None
        return None, reason
    if cp is None or cpx_sparse is None:
        return None, "CuPy/cuSPARSE is not available"
    skip_reason = _cusparse_baseline_skip_reason(value_dtype)
    if skip_reason is not None:
        return None, skip_reason
    return "cupy_cusparse", None


def _hipsparse_spmm_coo_skip_reason(value_dtype, index_dtype):
    if not _is_rocm_runtime():
        return "hipSPARSE COO SpMM reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCoo",
        "hipsparseCreateDnMat",
        "hipsparseDestroyDnMat",
        "hipsparseDestroySpMat",
        "hipsparseSpMM_bufferSize",
        "hipsparseSpMM_preprocess",
        "hipsparseSpMM",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE COO SpMM direct API is unavailable: missing {symbol}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_scalar(value_dtype, 0.0, 0.0)
        _ = _hipsparse_index_type(index_dtype, "hipSPARSE COO SpMM indices")
        _ = _hipsparse_spmm_order("row", "hipSPARSE COO SpMM")
        _ = _hipsparse_spmm_algorithm("coo")
        _ = _hipsparse_lookup(
            "hipsparseOperation_t",
            ("HIPSPARSE_OPERATION_NON_TRANSPOSE",),
        )
    except Exception as exc:
        return str(exc)
    return None


def _prepare_spmm_coo_ref_hipsparse(
    data,
    row,
    col,
    B,
    shape,
    out=None,
):
    skip_reason = _hipsparse_spmm_coo_skip_reason(data.dtype, row.dtype)
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(torch.is_tensor(t) for t in (data, row, col, B)):
        raise TypeError("data, row, col, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, B must all be CUDA tensors")
    if not all(t.device == data.device for t in (row, col, B)):
        raise ValueError("data, row, col, B must be on the same CUDA device")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must all be 1D tensors")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have the same length")
    if B.ndim != 2:
        raise ValueError("hipSPARSE COO SpMM reference expects a 2D dense RHS")

    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if int(B.shape[0]) != n_cols:
        raise ValueError(f"B.shape[0] must equal n_cols={n_cols}")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match sparse value dtype for direct hipSPARSE COO SpMM")
    if row.dtype != col.dtype:
        raise TypeError("row and col must use the same index dtype for direct hipSPARSE COO SpMM")
    if not B.is_contiguous():
        raise ValueError("hipSPARSE COO SpMM direct reference expects contiguous row-major B")

    n_dense_cols = int(B.shape[1])
    if n_dense_cols == 0:
        return {
            "backend": "hipsparse",
            "buffer_size": 0,
            "format": "coo",
            "C": torch.empty((n_rows, 0), dtype=data.dtype, device=data.device),
            "empty": True,
        }

    data = data.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    B = B.contiguous()
    value_type = _hipsparse_value_type(data.dtype)
    alpha = _hipsparse_scalar(data.dtype, 1.0, 0.0)
    beta = _hipsparse_scalar(data.dtype, 0.0, 0.0)
    index_type = _hipsparse_index_type(row.dtype, "hipSPARSE COO SpMM indices")
    op_enum = _hipsparse_lookup(
        "hipsparseOperation_t",
        ("HIPSPARSE_OPERATION_NON_TRANSPOSE",),
    )
    order = _hipsparse_spmm_order("row", "hipSPARSE COO SpMM")
    alg = _hipsparse_spmm_algorithm("coo")

    C = out
    if C is None:
        C = torch.empty((n_rows, n_dense_cols), dtype=data.dtype, device=data.device)
    else:
        if not torch.is_tensor(C):
            raise TypeError("out must be a torch.Tensor")
        if not C.is_cuda or C.device != data.device:
            raise ValueError("out must be a CUDA tensor on the same device as data")
        if C.dtype != data.dtype or C.shape != (n_rows, n_dense_cols):
            raise ValueError("out must match the result shape and dtype")
        if not C.is_contiguous():
            raise ValueError("out must be contiguous row-major")

    handle = None
    spmat = None
    matb = None
    matc = None
    workspace = 0
    workspace_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)

        spmat = ptr_type()
        matb = ptr_type()
        matc = ptr_type()
        spmat_ref = spmat.createRef()
        matb_ref = matb.createRef()
        matc_ref = matc.createRef()

        row_ptr = HipPointer.fromObj(row.data_ptr())
        col_ptr = HipPointer.fromObj(col.data_ptr())
        values_ptr = HipPointer.fromObj(data.data_ptr())
        b_ptr = HipPointer.fromObj(B.data_ptr())
        c_ptr = HipPointer.fromObj(C.data_ptr())

        index_base = _hipsparse_lookup(
            "hipsparseIndexBase_t",
            ("HIPSPARSE_INDEX_BASE_ZERO",),
        )

        _hipsparse_create_coo_descriptor(
            spmat_ref,
            n_rows,
            n_cols,
            int(data.numel()),
            row_ptr,
            col_ptr,
            values_ptr,
            index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_dnmat_descriptor(
            matb_ref,
            n_cols,
            n_dense_cols,
            int(B.stride(0)),
            b_ptr,
            value_type,
            order,
        )
        _hipsparse_create_dnmat_descriptor(
            matc_ref,
            n_rows,
            n_dense_cols,
            int(C.stride(0)),
            c_ptr,
            value_type,
            order,
        )

        size_out = ctypes.c_size_t()
        _hip_check_result(
            hipsparse.hipsparseSpMM_bufferSize(
                handle,
                op_enum,
                op_enum,
                alpha,
                spmat,
                matb,
                beta,
                matc,
                value_type,
                alg,
                size_out,
            ),
            "hipsparseSpMM_bufferSize",
        )
        buffer_size = int(size_out.value)
        if buffer_size > 0:
            workspace = _hip_check_result(hip.hipMalloc(buffer_size), "hipMalloc")
            workspace_allocated = True
        else:
            workspace = 0
        _hip_check_result(
            hipsparse.hipsparseSpMM_preprocess(
                handle,
                op_enum,
                op_enum,
                alpha,
                spmat,
                matb,
                beta,
                matc,
                value_type,
                alg,
                workspace,
            ),
            "hipsparseSpMM_preprocess",
        )
        return {
            "backend": "hipsparse",
            "buffer_size": buffer_size,
            "format": "coo",
            "handle": handle,
            "spmat": spmat,
            "matb": matb,
            "matc": matc,
            "workspace": workspace,
            "workspace_allocated": workspace_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "beta": beta,
            "value_type": value_type,
            "alg": alg,
            "C": C,
            "empty": False,
        }
    finally:
        if handle is None and matc is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnMat(matc), "hipsparseDestroyDnMat(C)"
                )
            except Exception:
                pass
        if handle is None and matb is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnMat(matb), "hipsparseDestroyDnMat(B)"
                )
            except Exception:
                pass
        if handle is None and spmat is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
                )
            except Exception:
                pass
        if handle is None and workspace_allocated:
            try:
                _hip_check_result(hip.hipFree(workspace), "hipFree")
            except Exception:
                pass
 

def _run_spmm_coo_ref_hipsparse_prepared(state):
    if state.get("empty"):
        return state["C"]
    _hip_check_result(
        hipsparse.hipsparseSpMM(
            state["handle"],
            state["op_enum"],
            state["op_enum"],
            state["alpha"],
            state["spmat"],
            state["matb"],
            state["beta"],
            state["matc"],
            state["value_type"],
            state["alg"],
            state["workspace"],
        ),
        "hipsparseSpMM",
    )
    return state["C"]


def _destroy_spmm_coo_ref_hipsparse_prepared(state):
    matc = state.get("matc")
    matb = state.get("matb")
    spmat = state.get("spmat")
    workspace_allocated = bool(state.get("workspace_allocated"))
    workspace = state.get("workspace", 0)
    handle = state.get("handle")
    if matc is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnMat(matc), "hipsparseDestroyDnMat(C)"
            )
        except Exception:
            pass
    if matb is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnMat(matb), "hipsparseDestroyDnMat(B)"
            )
        except Exception:
            pass
    if spmat is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
            )
        except Exception:
            pass
    if workspace_allocated:
        try:
            _hip_check_result(hip.hipFree(workspace), "hipFree")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def _spmm_coo_ref_hipsparse(
    data,
    row,
    col,
    B,
    shape,
    out=None,
    return_metadata=False,
):
    state = _prepare_spmm_coo_ref_hipsparse(data, row, col, B, shape, out=out)
    try:
        C = _run_spmm_coo_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse",
            "buffer_size": int(state.get("buffer_size", 0)),
            "format": "coo",
        }
        if return_metadata:
            return C, metadata
        return C
    finally:
        _destroy_spmm_coo_ref_hipsparse_prepared(state)


def _benchmark_spmm_coo_sparse_ref(data, row, col, B, shape, warmup, iters):
    backend, reason = _spmm_coo_sparse_ref_backend(data.dtype, row.dtype)
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result
    if backend == "hipsparse":
        values, ms = _benchmark_prepared_cuda_op(
            lambda: _prepare_spmm_coo_ref_hipsparse(data, row, col, B, shape),
            _run_spmm_coo_ref_hipsparse_prepared,
            _destroy_spmm_coo_ref_hipsparse_prepared,
            warmup=warmup,
            iters=iters,
        )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
        return result

    data_cp = _cupy_from_torch(data)
    row_cp = _cupy_from_torch(row.to(torch.int64))
    col_cp = _cupy_from_torch(col.to(torch.int64))
    B_cp = _cupy_from_torch(B)
    A_coo = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    values_cp, ms = _benchmark_cuda_op(
        lambda: A_coo @ B_cp,
        warmup=warmup,
        iters=iters,
    )
    result["values"] = _torch_from_cupy(values_cp)
    result["ms"] = ms
    result["reason"] = None
    return result


def _seg_starts_from_sorted_rows(row_i32, nnz, device):
    if nnz == 0:
        return None
    diff = row_i32[1:] != row_i32[:-1]
    breaks = torch.nonzero(diff, as_tuple=False).flatten().to(torch.int32) + 1
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            breaks,
            torch.tensor([nnz], dtype=torch.int32, device=device),
        ]
    )


@dataclass(frozen=True)
class SpmmCooAlgorithm:
    """Registered COO SpMM route for the route-based run API."""

    name: str
    display_name: str
    supported_ops: tuple
    supported_dtypes: tuple
    run: object


class SpmmCooAlgorithmUnavailable(RuntimeError):
    """Raised when a registered COO SpMM algorithm is unavailable."""


class PreparedCooSpmmRoute:
    """Matrix-level COO SpMM route preparation shared by registered algorithms."""

    __slots__ = (
        "data",
        "row",
        "col",
        "shape",
        "n_rows",
        "n_cols",
        "seg_starts",
        "row_lengths",
        "n_segs",
        "nnz",
        "max_row_nnz",
        "avg_nnz_per_row",
        "output_dtype",
        "compute_dtype",
        "op",
        "alg",
    )

    def __init__(
        self,
        data,
        row,
        col,
        shape,
        seg_starts,
        row_lengths,
        output_dtype,
        compute_dtype,
        op,
        alg,
    ):
        self.data = data
        self.row = row
        self.col = col
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(shape[0])
        self.n_cols = int(shape[1])
        self.seg_starts = seg_starts
        self.row_lengths = row_lengths
        self.n_segs = int(row_lengths.numel())
        self.nnz = int(data.numel())
        self.max_row_nnz = int(row_lengths.max().item()) if row_lengths.numel() else 0
        self.avg_nnz_per_row = float(self.nnz) / float(max(1, self.n_rows))
        self.output_dtype = output_dtype
        self.compute_dtype = compute_dtype
        self.op = str(op)
        self.alg = str(alg)


@triton.jit
def _spmm_coo_rowrun_real_kernel(
    data_ptr,
    row_ptr,
    col_ptr,
    b_ptr,
    c_ptr,
    seg_starts_ptr,
    n_segs,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    seg = tl.program_id(0)
    pid_n = tl.program_id(1)
    if seg >= n_segs:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_nnz = end - start
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(col_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)

    tl.store(c_ptr + row_id * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_coo_rowrun_complex_kernel(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    b_ri_ptr,
    c_ri_ptr,
    seg_starts_ptr,
    n_segs,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    seg = tl.program_id(0)
    pid_n = tl.program_id(1)
    if seg >= n_segs:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_nnz = end - start
    row_id = tl.load(row_ptr + start)
    acc_re = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc_im = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_re = tl.load(data_ri_ptr + idx * 2, mask=valid, other=0.0)
            a_im = tl.load(data_ri_ptr + idx * 2 + 1, mask=valid, other=0.0)
            a_col = tl.load(col_ptr + idx, mask=valid, other=0)
            b_re = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            b_im = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn + stride_br,
                mask=mask_n & valid,
                other=0.0,
            )
            acc_re = (
                acc_re
                + a_re.to(ACC_DTYPE) * b_re.to(ACC_DTYPE)
                - a_im.to(ACC_DTYPE) * b_im.to(ACC_DTYPE)
            )
            acc_im = (
                acc_im
                + a_re.to(ACC_DTYPE) * b_im.to(ACC_DTYPE)
                + a_im.to(ACC_DTYPE) * b_re.to(ACC_DTYPE)
            )

    tl.store(c_ri_ptr + row_id * stride_cm + offs_n * stride_cn, acc_re, mask=mask_n)
    tl.store(
        c_ri_ptr + row_id * stride_cm + offs_n * stride_cn + stride_cr,
        acc_im,
        mask=mask_n,
    )


@triton.jit
def _spmm_coo_alg1_process_count_kernel(
    row_lengths_ptr, counts_ptr, n_segs, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_segs
    lengths = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    b0 = mask & (lengths <= 32)
    b1 = mask & (lengths > 32) & (lengths <= 128)
    b2 = mask & (lengths > 128) & (lengths <= 512)
    b3 = mask & (lengths > 512) & (lengths <= 2048)
    b4 = mask & (lengths > 2048)
    tl.atomic_add(counts_ptr + 0, tl.sum(tl.where(b0, 1, 0)), sem="relaxed")
    tl.atomic_add(counts_ptr + 1, tl.sum(tl.where(b1, 1, 0)), sem="relaxed")
    tl.atomic_add(counts_ptr + 2, tl.sum(tl.where(b2, 1, 0)), sem="relaxed")
    tl.atomic_add(counts_ptr + 3, tl.sum(tl.where(b3, 1, 0)), sem="relaxed")
    tl.atomic_add(counts_ptr + 4, tl.sum(tl.where(b4, 1, 0)), sem="relaxed")


@triton.jit
def _spmm_coo_alg1_process_compact_kernel(
    row_lengths_ptr,
    offsets_ptr,
    write_counts_ptr,
    segs_flat_ptr,
    n_segs,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_segs
    lengths = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    bucket = tl.full([BLOCK_M], 4, tl.int32)
    bucket = tl.where(lengths <= 2048, 3, bucket)
    bucket = tl.where(lengths <= 512, 2, bucket)
    bucket = tl.where(lengths <= 128, 1, bucket)
    bucket = tl.where(lengths <= 32, 0, bucket)
    for b in tl.static_range(0, 5):
        in_bucket = mask & (bucket == b)
        local = tl.cumsum(tl.where(in_bucket, 1, 0), 0) - 1
        n_bucket = tl.sum(tl.where(in_bucket, 1, 0))
        base = tl.load(offsets_ptr + b) + tl.atomic_add(
            write_counts_ptr + b, n_bucket, sem="relaxed"
        )
        tl.store(segs_flat_ptr + base + local, offs, mask=in_bucket)


@triton.jit
def _spmm_coo_alg1_bucket_real_kernel(
    data_ptr,
    row_ptr,
    col_ptr,
    b_ptr,
    c_ptr,
    seg_starts_ptr,
    bucket_segs_ptr,
    n_bucket_segs,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    bucket_pos = tl.program_id(0)
    pid_n = tl.program_id(1)
    if bucket_pos >= n_bucket_segs:
        return

    seg = tl.load(bucket_segs_ptr + bucket_pos)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_nnz = end - start
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(col_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)

    tl.store(c_ptr + row_id * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_coo_atomic_real_kernel(
    data_ptr,
    row_ptr,
    col_ptr,
    b_ptr,
    c_ptr,
    nnz,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
):
    idx = tl.program_id(0)
    dense_col = tl.program_id(1)
    if idx >= nnz or dense_col >= n_dense_cols:
        return

    row_k = tl.load(row_ptr + idx)
    col_k = tl.load(col_ptr + idx)
    val_k = tl.load(data_ptr + idx)
    b_val = tl.load(b_ptr + col_k * stride_bk + dense_col * stride_bn)
    tl.atomic_add(
        c_ptr + row_k * stride_cm + dense_col * stride_cn,
        val_k.to(ACC_DTYPE) * b_val.to(ACC_DTYPE),
        sem="relaxed",
    )


@triton.jit
def _spmm_coo_atomic_complex_kernel(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    b_ri_ptr,
    c_ri_ptr,
    nnz,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    ACC_DTYPE: tl.constexpr,
):
    idx = tl.program_id(0)
    dense_col = tl.program_id(1)
    if idx >= nnz or dense_col >= n_dense_cols:
        return

    row_k = tl.load(row_ptr + idx)
    col_k = tl.load(col_ptr + idx)
    a_re = tl.load(data_ri_ptr + idx * 2)
    a_im = tl.load(data_ri_ptr + idx * 2 + 1)
    b_re = tl.load(b_ri_ptr + col_k * stride_bk + dense_col * stride_bn)
    b_im = tl.load(b_ri_ptr + col_k * stride_bk + dense_col * stride_bn + stride_br)
    contrib_re = a_re.to(ACC_DTYPE) * b_re.to(ACC_DTYPE) - a_im.to(ACC_DTYPE) * b_im.to(
        ACC_DTYPE
    )
    contrib_im = a_re.to(ACC_DTYPE) * b_im.to(ACC_DTYPE) + a_im.to(ACC_DTYPE) * b_re.to(
        ACC_DTYPE
    )
    tl.atomic_add(
        c_ri_ptr + row_k * stride_cm + dense_col * stride_cn,
        contrib_re,
        sem="relaxed",
    )
    tl.atomic_add(
        c_ri_ptr + row_k * stride_cm + dense_col * stride_cn + stride_cr,
        contrib_im,
        sem="relaxed",
    )


def _prepare_spmm_coo_inputs(data, row, col, B, shape, dense_layout="row"):
    dense_layout = _normalize_dense_layout(dense_layout)
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple: (n_rows, n_cols)")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, and col must be 1D tensors")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, and col must have the same length (nnz)")
    if B.shape[0] != n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={n_cols}, got {B.shape[0]}")

    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, and B must be CUDA tensors")
    if not all(t.device == data.device for t in (row, col, B)):
        raise ValueError("data, row, col, and B must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if row.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")

    nnz = data.numel()
    if nnz > _INDEX_LIMIT_INT32:
        raise ValueError(
            "nnz exceeds the int32 range supported by the Triton COO kernel"
        )
    if nnz > 0:
        min_row = int(row.min().item())
        max_row = int(row.max().item())
        min_col = int(col.min().item())
        max_col = int(col.max().item())
        if min_row < 0 or max_row >= n_rows:
            raise IndexError("row indices out of range for n_rows")
        if min_col < 0 or max_col >= n_cols:
            raise IndexError("col indices out of range for n_cols")
        if max_row > _INDEX_LIMIT_INT32:
            raise ValueError(
                "row indices exceed the int32 range supported by the Triton kernel"
            )
        if max_col > _INDEX_LIMIT_INT32:
            raise ValueError(
                "column indices exceed the int32 range supported by the Triton kernel"
            )

    data = data.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    B = _materialize_dense_layout(B, dense_layout)

    kernel_row = row.to(torch.int32) if row.dtype == torch.int64 else row
    kernel_col = col.to(torch.int32) if col.dtype == torch.int64 else col
    return data, kernel_row, kernel_col, B, n_rows, n_cols, int(B.shape[1])


def _resolve_spmm_coo_launch_config(n_dense_cols, nnz, block_n=None, block_nnz=256):
    warp_size, factor = _select_spmm_alg1_warp_and_factor(n_dense_cols)

    if block_n is None:
        block_n = warp_size * factor
    if block_nnz is None:
        block_nnz = 256

    if block_n <= 0 or block_nnz <= 0:
        raise ValueError("block_n and block_nnz must be positive when provided")

    return {
        "block_n": int(block_n),
        "block_nnz": int(block_nnz),
        "required_nnz_tiles": int(triton.cdiv(nnz, block_nnz) if nnz > 0 else 0),
        "heuristic_warp_size": int(warp_size),
        "heuristic_factor": int(factor),
    }


def _triton_spmm_coo_rowrun_impl(
    data,
    row,
    col,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
    output_dtype,
    out=None,
    dense_layout="row",
    seg_starts=None,
):
    device = data.device
    dtype = data.dtype
    dense_layout = _normalize_dense_layout(dense_layout)
    if out is not None:
        if out.shape != (int(n_rows), int(n_dense_cols)) or out.dtype != output_dtype:
            raise ValueError("out shape/dtype must match result")
        if out.device != device:
            raise ValueError("out must be on the same CUDA device as data")
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0 or data.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return _zeros_dense_layout(
            (n_rows, n_dense_cols), output_dtype, device, dense_layout
        )

    if seg_starts is None:
        seg_starts = _seg_starts_from_sorted_rows(row, int(data.numel()), device)
    n_segs = int(seg_starts.numel()) - 1 if seg_starts is not None else 0
    if n_segs == 0:
        if out is not None:
            out.zero_()
            return out
        return _zeros_dense_layout(
            (n_rows, n_dense_cols), output_dtype, device, dense_layout
        )

    grid = (n_segs, triton.cdiv(n_dense_cols, block_n))
    if not _is_complex_dtype(dtype):
        C_compute = (
            out
            if out is not None and dtype == output_dtype
            else _zeros_dense_layout(
                (n_rows, n_dense_cols), dtype, device, dense_layout
            )
        )
        if C_compute is out:
            C_compute.zero_()
        acc_dtype = tl.float64 if dtype == torch.float64 else tl.float32
        _spmm_coo_rowrun_real_kernel[grid](
            data,
            row,
            col,
            B,
            C_compute,
            seg_starts,
            n_segs,
            n_dense_cols,
            B.stride(0),
            B.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            BLOCK_N=block_n,
            BLOCK_NNZ=block_nnz,
            ACC_DTYPE=acc_dtype,
        )
        if dtype != output_dtype:
            C_cast = C_compute.to(output_dtype)
            if out is not None:
                out.copy_(C_cast)
                return out
            if dense_layout == "col":
                C_out = _empty_dense_layout(
                    (n_rows, n_dense_cols), output_dtype, device, dense_layout
                )
                C_out.copy_(C_cast)
                return C_out
            return C_cast
        return C_compute

    data_ri = torch.view_as_real(data).contiguous().reshape(-1)
    B_ri = torch.view_as_real(B)
    C_compute = (
        out
        if out is not None and dtype == output_dtype
        else _zeros_dense_layout(
            (n_rows, n_dense_cols),
            dtype,
            device,
            dense_layout,
        )
    )
    if C_compute is out:
        C_compute.zero_()
    C_ri = torch.view_as_real(C_compute)
    acc_dtype = tl.float64 if B_ri.dtype == torch.float64 else tl.float32
    _spmm_coo_rowrun_complex_kernel[grid](
        data_ri,
        row,
        col,
        B_ri,
        C_ri,
        seg_starts,
        n_segs,
        n_dense_cols,
        B_ri.stride(0),
        B_ri.stride(1),
        B_ri.stride(2),
        C_ri.stride(0),
        C_ri.stride(1),
        C_ri.stride(2),
        BLOCK_N=block_n,
        BLOCK_NNZ=block_nnz,
        ACC_DTYPE=acc_dtype,
    )
    if dtype != output_dtype:
        C_cast = C_compute.to(output_dtype)
        if out is not None:
            out.copy_(C_cast)
            return out
        if dense_layout == "col":
            C_out = _empty_dense_layout(
                (n_rows, n_dense_cols), output_dtype, device, dense_layout
            )
            C_out.copy_(C_cast)
            return C_out
        return C_cast
    return C_compute


def _triton_spmm_coo_atomic_impl(
    data,
    row,
    col,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
    output_dtype,
    out=None,
    dense_layout="row",
):
    device = data.device
    dtype = data.dtype
    dense_layout = _normalize_dense_layout(dense_layout)
    if out is not None:
        if out.shape != (int(n_rows), int(n_dense_cols)) or out.dtype != output_dtype:
            raise ValueError("out shape/dtype must match result")
        if out.device != device:
            raise ValueError("out must be on the same CUDA device as data")
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0 or data.numel() == 0:
        if out is not None:
            out.zero_()
            return out
        return _zeros_dense_layout(
            (n_rows, n_dense_cols), output_dtype, device, dense_layout
        )

    nnz = int(data.numel())
    if nnz == 0:
        if out is not None:
            out.zero_()
            return out
        return _zeros_dense_layout(
            (n_rows, n_dense_cols), output_dtype, device, dense_layout
        )

    if not _is_complex_dtype(dtype):
        C_compute = (
            out
            if out is not None and dtype == output_dtype
            else _zeros_dense_layout(
                (n_rows, n_dense_cols), dtype, device, dense_layout
            )
        )
        if C_compute is out:
            C_compute.zero_()
        acc_dtype = tl.float64 if dtype == torch.float64 else tl.float32
        _spmm_coo_atomic_real_kernel[(nnz, n_dense_cols)](
            data,
            row,
            col,
            B,
            C_compute,
            nnz,
            n_dense_cols,
            B.stride(0),
            B.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            ACC_DTYPE=acc_dtype,
        )
        if dtype != output_dtype:
            C_cast = C_compute.to(output_dtype)
            if out is not None:
                out.copy_(C_cast)
                return out
            if dense_layout == "col":
                C_out = _empty_dense_layout(
                    (n_rows, n_dense_cols), output_dtype, device, dense_layout
                )
                C_out.copy_(C_cast)
                return C_out
            return C_cast
        return C_compute

    data_ri = torch.view_as_real(data).contiguous().reshape(-1)
    B_ri = torch.view_as_real(B)
    C_compute = (
        out
        if out is not None and dtype == output_dtype
        else _zeros_dense_layout(
            (n_rows, n_dense_cols),
            dtype,
            device,
            dense_layout,
        )
    )
    if C_compute is out:
        C_compute.zero_()
    C_ri = torch.view_as_real(C_compute)
    acc_dtype = tl.float64 if B_ri.dtype == torch.float64 else tl.float32
    _spmm_coo_atomic_complex_kernel[(nnz, n_dense_cols)](
        data_ri,
        row,
        col,
        B_ri,
        C_ri,
        nnz,
        n_dense_cols,
        B_ri.stride(0),
        B_ri.stride(1),
        B_ri.stride(2),
        C_ri.stride(0),
        C_ri.stride(1),
        C_ri.stride(2),
        ACC_DTYPE=acc_dtype,
    )
    if dtype != output_dtype:
        C_cast = C_compute.to(output_dtype)
        if out is not None:
            out.copy_(C_cast)
            return out
        if dense_layout == "col":
            C_out = _empty_dense_layout(
                (n_rows, n_dense_cols), output_dtype, device, dense_layout
            )
            C_out.copy_(C_cast)
            return C_out
        return C_cast
    return C_compute


def _normalize_spmm_coo_route(route):
    route = "rowrun" if route is None else str(route).lower()
    if route not in ("rowrun", "atomic"):
        raise ValueError("route must be 'rowrun' or 'atomic'")
    return route


def _triton_spmm_coo_impl(
    data,
    row,
    col,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
    route="rowrun",
    output_dtype=None,
    out=None,
    dense_layout="row",
):
    route = _normalize_spmm_coo_route(route)
    dense_layout = _normalize_dense_layout(dense_layout)
    resolved_output_dtype = output_dtype if output_dtype is not None else data.dtype
    if route == "rowrun":
        return _triton_spmm_coo_rowrun_impl(
            data,
            row,
            col,
            B,
            n_rows,
            n_dense_cols,
            block_n,
            block_nnz,
            output_dtype=resolved_output_dtype,
            out=out,
            dense_layout=dense_layout,
        )
    return _triton_spmm_coo_atomic_impl(
        data,
        row,
        col,
        B,
        n_rows,
        n_dense_cols,
        block_n,
        block_nnz,
        output_dtype=resolved_output_dtype,
        out=out,
        dense_layout=dense_layout,
    )


def _normalize_spmm_coo_alg(alg):
    token = "auto" if alg is None else str(alg).strip().lower()
    aliases = {
        "rowrun": "coo_rowrun",
        "coo": "coo_rowrun",
        "base": "coo_rowrun",
        "coo_base": "coo_rowrun",
        "atomic": "coo_atomic",
        "alg1": "spmm_coo_alg1",
        "coo_alg1": "spmm_coo_alg1",
    }
    if token == "auto":
        return "auto"
    return aliases.get(token, token)


def _prepare_spmm_coo_matrix(data, row, col, shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple: (n_rows, n_cols)")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, and col must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, and col must have the same length (nnz)")
    if not all(t.is_cuda for t in (data, row, col)):
        raise ValueError("data, row, and col must be CUDA tensors")
    if not all(t.device == data.device for t in (row, col)):
        raise ValueError("data, row, and col must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if row.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    nnz = int(data.numel())
    if nnz > _INDEX_LIMIT_INT32:
        raise ValueError(
            "nnz exceeds the int32 range supported by the Triton COO kernel"
        )
    if nnz > 0:
        min_row = int(row.min().item())
        max_row = int(row.max().item())
        min_col = int(col.min().item())
        max_col = int(col.max().item())
        if min_row < 0 or max_row >= n_rows:
            raise IndexError("row indices out of range for n_rows")
        if min_col < 0 or max_col >= n_cols:
            raise IndexError("col indices out of range for n_cols")
        if max_row > _INDEX_LIMIT_INT32:
            raise ValueError(
                "row indices exceed the int32 range supported by the Triton kernel"
            )
        if max_col > _INDEX_LIMIT_INT32:
            raise ValueError(
                "column indices exceed the int32 range supported by the Triton kernel"
            )
    kernel_row = (
        row.contiguous().to(torch.int32)
        if row.dtype == torch.int64
        else row.contiguous()
    )
    kernel_col = (
        col.contiguous().to(torch.int32)
        if col.dtype == torch.int64
        else col.contiguous()
    )
    return data.contiguous(), kernel_row, kernel_col, (n_rows, n_cols)


def _validate_spmm_coo_route_runtime_inputs(prepared, B, dense_layout):
    if B is None or not torch.is_tensor(B):
        raise TypeError("B must be a torch.Tensor")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if B.dtype != prepared.output_dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if int(B.shape[0]) != prepared.n_cols:
        raise ValueError(
            f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}"
        )
    B_compute = (
        B
        if prepared.compute_dtype == prepared.output_dtype
        else B.to(prepared.compute_dtype)
    )
    return _materialize_dense_layout(B_compute, dense_layout)


def _run_spmm_coo_rowrun_route(
    prepared, B, *, timing=False, diagnostics=False, dense_layout="row"
):
    dense_layout = _normalize_dense_layout(dense_layout)
    B = _validate_spmm_coo_route_runtime_inputs(prepared, B, dense_layout)
    launch = _resolve_spmm_coo_launch_config(int(B.shape[1]), prepared.nnz)
    compute_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _triton_spmm_coo_rowrun_impl(
        prepared.data,
        prepared.row,
        prepared.col,
        B,
        prepared.n_rows,
        int(B.shape[1]),
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        output_dtype=prepared.output_dtype,
        dense_layout=dense_layout,
        seg_starts=prepared.seg_starts,
    )
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    meta = {
        "alg": "coo_rowrun",
        "display_name": "COORowRun",
        "op": prepared.op,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": compute_ms,
        "dense_layout": dense_layout,
        "b_stride": tuple(int(v) for v in B.stride()),
        "c_stride": tuple(int(v) for v in C.stride()),
        "output_layout": _dense_layout_name(C),
    }
    if diagnostics:
        meta["diagnostics"] = {
            "launch_config_scope": "matrix",
            "launch_config_count": 1,
            "bucket_count": 0,
            "long_row_count": 0,
            "long_part_count": 0,
            "launch_version": "coo_rowrun_v1",
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "warp_size": launch["heuristic_warp_size"],
            "factor": launch["heuristic_factor"],
            "grid_m": prepared.n_segs,
            "grid_n": triton.cdiv(int(B.shape[1]), launch["block_n"]),
            "dense_layout": dense_layout,
            "b_stride": tuple(int(v) for v in B.stride()),
            "c_stride": tuple(int(v) for v in C.stride()),
            "output_layout": _dense_layout_name(C),
        }
    return C, meta


def _run_spmm_coo_atomic_route(
    prepared, B, *, timing=False, diagnostics=False, dense_layout="row"
):
    dense_layout = _normalize_dense_layout(dense_layout)
    B = _validate_spmm_coo_route_runtime_inputs(prepared, B, dense_layout)
    launch = _resolve_spmm_coo_launch_config(int(B.shape[1]), prepared.nnz)
    compute_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _triton_spmm_coo_atomic_impl(
        prepared.data,
        prepared.row,
        prepared.col,
        B,
        prepared.n_rows,
        int(B.shape[1]),
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        output_dtype=prepared.output_dtype,
        dense_layout=dense_layout,
    )
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    meta = {
        "alg": "coo_atomic",
        "display_name": "COOAtomic",
        "op": prepared.op,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": compute_ms,
        "dense_layout": dense_layout,
        "b_stride": tuple(int(v) for v in B.stride()),
        "c_stride": tuple(int(v) for v in C.stride()),
        "output_layout": _dense_layout_name(C),
    }
    if diagnostics:
        meta["diagnostics"] = {
            "launch_config_scope": "matrix",
            "launch_config_count": 1,
            "bucket_count": 0,
            "long_row_count": 0,
            "long_part_count": 0,
            "launch_version": "coo_atomic_v1",
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "warp_size": launch["heuristic_warp_size"],
            "factor": launch["heuristic_factor"],
            "grid_m": prepared.nnz,
            "grid_n": int(B.shape[1]),
            "dense_layout": dense_layout,
            "b_stride": tuple(int(v) for v in B.stride()),
            "c_stride": tuple(int(v) for v in C.stride()),
            "output_layout": _dense_layout_name(C),
        }
    return C, meta


def _spmm_coo_alg1_build_bucket_descriptors(segs_flat, counts, offsets):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    counts_cpu = counts.detach().cpu().tolist()
    offsets_cpu = offsets.detach().cpu().tolist()
    buckets = []
    for bucket_id, count in enumerate(counts_cpu):
        offset = int(offsets_cpu[bucket_id])
        count = int(count)
        buckets.append(
            {
                "bucket_id": bucket_id,
                "rows": segs_flat.narrow(0, offset, count),
                "count": count,
                # Keep large row-run buckets logically separate, but cap the
                # static inner loop to avoid enormous Triton specializations.
                "block_nnz": (32, 64, 128, 128, 256)[bucket_id],
            }
        )
    process_cpu_ms = (time.perf_counter() - t0) * 1000.0
    return buckets, process_cpu_ms


def _run_spmm_coo_alg1_route(
    prepared, B, *, timing=False, diagnostics=False, dense_layout="row"
):
    if prepared.output_dtype not in (torch.float32, torch.float64):
        raise TypeError("spmm_coo_alg1 only supports float32 and float64")
    dense_layout = _normalize_dense_layout(dense_layout)
    B = _validate_spmm_coo_route_runtime_inputs(prepared, B, dense_layout)
    n_dense_cols = int(B.shape[1])
    device = prepared.data.device
    bucket_count = 5
    counts = torch.zeros((bucket_count,), dtype=torch.int64, device=device)
    offsets = torch.empty_like(counts)
    write_counts = torch.zeros_like(counts)
    segs_flat = torch.empty((prepared.n_segs,), dtype=torch.int32, device=device)
    block_m = 256
    grid = (triton.cdiv(prepared.n_segs, block_m),)
    process_gpu_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    if prepared.n_segs > 0:
        _spmm_coo_alg1_process_count_kernel[grid](
            prepared.row_lengths,
            counts,
            prepared.n_segs,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    offsets[0] = 0
    if bucket_count > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    if prepared.n_segs > 0:
        _spmm_coo_alg1_process_compact_kernel[grid](
            prepared.row_lengths,
            offsets,
            write_counts,
            segs_flat,
            prepared.n_segs,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    if timing:
        end.record()
        torch.cuda.synchronize()
        process_gpu_ms = start.elapsed_time(end)
    else:
        torch.cuda.synchronize()
    buckets, process_cpu_ms = _spmm_coo_alg1_build_bucket_descriptors(
        segs_flat, counts, offsets
    )

    C_compute = _zeros_dense_layout(
        (prepared.n_rows, n_dense_cols), prepared.compute_dtype, device, dense_layout
    )
    launch = _resolve_spmm_coo_launch_config(n_dense_cols, prepared.nnz)
    acc_dtype = tl.float64 if prepared.compute_dtype == torch.float64 else tl.float32
    compute_ms = None
    if timing:
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record()
    for bucket in buckets:
        n_bucket = int(bucket["count"])
        if n_bucket <= 0:
            continue
        block_nnz = int(bucket["block_nnz"])
        grid_bucket = (n_bucket, triton.cdiv(n_dense_cols, launch["block_n"]))
        _spmm_coo_alg1_bucket_real_kernel[grid_bucket](
            prepared.data,
            prepared.row,
            prepared.col,
            B,
            C_compute,
            prepared.seg_starts,
            bucket["rows"],
            n_bucket,
            n_dense_cols,
            B.stride(0),
            B.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            BLOCK_N=launch["block_n"],
            BLOCK_NNZ=block_nnz,
            ACC_DTYPE=acc_dtype,
        )
    if timing:
        compute_end.record()
        torch.cuda.synchronize()
        compute_ms = compute_start.elapsed_time(compute_end)
    if prepared.compute_dtype != prepared.output_dtype:
        C = C_compute.to(prepared.output_dtype)
        if dense_layout == "col":
            C_out = _empty_dense_layout(
                (prepared.n_rows, n_dense_cols),
                prepared.output_dtype,
                device,
                dense_layout,
            )
            C_out.copy_(C)
            C = C_out
    else:
        C = C_compute
    counts_cpu = [int(bucket["count"]) for bucket in buckets]
    meta = {
        "alg": "spmm_coo_alg1",
        "display_name": "COOAlg1",
        "op": prepared.op,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
        "dense_layout": dense_layout,
        "b_stride": tuple(int(v) for v in B.stride()),
        "c_stride": tuple(int(v) for v in C.stride()),
        "output_layout": _dense_layout_name(C),
    }
    if diagnostics:
        meta["diagnostics"] = {
            "launch_config_scope": "bucket",
            "launch_config_count": sum(1 for count in counts_cpu if count > 0),
            "bucket_count": sum(1 for count in counts_cpu if count > 0),
            "bucket_counts": "|".join(str(v) for v in counts_cpu),
            "long_row_count": counts_cpu[-1],
            "long_part_count": counts_cpu[-1],
            "launch_version": "spmm_coo_alg1_v1",
            "block_n": launch["block_n"],
            "block_nnz": "32|64|128|128|256",
            "warp_size": launch["heuristic_warp_size"],
            "factor": launch["heuristic_factor"],
            "grid_m": prepared.n_segs,
            "grid_n": triton.cdiv(n_dense_cols, launch["block_n"]),
            "dense_layout": dense_layout,
            "b_stride": tuple(int(v) for v in B.stride()),
            "c_stride": tuple(int(v) for v in C.stride()),
            "output_layout": _dense_layout_name(C),
        }
    return C, meta


SPMM_COO_ALGORITHMS = {
    "coo_rowrun": SpmmCooAlgorithm(
        name="coo_rowrun",
        display_name="COORowRun",
        supported_ops=tuple(SPMM_COO_OP_NAMES.values()),
        supported_dtypes=SUPPORTED_SPMM_VALUE_DTYPES,
        run=_run_spmm_coo_rowrun_route,
    ),
    "coo_atomic": SpmmCooAlgorithm(
        name="coo_atomic",
        display_name="COOAtomic",
        supported_ops=tuple(SPMM_COO_OP_NAMES.values()),
        supported_dtypes=SUPPORTED_SPMM_VALUE_DTYPES,
        run=_run_spmm_coo_atomic_route,
    ),
    "spmm_coo_alg1": SpmmCooAlgorithm(
        name="spmm_coo_alg1",
        display_name="COOAlg1",
        supported_ops=tuple(SPMM_COO_OP_NAMES.values()),
        supported_dtypes=(torch.float32, torch.float64),
        run=_run_spmm_coo_alg1_route,
    ),
}


def resolve_spmm_coo_algorithm(alg, op, dtype):
    token = _normalize_spmm_coo_alg(alg)
    if token == "auto":
        token = "coo_rowrun"
    if token not in SPMM_COO_ALGORITHMS:
        supported = ", ".join(sorted(SPMM_COO_ALGORITHMS))
        raise ValueError(
            f"unsupported COO SpMM algorithm {alg!r}; supported: auto, {supported}"
        )
    algorithm = SPMM_COO_ALGORITHMS[token]
    op_name = _spmm_coo_op_to_name(op)
    if op_name not in algorithm.supported_ops:
        raise ValueError(f"algorithm {token!r} does not support op {op_name!r}")
    if dtype not in algorithm.supported_dtypes:
        raise TypeError(f"algorithm {token!r} does not support dtype {dtype}")
    return algorithm


def list_spmm_coo_algorithms(op=None, dtype=None):
    op_name = None if op is None else _spmm_coo_op_to_name(op)
    names = []
    for name, algorithm in SPMM_COO_ALGORITHMS.items():
        if op_name is not None and op_name not in algorithm.supported_ops:
            continue
        if dtype is not None and dtype not in algorithm.supported_dtypes:
            continue
        names.append(name)
    return tuple(names)


def prepare_spmm_coo_route(data, row, col, shape, *, op="non", alg="auto"):
    """Prepare matrix-level canonical COO metadata for registered SpMM algorithms."""
    op_code = _normalize_spmm_coo_op(op)
    op_name = _spmm_coo_op_to_name(op_code)
    data, row, col, shape = _materialize_spmm_coo_op(data, row, col, shape, op_code)
    output_dtype = data.dtype
    compute_dtype = _spmm_coo_compute_dtype(output_dtype)
    data, row, col, shape = _prepare_spmm_coo_matrix(data, row, col, shape)
    data_compute = data if compute_dtype == output_dtype else data.to(compute_dtype)
    canonical_data, canonical_row, canonical_col = _coalesce_coo_entries(
        data_compute, row, col, shape
    )
    canonical_data, canonical_row, canonical_col = _sort_coo_lex_inplace(
        canonical_data,
        canonical_row,
        canonical_col,
        shape[1],
    )
    canonical_row = canonical_row.to(torch.int32)
    canonical_col = canonical_col.to(torch.int32)
    seg_starts = _seg_starts_from_sorted_rows(
        canonical_row, int(canonical_data.numel()), canonical_data.device
    )
    if seg_starts is None:
        row_lengths = torch.empty((0,), dtype=torch.int32, device=canonical_data.device)
    else:
        row_lengths = (seg_starts[1:] - seg_starts[:-1]).contiguous()
    resolved_alg = _normalize_spmm_coo_alg(alg)
    if resolved_alg != "auto":
        resolve_spmm_coo_algorithm(resolved_alg, op_name, output_dtype)
    return PreparedCooSpmmRoute(
        canonical_data,
        canonical_row,
        canonical_col,
        shape,
        seg_starts,
        row_lengths,
        output_dtype,
        compute_dtype,
        op_name,
        resolved_alg,
    )


def flagsparse_spmm_coo_run(
    prepared,
    B,
    *,
    alg=None,
    dense_layout="auto",
    return_time=False,
    return_meta=False,
    timing=False,
    diagnostics=False,
):
    """Run a registered COO SpMM algorithm with CSR-style timing metadata."""
    if not isinstance(prepared, PreparedCooSpmmRoute):
        raise TypeError("prepared must be a PreparedCooSpmmRoute instance")
    alg_name = prepared.alg if alg is None else _normalize_spmm_coo_alg(alg)
    algorithm = resolve_spmm_coo_algorithm(alg_name, prepared.op, prepared.output_dtype)
    dense_layout = _normalize_dense_layout(dense_layout)
    start = (
        torch.cuda.Event(enable_timing=True) if (return_time or return_meta) else None
    )
    end = torch.cuda.Event(enable_timing=True) if (return_time or return_meta) else None
    if start is not None:
        torch.cuda.synchronize()
        start.record()
    C, route_meta = algorithm.run(
        prepared,
        B,
        timing=bool(timing),
        diagnostics=bool(diagnostics),
        dense_layout=dense_layout,
    )
    if end is not None:
        end.record()
        torch.cuda.synchronize()
        gpu_ms = start.elapsed_time(end)
    else:
        gpu_ms = None
    process_cpu_ms = float(route_meta.get("process_cpu_ms", 0.0) or 0.0)
    operator_ms = (process_cpu_ms + float(gpu_ms)) if gpu_ms is not None else None
    meta = None
    if return_meta:
        meta = {
            "alg": algorithm.name,
            "display_name": algorithm.display_name,
            "op": prepared.op,
            "operator_ms": operator_ms,
            "gpu_ms": gpu_ms,
            "process_cpu_ms": process_cpu_ms,
            "dense_layout": route_meta.get("dense_layout", dense_layout),
            "b_stride": route_meta.get("b_stride"),
            "c_stride": route_meta.get("c_stride"),
            "output_layout": route_meta.get("output_layout"),
        }
        if timing:
            meta["process_gpu_ms"] = route_meta.get("process_gpu_ms")
            meta["compute_ms"] = route_meta.get("compute_ms")
        if diagnostics and "diagnostics" in route_meta:
            meta["diagnostics"] = route_meta["diagnostics"]
    if return_time and return_meta:
        return C, operator_ms, meta
    if return_time:
        return C, operator_ms
    if return_meta:
        return C, meta
    return C


def _run_spmm_coo_canonical_route(
    canonical_data,
    canonical_row,
    canonical_col,
    canonical_B,
    n_rows,
    n_dense_cols,
    output_dtype,
    block_n=None,
    block_nnz=256,
    out=None,
    return_time=False,
    route="rowrun",
    dense_layout="row",
):
    route = _normalize_spmm_coo_route(route)
    dense_layout = _normalize_dense_layout(dense_layout)
    launch = _resolve_spmm_coo_launch_config(
        n_dense_cols,
        canonical_data.numel(),
        block_n=block_n,
        block_nnz=block_nnz,
    )

    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != canonical_data.device:
            raise ValueError("out must be on the same CUDA device as the inputs")
        if out.shape != (n_rows, n_dense_cols) or out.dtype != output_dtype:
            raise ValueError("out shape/dtype must match result")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = _triton_spmm_coo_impl(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        route=route,
        output_dtype=output_dtype,
        out=out,
        dense_layout=dense_layout,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if return_time:
        return C, elapsed_ms
    return C


def _run_spmm_coo_route(
    data,
    row,
    col,
    B,
    shape,
    block_n=None,
    block_nnz=256,
    out=None,
    return_time=False,
    return_meta=False,
    route="rowrun",
    op=None,
    transpose=None,
    dense_layout="row",
):
    route = _normalize_spmm_coo_route(route)
    dense_layout = _normalize_dense_layout(dense_layout)
    op_explicit = op is not None
    op_code = _normalize_spmm_coo_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmm_coo_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    op_name = _spmm_coo_op_to_name(op_code)
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n must be positive when provided")
    if block_nnz is not None and block_nnz <= 0:
        raise ValueError("block_nnz must be positive when provided")

    do_timing = bool(return_time or return_meta)
    symbolic_ms = 0.0 if do_timing else None
    compute_ms = None
    op_total_ms = None

    if do_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    data, row, col, shape = _materialize_spmm_coo_op(data, row, col, shape, op_code)
    if do_timing:
        torch.cuda.synchronize()
        symbolic_ms = (
            (time.perf_counter() - t0) * 1000.0
            if _spmm_coo_op_transposes(op_code)
            else 0.0
        )

    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        _,
        n_dense_cols,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_inputs(
        data,
        row,
        col,
        B,
        shape,
        dense_layout=dense_layout,
    )

    result = _run_spmm_coo_canonical_route(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        output_dtype,
        block_n=block_n,
        block_nnz=block_nnz,
        out=out,
        return_time=do_timing,
        route=route,
        dense_layout=dense_layout,
    )
    if do_timing:
        C, compute_ms = result
        op_total_ms = symbolic_ms + compute_ms
    else:
        C = result

    meta = None
    if return_meta:
        meta = {
            "op": op_name,
            "route": route,
            "symbolic_ms": symbolic_ms,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "dense_layout": dense_layout,
            "b_stride": tuple(int(v) for v in canonical_B.stride()),
            "c_stride": tuple(int(v) for v in C.stride()),
            "output_layout": _dense_layout_name(C),
        }
    if return_time and return_meta:
        return C, op_total_ms, meta
    if return_time:
        return C, op_total_ms
    if return_meta:
        return C, meta
    return C


def flagsparse_spmm_coo(
    data,
    row,
    col,
    B,
    shape,
    block_n=None,
    block_nnz=256,
    out=None,
    return_time=False,
    transpose=None,
    op=None,
    return_meta=False,
    dense_layout="auto",
):
    """COO SpMM using a native Triton COO row-run kernel by default.

    op: 0/'non' for A @ B, 1/'trans' for A.T @ B,
    2/'conj' for A.conj().T @ B.
    """
    return _run_spmm_coo_route(
        data,
        row,
        col,
        B,
        shape,
        block_n=block_n,
        block_nnz=block_nnz,
        out=out,
        return_time=return_time,
        return_meta=return_meta,
        route="rowrun",
        op=op,
        transpose=transpose,
        dense_layout=dense_layout,
    )


def _build_spmm_coo_pytorch_reference_from_canonical(
    canonical_data,
    canonical_row,
    canonical_col,
    canonical_B,
    shape,
    output_dtype,
):
    canonical_coo = _build_torch_sparse_coo(
        canonical_data,
        canonical_row,
        canonical_col,
        shape,
    )
    expected = torch.sparse.mm(canonical_coo, canonical_B)
    return expected if expected.dtype == output_dtype else expected.to(output_dtype)


def _build_spmm_coo_pytorch_reference(data, row, col, B, shape, op="non"):
    op_code = _normalize_spmm_coo_op(op)
    data, row, col, shape = _materialize_spmm_coo_op(data, row, col, shape, op_code)
    native_data, native_row, native_col, native_B, n_rows, n_cols, n_dense_cols = (
        _prepare_spmm_coo_inputs(data, row, col, B, shape)
    )
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        _,
        _,
        _,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_prepared(
        native_data,
        native_row,
        native_col,
        native_B,
        n_rows,
        n_cols,
        n_dense_cols,
    )
    native_coo = _build_torch_sparse_coo(native_data, native_row, native_col, shape)
    pytorch_format = "COO"
    pytorch_reason = None
    pytorch_op = lambda: torch.sparse.mm(native_coo, native_B)
    expected = _build_spmm_coo_pytorch_reference_from_canonical(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        shape,
        output_dtype,
    )
    return expected, pytorch_op, pytorch_format, pytorch_reason


def _benchmark_spmm_coo_canonical_route(
    canonical_data,
    canonical_row,
    canonical_col,
    canonical_B,
    n_rows,
    n_dense_cols,
    output_dtype,
    warmup,
    iters,
    block_n,
    block_nnz,
    route,
):
    route = _normalize_spmm_coo_route(route)
    op = lambda: _run_spmm_coo_canonical_route(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_dense_cols,
        output_dtype,
        block_n=block_n,
        block_nnz=block_nnz,
        return_time=False,
        route=route,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = op()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000.0
    values, steady_ms = _benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return values, steady_ms, first_call_ms


def _benchmark_spmm_coo_route(
    data,
    row,
    col,
    B,
    shape,
    warmup,
    iters,
    block_n,
    block_nnz,
    route,
    op="non",
    dense_layout="row",
):
    route = _normalize_spmm_coo_route(route)
    dense_layout = _normalize_dense_layout(dense_layout)
    run = lambda: _run_spmm_coo_route(
        data,
        row,
        col,
        B,
        shape,
        block_n=block_n,
        block_nnz=block_nnz,
        return_time=False,
        route=route,
        op=op,
        dense_layout=dense_layout,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = run()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000.0
    values, steady_ms = _benchmark_cuda_op(run, warmup=warmup, iters=iters)
    return values, steady_ms, first_call_ms


def _spmm_coo_pairwise_summary(candidate, reference, value_dtype):
    metrics = _spmm_validation_metrics(candidate, reference)
    atol, rtol = _spmm_coo_reference_tolerance(value_dtype)
    if candidate.numel() == 0:
        error_ratio = 0.0
    else:
        diff = torch.abs(candidate - reference)
        denom = atol + rtol * torch.abs(reference)
        error_ratio = float(torch.max(diff / denom).item())
    return {
        "match": torch.allclose(candidate, reference, atol=atol, rtol=rtol),
        "error_ratio": error_ratio,
        "max_abs_error": metrics["max_abs_error"],
        "max_relative_error": metrics["max_relative_error"],
        "sum_relative_error": metrics["sum_relative_error"],
    }


def benchmark_spmm_coo_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=256,
    run_cusparse=True,
    route="rowrun",
    compare_routes=False,
    op="non",
    dense_layout="row",
):
    """Benchmark native COO SpMM vs PyTorch COO sparse.mm and CuPy/cuSPARSE COO @ dense."""
    selected_route = _normalize_spmm_coo_route(route)
    dense_layout = _normalize_dense_layout(dense_layout)
    op_code = _normalize_spmm_coo_op(op)
    op_name = _spmm_coo_op_to_name(op_code)
    device = torch.device("cuda")
    data, row, col = _build_random_coo(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    shape = (n_rows, n_cols)
    b_rows = n_rows if _spmm_coo_op_transposes(op_code) else n_cols
    B = _materialize_dense_layout(
        _build_random_dense((b_rows, n_dense_cols), value_dtype, device),
        dense_layout,
    )
    effective_data, effective_row, effective_col, effective_shape = (
        _materialize_spmm_coo_op(
            data,
            row,
            col,
            shape,
            op_code,
        )
    )

    native_data, native_row, native_col, native_B, _, _, _ = _prepare_spmm_coo_inputs(
        effective_data,
        effective_row,
        effective_col,
        B,
        effective_shape,
        dense_layout=dense_layout,
    )
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        n_rows,
        n_cols,
        n_dense_cols,
        output_dtype,
        _,
    ) = _prepare_spmm_coo_canonical_prepared(
        native_data,
        native_row,
        native_col,
        native_B,
        n_rows,
        n_cols,
        native_B.shape[1],
        dense_layout=dense_layout,
    )
    launch = _resolve_spmm_coo_launch_config(
        n_dense_cols,
        canonical_data.numel(),
        block_n=block_n,
        block_nnz=block_nnz,
    )
    seg_starts = _seg_starts_from_sorted_rows(
        canonical_row, canonical_data.numel(), device
    )
    n_row_runs = int(seg_starts.numel()) - 1 if seg_starts is not None else 0
    cusparse_data, cusparse_row, cusparse_col = _coalesce_coo_entries(
        native_data,
        native_row,
        native_col,
        effective_shape,
    )
    cusparse_data, cusparse_row, cusparse_col = _sort_coo_lex_inplace(
        cusparse_data,
        cusparse_row,
        cusparse_col,
        effective_shape[1],
    )

    expected = _build_spmm_coo_pytorch_reference_from_canonical(
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        effective_shape,
        output_dtype,
    )
    pytorch_coo = _build_torch_sparse_coo(
        native_data, native_row, native_col, effective_shape
    )
    pytorch_op = lambda: torch.sparse.mm(pytorch_coo, native_B)
    pytorch_format = "COO"
    pytorch_reason = None

    triton_C, triton_ms, triton_first_call_ms = _benchmark_spmm_coo_route(
        data,
        row,
        col,
        B,
        shape,
        warmup,
        iters,
        launch["block_n"],
        launch["block_nnz"],
        selected_route,
        op=op_name,
        dense_layout=dense_layout,
    )
    triton_summary = _spmm_coo_pairwise_summary(triton_C, expected, value_dtype)
    triton_match = triton_summary["match"]

    pytorch_values = expected
    pytorch_ms = None
    try:
        pytorch_values, pytorch_ms = _benchmark_cuda_op(
            pytorch_op, warmup=warmup, iters=iters
        )
    except Exception as exc:
        pytorch_reason = (
            str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"
        )

    cusparse_ms = None
    cusparse_match = None
    cusparse_reason = None
    cusparse_values = None
    cusparse_summary = None
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    sparse_ref_backend, sparse_ref_reason = _spmm_coo_sparse_ref_backend(
        value_dtype, cusparse_row.dtype
    )
    if run_cusparse and sparse_ref_backend == "hipsparse":
        # DCU/ROCm: hipSPARSE COO SpMM stands in for the cuSPARSE baseline. The
        # op has already been materialized into cusparse_row/col/data above, so the
        # non-transpose hipSPARSE entry point covers every op here.
        try:
            cusparse_values, cusparse_ms = _benchmark_prepared_cuda_op(
                lambda: _prepare_spmm_coo_ref_hipsparse(
                    cusparse_data, cusparse_row, cusparse_col, native_B, effective_shape
                ),
                _run_spmm_coo_ref_hipsparse_prepared,
                _destroy_spmm_coo_ref_hipsparse_prepared,
                warmup=warmup,
                iters=iters,
            )
            cusparse_summary = _spmm_coo_pairwise_summary(
                cusparse_values, expected, value_dtype
            )
            cusparse_match = cusparse_summary["match"]
        except Exception as exc:
            cusparse_reason = str(exc)
    elif run_cusparse:
        if cp is None or cpx_sparse is None:
            cusparse_reason = sparse_ref_reason or "CuPy/cuSPARSE is not available"
        elif value_dtype not in _cupy_supported_dtypes:
            cusparse_reason = "float16/bfloat16 not supported by CuPy sparse; skipped"
        else:
            try:
                data_cp = _cupy_from_torch(cusparse_data)
                row_cp = _cupy_from_torch(cusparse_row.to(torch.int64))
                col_cp = _cupy_from_torch(cusparse_col.to(torch.int64))
                B_cp = _cupy_from_torch(native_B)
                A_coo = cpx_sparse.coo_matrix(
                    (data_cp, (row_cp, col_cp)), shape=effective_shape
                )
                cusparse_values_cp, cusparse_ms = _benchmark_cuda_op(
                    lambda: A_coo @ B_cp, warmup=warmup, iters=iters
                )
                cusparse_values = _torch_from_cupy(cusparse_values_cp)
                cusparse_summary = _spmm_coo_pairwise_summary(
                    cusparse_values, expected, value_dtype
                )
                cusparse_match = cusparse_summary["match"]
            except Exception as exc:
                cusparse_reason = str(exc)

    route_results = None
    parity = None
    route_samples = None
    if compare_routes:
        route_outputs = {selected_route: triton_C}
        route_results = {
            selected_route: {
                "route": selected_route,
                "ms": triton_ms,
                "first_call_ms": triton_first_call_ms,
                "match_reference": triton_summary["match"],
                "error_ratio": triton_summary["error_ratio"],
                "max_abs_error": triton_summary["max_abs_error"],
                "max_relative_error": triton_summary["max_relative_error"],
                "match_cusparse": (
                    None
                    if cusparse_values is None
                    else torch.allclose(
                        triton_C,
                        cusparse_values,
                        atol=_spmm_coo_reference_tolerance(value_dtype)[0],
                        rtol=_spmm_coo_reference_tolerance(value_dtype)[1],
                    )
                ),
                "error": None,
            }
        }

        for extra_route in ("rowrun", "atomic"):
            if extra_route in route_outputs:
                continue
            try:
                extra_values, extra_ms, extra_first_call_ms = _benchmark_spmm_coo_route(
                    data,
                    row,
                    col,
                    B,
                    shape,
                    warmup,
                    iters,
                    launch["block_n"],
                    launch["block_nnz"],
                    extra_route,
                    op=op_name,
                    dense_layout=dense_layout,
                )
                extra_summary = _spmm_coo_pairwise_summary(
                    extra_values, expected, value_dtype
                )
                route_outputs[extra_route] = extra_values
                route_results[extra_route] = {
                    "route": extra_route,
                    "ms": extra_ms,
                    "first_call_ms": extra_first_call_ms,
                    "match_reference": extra_summary["match"],
                    "error_ratio": extra_summary["error_ratio"],
                    "max_abs_error": extra_summary["max_abs_error"],
                    "max_relative_error": extra_summary["max_relative_error"],
                    "match_cusparse": (
                        None
                        if cusparse_values is None
                        else torch.allclose(
                            extra_values,
                            cusparse_values,
                            atol=_spmm_coo_reference_tolerance(value_dtype)[0],
                            rtol=_spmm_coo_reference_tolerance(value_dtype)[1],
                        )
                    ),
                    "error": None,
                }
            except Exception as exc:
                route_results[extra_route] = {
                    "route": extra_route,
                    "ms": None,
                    "first_call_ms": None,
                    "match_reference": False,
                    "error_ratio": None,
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "match_cusparse": None,
                    "error": str(exc),
                }

        def _safe_parity(lhs, rhs):
            if lhs in route_outputs and rhs in route_outputs:
                return _spmm_coo_pairwise_summary(
                    route_outputs[lhs], route_outputs[rhs], value_dtype
                )
            return {
                "match": None,
                "error_ratio": None,
                "max_abs_error": None,
                "max_relative_error": None,
                "sum_relative_error": None,
            }

        parity = {
            "rowrun_vs_atomic": _safe_parity("rowrun", "atomic"),
        }
        route_samples = route_outputs
    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if (pytorch_ms is not None and triton_ms > 0) else None
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms if (cusparse_ms is not None and triton_ms > 0) else None
    )
    threshold = _spmm_relative_threshold(value_dtype)
    return {
        "parameters": {
            "format": "coo",
            "internal_format": f"native-{selected_route}",
            "route": selected_route,
            "op": op_name,
            "dense_layout": dense_layout,
            "b_stride": tuple(int(v) for v in native_B.stride()),
            "c_stride": tuple(int(v) for v in triton_C.stride()),
            "output_layout": _dense_layout_name(triton_C),
            "compare_routes": bool(compare_routes),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "required_nnz_tiles": launch["required_nnz_tiles"],
            "heuristic_warp_size": launch["heuristic_warp_size"],
            "heuristic_factor": launch["heuristic_factor"],
            "n_row_runs": n_row_runs,
            "run_cusparse": run_cusparse,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "triton_first_call_ms": triton_first_call_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_summary["max_abs_error"],
            "triton_max_abs_error": triton_summary["max_abs_error"],
            "triton_max_relative_error": triton_summary["max_relative_error"],
            "triton_sum_relative_error": triton_summary["sum_relative_error"],
            "triton_relative_threshold": threshold,
            "triton_strict_allclose_match": triton_match,
            "pytorch_match_reference": True,
            "pytorch_max_error": 0.0,
            "pytorch_max_abs_error": 0.0,
            "pytorch_max_relative_error": 0.0,
            "pytorch_sum_relative_error": 0.0,
            "pytorch_relative_threshold": threshold,
            "cusparse_match_reference": cusparse_match,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": (
                cusparse_summary["max_abs_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_max_abs_error": (
                cusparse_summary["max_abs_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_max_relative_error": (
                cusparse_summary["max_relative_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_sum_relative_error": (
                cusparse_summary["sum_relative_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_relative_threshold": threshold,
            "cusparse_strict_allclose_match": cusparse_match,
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "pytorch_sparse_format": pytorch_format,
            "cusparse_unavailable_reason": cusparse_reason,
            "flagsparse_internal_route": f"coo-native-{selected_route}",
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_C,
            "reference": expected,
            "cusparse": cusparse_values,
        },
        "route_results": route_results,
        "parity": parity,
        "route_samples": route_samples,
    }


def comprehensive_spmm_coo_test(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=256,
    run_cusparse=True,
    op="non",
    dense_layout="row",
):
    """Full COO SpMM benchmark entry for one configuration."""
    return benchmark_spmm_coo_case(
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        n_dense_cols=n_dense_cols,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        block_n=block_n,
        block_nnz=block_nnz,
        run_cusparse=run_cusparse,
        op=op,
        dense_layout=dense_layout,
    )
