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

"""Native Blocked-ELL (BELL) SpMM kernels and route helpers."""

from dataclasses import dataclass

from ._common import *

import triton
import triton.language as tl


SUPPORTED_SPMM_BELL_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SUPPORTED_SPMM_BELL_INDEX_DTYPES = SUPPORTED_INDEX_DTYPES

SPMM_BELL_OP_NON = 0
SPMM_BELL_OP_TRANS = 1
SPMM_BELL_OP_CONJ_TRANS = 2
SPMM_BELL_OP_NAMES = {
    SPMM_BELL_OP_NON: "non",
    SPMM_BELL_OP_TRANS: "trans",
    SPMM_BELL_OP_CONJ_TRANS: "conj",
}
SPMM_BELL_SUPPORTED_OP_NAMES = ("non",)
_SPMM_BELL_OP_NAME_TO_CODE = {
    name: code for code, name in SPMM_BELL_OP_NAMES.items()
}

SPMM_BELL_ALG_BASE = "spmm_bell_base"
_SPMM_BELL_RESERVED_OP_MESSAGE = (
    "spmm_bell_base supports op='non' only; trans/conj are reserved"
)


class SpmmBellAlgorithmUnavailable(RuntimeError):
    """Raised when a requested BELL SpMM route is unavailable."""


@dataclass(frozen=True)
class SpmmBellAlgorithm:
    name: str
    display_name: str
    supported_ops: tuple[str, ...]
    supported_dtypes: tuple
    run: object


def _normalize_spmm_bell_op(op=None, transpose=False):
    if op is None:
        return SPMM_BELL_OP_TRANS if bool(transpose) else SPMM_BELL_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMM_BELL_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMM_BELL_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMM_BELL_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmm_bell_op_to_name(op):
    return SPMM_BELL_OP_NAMES[_normalize_spmm_bell_op(op)]


def _spmm_bell_op_transposes(op):
    return _normalize_spmm_bell_op(op) in (
        SPMM_BELL_OP_TRANS,
        SPMM_BELL_OP_CONJ_TRANS,
    )


def _ensure_spmm_bell_supported_op(op_code):
    op_name = _spmm_bell_op_to_name(op_code)
    if op_name not in SPMM_BELL_SUPPORTED_OP_NAMES:
        raise ValueError(_SPMM_BELL_RESERVED_OP_MESSAGE)


def _normalize_spmm_bell_alg(alg):
    token = "auto" if alg is None else str(alg).strip().lower().replace("-", "_")
    if token in ("auto", "base", "bell_base", "spmm_bell_base"):
        return "auto" if token == "auto" else SPMM_BELL_ALG_BASE
    raise ValueError("unsupported BELL SpMM algorithm; supported: auto, spmm_bell_base")


def _select_block_n(n_dense_cols, dtype, device=None):
    rocm_launch = _spmm_rocm_launch_overrides(
        n_dense_cols=n_dense_cols,
        fmt="bell",
        dtype=dtype,
        device=device,
    )
    if rocm_launch is not None and rocm_launch.get("block_n") is not None:
        return int(rocm_launch["block_n"])
    if dtype in (torch.float64, torch.complex128):
        return 16 if n_dense_cols >= 16 else 8
    return 32 if n_dense_cols >= 32 else 16


def _resolve_spmm_bell_launch(prepared, n_dense_cols):
    backend_info = _get_device_backend_info(prepared.data.device)
    rocm_launch = _spmm_rocm_launch_overrides(
        n_dense_cols=n_dense_cols,
        nnz=prepared.stored_nnz,
        fmt="bell",
        dtype=prepared.data.dtype,
        device=prepared.data.device,
    )
    if rocm_launch is None:
        block_n = (
            prepared.block_n
            if prepared.block_n_override
            else _select_block_n(n_dense_cols, prepared.data.dtype, prepared.data.device)
        )
        return {
            "block_n": int(block_n),
            "num_warps": 4,
            "num_stages": 1,
            "launch_backend": backend_info["backend"],
            "device_warp_size": int(backend_info["device_warp_size"]),
        }
    block_n = prepared.block_n if prepared.block_n_override else rocm_launch["block_n"]
    return {
        "block_n": int(block_n),
        "num_warps": int(rocm_launch["num_warps"]),
        "num_stages": int(rocm_launch["num_stages"]),
        "launch_backend": rocm_launch["backend"],
        "device_warp_size": int(rocm_launch["device_warp_size"]),
    }


class PreparedBellSpmm:
    """Prepared BELL metadata for native SpMM routes."""

    __slots__ = (
        "data",
        "indices",
        "shape",
        "n_rows",
        "n_cols",
        "n_block_rows",
        "n_block_cols",
        "ell_width_blocks",
        "block_dim",
        "stored_nnz",
        "block_n",
        "block_n_override",
        "op",
        "alg",
    )

    def __init__(
        self,
        *,
        data,
        indices,
        shape,
        n_rows,
        n_cols,
        n_block_rows,
        n_block_cols,
        ell_width_blocks,
        block_dim,
        stored_nnz,
        block_n,
        op="non",
        alg="auto",
    ):
        self.data = data
        self.indices = indices
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.n_block_rows = int(n_block_rows)
        self.n_block_cols = int(n_block_cols)
        self.ell_width_blocks = int(ell_width_blocks)
        self.block_dim = int(block_dim)
        self.stored_nnz = int(stored_nnz)
        self.block_n = int(block_n) if block_n is not None else 0
        self.block_n_override = block_n is not None
        self.op = _spmm_bell_op_to_name(op)
        self.alg = _normalize_spmm_bell_alg(alg)


@triton.jit
def _spmm_bell_non_real_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    c_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    dense_cols: tl.constexpr,
    ell_width_blocks: tl.constexpr,
    block_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    tile = tl.program_id(2)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    out_row = brow * block_dim + inner_row
    mask_n = cols < dense_cols
    bcol0 = tl.load(indices_ptr + brow * ell_width_blocks)
    k0 = bcol0 * block_dim
    valid0 = (bcol0 >= 0) & (k0 < n_cols)
    a0 = tl.load(
        data_ptr + ((brow * ell_width_blocks) * block_dim + inner_row) * block_dim,
        mask=valid0,
        other=0.0,
    )
    b0 = tl.load(
        b_ptr + k0 * dense_cols + cols,
        mask=valid0 & mask_n,
        other=0.0,
    )
    acc = a0 * b0

    for inner_col in range(1, block_dim):
        k = bcol0 * block_dim + inner_col
        valid = (bcol0 >= 0) & (k < n_cols)
        a_off = ((brow * ell_width_blocks) * block_dim + inner_row) * block_dim + inner_col
        a = tl.load(data_ptr + a_off, mask=valid, other=0.0)
        b = tl.load(b_ptr + k * dense_cols + cols, mask=valid & mask_n, other=0.0)
        acc += a * b

    for slot in range(1, ell_width_blocks):
        bcol = tl.load(indices_ptr + brow * ell_width_blocks + slot)
        for inner_col in range(0, block_dim):
            k = bcol * block_dim + inner_col
            valid = (bcol >= 0) & (k < n_cols)
            a_off = ((brow * ell_width_blocks + slot) * block_dim + inner_row) * block_dim + inner_col
            a = tl.load(data_ptr + a_off, mask=valid, other=0.0)
            b = tl.load(b_ptr + k * dense_cols + cols, mask=valid & mask_n, other=0.0)
            acc += a * b

    tl.store(c_ptr + out_row * dense_cols + cols, acc, mask=(out_row < n_rows) & mask_n)


@triton.jit
def _spmm_bell_non_complex_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    c_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    dense_cols: tl.constexpr,
    ell_width_blocks: tl.constexpr,
    block_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    tile = tl.program_id(2)
    cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    out_row = brow * block_dim + inner_row
    mask_n = cols < dense_cols
    bcol0 = tl.load(indices_ptr + brow * ell_width_blocks)
    k0 = bcol0 * block_dim
    valid0 = (bcol0 >= 0) & (k0 < n_cols)
    a_off0 = ((brow * ell_width_blocks) * block_dim + inner_row) * block_dim * 2
    b_off0 = (k0 * dense_cols + cols) * 2
    ar0 = tl.load(data_ptr + a_off0, mask=valid0, other=0.0)
    ai0 = tl.load(data_ptr + a_off0 + 1, mask=valid0, other=0.0)
    br0 = tl.load(b_ptr + b_off0, mask=valid0 & mask_n, other=0.0)
    bi0 = tl.load(b_ptr + b_off0 + 1, mask=valid0 & mask_n, other=0.0)
    acc_r = ar0 * br0 - ai0 * bi0
    acc_i = ar0 * bi0 + ai0 * br0

    for inner_col in range(1, block_dim):
        k = bcol0 * block_dim + inner_col
        valid = (bcol0 >= 0) & (k < n_cols)
        a_off = (((brow * ell_width_blocks) * block_dim + inner_row) * block_dim + inner_col) * 2
        b_off = (k * dense_cols + cols) * 2
        ar = tl.load(data_ptr + a_off, mask=valid, other=0.0)
        ai = tl.load(data_ptr + a_off + 1, mask=valid, other=0.0)
        br = tl.load(b_ptr + b_off, mask=valid & mask_n, other=0.0)
        bi = tl.load(b_ptr + b_off + 1, mask=valid & mask_n, other=0.0)
        acc_r += ar * br - ai * bi
        acc_i += ar * bi + ai * br

    for slot in range(1, ell_width_blocks):
        bcol = tl.load(indices_ptr + brow * ell_width_blocks + slot)
        for inner_col in range(0, block_dim):
            k = bcol * block_dim + inner_col
            valid = (bcol >= 0) & (k < n_cols)
            a_off = (
                ((brow * ell_width_blocks + slot) * block_dim + inner_row)
                * block_dim
                + inner_col
            ) * 2
            b_off = (k * dense_cols + cols) * 2
            ar = tl.load(data_ptr + a_off, mask=valid, other=0.0)
            ai = tl.load(data_ptr + a_off + 1, mask=valid, other=0.0)
            br = tl.load(b_ptr + b_off, mask=valid & mask_n, other=0.0)
            bi = tl.load(b_ptr + b_off + 1, mask=valid & mask_n, other=0.0)
            acc_r += ar * br - ai * bi
            acc_i += ar * bi + ai * br

    c_off = (out_row * dense_cols + cols) * 2
    mask = (out_row < n_rows) & mask_n
    tl.store(c_ptr + c_off, acc_r, mask=mask)
    tl.store(c_ptr + c_off + 1, acc_i, mask=mask)


def _prepare_spmm_bell_matrix(data, indices, shape, block_dim=None):
    if not isinstance(data, torch.Tensor) or not isinstance(indices, torch.Tensor):
        raise TypeError("BELL SpMM inputs must be torch.Tensor")
    if data.dim() != 4:
        raise ValueError("BELL data must have shape (mb, ell_width_blocks, block_dim, block_dim)")
    if indices.dim() != 2:
        raise ValueError("BELL indices must have shape (mb, ell_width_blocks)")
    if data.device.type != "cuda" or indices.device.type != "cuda":
        raise ValueError("BELL SpMM inputs must be CUDA tensors")
    if data.device != indices.device:
        raise ValueError("BELL data and indices must be on the same device")
    if data.dtype not in SUPPORTED_SPMM_BELL_VALUE_DTYPES:
        raise TypeError("BELL data must be float32, float64, complex64, or complex128")
    if indices.dtype not in SUPPORTED_SPMM_BELL_INDEX_DTYPES:
        raise TypeError("BELL indices must be int32 or int64")
    if len(shape) != 2:
        raise ValueError("shape must be a pair (M, K)")

    n_rows = int(shape[0])
    n_cols = int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    n_block_rows = int(data.shape[0])
    ell_width_blocks = int(data.shape[1])
    inferred_block_dim = int(data.shape[2])
    if int(data.shape[3]) != inferred_block_dim:
        raise ValueError("BELL SpMM v1 requires square blocks")
    if block_dim is not None and int(block_dim) != inferred_block_dim:
        raise ValueError("block_dim must match data.shape[2:]")
    if inferred_block_dim <= 0:
        raise ValueError("block_dim must be positive")
    if tuple(indices.shape) != (n_block_rows, ell_width_blocks):
        raise ValueError("indices shape must match data block row and ell_width_blocks")

    expected_block_rows = (n_rows + inferred_block_dim - 1) // inferred_block_dim
    expected_block_cols = (n_cols + inferred_block_dim - 1) // inferred_block_dim
    if n_block_rows != expected_block_rows:
        raise ValueError("data.shape[0] must equal ceil(shape[0] / block_dim)")
    if ell_width_blocks <= 0:
        raise ValueError("ell_width_blocks must be positive")

    if indices.numel():
        min_index = int(torch.min(indices).item())
        max_index = int(torch.max(indices).item())
        if min_index < -1:
            raise IndexError("BELL padding must use column index -1")
        if max_index >= expected_block_cols:
            raise IndexError("BELL block column index is out of range")
        invalid_seen = torch.cumsum((indices < 0).to(torch.int64), dim=1) > 0
        if bool(torch.any((indices >= 0) & invalid_seen).item()):
            raise IndexError("BELL padding (-1) must be trailing within each block row")

    if not data.is_contiguous():
        data = data.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    valid_blocks = int(torch.count_nonzero(indices >= 0).item()) if indices.numel() else 0
    stored_nnz = valid_blocks * inferred_block_dim * inferred_block_dim
    return (
        data,
        indices,
        (n_rows, n_cols),
        n_block_rows,
        expected_block_cols,
        ell_width_blocks,
        inferred_block_dim,
        stored_nnz,
    )


def prepare_spmm_bell_route(
    data,
    indices,
    shape,
    *,
    block_dim=None,
    op="non",
    alg="auto",
    block_n=None,
):
    op_code = _normalize_spmm_bell_op(op)
    _ensure_spmm_bell_supported_op(op_code)
    resolved_alg = _normalize_spmm_bell_alg(alg)
    (
        data,
        indices,
        shape,
        n_block_rows,
        n_block_cols,
        ell_width_blocks,
        block_dim,
        stored_nnz,
    ) = _prepare_spmm_bell_matrix(data, indices, shape, block_dim)
    block_n = int(block_n) if block_n is not None else None
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n must be positive")
    resolve_spmm_bell_algorithm(resolved_alg, _spmm_bell_op_to_name(op_code), data.dtype)
    return PreparedBellSpmm(
        data=data,
        indices=indices,
        shape=shape,
        n_rows=shape[0],
        n_cols=shape[1],
        n_block_rows=n_block_rows,
        n_block_cols=n_block_cols,
        ell_width_blocks=ell_width_blocks,
        block_dim=block_dim,
        stored_nnz=stored_nnz,
        block_n=block_n,
        op=_spmm_bell_op_to_name(op_code),
        alg=resolved_alg,
    )


def _validate_spmm_bell_B(B, prepared):
    if not isinstance(B, torch.Tensor):
        raise TypeError("B must be a torch.Tensor")
    if B.dim() != 2:
        raise ValueError("B must be a 2D dense matrix")
    if B.device != prepared.data.device:
        raise ValueError("B device must match the prepared BELL matrix")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match BELL data dtype")
    if int(B.shape[0]) != prepared.n_cols:
        raise ValueError("B must have shape (K, N), where K matches shape[1]")
    if not B.is_contiguous():
        B = B.contiguous()
    return B


def _triton_spmm_bell_base_kernel(prepared, B):
    dense_cols = int(B.shape[1])
    C = torch.empty((prepared.n_rows, dense_cols), device=B.device, dtype=B.dtype)
    if dense_cols == 0 or prepared.n_rows == 0:
        return C
    launch = _resolve_spmm_bell_launch(prepared, dense_cols)
    block_n = launch["block_n"]
    grid = (
        prepared.n_block_rows,
        prepared.block_dim,
        triton.cdiv(dense_cols, block_n),
    )
    if _is_complex_dtype(prepared.data.dtype):
        data_view = torch.view_as_real(prepared.data)
        B_view = torch.view_as_real(B)
        C_view = torch.view_as_real(C)
        _spmm_bell_non_complex_kernel[grid](
            data_view,
            prepared.indices,
            B_view,
            C_view,
            prepared.n_rows,
            prepared.n_cols,
            dense_cols,
            prepared.ell_width_blocks,
            prepared.block_dim,
            BLOCK_N=block_n,
            num_warps=launch["num_warps"],
            num_stages=launch["num_stages"],
        )
    else:
        _spmm_bell_non_real_kernel[grid](
            prepared.data,
            prepared.indices,
            B,
            C,
            prepared.n_rows,
            prepared.n_cols,
            dense_cols,
            prepared.ell_width_blocks,
            prepared.block_dim,
            BLOCK_N=block_n,
            num_warps=launch["num_warps"],
            num_stages=launch["num_stages"],
        )
    return C


def _run_spmm_bell_base_route(prepared, B, *, timing=False, diagnostics=False):
    del diagnostics
    compute_ms = None
    launch = _resolve_spmm_bell_launch(prepared, int(B.shape[1]))
    if timing:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _triton_spmm_bell_base_kernel(prepared, B)
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    process_cpu_ms = 0.0
    meta = {
        "alg": SPMM_BELL_ALG_BASE,
        "op": prepared.op,
        "logical_shape": prepared.shape,
        "block_dim": prepared.block_dim,
        "n_block_rows": prepared.n_block_rows,
        "n_block_cols": prepared.n_block_cols,
        "ell_width_blocks": prepared.ell_width_blocks,
        "stored_nnz": prepared.stored_nnz,
        "block_n": launch["block_n"],
        "num_warps": launch["num_warps"],
        "num_stages": launch["num_stages"],
        "launch_backend": launch["launch_backend"],
        "device_warp_size": launch["device_warp_size"],
        "process_cpu_ms": process_cpu_ms,
        "op_total_ms": compute_ms,
        "compute_ms": compute_ms if timing else None,
        "process_gpu_ms": 0.0 if timing else None,
    }
    return C, meta


SPMM_BELL_ALGORITHMS = {
    SPMM_BELL_ALG_BASE: SpmmBellAlgorithm(
        name=SPMM_BELL_ALG_BASE,
        display_name="BELL base",
        supported_ops=SPMM_BELL_SUPPORTED_OP_NAMES,
        supported_dtypes=SUPPORTED_SPMM_BELL_VALUE_DTYPES,
        run=_run_spmm_bell_base_route,
    )
}


def resolve_spmm_bell_algorithm(alg, op, dtype):
    token = _normalize_spmm_bell_alg(alg)
    if token == "auto":
        token = SPMM_BELL_ALG_BASE
    if token not in SPMM_BELL_ALGORITHMS:
        supported = ", ".join(sorted(SPMM_BELL_ALGORITHMS))
        raise SpmmBellAlgorithmUnavailable(f"unsupported BELL SpMM algorithm: {token}; supported: {supported}")
    algorithm = SPMM_BELL_ALGORITHMS[token]
    op_name = _spmm_bell_op_to_name(op)
    if op_name not in algorithm.supported_ops:
        raise SpmmBellAlgorithmUnavailable(_SPMM_BELL_RESERVED_OP_MESSAGE)
    if dtype not in algorithm.supported_dtypes:
        raise SpmmBellAlgorithmUnavailable(f"{algorithm.name} does not support dtype {dtype}")
    return algorithm


def list_spmm_bell_algorithms(op=None, dtype=None):
    op_name = None if op is None else _spmm_bell_op_to_name(op)
    out = []
    for name, algorithm in SPMM_BELL_ALGORITHMS.items():
        if op_name is not None and op_name not in algorithm.supported_ops:
            continue
        if dtype is not None and dtype not in algorithm.supported_dtypes:
            continue
        out.append(name)
    return tuple(out)


def flagsparse_spmm_bell_run(
    prepared,
    B,
    *,
    op=None,
    alg=None,
    return_time=False,
    return_meta=False,
    timing=False,
):
    if not isinstance(prepared, PreparedBellSpmm):
        raise TypeError("prepared must be a PreparedBellSpmm")
    op_name = prepared.op if op is None else _spmm_bell_op_to_name(op)
    _ensure_spmm_bell_supported_op(_normalize_spmm_bell_op(op_name))
    if op_name != prepared.op:
        raise ValueError(f"op={op_name} does not match prepared.op={prepared.op}")
    alg_name = prepared.alg if alg is None else _normalize_spmm_bell_alg(alg)
    algorithm = resolve_spmm_bell_algorithm(alg_name, op_name, prepared.data.dtype)
    B = _validate_spmm_bell_B(B, prepared)
    collect_timing = bool(return_time or return_meta)
    if collect_timing:
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
    C, route_meta = algorithm.run(prepared, B, timing=timing)
    if collect_timing:
        event_end.record()
        torch.cuda.synchronize()
        gpu_ms = event_start.elapsed_time(event_end)
    else:
        gpu_ms = None
    process_cpu_ms = float(route_meta.get("process_cpu_ms", 0.0) or 0.0)
    operator_ms = process_cpu_ms + float(gpu_ms) if gpu_ms is not None else None
    if return_meta:
        meta = dict(route_meta)
        meta.update(
            {
                "alg": algorithm.name,
                "display_name": algorithm.display_name,
                "op": op_name,
                "operator_ms": operator_ms,
                "gpu_ms": gpu_ms,
                "process_cpu_ms": process_cpu_ms,
            }
        )
        if timing:
            meta["op_total_ms"] = (
                process_cpu_ms
                + float(meta.get("process_gpu_ms", 0.0) or 0.0)
                + float(meta.get("compute_ms", 0.0) or 0.0)
            )
        else:
            meta["op_total_ms"] = operator_ms
        if return_time:
            return C, operator_ms, meta
        return C, meta
    if return_time:
        return C, operator_ms
    return C


def flagsparse_spmm_bell(
    data=None,
    indices=None,
    B=None,
    shape=None,
    *,
    prepared=None,
    block_dim=None,
    op=None,
    transpose=None,
    alg="auto",
    block_n=None,
    return_time=False,
    return_meta=False,
    timing=False,
):
    op_explicit = op is not None
    op_code = _normalize_spmm_bell_op(
        op if op_explicit else None,
        transpose=False if transpose is None else bool(transpose),
    )
    if op_explicit and transpose is not None and bool(transpose) != _spmm_bell_op_transposes(op_code):
        raise ValueError("transpose conflicts with op")
    _ensure_spmm_bell_supported_op(op_code)
    if prepared is None:
        if data is None or indices is None or shape is None:
            raise ValueError("data, indices, and shape are required when prepared is not provided")
        prepared = prepare_spmm_bell_route(
            data,
            indices,
            shape,
            block_dim=block_dim,
            op=_spmm_bell_op_to_name(op_code),
            alg=alg,
            block_n=block_n,
        )
    else:
        if op_explicit and _spmm_bell_op_to_name(op_code) != prepared.op:
            raise ValueError(f"op={_spmm_bell_op_to_name(op_code)} does not match prepared.op={prepared.op}")
        if transpose is not None and bool(transpose) != _spmm_bell_op_transposes(prepared.op):
            raise ValueError("transpose conflicts with prepared op")
        op_code = _normalize_spmm_bell_op(prepared.op)
    return flagsparse_spmm_bell_run(
        prepared,
        B,
        op=_spmm_bell_op_to_name(op_code),
        alg=alg,
        return_time=return_time,
        return_meta=return_meta,
        timing=timing,
    )
