"""Experimental Triton port of AlphaSparse CUDA CSR ALG1 for SpMM."""

import importlib

from ._common import *
from ._alpha_spmm_alg1_common import (
    _build_alpha_spmm_alg1_launch_meta,
    _select_alpha_spmm_alg1_warp_and_factor,
)
from .spmm_csr import _prepare_spmm_csr_matrix


def _load_tle_language():
    errors = []
    for module_name in (
        "triton.experimental.tle.language",
        "triton.experimental.tle",
        "triton.language.extra",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on FlagTree runtime.
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if hasattr(module, "gpu"):
            return module, None
        errors.append(f"{module_name}: imported, but has no gpu namespace")
    return None, RuntimeError("No FlagTree TLE module with a gpu namespace was found: " + "; ".join(errors))


tle, _TLE_IMPORT_ERROR = _load_tle_language()


SUPPORTED_ALPHA_SPMM_ALG1_DTYPES = (torch.float32, torch.float64)
_ALPHA_SPMM_ALG1_NUM_WARPS = 8
_ALPHA_SPMM_ALG1_NUM_STAGES = 1


class PreparedAlphaSpmmAlg1:
    """Cached CSR metadata for repeated alpha_spmm_alg1 calls."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_lengths",
        "max_row_nnz",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.row_lengths = row_lengths
        self.max_row_nnz = int(max_row_nnz)


def _prepare_alpha_spmm_alg1_common(data, indices, indptr, shape):
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    if data.dtype not in SUPPORTED_ALPHA_SPMM_ALG1_DTYPES:
        raise TypeError("alpha_spmm_alg1 TLE routes only support float32 and float64")
    return PreparedAlphaSpmmAlg1(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
    )


def prepare_alpha_spmm_alg1(data, indices, indptr, shape):
    """Prepare CSR metadata for the baseline ALG1 route."""
    return _prepare_alpha_spmm_alg1_common(data, indices, indptr, shape)


def prepare_alpha_spmm_alg1_tle(data, indices, indptr, shape):
    """Prepare CSR metadata for the TLE-Struct ALG1 route."""
    return _prepare_alpha_spmm_alg1_common(data, indices, indptr, shape)


def prepare_alpha_spmm_alg1_tle_opt(data, indices, indptr, shape):
    """Prepare CSR metadata for the TLE-Struct ALG1 optimization route."""
    return _prepare_alpha_spmm_alg1_common(data, indices, indptr, shape)


def prepare_alpha_spmm_alg1_tle_opt2(data, indices, indptr, shape):
    """Prepare CSR metadata for the TLE-Struct ALG1 shape-bucket optimization route."""
    return _prepare_alpha_spmm_alg1_common(data, indices, indptr, shape)


def _alpha_spmm_alg1_acc_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _normalize_alpha_spmm_alg1_device_props(device):
    props = torch.cuda.get_device_properties(device)
    return {
        "device_name": str(getattr(props, "name", "cuda")),
        "sm_count": int(getattr(props, "multi_processor_count", 0) or 0),
    }


def _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, out):
    if B is None:
        raise ValueError("B is required")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != prepared.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != prepared.data.device:
            raise ValueError("out must be on the same CUDA device as sparse matrix data")
        if out.shape != (prepared.n_rows, int(B.shape[1])) or out.dtype != prepared.data.dtype:
            raise ValueError("out shape/dtype must match result")
    return B.contiguous()


def _build_alpha_spmm_alg1_runtime_meta(prepared, B):
    warp_size, factor = _select_alpha_spmm_alg1_warp_and_factor(
        int(B.shape[1]),
        order_row=True,
    )
    launch = _build_alpha_spmm_alg1_launch_meta(
        prepared.n_rows,
        int(B.shape[1]),
        warp_size,
        factor,
        block_size=256,
    )
    device_props = _normalize_alpha_spmm_alg1_device_props(prepared.data.device)
    meta = {
        **device_props,
        **launch,
        "warp_size": int(warp_size),
        "factor": int(factor),
        "num_warps": _ALPHA_SPMM_ALG1_NUM_WARPS,
        "num_stages": _ALPHA_SPMM_ALG1_NUM_STAGES,
        "order_row_path": True,
        "alpha": 1,
        "beta": 0,
        "max_row_nnz": int(prepared.max_row_nnz),
    }
    return meta


def _resolve_alpha_spmm_alg1_tle_opt_launch_v0(dtype, n_dense_cols, max_row_nnz, block_cols):
    """Rollback baseline used by the first TLEOpt route before P1a tuning."""
    n_dense_cols = int(n_dense_cols)
    max_row_nnz = int(max_row_nnz)
    block_cols = int(block_cols)
    if n_dense_cols <= 16:
        num_warps = 2
    elif n_dense_cols <= 64:
        num_warps = 4
    else:
        num_warps = 8 if max_row_nnz >= 512 else 4
    if block_cols >= 128:
        num_warps = max(num_warps, 4)
    if max_row_nnz >= 1024 and n_dense_cols > 32:
        num_warps = max(num_warps, 8)
    if dtype == torch.float64:
        num_warps = min(num_warps, 8)
    else:
        num_warps = min(num_warps, 8)
    num_stages = 1 if (n_dense_cols > 64 or block_cols >= 128 or max_row_nnz >= 512) else 2
    return int(num_warps), int(num_stages)


def _resolve_alpha_spmm_alg1_tle_opt_launch_v1(dtype, n_dense_cols, max_row_nnz, block_cols):
    n_dense_cols = int(n_dense_cols)
    max_row_nnz = int(max_row_nnz)
    block_cols = int(block_cols)
    if n_dense_cols <= 8 and max_row_nnz <= 64:
        num_warps = 1
    elif n_dense_cols <= 16:
        num_warps = 2
    elif n_dense_cols <= 64:
        num_warps = 4
    else:
        num_warps = 8 if max_row_nnz >= 256 else 4
    if block_cols >= 128:
        num_warps = max(num_warps, 4)
    if max_row_nnz >= 512 and n_dense_cols > 16:
        num_warps = max(num_warps, 8)
    if dtype == torch.float64 and max_row_nnz <= 64 and n_dense_cols <= 16:
        num_warps = min(num_warps, 2)

    if max_row_nnz <= 32 and block_cols <= 16:
        num_stages = 4
    elif max_row_nnz <= 128 and n_dense_cols <= 32:
        num_stages = 3
    elif n_dense_cols <= 64 and max_row_nnz < 512:
        num_stages = 2
    else:
        num_stages = 1
    return int(max(1, min(num_warps, 8))), int(max(1, min(num_stages, 4)))


def _resolve_alpha_spmm_alg1_tle_opt2_launch(
    dtype,
    n_rows,
    n_cols,
    nnz,
    n_dense_cols,
    max_row_nnz,
    block_cols,
):
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    nnz = int(nnz)
    n_dense_cols = int(n_dense_cols)
    max_row_nnz = int(max_row_nnz)
    block_cols = int(block_cols)
    avg_row_nnz = float(nnz) / float(max(1, n_rows))
    density = float(nnz) / float(max(1, n_rows * n_cols))

    if avg_row_nnz <= 4.0 and max_row_nnz <= 64 and n_dense_cols <= 16:
        num_warps = 1
    elif avg_row_nnz <= 16.0 and n_dense_cols <= 32:
        num_warps = 2
    elif avg_row_nnz <= 64.0 and max_row_nnz <= 512:
        num_warps = 4
    else:
        num_warps = 8
    if block_cols >= 128 or n_dense_cols > 64:
        num_warps = max(num_warps, 4)
    if max_row_nnz >= 512 or density > 0.05:
        num_warps = max(num_warps, 8)
    if dtype == torch.float64 and avg_row_nnz <= 8.0 and n_dense_cols <= 16:
        num_warps = min(num_warps, 2)

    if avg_row_nnz <= 4.0 and max_row_nnz <= 32 and block_cols <= 16:
        num_stages = 4
    elif avg_row_nnz <= 16.0 and max_row_nnz <= 128 and n_dense_cols <= 32:
        num_stages = 3
    elif max_row_nnz < 512 and n_dense_cols <= 64:
        num_stages = 2
    else:
        num_stages = 1
    return int(max(1, min(num_warps, 8))), int(max(1, min(num_stages, 4)))


def _build_alpha_spmm_alg1_tle_opt_runtime_meta(prepared, B):
    meta = _build_alpha_spmm_alg1_runtime_meta(prepared, B)
    num_warps, num_stages = _resolve_alpha_spmm_alg1_tle_opt_launch_v1(
        prepared.data.dtype,
        int(B.shape[1]),
        prepared.max_row_nnz,
        meta["block_cols"],
    )
    meta.update(
        {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "loop_strategy": "block_local_row_nnz",
            "launch_version": "p1a_v1",
        }
    )
    return meta


def _build_alpha_spmm_alg1_tle_opt2_runtime_meta(prepared, B):
    meta = _build_alpha_spmm_alg1_runtime_meta(prepared, B)
    num_warps, num_stages = _resolve_alpha_spmm_alg1_tle_opt2_launch(
        prepared.data.dtype,
        prepared.n_rows,
        prepared.n_cols,
        int(prepared.data.numel()),
        int(B.shape[1]),
        prepared.max_row_nnz,
        meta["block_cols"],
    )
    meta.update(
        {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "loop_strategy": "block_local_row_nnz",
            "launch_version": "p1b_shape_bucket_v1",
        }
    )
    return meta


def _with_alpha_spmm_alg1_route(meta, route):
    out = dict(meta)
    out["route"] = route
    return out


@triton.jit
def _alpha_spmm_alg1_rowmajor_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    WARP_SIZE: tl.constexpr,
    FACTOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lane_offsets = tl.arange(0, WARP_SIZE)
    block_row_start = pid_m * BLOCK_ROWS
    block_col_start = pid_n * BLOCK_COLS
    offs0 = block_col_start + lane_offsets
    mask0 = offs0 < n_dense_cols
    offs1 = block_col_start + WARP_SIZE + lane_offsets
    mask1 = offs1 < n_dense_cols
    offs2 = block_col_start + 2 * WARP_SIZE + lane_offsets
    mask2 = offs2 < n_dense_cols
    offs3 = block_col_start + 3 * WARP_SIZE + lane_offsets
    mask3 = offs3 < n_dense_cols

    for local_row in tl.static_range(0, BLOCK_ROWS):
        row = block_row_start + local_row
        if row < n_rows:
            row_start = tl.load(indptr_ptr + row)
            row_end = tl.load(indptr_ptr + row + 1)
            acc0 = tl.zeros([WARP_SIZE], dtype=ACC_DTYPE)
            acc1 = tl.zeros([WARP_SIZE], dtype=ACC_DTYPE)
            acc2 = tl.zeros([WARP_SIZE], dtype=ACC_DTYPE)
            acc3 = tl.zeros([WARP_SIZE], dtype=ACC_DTYPE)

            # AlphaSparse: each warp consumes one WARP_SIZE-sized chunk of CSR entries.
            # Triton adaptation: one scalar element per inner iteration (tensor
            # blocks do not support constexpr indexing, so we load a_col/a_val as
            # scalars directly, matching the per-element data flow of the CUDA
            # sequential reduce). The valid guard protects both sparse loads and
            # dense B loads so tail elements neither contribute numerically nor
            # trigger extra reads from B.
            for chunk_start in tl.range(row_start, row_end, WARP_SIZE):
                for jj in tl.static_range(0, WARP_SIZE):
                    elem_idx = chunk_start + jj
                    valid = elem_idx < row_end
                    a_col = tl.load(indices_ptr + elem_idx, mask=valid, other=0)
                    a_val = tl.load(data_ptr + elem_idx, mask=valid, other=0.0).to(
                        ACC_DTYPE
                    )
                    if FACTOR > 0:
                        b0 = tl.load(
                            b_ptr + a_col * stride_bk + offs0 * stride_bn,
                            mask=mask0 & valid,
                            other=0.0,
                        )
                        acc0 = acc0 + a_val * b0.to(ACC_DTYPE)
                    if FACTOR > 1:
                        b1 = tl.load(
                            b_ptr + a_col * stride_bk + offs1 * stride_bn,
                            mask=mask1 & valid,
                            other=0.0,
                        )
                        acc1 = acc1 + a_val * b1.to(ACC_DTYPE)
                    if FACTOR > 2:
                        b2 = tl.load(
                            b_ptr + a_col * stride_bk + offs2 * stride_bn,
                            mask=mask2 & valid,
                            other=0.0,
                        )
                        acc2 = acc2 + a_val * b2.to(ACC_DTYPE)
                    if FACTOR > 3:
                        b3 = tl.load(
                            b_ptr + a_col * stride_bk + offs3 * stride_bn,
                            mask=mask3 & valid,
                            other=0.0,
                        )
                        acc3 = acc3 + a_val * b3.to(ACC_DTYPE)

            tl.store(c_ptr + row * stride_cm + offs0 * stride_cn, acc0, mask=mask0)
            if FACTOR > 1:
                tl.store(c_ptr + row * stride_cm + offs1 * stride_cn, acc1, mask=mask1)
            if FACTOR > 2:
                tl.store(c_ptr + row * stride_cm + offs2 * stride_cn, acc2, mask=mask2)
            if FACTOR > 3:
                tl.store(c_ptr + row * stride_cm + offs3 * stride_cn, acc3, mask=mask3)


def _run_alpha_spmm_alg1(prepared, B, meta):
    if prepared.n_rows == 0 or int(B.shape[1]) == 0:
        return torch.zeros(
            (prepared.n_rows, int(B.shape[1])),
            dtype=prepared.data.dtype,
            device=prepared.data.device,
        )

    C_out = torch.empty(
        (prepared.n_rows, int(B.shape[1])),
        dtype=prepared.data.dtype,
        device=prepared.data.device,
    )
    grid = (meta["grid_m"], meta["grid_n"])
    _alpha_spmm_alg1_rowmajor_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        B,
        C_out,
        prepared.n_rows,
        int(B.shape[1]),
        B.stride(0),
        B.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        WARP_SIZE=meta["warp_size"],
        FACTOR=meta["factor"],
        BLOCK_ROWS=meta["block_rows"],
        BLOCK_COLS=meta["block_cols"],
        ACC_DTYPE=_alpha_spmm_alg1_acc_dtype(prepared.data.dtype),
        num_warps=meta["num_warps"],
        num_stages=meta["num_stages"],
    )
    return C_out


@triton.jit
def _alpha_spmm_alg1_tle_rowmajor_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    WARP_SIZE: tl.constexpr,
    FACTOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    MAX_ROW_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lane_offsets = tl.arange(0, WARP_SIZE)
    local_row_offsets = tl.arange(0, BLOCK_ROWS)
    block_row_start = pid_m * BLOCK_ROWS
    block_col_start = pid_n * BLOCK_COLS
    offs0 = block_col_start + lane_offsets
    mask0 = offs0 < n_dense_cols
    offs1 = block_col_start + WARP_SIZE + lane_offsets
    mask1 = offs1 < n_dense_cols
    offs2 = block_col_start + 2 * WARP_SIZE + lane_offsets
    mask2 = offs2 < n_dense_cols
    offs3 = block_col_start + 3 * WARP_SIZE + lane_offsets
    mask3 = offs3 < n_dense_cols

    if ACC_DTYPE == tl.float64:
        s_val = tle.gpu.alloc(
            (BLOCK_ROWS, WARP_SIZE),
            dtype=tl.float64,
            scope=tle.gpu.smem,
        )
    else:
        s_val = tle.gpu.alloc(
            (BLOCK_ROWS, WARP_SIZE),
            dtype=tl.float32,
            scope=tle.gpu.smem,
        )
    s_col = tle.gpu.alloc(
        (BLOCK_ROWS, WARP_SIZE),
        dtype=tl.int32,
        scope=tle.gpu.smem,
    )

    rows = block_row_start + local_row_offsets
    row_valid = rows < n_rows
    row_start = tl.load(indptr_ptr + rows, mask=row_valid, other=0)
    row_end = tl.load(indptr_ptr + rows + 1, mask=row_valid, other=0)
    row_nnz = row_end - row_start

    zeros_2d = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=tl.int32)
    row_ids_2d = local_row_offsets[:, None] + zeros_2d
    lane_ids_2d = lane_offsets[None, :] + zeros_2d
    acc0 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc1 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc2 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc3 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)

    for chunk_offset in tl.range(0, MAX_ROW_NNZ, WARP_SIZE):
        elem_offsets = row_start[:, None] + chunk_offset + lane_ids_2d
        valid_offsets = row_valid[:, None] & ((chunk_offset + lane_ids_2d) < row_nnz[:, None])
        staged_cols = tl.load(indices_ptr + elem_offsets, mask=valid_offsets, other=0)
        staged_vals = tl.load(data_ptr + elem_offsets, mask=valid_offsets, other=0.0).to(ACC_DTYPE)
        tl.store(tle.gpu.local_ptr(s_col, indices=(row_ids_2d, lane_ids_2d)), staged_cols, mask=valid_offsets)
        tl.store(tle.gpu.local_ptr(s_val, indices=(row_ids_2d, lane_ids_2d)), staged_vals, mask=valid_offsets)

        for jj in tl.static_range(0, WARP_SIZE):
            valid = row_valid & ((chunk_offset + jj) < row_nnz)
            local_jj = tl.full((BLOCK_ROWS,), jj, dtype=tl.int32)
            a_col = tl.load(tle.gpu.local_ptr(s_col, indices=(local_row_offsets, local_jj)), mask=valid, other=0)
            a_val = tl.load(tle.gpu.local_ptr(s_val, indices=(local_row_offsets, local_jj)), mask=valid, other=0.0).to(ACC_DTYPE)
            if FACTOR > 0:
                b0 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs0[None, :] * stride_bn,
                    mask=valid[:, None] & mask0[None, :],
                    other=0.0,
                )
                acc0 = acc0 + a_val[:, None] * b0.to(ACC_DTYPE)
            if FACTOR > 1:
                b1 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs1[None, :] * stride_bn,
                    mask=valid[:, None] & mask1[None, :],
                    other=0.0,
                )
                acc1 = acc1 + a_val[:, None] * b1.to(ACC_DTYPE)
            if FACTOR > 2:
                b2 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs2[None, :] * stride_bn,
                    mask=valid[:, None] & mask2[None, :],
                    other=0.0,
                )
                acc2 = acc2 + a_val[:, None] * b2.to(ACC_DTYPE)
            if FACTOR > 3:
                b3 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs3[None, :] * stride_bn,
                    mask=valid[:, None] & mask3[None, :],
                    other=0.0,
                )
                acc3 = acc3 + a_val[:, None] * b3.to(ACC_DTYPE)

    tl.store(
        c_ptr + rows[:, None] * stride_cm + offs0[None, :] * stride_cn,
        acc0,
        mask=row_valid[:, None] & mask0[None, :],
    )
    if FACTOR > 1:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs1[None, :] * stride_cn,
            acc1,
            mask=row_valid[:, None] & mask1[None, :],
        )
    if FACTOR > 2:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs2[None, :] * stride_cn,
            acc2,
            mask=row_valid[:, None] & mask2[None, :],
        )
    if FACTOR > 3:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs3[None, :] * stride_cn,
            acc3,
            mask=row_valid[:, None] & mask3[None, :],
        )


@triton.jit
def _alpha_spmm_alg1_tle_opt_rowmajor_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    WARP_SIZE: tl.constexpr,
    FACTOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    MAX_ROW_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lane_offsets = tl.arange(0, WARP_SIZE)
    local_row_offsets = tl.arange(0, BLOCK_ROWS)
    block_row_start = pid_m * BLOCK_ROWS
    block_col_start = pid_n * BLOCK_COLS
    offs0 = block_col_start + lane_offsets
    mask0 = offs0 < n_dense_cols
    offs1 = block_col_start + WARP_SIZE + lane_offsets
    mask1 = offs1 < n_dense_cols
    offs2 = block_col_start + 2 * WARP_SIZE + lane_offsets
    mask2 = offs2 < n_dense_cols
    offs3 = block_col_start + 3 * WARP_SIZE + lane_offsets
    mask3 = offs3 < n_dense_cols

    if ACC_DTYPE == tl.float64:
        s_val = tle.gpu.alloc(
            (BLOCK_ROWS, WARP_SIZE),
            dtype=tl.float64,
            scope=tle.gpu.smem,
        )
    else:
        s_val = tle.gpu.alloc(
            (BLOCK_ROWS, WARP_SIZE),
            dtype=tl.float32,
            scope=tle.gpu.smem,
        )
    s_col = tle.gpu.alloc(
        (BLOCK_ROWS, WARP_SIZE),
        dtype=tl.int32,
        scope=tle.gpu.smem,
    )

    rows = block_row_start + local_row_offsets
    row_valid = rows < n_rows
    row_start = tl.load(indptr_ptr + rows, mask=row_valid, other=0)
    row_end = tl.load(indptr_ptr + rows + 1, mask=row_valid, other=0)
    row_nnz = row_end - row_start
    block_row_nnz = tl.max(tl.where(row_valid, row_nnz, 0), axis=0)

    zeros_2d = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=tl.int32)
    row_ids_2d = local_row_offsets[:, None] + zeros_2d
    lane_ids_2d = lane_offsets[None, :] + zeros_2d
    acc0 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc1 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc2 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc3 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)

    # TLEOpt keeps the ALG1 staging structure but limits this program's loop to
    # the current row block instead of the matrix-wide max row nnz.
    for chunk_offset in tl.range(0, block_row_nnz, WARP_SIZE):
        elem_offsets = row_start[:, None] + chunk_offset + lane_ids_2d
        valid_offsets = row_valid[:, None] & ((chunk_offset + lane_ids_2d) < row_nnz[:, None])
        staged_cols = tl.load(indices_ptr + elem_offsets, mask=valid_offsets, other=0)
        staged_vals = tl.load(data_ptr + elem_offsets, mask=valid_offsets, other=0.0).to(ACC_DTYPE)
        tl.store(tle.gpu.local_ptr(s_col, indices=(row_ids_2d, lane_ids_2d)), staged_cols, mask=valid_offsets)
        tl.store(tle.gpu.local_ptr(s_val, indices=(row_ids_2d, lane_ids_2d)), staged_vals, mask=valid_offsets)

        for jj in tl.static_range(0, WARP_SIZE):
            valid = row_valid & ((chunk_offset + jj) < row_nnz)
            local_jj = tl.full((BLOCK_ROWS,), jj, dtype=tl.int32)
            a_col = tl.load(tle.gpu.local_ptr(s_col, indices=(local_row_offsets, local_jj)), mask=valid, other=0)
            a_val = tl.load(tle.gpu.local_ptr(s_val, indices=(local_row_offsets, local_jj)), mask=valid, other=0.0).to(ACC_DTYPE)
            if FACTOR > 0:
                b0 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs0[None, :] * stride_bn,
                    mask=valid[:, None] & mask0[None, :],
                    other=0.0,
                )
                acc0 = acc0 + a_val[:, None] * b0.to(ACC_DTYPE)
            if FACTOR > 1:
                b1 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs1[None, :] * stride_bn,
                    mask=valid[:, None] & mask1[None, :],
                    other=0.0,
                )
                acc1 = acc1 + a_val[:, None] * b1.to(ACC_DTYPE)
            if FACTOR > 2:
                b2 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs2[None, :] * stride_bn,
                    mask=valid[:, None] & mask2[None, :],
                    other=0.0,
                )
                acc2 = acc2 + a_val[:, None] * b2.to(ACC_DTYPE)
            if FACTOR > 3:
                b3 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs3[None, :] * stride_bn,
                    mask=valid[:, None] & mask3[None, :],
                    other=0.0,
                )
                acc3 = acc3 + a_val[:, None] * b3.to(ACC_DTYPE)

    tl.store(
        c_ptr + rows[:, None] * stride_cm + offs0[None, :] * stride_cn,
        acc0,
        mask=row_valid[:, None] & mask0[None, :],
    )
    if FACTOR > 1:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs1[None, :] * stride_cn,
            acc1,
            mask=row_valid[:, None] & mask1[None, :],
        )
    if FACTOR > 2:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs2[None, :] * stride_cn,
            acc2,
            mask=row_valid[:, None] & mask2[None, :],
        )
    if FACTOR > 3:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs3[None, :] * stride_cn,
            acc3,
            mask=row_valid[:, None] & mask3[None, :],
        )


@triton.jit
def _alpha_spmm_alg1_tle_opt2_rowmajor_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    WARP_SIZE: tl.constexpr,
    FACTOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    MAX_ROW_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lane_offsets = tl.arange(0, WARP_SIZE)
    local_row_offsets = tl.arange(0, BLOCK_ROWS)
    block_row_start = pid_m * BLOCK_ROWS
    block_col_start = pid_n * BLOCK_COLS
    offs0 = block_col_start + lane_offsets
    mask0 = offs0 < n_dense_cols
    offs1 = block_col_start + WARP_SIZE + lane_offsets
    mask1 = offs1 < n_dense_cols
    offs2 = block_col_start + 2 * WARP_SIZE + lane_offsets
    mask2 = offs2 < n_dense_cols
    offs3 = block_col_start + 3 * WARP_SIZE + lane_offsets
    mask3 = offs3 < n_dense_cols

    if ACC_DTYPE == tl.float64:
        s_val = tle.gpu.alloc(
            (BLOCK_ROWS, WARP_SIZE),
            dtype=tl.float64,
            scope=tle.gpu.smem,
        )
    else:
        s_val = tle.gpu.alloc(
            (BLOCK_ROWS, WARP_SIZE),
            dtype=tl.float32,
            scope=tle.gpu.smem,
        )
    s_col = tle.gpu.alloc(
        (BLOCK_ROWS, WARP_SIZE),
        dtype=tl.int32,
        scope=tle.gpu.smem,
    )

    rows = block_row_start + local_row_offsets
    row_valid = rows < n_rows
    row_start = tl.load(indptr_ptr + rows, mask=row_valid, other=0)
    row_end = tl.load(indptr_ptr + rows + 1, mask=row_valid, other=0)
    row_nnz = row_end - row_start
    block_row_nnz = tl.max(tl.where(row_valid, row_nnz, 0), axis=0)

    zeros_2d = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=tl.int32)
    row_ids_2d = local_row_offsets[:, None] + zeros_2d
    lane_ids_2d = lane_offsets[None, :] + zeros_2d
    acc0 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc1 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc2 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)
    acc3 = tl.zeros((BLOCK_ROWS, WARP_SIZE), dtype=ACC_DTYPE)

    # TLEOpt2 keeps the same ALG1 staging dataflow as TLEOpt; only the host-side
    # launch heuristic is independently bucketed by matrix shape.
    for chunk_offset in tl.range(0, block_row_nnz, WARP_SIZE):
        elem_offsets = row_start[:, None] + chunk_offset + lane_ids_2d
        valid_offsets = row_valid[:, None] & ((chunk_offset + lane_ids_2d) < row_nnz[:, None])
        staged_cols = tl.load(indices_ptr + elem_offsets, mask=valid_offsets, other=0)
        staged_vals = tl.load(data_ptr + elem_offsets, mask=valid_offsets, other=0.0).to(ACC_DTYPE)
        tl.store(tle.gpu.local_ptr(s_col, indices=(row_ids_2d, lane_ids_2d)), staged_cols, mask=valid_offsets)
        tl.store(tle.gpu.local_ptr(s_val, indices=(row_ids_2d, lane_ids_2d)), staged_vals, mask=valid_offsets)

        for jj in tl.static_range(0, WARP_SIZE):
            valid = row_valid & ((chunk_offset + jj) < row_nnz)
            local_jj = tl.full((BLOCK_ROWS,), jj, dtype=tl.int32)
            a_col = tl.load(tle.gpu.local_ptr(s_col, indices=(local_row_offsets, local_jj)), mask=valid, other=0)
            a_val = tl.load(tle.gpu.local_ptr(s_val, indices=(local_row_offsets, local_jj)), mask=valid, other=0.0).to(ACC_DTYPE)
            if FACTOR > 0:
                b0 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs0[None, :] * stride_bn,
                    mask=valid[:, None] & mask0[None, :],
                    other=0.0,
                )
                acc0 = acc0 + a_val[:, None] * b0.to(ACC_DTYPE)
            if FACTOR > 1:
                b1 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs1[None, :] * stride_bn,
                    mask=valid[:, None] & mask1[None, :],
                    other=0.0,
                )
                acc1 = acc1 + a_val[:, None] * b1.to(ACC_DTYPE)
            if FACTOR > 2:
                b2 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs2[None, :] * stride_bn,
                    mask=valid[:, None] & mask2[None, :],
                    other=0.0,
                )
                acc2 = acc2 + a_val[:, None] * b2.to(ACC_DTYPE)
            if FACTOR > 3:
                b3 = tl.load(
                    b_ptr + a_col[:, None] * stride_bk + offs3[None, :] * stride_bn,
                    mask=valid[:, None] & mask3[None, :],
                    other=0.0,
                )
                acc3 = acc3 + a_val[:, None] * b3.to(ACC_DTYPE)

    tl.store(
        c_ptr + rows[:, None] * stride_cm + offs0[None, :] * stride_cn,
        acc0,
        mask=row_valid[:, None] & mask0[None, :],
    )
    if FACTOR > 1:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs1[None, :] * stride_cn,
            acc1,
            mask=row_valid[:, None] & mask1[None, :],
        )
    if FACTOR > 2:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs2[None, :] * stride_cn,
            acc2,
            mask=row_valid[:, None] & mask2[None, :],
        )
    if FACTOR > 3:
        tl.store(
            c_ptr + rows[:, None] * stride_cm + offs3[None, :] * stride_cn,
            acc3,
            mask=row_valid[:, None] & mask3[None, :],
        )


def is_alpha_spmm_alg1_tle_available():
    return tle is not None


def alpha_spmm_alg1_tle_unavailable_reason():
    if _TLE_IMPORT_ERROR is None:
        return "TLE kernel is not available in this runtime"
    return f"{type(_TLE_IMPORT_ERROR).__name__}: {_TLE_IMPORT_ERROR}"


def is_alpha_spmm_alg1_tle_opt_available():
    return is_alpha_spmm_alg1_tle_available()


def alpha_spmm_alg1_tle_opt_unavailable_reason():
    return alpha_spmm_alg1_tle_unavailable_reason()


def is_alpha_spmm_alg1_tle_opt2_available():
    return is_alpha_spmm_alg1_tle_available()


def alpha_spmm_alg1_tle_opt2_unavailable_reason():
    return alpha_spmm_alg1_tle_unavailable_reason()


def build_alpha_spmm_alg1_tle_opt_meta(prepared, B):
    B = _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, None)
    return _with_alpha_spmm_alg1_route(
        _build_alpha_spmm_alg1_tle_opt_runtime_meta(prepared, B),
        "alpha_spmm_alg1_tle_opt",
    )


def build_alpha_spmm_alg1_tle_opt2_meta(prepared, B):
    B = _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, None)
    return _with_alpha_spmm_alg1_route(
        _build_alpha_spmm_alg1_tle_opt2_runtime_meta(prepared, B),
        "alpha_spmm_alg1_tle_opt2",
    )


def _run_alpha_spmm_alg1_tle(prepared, B, meta):
    if tle is None:
        raise RuntimeError(
            "flagsparse_alpha_spmm_alg1_tle requires FlagTree/TLE-Struct runtime "
            f"support ({alpha_spmm_alg1_tle_unavailable_reason()})"
        )
    if prepared.n_rows == 0 or int(B.shape[1]) == 0:
        return torch.zeros(
            (prepared.n_rows, int(B.shape[1])),
            dtype=prepared.data.dtype,
            device=prepared.data.device,
        )

    C_out = torch.empty(
        (prepared.n_rows, int(B.shape[1])),
        dtype=prepared.data.dtype,
        device=prepared.data.device,
    )
    grid = (meta["grid_m"], meta["grid_n"])
    _alpha_spmm_alg1_tle_rowmajor_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        B,
        C_out,
        prepared.n_rows,
        int(B.shape[1]),
        B.stride(0),
        B.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        WARP_SIZE=meta["warp_size"],
        FACTOR=meta["factor"],
        BLOCK_ROWS=meta["block_rows"],
        BLOCK_COLS=meta["block_cols"],
        MAX_ROW_NNZ=meta["max_row_nnz"],
        ACC_DTYPE=_alpha_spmm_alg1_acc_dtype(prepared.data.dtype),
        num_warps=meta["num_warps"],
        num_stages=meta["num_stages"],
    )
    return C_out


def _run_alpha_spmm_alg1_tle_opt(prepared, B, meta):
    if tle is None:
        raise RuntimeError(
            "flagsparse_alpha_spmm_alg1_tle_opt requires FlagTree/TLE-Struct runtime "
            f"support ({alpha_spmm_alg1_tle_opt_unavailable_reason()})"
        )
    if prepared.n_rows == 0 or int(B.shape[1]) == 0:
        return torch.zeros(
            (prepared.n_rows, int(B.shape[1])),
            dtype=prepared.data.dtype,
            device=prepared.data.device,
        )

    C_out = torch.empty(
        (prepared.n_rows, int(B.shape[1])),
        dtype=prepared.data.dtype,
        device=prepared.data.device,
    )
    grid = (meta["grid_m"], meta["grid_n"])
    _alpha_spmm_alg1_tle_opt_rowmajor_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        B,
        C_out,
        prepared.n_rows,
        int(B.shape[1]),
        B.stride(0),
        B.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        WARP_SIZE=meta["warp_size"],
        FACTOR=meta["factor"],
        BLOCK_ROWS=meta["block_rows"],
        BLOCK_COLS=meta["block_cols"],
        MAX_ROW_NNZ=meta["max_row_nnz"],
        ACC_DTYPE=_alpha_spmm_alg1_acc_dtype(prepared.data.dtype),
        num_warps=meta["num_warps"],
        num_stages=meta["num_stages"],
    )
    return C_out


def _run_alpha_spmm_alg1_tle_opt2(prepared, B, meta):
    if tle is None:
        raise RuntimeError(
            "flagsparse_alpha_spmm_alg1_tle_opt2 requires FlagTree/TLE-Struct runtime "
            f"support ({alpha_spmm_alg1_tle_opt2_unavailable_reason()})"
        )
    if prepared.n_rows == 0 or int(B.shape[1]) == 0:
        return torch.zeros(
            (prepared.n_rows, int(B.shape[1])),
            dtype=prepared.data.dtype,
            device=prepared.data.device,
        )

    C_out = torch.empty(
        (prepared.n_rows, int(B.shape[1])),
        dtype=prepared.data.dtype,
        device=prepared.data.device,
    )
    grid = (meta["grid_m"], meta["grid_n"])
    _alpha_spmm_alg1_tle_opt2_rowmajor_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        B,
        C_out,
        prepared.n_rows,
        int(B.shape[1]),
        B.stride(0),
        B.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        WARP_SIZE=meta["warp_size"],
        FACTOR=meta["factor"],
        BLOCK_ROWS=meta["block_rows"],
        BLOCK_COLS=meta["block_cols"],
        MAX_ROW_NNZ=meta["max_row_nnz"],
        ACC_DTYPE=_alpha_spmm_alg1_acc_dtype(prepared.data.dtype),
        num_warps=meta["num_warps"],
        num_stages=meta["num_stages"],
    )
    return C_out


def flagsparse_alpha_spmm_alg1(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_meta=False,
):
    """Experimental Triton baseline port of AlphaSparse CUDA CSR ALG1."""
    if prepared is not None and not isinstance(prepared, PreparedAlphaSpmmAlg1):
        raise TypeError("prepared must be a PreparedAlphaSpmmAlg1 instance")
    raw_inputs_provided = any(arg is not None for arg in (data, indices, indptr, shape))
    if prepared is not None and raw_inputs_provided:
        raise ValueError("Pass either raw CSR inputs or prepared, not both")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_alpha_spmm_alg1(data, indices, indptr, shape)

    B = _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, out)
    meta = _with_alpha_spmm_alg1_route(
        _build_alpha_spmm_alg1_runtime_meta(prepared, B),
        "alpha_spmm_alg1",
    )
    C = _run_alpha_spmm_alg1(prepared, B, meta)
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        return C, meta
    return C


def flagsparse_alpha_spmm_alg1_tle(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_meta=False,
):
    """Experimental TLE-Struct port of AlphaSparse CUDA CSR ALG1."""
    if prepared is not None and not isinstance(prepared, PreparedAlphaSpmmAlg1):
        raise TypeError("prepared must be a PreparedAlphaSpmmAlg1 instance")
    raw_inputs_provided = any(arg is not None for arg in (data, indices, indptr, shape))
    if prepared is not None and raw_inputs_provided:
        raise ValueError("Pass either raw CSR inputs or prepared, not both")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_alpha_spmm_alg1_tle(data, indices, indptr, shape)

    B = _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, out)
    meta = _with_alpha_spmm_alg1_route(
        _build_alpha_spmm_alg1_runtime_meta(prepared, B),
        "alpha_spmm_alg1_tle",
    )
    C = _run_alpha_spmm_alg1_tle(prepared, B, meta)
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        return C, meta
    return C


def flagsparse_alpha_spmm_alg1_tle_opt(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_meta=False,
    meta=None,
):
    """Experimental TLE-Struct optimized port of AlphaSparse CUDA CSR ALG1."""
    if prepared is not None and not isinstance(prepared, PreparedAlphaSpmmAlg1):
        raise TypeError("prepared must be a PreparedAlphaSpmmAlg1 instance")
    raw_inputs_provided = any(arg is not None for arg in (data, indices, indptr, shape))
    if prepared is not None and raw_inputs_provided:
        raise ValueError("Pass either raw CSR inputs or prepared, not both")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_alpha_spmm_alg1_tle_opt(data, indices, indptr, shape)

    B = _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, out)
    if meta is None:
        meta = _build_alpha_spmm_alg1_tle_opt_runtime_meta(prepared, B)
    meta = _with_alpha_spmm_alg1_route(meta, "alpha_spmm_alg1_tle_opt")
    C = _run_alpha_spmm_alg1_tle_opt(prepared, B, meta)
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        return C, meta
    return C


def flagsparse_alpha_spmm_alg1_tle_opt2(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_meta=False,
    meta=None,
):
    """Experimental TLE-Struct optimized port with shape-bucket launch tuning."""
    if prepared is not None and not isinstance(prepared, PreparedAlphaSpmmAlg1):
        raise TypeError("prepared must be a PreparedAlphaSpmmAlg1 instance")
    raw_inputs_provided = any(arg is not None for arg in (data, indices, indptr, shape))
    if prepared is not None and raw_inputs_provided:
        raise ValueError("Pass either raw CSR inputs or prepared, not both")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_alpha_spmm_alg1_tle_opt2(data, indices, indptr, shape)

    B = _validate_alpha_spmm_alg1_runtime_inputs(prepared, B, out)
    if meta is None:
        meta = _build_alpha_spmm_alg1_tle_opt2_runtime_meta(prepared, B)
    meta = _with_alpha_spmm_alg1_route(meta, "alpha_spmm_alg1_tle_opt2")
    C = _run_alpha_spmm_alg1_tle_opt2(prepared, B, meta)
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        return C, meta
    return C
