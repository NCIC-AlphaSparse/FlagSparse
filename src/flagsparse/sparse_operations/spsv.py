"""Sparse triangular solve (SpSV) CSR/COO."""

from ._common import *

from collections import OrderedDict
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
def _spsv_env_flag(name, default="0"):
    return str(os.environ.get(name, default)).lower() in ("1", "true", "yes", "on")


SPSV_PROMOTE_FP32_TO_FP64 = _spsv_env_flag("FLAGSPARSE_SPSV_PROMOTE_FP32_TO_FP64", "0")
SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64", "0"
)
SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128 = _spsv_env_flag(
    "FLAGSPARSE_SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128", "0"
)
SPSV_ENABLE_CSR_CW = _spsv_env_flag("FLAGSPARSE_SPSV_ENABLE_CSR_CW", "0")
SPSV_ENABLE_TRANSPOSE_CW = _spsv_env_flag("FLAGSPARSE_SPSV_ENABLE_TRANSPOSE_CW", "0")
SPSV_ENABLE_LEVEL_FRONTIERS = _spsv_env_flag("FLAGSPARSE_SPSV_ENABLE_LEVEL_FRONTIERS", "0")
SPSV_ENABLE_REVERSE_FRONTIERS = _spsv_env_flag("FLAGSPARSE_SPSV_ENABLE_REVERSE_FRONTIERS", "0")
_SPSV_CSR_PREPROCESS_CACHE = OrderedDict()
_SPSV_CSR_PREPROCESS_CACHE_SIZE = 8


def _clear_spsv_csr_preprocess_cache():
    _SPSV_CSR_PREPROCESS_CACHE.clear()


def _as_strided_contiguous(tensor):
    if tensor is None:
        return None
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
    transpose_diag = plan.get("transpose_diag")
    if transpose_diag is not None and torch.is_complex(transpose_diag):
        plan["transpose_diag_ri"] = _complex_interleaved_view(transpose_diag)
    cw_diag = plan.get("cw_diag")
    if cw_diag is not None and torch.is_complex(cw_diag):
        plan["cw_diag_ri"] = _complex_interleaved_view(cw_diag)
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
    if b.ndim not in (1, 2):
        raise ValueError("b must be 1D or 2D (vector or multiple RHS)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if b.ndim == 1 and b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")
    if b.ndim == 2 and b.shape[0] != n_rows:
        raise ValueError(f"b.shape[0] must equal n_rows={n_rows}")

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


def _csr_preprocess_cache_key(data, indices, indptr, shape, lower, trans_mode, unit_diagonal):
    return (
        "csr_preprocess",
        trans_mode,
        bool(lower),
        bool(unit_diagonal),
        int(shape[0]),
        int(shape[1]),
        _tensor_cache_token(data),
        _tensor_cache_token(indices),
        _tensor_cache_token(indptr),
    )


def _build_spsv_frontiers(indptr, indices, levels, lower=True):
    """Greedily merge rows from adjacent levels when they do not depend on the
    currently active frontier.

    This keeps the same correctness contract as strict level scheduling while
    trimming some kernel launches on matrices with narrow but not fully
    serialized dependency wavefronts.
    """
    if not levels:
        return []

    indptr_h = indptr.to(torch.int64).cpu()
    indices_h = indices.to(torch.int64).cpu()
    device = indptr.device
    frontier_rows = []
    frontier_row_set = set()
    merged = []

    def _flush_frontier():
        nonlocal frontier_rows, frontier_row_set
        if frontier_rows:
            merged.append(torch.tensor(frontier_rows, dtype=torch.int32, device=device))
            frontier_rows = []
            frontier_row_set = set()

    for rows_lv in levels:
        for row in rows_lv.to(torch.int64).cpu().tolist():
            start = int(indptr_h[row].item())
            end = int(indptr_h[row + 1].item())
            depends_on_frontier = False
            for p in range(start, end):
                col = int(indices_h[p].item())
                if lower:
                    is_dep = col < row
                else:
                    is_dep = col > row
                if is_dep and col in frontier_row_set:
                    depends_on_frontier = True
                    break
            if depends_on_frontier:
                _flush_frontier()
            frontier_rows.append(int(row))
            frontier_row_set.add(int(row))
    _flush_frontier()
    return merged


def _build_spsv_reverse_frontiers(indptr, indices, levels, lower=True):
    """Greedily merge reverse-topological launch groups for transpose push.

    Rows processed by the currently active reverse frontier push residual
    updates into their dependency targets. A candidate row can be merged only
    when no active row would update it; otherwise it must be delayed until the
    current frontier completes.
    """
    if not levels:
        return []

    indptr_h = indptr.to(torch.int64).cpu()
    indices_h = indices.to(torch.int64).cpu()
    device = indptr.device
    dependency_targets = {}
    for rows_lv in levels:
        for row in rows_lv.to(torch.int64).cpu().tolist():
            start = int(indptr_h[row].item())
            end = int(indptr_h[row + 1].item())
            targets = set()
            for p in range(start, end):
                col = int(indices_h[p].item())
                is_dep = (col < row) if lower else (col > row)
                if is_dep:
                    targets.add(col)
            dependency_targets[int(row)] = targets

    frontier_rows = []
    frontier_targets = set()
    merged = []

    def _flush_frontier():
        nonlocal frontier_rows, frontier_targets
        if frontier_rows:
            merged.append(torch.tensor(frontier_rows, dtype=torch.int32, device=device))
            frontier_rows = []
            frontier_targets = set()

    for rows_lv in reversed(levels):
        for row in rows_lv.to(torch.int64).cpu().tolist():
            if int(row) in frontier_targets:
                _flush_frontier()
            frontier_rows.append(int(row))
            frontier_targets.update(dependency_targets.get(int(row), ()))
    _flush_frontier()
    return merged


def _prepare_spsv_transpose_cw_metadata(data, indices64, indptr64, n_rows, lower, unit_diagonal=False):
    indegree = torch.zeros(n_rows, dtype=torch.int32, device=data.device)
    diag = torch.ones(n_rows, dtype=data.dtype, device=data.device)
    if n_rows == 0:
        return diag, indegree
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    if dep_mask.numel() > 0:
        dep_rows = row_ids[dep_mask]
        dep_counts = torch.bincount(dep_rows, minlength=n_rows)
        indegree.copy_(dep_counts.to(torch.int32))
    if not unit_diagonal:
        indegree.add_(1)
        diag_mask = indices64 == row_ids
        if bool(torch.any(diag_mask).item()):
            diag.scatter_(0, row_ids[diag_mask], data[diag_mask])
    return diag, indegree


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


def _prepare_spsv_nontrans_cw_metadata(data, indices64, indptr64, n_rows, unit_diagonal=False):
    diag = torch.ones(n_rows, dtype=data.dtype, device=data.device)
    if n_rows == 0:
        return diag
    if unit_diagonal:
        return diag
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    diag_mask = indices64 == row_ids
    if bool(torch.any(diag_mask).item()):
        diag.scatter_(0, row_ids[diag_mask], data[diag_mask])
    return diag


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
    target = max(256, min(n_rows, 4096))
    if max_frontier > 0:
        target = min(target, max(128, min(n_rows, max_frontier * 8)))
    if avg_nnz_per_row > 2048:
        target = max(128, target // 2)
    if rhs_bucket >= 16:
        target = max(64, target // 4)
    elif rhs_bucket >= 8:
        target = max(64, target // 2)
    elif rhs_bucket >= 4:
        target = max(128, (target * 3) // 4)
    return _snap_cw_worker_count(target, n_rows)


def _resolve_cw_worker_count(n_rows, matrix_stats, n_rhs, cached_worker_count=None):
    rhs_bucket = _cw_rhs_bucket(n_rhs)
    if cached_worker_count is not None and rhs_bucket == 1:
        return int(max(1, min(int(cached_worker_count), int(max(n_rows, 1)))))
    return _cw_worker_count(
        n_rows,
        int(matrix_stats.get("max_frontier", n_rows)),
        float(matrix_stats.get("avg_nnz_per_row", 0.0)),
        rhs_bucket,
    )


def _spsv_level_stats(levels, n_rows):
    if not levels:
        return {
            "num_levels": 0,
            "max_frontier": 0,
            "avg_frontier": 0.0,
            "frontier_ratio": 0.0,
        }
    num_levels = len(levels)
    max_frontier = max(int(rows.numel()) for rows in levels)
    avg_frontier = float(n_rows) / float(max(num_levels, 1))
    frontier_ratio = float(max_frontier) / float(max(n_rows, 1))
    return {
        "num_levels": num_levels,
        "max_frontier": max_frontier,
        "avg_frontier": avg_frontier,
        "frontier_ratio": frontier_ratio,
    }


def _build_spsv_matrix_stats(indptr64, levels, n_rows):
    stats = _spsv_level_stats(levels, n_rows)
    if indptr64.numel() <= 1:
        avg_nnz_per_row = 0.0
        max_nnz_per_row = 0
    else:
        row_lengths = indptr64[1:] - indptr64[:-1]
        avg_nnz_per_row = float(row_lengths.to(torch.float32).mean().item())
        max_nnz_per_row = int(row_lengths.max().item())
    stats["avg_nnz_per_row"] = avg_nnz_per_row
    stats["max_nnz_per_row"] = max_nnz_per_row
    stats["n_rows"] = int(n_rows)
    return stats


def _choose_spsv_block_rhs(n_rhs, matrix_stats, complex_mode=False):
    avg_nnz = float(matrix_stats.get("avg_nnz_per_row", 0.0))
    max_nnz = int(matrix_stats.get("max_nnz_per_row", 0))
    frontier_ratio = float(matrix_stats.get("frontier_ratio", 0.0))
    if n_rhs <= 1:
        return 1
    if complex_mode:
        if avg_nnz > 512 or max_nnz > 4096:
            return 4 if n_rhs >= 4 else n_rhs
        return 8 if n_rhs >= 8 else n_rhs
    if avg_nnz > 2048 or max_nnz > 16384:
        return 4 if n_rhs >= 4 else n_rhs
    if avg_nnz > 512 or frontier_ratio < 0.02:
        return 8 if n_rhs >= 8 else n_rhs
    if avg_nnz > 128:
        return 16 if n_rhs >= 16 else n_rhs
    if n_rhs <= 8:
        return n_rhs
    if n_rhs <= 16:
        return 16
    if n_rhs <= 32:
        return 32
    return 64


def _score_nontrans_levels(matrix_stats, n_rhs, complex_mode):
    score = 0.0
    score += float(matrix_stats["frontier_ratio"]) * 10.0
    score += min(float(matrix_stats["avg_frontier"]) / 64.0, 6.0)
    score += min(float(matrix_stats["avg_nnz_per_row"]) / 256.0, 4.0)
    if n_rhs >= 8:
        score += 2.5
    if complex_mode:
        score += 2.0
    return score


def _score_nontrans_cw(matrix_stats, n_rhs, complex_mode):
    score = 0.0
    if matrix_stats["num_levels"] > 1024:
        score += 2.0
    if matrix_stats["num_levels"] > 4096:
        score += 2.0
    if matrix_stats["frontier_ratio"] < 0.03:
        score += 4.0
    elif matrix_stats["frontier_ratio"] < 0.06:
        score += 2.0
    if matrix_stats["avg_frontier"] < 16.0:
        score += 3.0
    elif matrix_stats["avg_frontier"] < 32.0:
        score += 1.5
    if matrix_stats["avg_nnz_per_row"] <= 128.0:
        score += 1.5
    elif matrix_stats["avg_nnz_per_row"] <= 160.0:
        score += 0.5
    if matrix_stats["max_nnz_per_row"] > 4096:
        score -= 2.5
    elif matrix_stats["max_nnz_per_row"] > 1536:
        score -= 1.5
    if n_rhs >= 8:
        score -= 3.0
    elif n_rhs >= 4:
        score -= 1.5
    if complex_mode:
        score -= 2.0
    return score


def _score_transpose_push(matrix_stats, n_rhs, complex_mode):
    score = 0.0
    score += float(matrix_stats["frontier_ratio"]) * 12.0
    score += min(float(matrix_stats["avg_frontier"]) / 48.0, 6.0)
    score += min(float(matrix_stats["avg_nnz_per_row"]) / 256.0, 4.0)
    if n_rhs >= 4:
        score += 2.0
    if complex_mode:
        score += 2.5
    return score


def _score_transpose_cw(matrix_stats, n_rhs, complex_mode):
    score = 0.0
    if matrix_stats["num_levels"] > 2048:
        score += 2.0
    if matrix_stats["num_levels"] > 8192:
        score += 2.0
    if matrix_stats["frontier_ratio"] < 0.02:
        score += 4.0
    elif matrix_stats["frontier_ratio"] < 0.05:
        score += 2.0
    if matrix_stats["avg_frontier"] < 12.0:
        score += 3.0
    elif matrix_stats["avg_frontier"] < 24.0:
        score += 1.5
    if matrix_stats["avg_nnz_per_row"] <= 96.0:
        score += 1.0
    if matrix_stats["max_nnz_per_row"] > 2048:
        score -= 2.5
    if n_rhs >= 4:
        score -= 2.5
    elif n_rhs >= 2:
        score -= 1.0
    if complex_mode:
        score -= 2.5
    return score


def _nontrans_cw_eligible(matrix_stats, n_rhs, complex_mode):
    if complex_mode:
        return False
    if n_rhs != 1:
        return False
    if int(matrix_stats.get("num_levels", 0)) < 1024:
        return False
    if float(matrix_stats.get("avg_frontier", 0.0)) > 16.0:
        return False
    if float(matrix_stats.get("frontier_ratio", 1.0)) > 0.05:
        return False
    avg_nnz = float(matrix_stats.get("avg_nnz_per_row", 0.0))
    max_nnz = int(matrix_stats.get("max_nnz_per_row", 0))
    if avg_nnz <= 0.0 or avg_nnz > 160.0:
        return False
    if max_nnz > 1536:
        return False
    return True


def _transpose_cw_eligible(matrix_stats, n_rhs, complex_mode, trans_mode):
    if trans_mode != "T":
        return False
    if complex_mode:
        return False
    if n_rhs != 1:
        return False
    if int(matrix_stats.get("num_levels", 0)) < 512:
        return False
    if float(matrix_stats.get("avg_frontier", 0.0)) > 16.0:
        return False
    if float(matrix_stats.get("frontier_ratio", 1.0)) > 0.04:
        return False
    avg_nnz = float(matrix_stats.get("avg_nnz_per_row", 0.0))
    max_nnz = int(matrix_stats.get("max_nnz_per_row", 0))
    if avg_nnz <= 0.0 or avg_nnz > 160.0:
        return False
    if max_nnz > 1024:
        return False
    return True


def _select_nontrans_route(matrix_stats, n_rhs, complex_mode):
    if not _nontrans_cw_eligible(matrix_stats, n_rhs, complex_mode):
        return "csr_levels"
    levels_score = _score_nontrans_levels(matrix_stats, n_rhs, complex_mode)
    cw_score = _score_nontrans_cw(matrix_stats, n_rhs, complex_mode)
    return "csr_cw" if cw_score >= (levels_score + 2.0) else "csr_levels"


def _select_transpose_route(matrix_stats, n_rhs, complex_mode, trans_mode):
    if not _transpose_cw_eligible(matrix_stats, n_rhs, complex_mode, trans_mode):
        return "transpose_push"
    push_score = _score_transpose_push(matrix_stats, n_rhs, complex_mode)
    cw_score = _score_transpose_cw(matrix_stats, n_rhs, complex_mode)
    return "transpose_cw" if cw_score >= (push_score + 1.0) else "transpose_push"


def _prepare_spsv_csr_system(data, indices64, indptr64, n_rows, n_cols, lower, trans_mode, unit_diagonal):
    if trans_mode == "N":
        levels = _build_spsv_levels(indptr64, indices64, n_rows, lower=lower)
        matrix_stats = _build_spsv_matrix_stats(indptr64, levels, n_rows)
        default_block_nnz, default_max_segments = _auto_spsv_launch_config(indptr64)
        levels_plan = {
            "solve_kind": "csr_levels",
            "kernel_data": data,
            "kernel_indices32": indices64.to(torch.int32),
            "kernel_indptr64": indptr64,
            "lower_eff": lower,
            "launch_groups": (
                _build_spsv_frontiers(indptr64, indices64, levels, lower=lower)
                if SPSV_ENABLE_LEVEL_FRONTIERS
                else levels
            ),
            "default_block_nnz": default_block_nnz,
            "default_max_segments": default_max_segments,
            "transpose_conjugate": False,
            "cw_worker_count": None,
            "matrix_stats": matrix_stats,
            "route_name": "csr_levels",
            "alt_plan": None,
        }
        _attach_spsv_complex_plan_views(levels_plan)
        if not SPSV_ENABLE_CSR_CW:
            return levels_plan
        data_sorted, indices_sorted64, indptr_sorted64 = _sort_csr_rows(
            data, indices64, indptr64, n_rows, n_cols, lower=lower
        )
        default_block_nnz, default_max_segments = _auto_spsv_launch_config(indptr_sorted64)
        cw_plan = {
            "solve_kind": "csr_cw",
            "kernel_data": data_sorted,
            "kernel_indices32": indices_sorted64.to(torch.int32),
            "kernel_indptr64": indptr_sorted64,
            "lower_eff": lower,
            "launch_groups": None,
            "default_block_nnz": default_block_nnz,
            "default_max_segments": default_max_segments,
            "transpose_conjugate": False,
            "cw_diag": _prepare_spsv_nontrans_cw_metadata(
                data_sorted, indices_sorted64, indptr_sorted64, n_rows, unit_diagonal=unit_diagonal
            ),
            "cw_worker_count": _cw_worker_count(
                n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
            ),
            "matrix_stats": matrix_stats,
            "route_name": "csr_cw",
            "alt_plan": None,
        }
        _attach_spsv_complex_plan_views(cw_plan)
        levels_plan["alt_plan"] = cw_plan
        return levels_plan

    levels = _build_spsv_levels(indptr64, indices64, n_rows, lower=lower)
    matrix_stats = _build_spsv_matrix_stats(indptr64, levels, n_rows)
    default_block_nnz, default_max_segments = _choose_transpose_family_launch_config(
        indptr64
    )
    push_plan = {
        "solve_kind": "transpose_push",
        "kernel_data": data,
        "kernel_indices32": indices64.to(torch.int32),
        "kernel_indptr64": indptr64,
        "lower_eff": lower,
        "launch_groups": (
            _build_spsv_reverse_frontiers(indptr64, indices64, levels, lower=lower)
            if SPSV_ENABLE_REVERSE_FRONTIERS
            else list(reversed(levels))
        ),
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "transpose_conjugate": trans_mode == "C",
        "cw_worker_count": None,
        "matrix_stats": matrix_stats,
        "route_name": "transpose_push",
        "alt_plan": None,
    }
    _attach_spsv_complex_plan_views(push_plan)
    if not SPSV_ENABLE_TRANSPOSE_CW:
        return push_plan

    diag, indegree_init = _prepare_spsv_transpose_cw_metadata(
        data, indices64, indptr64, n_rows, lower, unit_diagonal=unit_diagonal
    )
    default_block_nnz, default_max_segments = _choose_transpose_family_launch_config(
        indptr64
    )
    cw_plan = {
        "solve_kind": "transpose_cw",
        "kernel_data": data,
        "kernel_indices32": indices64.to(torch.int32),
        "kernel_indptr64": indptr64,
        "lower_eff": lower,
        "launch_groups": None,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "transpose_conjugate": trans_mode == "C",
        "transpose_diag": diag,
        "transpose_indegree_init": indegree_init,
        "cw_worker_count": _cw_worker_count(
            n_rows, matrix_stats["max_frontier"], matrix_stats["avg_nnz_per_row"], 1
        ),
        "matrix_stats": matrix_stats,
        "route_name": "transpose_cw",
        "alt_plan": None,
    }
    _attach_spsv_complex_plan_views(cw_plan)
    push_plan["alt_plan"] = cw_plan
    return push_plan


def _resolve_spsv_csr_runtime(
    data,
    indices,
    indptr,
    b,
    shape,
    lower,
    transpose,
    unit_diagonal=False,
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
        input_data, input_indices, input_indptr, (n_rows, n_cols), lower, trans_mode, unit_diagonal
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


def _select_spsv_runtime_plan(solve_plan, rhs_cols, compute_dtype, trans_mode):
    matrix_stats = solve_plan.get("matrix_stats", {})
    route_name = solve_plan.get("route_name", solve_plan["solve_kind"])
    alt_plan = solve_plan.get("alt_plan")
    complex_mode = compute_dtype in (torch.complex64, torch.complex128)
    if trans_mode == "N":
        desired = _select_nontrans_route(matrix_stats, rhs_cols, complex_mode)
    else:
        desired = _select_transpose_route(
            matrix_stats, rhs_cols, complex_mode, trans_mode
        )
    if desired == route_name or alt_plan is None:
        return solve_plan
    if alt_plan.get("route_name", alt_plan["solve_kind"]) == desired:
        return alt_plan
    return solve_plan


@triton.jit
def _spsv_csr_level_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_level_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    diag = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    if UNIT_DIAG:
        diag = diag + 1.0

    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_vals = tl.load(x_ptr + col, mask=mask, other=0.0)

        if LOWER:
            solved = col < row
        else:
            solved = col > row
        is_diag = col == row

        acc = acc + tl.sum(tl.where(mask & solved, a * x_vals, 0.0))
        if not UNIT_DIAG:
            diag = diag + tl.sum(tl.where(mask & is_diag, a, 0.0))

    rhs = tl.load(b_ptr + row)
    diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
    x_row = (rhs - acc) / diag_safe
    # Prevent NaN propagation in ill-conditioned rows.
    x_row = tl.where(x_row == x_row, x_row, 0.0)
    tl.store(x_ptr + row, x_row)


@triton.jit
def _spsv_csr_level_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    x_ri_ptr,
    rows_ptr,
    n_level_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_level_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc_re = tl.zeros((1,), dtype=tl.float64)
        acc_im = tl.zeros((1,), dtype=tl.float64)
        diag_re = tl.zeros((1,), dtype=tl.float64)
        diag_im = tl.zeros((1,), dtype=tl.float64)
    else:
        acc_re = tl.zeros((1,), dtype=tl.float32)
        acc_im = tl.zeros((1,), dtype=tl.float32)
        diag_re = tl.zeros((1,), dtype=tl.float32)
        diag_im = tl.zeros((1,), dtype=tl.float32)

    if UNIT_DIAG:
        diag_re = diag_re + 1.0

    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end

        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
        a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
        x_re = tl.load(x_ri_ptr + col * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=mask, other=0.0)

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

        if LOWER:
            solved = col < row
        else:
            solved = col > row
        is_diag = col == row

        prod_re = a_re * x_re - a_im * x_im
        prod_im = a_re * x_im + a_im * x_re
        acc_re = acc_re + tl.sum(tl.where(mask & solved, prod_re, 0.0))
        acc_im = acc_im + tl.sum(tl.where(mask & solved, prod_im, 0.0))

        if not UNIT_DIAG:
            diag_re = diag_re + tl.sum(tl.where(mask & is_diag, a_re, 0.0))
            diag_im = diag_im + tl.sum(tl.where(mask & is_diag, a_im, 0.0))

    rhs_re = tl.load(b_ri_ptr + row * 2)
    rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)

    num_re = rhs_re - acc_re
    num_im = rhs_im - acc_im
    den = diag_re * diag_re + diag_im * diag_im
    den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)

    x_re_out = (num_re * diag_re + num_im * diag_im) / den_safe
    x_im_out = (num_im * diag_re - num_re * diag_im) / den_safe
    x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
    x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

    offs1 = tl.arange(0, 1)
    tl.store(x_ri_ptr + row * 2 + offs1, x_re_out)
    tl.store(x_ri_ptr + row * 2 + 1 + offs1, x_im_out)


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


@triton.jit
def _spsv_csr_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ri_ptr,
    b_ri_ptr,
    x_ri_ptr,
    ready_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    row = tl.atomic_add(row_counter_ptr, 1)
    while row < n_rows:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        if UNIT_DIAG:
            diag_re = 1.0
            diag_im = 0.0
        else:
            diag_re = tl.load(diag_ri_ptr + row * 2)
            diag_im = tl.load(diag_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            diag_re = diag_re.to(tl.float64)
            diag_im = diag_im.to(tl.float64)
            acc_re = tl.zeros((1,), dtype=tl.float64)
            acc_im = tl.zeros((1,), dtype=tl.float64)
        else:
            diag_re = diag_re.to(tl.float32)
            diag_im = diag_im.to(tl.float32)
            acc_re = tl.zeros((1,), dtype=tl.float32)
            acc_im = tl.zeros((1,), dtype=tl.float32)

        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
            a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
            if USE_FP64_ACC:
                a_re = a_re.to(tl.float64)
                a_im = a_im.to(tl.float64)
            else:
                a_re = a_re.to(tl.float32)
                a_im = a_im.to(tl.float32)
            if LOWER:
                dep_mask = mask & (col < row)
            else:
                dep_mask = mask & (col > row)

            for k in range(BLOCK_NNZ):
                if dep_mask[k]:
                    dep_col = col[k]
                    while tl.load(ready_ptr + dep_col) == 0:
                        pass
                    x_re = tl.load(x_ri_ptr + dep_col * 2)
                    x_im = tl.load(x_ri_ptr + dep_col * 2 + 1)
                    if USE_FP64_ACC:
                        x_re = x_re.to(tl.float64)
                        x_im = x_im.to(tl.float64)
                    else:
                        x_re = x_re.to(tl.float32)
                        x_im = x_im.to(tl.float32)
                    acc_re += a_re[k] * x_re - a_im[k] * x_im
                    acc_im += a_re[k] * x_im + a_im[k] * x_re

        rhs_re = tl.load(b_ri_ptr + row * 2)
        rhs_im = tl.load(b_ri_ptr + row * 2 + 1)
        if USE_FP64_ACC:
            rhs_re = rhs_re.to(tl.float64)
            rhs_im = rhs_im.to(tl.float64)
        else:
            rhs_re = rhs_re.to(tl.float32)
            rhs_im = rhs_im.to(tl.float32)

        num_re = rhs_re - acc_re
        num_im = rhs_im - acc_im
        den = diag_re * diag_re + diag_im * diag_im
        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
        x_re_out = (num_re * diag_re + num_im * diag_im) / den_safe
        x_im_out = (num_im * diag_re - num_re * diag_im) / den_safe
        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

        tl.store(x_ri_ptr + row * 2, x_re_out)
        tl.store(x_ri_ptr + row * 2 + 1, x_im_out)
        tl.debug_barrier()
        tl.store(ready_ptr + row, 1)
        row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_transpose_push_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    residual_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_level_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    diag = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    if UNIT_DIAG:
        diag = diag + 1.0
    else:
        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            is_diag = col == row
            diag = diag + tl.sum(tl.where(mask & is_diag, a, 0.0))

    rhs = tl.load(residual_ptr + row)
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
            target_mask = mask & (col < row)
        else:
            target_mask = mask & (col > row)
        tl.atomic_add(residual_ptr + col, -a * x_row, mask=target_mask)


@triton.jit
def _spsv_csr_transpose_push_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    residual_ri_ptr,
    x_ri_ptr,
    rows_ptr,
    n_level_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    CONJ_TRANS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_level_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        diag_re = tl.zeros((1,), dtype=tl.float64)
        diag_im = tl.zeros((1,), dtype=tl.float64)
    else:
        diag_re = tl.zeros((1,), dtype=tl.float32)
        diag_im = tl.zeros((1,), dtype=tl.float32)

    if UNIT_DIAG:
        diag_re = diag_re + 1.0
    else:
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
            is_diag = col == row
            diag_re = diag_re + tl.sum(tl.where(mask & is_diag, a_re, 0.0))
            diag_im = diag_im + tl.sum(tl.where(mask & is_diag, a_im, 0.0))

    rhs_re = tl.load(residual_ri_ptr + row * 2)
    rhs_im = tl.load(residual_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)

    den = diag_re * diag_re + diag_im * diag_im
    den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
    x_re_out = (rhs_re * diag_re + rhs_im * diag_im) / den_safe
    x_im_out = (rhs_im * diag_re - rhs_re * diag_im) / den_safe
    x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
    x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

    offs1 = tl.arange(0, 1)
    tl.store(x_ri_ptr + row * 2 + offs1, x_re_out)
    tl.store(x_ri_ptr + row * 2 + 1 + offs1, x_im_out)

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
            target_mask = mask & (col < row)
        else:
            target_mask = mask & (col > row)
        prod_re = a_re * x_re_out - a_im * x_im_out
        prod_im = a_re * x_im_out + a_im * x_re_out
        tl.atomic_add(residual_ri_ptr + col * 2, -prod_re, mask=target_mask)
        tl.atomic_add(residual_ri_ptr + col * 2 + 1, -prod_im, mask=target_mask)


@triton.jit
def _spsv_csr_transpose_cw_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    indegree_ptr,
    residual_ptr,
    x_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    row = tl.atomic_add(row_counter_ptr, 1)
    while row < n_rows:
        ready_value = 0 if UNIT_DIAG else 1
        while tl.load(indegree_ptr + row) != ready_value:
            pass

        rhs = tl.load(residual_ptr + row)
        if UNIT_DIAG:
            diag = rhs * 0 + 1.0
        else:
            diag = tl.load(diag_ptr + row)
        diag_safe = tl.where(tl.abs(diag) < DIAG_EPS, 1.0, diag)
        x_row = rhs / diag_safe
        x_row = tl.where(x_row == x_row, x_row, 0.0)
        tl.store(x_ptr + row, x_row)

        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        for seg in range(MAX_SEGMENTS):
            idx = start + seg * BLOCK_NNZ
            offsets = idx + tl.arange(0, BLOCK_NNZ)
            mask = offsets < end
            a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offsets, mask=mask, other=0)
            if LOWER:
                target_mask = mask & (col < row)
            else:
                target_mask = mask & (col > row)
            tl.atomic_add(residual_ptr + col, -a * x_row, mask=target_mask)
            tl.atomic_add(indegree_ptr + col, -1, mask=target_mask)
        row = tl.atomic_add(row_counter_ptr, 1)


@triton.jit
def _spsv_csr_transpose_cw_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ri_ptr,
    indegree_ptr,
    residual_ri_ptr,
    x_ri_ptr,
    row_counter_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
    CONJ_TRANS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    DIAG_EPS: tl.constexpr,
):
    row = tl.atomic_add(row_counter_ptr, 1)
    while row < n_rows:
        ready_value = 0 if UNIT_DIAG else 1
        while tl.load(indegree_ptr + row) != ready_value:
            pass

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
            diag_re = tl.load(diag_ri_ptr + row * 2)
            diag_im = tl.load(diag_ri_ptr + row * 2 + 1)
            if CONJ_TRANS:
                diag_im = -diag_im
            if USE_FP64_ACC:
                diag_re = diag_re.to(tl.float64)
                diag_im = diag_im.to(tl.float64)
            else:
                diag_re = diag_re.to(tl.float32)
                diag_im = diag_im.to(tl.float32)

        den = diag_re * diag_re + diag_im * diag_im
        den_safe = tl.where(den < (DIAG_EPS * DIAG_EPS), 1.0, den)
        x_re_out = (rhs_re * diag_re + rhs_im * diag_im) / den_safe
        x_im_out = (rhs_im * diag_re - rhs_re * diag_im) / den_safe
        x_re_out = tl.where(x_re_out == x_re_out, x_re_out, 0.0)
        x_im_out = tl.where(x_im_out == x_im_out, x_im_out, 0.0)

        offs1 = tl.arange(0, 1)
        tl.store(x_ri_ptr + row * 2 + offs1, x_re_out)
        tl.store(x_ri_ptr + row * 2 + 1 + offs1, x_im_out)

        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
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
                target_mask = mask & (col < row)
            else:
                target_mask = mask & (col > row)
            prod_re = a_re * x_re_out - a_im * x_im_out
            prod_im = a_re * x_im_out + a_im * x_re_out
            tl.atomic_add(residual_ri_ptr + col * 2, -prod_re, mask=target_mask)
            tl.atomic_add(residual_ri_ptr + col * 2 + 1, -prod_im, mask=target_mask)
            tl.atomic_add(indegree_ptr + col, -1, mask=target_mask)
        row = tl.atomic_add(row_counter_ptr, 1)


def _build_spsv_levels(indptr, indices, n_rows, lower=True):
    """Build dependency levels for triangular solve so each level can run in parallel."""
    if n_rows == 0:
        return []
    indptr_h = indptr.to(torch.int64).cpu()
    indices_h = indices.to(torch.int64).cpu()
    levels = [0] * n_rows
    if lower:
        for i in range(n_rows):
            s = int(indptr_h[i].item())
            e = int(indptr_h[i + 1].item())
            lvl = 0
            for p in range(s, e):
                c = int(indices_h[p].item())
                if c < i:
                    lvl = max(lvl, levels[c] + 1)
            levels[i] = lvl
    else:
        for i in range(n_rows - 1, -1, -1):
            s = int(indptr_h[i].item())
            e = int(indptr_h[i + 1].item())
            lvl = 0
            for p in range(s, e):
                c = int(indices_h[p].item())
                if c > i:
                    lvl = max(lvl, levels[c] + 1)
            levels[i] = lvl

    max_level = max(levels)
    buckets = [[] for _ in range(max_level + 1)]
    for r, lv in enumerate(levels):
        buckets[lv].append(r)

    device = indptr.device
    return [
        torch.tensor(rows, dtype=torch.int32, device=device)
        for rows in buckets
        if rows
    ]


def _auto_spsv_launch_config(indptr, block_nnz=None, max_segments=None):
    if indptr.numel() <= 1:
        max_nnz_per_row = 0
    else:
        row_lengths = indptr[1:] - indptr[:-1]
        max_nnz_per_row = int(row_lengths.max().item())

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


def _triton_spsv_csr_vector(
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
    levels=None,
    block_nnz_use=None,
    max_segments_use=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if levels is None:
        levels = _build_spsv_levels(indptr, indices, n_rows, lower=lower)
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsv_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    for rows_lv in levels:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv,)
        _spsv_csr_level_kernel[grid](
            data,
            indices,
            indptr,
            b_vec,
            x,
            rows_lv,
            n_level_rows=n_lv,
            BLOCK_NNZ=block_nnz_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            UNIT_DIAG=unit_diagonal,
            DIAG_EPS=diag_eps,
        )
    return x


def _triton_spsv_csr_vector_complex(
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
    levels=None,
    block_nnz_use=None,
    max_segments_use=None,
    data_ri_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if levels is None:
        levels = _build_spsv_levels(indptr, indices, n_rows, lower=lower)
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsv_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    # Some PyTorch builds return CSR values with a non-strided layout wrapper.
    # Materialize a plain 1D strided buffer before splitting into real/imag parts.
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    b_ri = torch.view_as_real(b_vec.contiguous()).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    if component_dtype == torch.float16:
        x_ri_work = torch.zeros((n_rows, 2), dtype=torch.float32, device=b_vec.device)
        x_ri = x_ri_work.reshape(-1).contiguous()
    else:
        x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    for rows_lv in levels:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv,)
        _spsv_csr_level_kernel_complex[grid](
            data_ri,
            indices,
            indptr,
            b_ri,
            x_ri,
            rows_lv,
            n_level_rows=n_lv,
            BLOCK_NNZ=block_nnz_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            UNIT_DIAG=unit_diagonal,
            USE_FP64_ACC=use_fp64,
            DIAG_EPS=diag_eps,
        )
    if component_dtype == torch.float16:
        return torch.view_as_complex(x_ri_work.contiguous())
    return x


def _triton_spsv_csr_cw_vector(
    data,
    indices,
    indptr,
    diag,
    b_vec,
    n_rows,
    lower=True,
    block_nnz=None,
    max_segments=None,
    diag_eps=1e-12,
    block_nnz_use=None,
    max_segments_use=None,
    worker_count=None,
    matrix_stats=None,
):
    if b_vec.ndim == 1:
        b_mat = b_vec.unsqueeze(1).contiguous()
    else:
        b_mat = b_vec.contiguous()
    x = torch.zeros_like(b_mat)
    ready = torch.zeros(n_rows, dtype=torch.int32, device=b_mat.device)
    row_counter = torch.zeros(1, dtype=torch.int32, device=b_mat.device)
    n_rhs = int(b_mat.shape[1])
    if n_rows == 0 or n_rhs == 0:
        return x.squeeze(1) if b_vec.ndim == 1 else x
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsv_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    matrix_stats = matrix_stats or {}
    block_rhs = _choose_spsv_block_rhs(n_rhs, matrix_stats, complex_mode=False)
    if worker_count is None:
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, n_rhs)
    grid = (worker_count,)
    _spsv_csr_cw_kernel[grid](
        data,
        indices,
        indptr,
        diag,
        b_mat,
        x,
        ready,
        row_counter,
        n_rows,
        n_rhs,
        b_mat.stride(0),
        x.stride(0),
        BLOCK_RHS=block_rhs,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        DIAG_EPS=diag_eps,
    )
    return x.squeeze(1) if b_vec.ndim == 1 else x


def _triton_spsv_csr_cw_vector_complex(
    data,
    indices,
    indptr,
    diag,
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
    data_ri_in=None,
    diag_ri_in=None,
):
    if b_vec.ndim != 1:
        shared_b = b_vec if b_vec.is_contiguous() else b_vec.contiguous()
        cols = []
        for bj in torch.unbind(shared_b, dim=1):
            cols.append(
                _triton_spsv_csr_cw_vector_complex(
                    data,
                    indices,
                    indptr,
                    diag,
                    bj,
                    n_rows,
                    lower=lower,
                    unit_diagonal=unit_diagonal,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
                    diag_eps=diag_eps,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                    worker_count=worker_count,
                    matrix_stats=matrix_stats,
                    data_ri_in=data_ri_in,
                    diag_ri_in=diag_ri_in,
                )
            )
        return torch.stack(cols, dim=1)

    x = torch.zeros_like(b_vec)
    ready = torch.zeros(n_rows, dtype=torch.int32, device=b_vec.device)
    row_counter = torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    if n_rows == 0:
        return x
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _auto_spsv_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    diag_ri = diag_ri_in if diag_ri_in is not None else _complex_interleaved_view(diag)
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
        diag_ri,
        b_ri,
        x_ri,
        ready,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        UNIT_DIAG=unit_diagonal,
        USE_FP64_ACC=use_fp64,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_transpose_push_vector(
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
    launch_groups=None,
    block_nnz_use=None,
    max_segments_use=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    residual = b_vec.clone()
    if launch_groups is None:
        levels = _build_spsv_levels(indptr, indices, n_rows, lower=lower)
        launch_groups = list(reversed(levels))
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    for rows_lv in launch_groups:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv,)
        _spsv_csr_transpose_push_kernel[grid](
            data,
            indices,
            indptr,
            residual,
            x,
            rows_lv,
            n_level_rows=n_lv,
            BLOCK_NNZ=block_nnz_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            UNIT_DIAG=unit_diagonal,
            DIAG_EPS=diag_eps,
        )
    return x


def _triton_spsv_csr_transpose_push_vector_complex(
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
    launch_groups=None,
    block_nnz_use=None,
    max_segments_use=None,
    data_ri_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if launch_groups is None:
        levels = _build_spsv_levels(indptr, indices, n_rows, lower=lower)
        launch_groups = list(reversed(levels))
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    residual_work = b_vec.contiguous().clone()
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    residual_ri = torch.view_as_real(residual_work).reshape(-1).contiguous()
    component_dtype = _component_dtype_for_complex(data.dtype)
    use_fp64 = component_dtype == torch.float64
    if component_dtype == torch.float16:
        x_ri_work = torch.zeros((n_rows, 2), dtype=torch.float32, device=b_vec.device)
        x_ri = x_ri_work.reshape(-1).contiguous()
    else:
        x_ri = torch.view_as_real(x.contiguous()).reshape(-1).contiguous()

    for rows_lv in launch_groups:
        n_lv = rows_lv.numel()
        if n_lv == 0:
            continue
        grid = (n_lv,)
        _spsv_csr_transpose_push_kernel_complex[grid](
            data_ri,
            indices,
            indptr,
            residual_ri,
            x_ri,
            rows_lv,
            n_level_rows=n_lv,
            BLOCK_NNZ=block_nnz_use,
            MAX_SEGMENTS=max_segments_use,
            LOWER=lower,
            UNIT_DIAG=unit_diagonal,
            CONJ_TRANS=conjugate,
            USE_FP64_ACC=use_fp64,
            DIAG_EPS=diag_eps,
        )
    if component_dtype == torch.float16:
        return torch.view_as_complex(x_ri_work.contiguous())
    return x


def _triton_spsv_csr_transpose_cw_vector(
    data,
    indices,
    indptr,
    diag,
    indegree_init,
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
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    residual = b_vec.clone()
    indegree = indegree_init.clone()
    row_counter = torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )
    if worker_count is None:
        matrix_stats = matrix_stats or {}
        worker_count = _resolve_cw_worker_count(n_rows, matrix_stats, 1)
    grid = (worker_count,)
    _spsv_csr_transpose_cw_kernel[grid](
        data,
        indices,
        indptr,
        diag,
        indegree,
        residual,
        x,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
        UNIT_DIAG=unit_diagonal,
        DIAG_EPS=diag_eps,
    )
    return x


def _triton_spsv_csr_transpose_cw_vector_complex(
    data,
    indices,
    indptr,
    diag,
    indegree_init,
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
    diag_ri_in=None,
):
    x = torch.zeros_like(b_vec)
    if n_rows == 0:
        return x
    if block_nnz_use is None or max_segments_use is None:
        block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
            indptr, block_nnz=block_nnz, max_segments=max_segments
        )

    residual_work = b_vec.contiguous().clone()
    indegree = indegree_init.clone()
    row_counter = torch.zeros(1, dtype=torch.int32, device=b_vec.device)
    data_ri = data_ri_in if data_ri_in is not None else _complex_interleaved_view(data)
    diag_ri = diag_ri_in if diag_ri_in is not None else _complex_interleaved_view(diag)
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
        diag_ri,
        indegree,
        residual_ri,
        x_ri,
        row_counter,
        n_rows,
        BLOCK_NNZ=block_nnz_use,
        MAX_SEGMENTS=max_segments_use,
        LOWER=lower,
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


def _prepare_spsv_coo_inputs(data, row, col, b, shape):
    if not all(torch.is_tensor(t) for t in (data, row, col, b)):
        raise TypeError("data, row, col, b must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, b)):
        raise ValueError("data, row, col, b must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if row.numel() != data.numel() or col.numel() != data.numel():
        raise ValueError("data, row, col must have the same length")
    if b.ndim not in (1, 2):
        raise ValueError("b must be 1D or 2D (vector or multiple RHS)")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if b.ndim == 1 and b.numel() != n_rows:
        raise ValueError(f"b length must equal n_rows={n_rows}")
    if b.ndim == 2 and b.shape[0] != n_rows:
        raise ValueError(f"b.shape[0] must equal n_rows={n_rows}")

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


def _csr_transpose(data, indices64, indptr64, n_rows, n_cols, conjugate=False):
    if data.numel() == 0:
        out_data = data
        out_indices = torch.empty(0, dtype=torch.int64, device=data.device)
        out_indptr = torch.zeros(n_cols + 1, dtype=torch.int64, device=data.device)
        return out_data, out_indices, out_indptr

    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )
    new_row = indices64
    new_col = row_ids
    data_eff = data.conj() if conjugate and torch.is_complex(data) else data
    data_t, indices_t, indptr_t = _coo_to_csr_sorted_unique(
        data_eff, new_row, new_col, n_cols, n_rows
    )
    return data_t, indices_t, indptr_t


def _build_coo_row_ptr(row_sorted, n_rows):
    row_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=row_sorted.device)
    if row_sorted.numel() > 0:
        nnz_per_row = torch.bincount(row_sorted, minlength=n_rows)
        row_ptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return row_ptr


def _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols):
    nnz = data.numel()
    if nnz == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        indices = torch.empty(0, dtype=torch.int64, device=data.device)
        return data, indices, indptr

    key = row64 * max(1, n_cols) + col64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_s = key[order]
    data_s = data[order]

    unique_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    out_nnz = unique_key.numel()

    data_u = torch.zeros(out_nnz, dtype=data.dtype, device=data.device)
    data_u.scatter_add_(0, inverse, data_s)

    row_u = torch.div(unique_key, max(1, n_cols), rounding_mode="floor")
    col_u = unique_key - row_u * max(1, n_cols)
    indptr = _build_coo_row_ptr(row_u, n_rows)
    indices = col_u.to(torch.int64)
    return data_u, indices, indptr


def flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
):
    """Sparse triangular solve using Triton CSR kernels.

    Primary support matrix:
    - NON_TRANS: float32/float64/complex64/complex128 with int32/int64 indices
    - TRANS/CONJ: float32/float64/complex64/complex128 with int32/int64 indices
    """
    (
        data,
        b,
        original_output_dtype,
        trans_mode,
        n_rows,
        n_cols,
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
    )

    rhs_cols = 1 if b.ndim == 1 else int(b.shape[1])
    solve_plan = _select_spsv_runtime_plan(
        solve_plan, rhs_cols, data.dtype, trans_mode
    )
    solve_kind = solve_plan["solve_kind"]
    kernel_data = solve_plan["kernel_data"]
    kernel_indices32 = solve_plan["kernel_indices32"]
    kernel_indptr64 = solve_plan["kernel_indptr64"]
    lower_eff = solve_plan["lower_eff"]
    launch_groups = solve_plan["launch_groups"]
    transpose_conjugate = solve_plan["transpose_conjugate"]
    default_block_nnz = solve_plan["default_block_nnz"]
    default_max_segments = solve_plan["default_max_segments"]
    transpose_diag = solve_plan.get("transpose_diag")
    transpose_indegree_init = solve_plan.get("transpose_indegree_init")
    cw_diag = solve_plan.get("cw_diag")
    cw_worker_count = solve_plan.get("cw_worker_count")
    matrix_stats = solve_plan.get("matrix_stats", {})
    kernel_indices = kernel_indices32
    kernel_indptr = kernel_indptr64
    compute_dtype = data.dtype
    data_in = kernel_data
    b_in = b
    if (
        data.dtype == torch.complex64
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_COMPLEX64_TO_COMPLEX128
    ):
        compute_dtype = torch.complex128
        data_in = kernel_data.to(torch.complex128)
        b_in = b.to(torch.complex128)
    elif data.dtype == torch.float32 and SPSV_PROMOTE_FP32_TO_FP64:
        compute_dtype = torch.float64
        data_in = kernel_data.to(torch.float64)
        b_in = b.to(torch.float64)
    elif (
        data.dtype == torch.float32
        and trans_mode in ("T", "C")
        and SPSV_PROMOTE_TRANSPOSE_FP32_TO_FP64
    ):
        compute_dtype = torch.float64
        data_in = kernel_data.to(torch.float64)
        b_in = b.to(torch.float64)

    if solve_kind in ("transpose_push", "transpose_cw"):
        if block_nnz is None and max_segments is None:
            block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        else:
            block_nnz_use, max_segments_use = _choose_transpose_family_launch_config(
                kernel_indptr, block_nnz=block_nnz, max_segments=max_segments
            )
        if solve_kind == "transpose_cw":
            vec_real = _triton_spsv_csr_transpose_cw_vector
            vec_complex = _triton_spsv_csr_transpose_cw_vector_complex
        else:
            vec_real = _triton_spsv_csr_transpose_push_vector
            vec_complex = _triton_spsv_csr_transpose_push_vector_complex
    elif solve_kind == "csr_cw":
        if block_nnz is None and max_segments is None:
            block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        else:
            block_nnz_use, max_segments_use = _auto_spsv_launch_config(
                kernel_indptr, block_nnz=block_nnz, max_segments=max_segments
            )
        vec_real = _triton_spsv_csr_cw_vector
        vec_complex = _triton_spsv_csr_cw_vector_complex
    else:
        if block_nnz is None and max_segments is None:
            block_nnz_use, max_segments_use = default_block_nnz, default_max_segments
        else:
            block_nnz_use, max_segments_use = _auto_spsv_launch_config(
                kernel_indptr, block_nnz=block_nnz, max_segments=max_segments
            )
        vec_real = _triton_spsv_csr_vector
        vec_complex = _triton_spsv_csr_vector_complex
    diag_eps = _spsv_diag_eps_for_dtype(compute_dtype)

    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    transpose_diag_in = transpose_diag
    if transpose_diag is not None and compute_dtype != data.dtype:
        transpose_diag_in = transpose_diag.to(compute_dtype)
    cw_diag_in = cw_diag
    if cw_diag is not None and compute_dtype != data.dtype:
        cw_diag_in = cw_diag.to(compute_dtype)
    worker_count_use = cw_worker_count
    rhs_cols = 1 if b_in.ndim == 1 else int(b_in.shape[1])
    matrix_stats_use = dict(matrix_stats)
    if solve_kind in ("csr_cw", "transpose_cw"):
        worker_count_use = _resolve_cw_worker_count(
            n_rows,
            matrix_stats_use,
            rhs_cols,
            cached_worker_count=cw_worker_count,
        )
    complex_kernel_data_ri = None
    complex_transpose_diag_ri = None
    complex_cw_diag_ri = None
    if torch.is_complex(data_in):
        if compute_dtype == solve_plan["kernel_data"].dtype:
            complex_kernel_data_ri = solve_plan.get("kernel_data_ri")
            complex_transpose_diag_ri = solve_plan.get("transpose_diag_ri")
            complex_cw_diag_ri = solve_plan.get("cw_diag_ri")
        if complex_kernel_data_ri is None:
            complex_kernel_data_ri = _complex_interleaved_view(data_in)
        if transpose_diag_in is not None and complex_transpose_diag_ri is None:
            complex_transpose_diag_ri = _complex_interleaved_view(transpose_diag_in)
        if cw_diag_in is not None and complex_cw_diag_ri is None:
            complex_cw_diag_ri = _complex_interleaved_view(cw_diag_in)
    if b_in.ndim == 1:
        if torch.is_complex(data_in):
            if solve_kind == "transpose_push":
                x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    unit_diagonal=unit_diagonal,
                    conjugate=transpose_conjugate,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
                    diag_eps=diag_eps,
                    launch_groups=launch_groups,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                    data_ri_in=complex_kernel_data_ri,
                )
            elif solve_kind == "transpose_cw":
                x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    transpose_diag_in,
                    transpose_indegree_init,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    unit_diagonal=unit_diagonal,
                    conjugate=transpose_conjugate,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
                    diag_eps=diag_eps,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                    worker_count=worker_count_use,
                    matrix_stats=matrix_stats_use,
                    data_ri_in=complex_kernel_data_ri,
                    diag_ri_in=complex_transpose_diag_ri,
                )
            elif solve_kind == "csr_cw":
                x = vec_complex(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    cw_diag_in,
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
                    data_ri_in=complex_kernel_data_ri,
                    diag_ri_in=complex_cw_diag_ri,
                )
            else:
                x = vec_complex(
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
                    levels=launch_groups,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                    data_ri_in=complex_kernel_data_ri,
                )
        else:
            if solve_kind == "transpose_push":
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
                    launch_groups=launch_groups,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                )
            elif solve_kind == "transpose_cw":
                x = vec_real(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    transpose_diag_in,
                    transpose_indegree_init,
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
                )
            elif solve_kind == "csr_cw":
                x = vec_real(
                    data_in,
                    kernel_indices,
                    kernel_indptr,
                    cw_diag_in,
                    b_in,
                    n_rows,
                    lower=lower_eff,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
                    diag_eps=diag_eps,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                    worker_count=worker_count_use,
                    matrix_stats=matrix_stats_use,
                )
            else:
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
                    levels=launch_groups,
                    block_nnz_use=block_nnz_use,
                    max_segments_use=max_segments_use,
                )
    else:
        b_cols = b_in if b_in.is_contiguous() else b_in.contiguous()
        cols = []
        for bj in torch.unbind(b_cols, dim=1):
            if torch.is_complex(data_in):
                if solve_kind == "transpose_push":
                    cols.append(
                        vec_complex(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            bj,
                            n_rows,
                            lower=lower_eff,
                            unit_diagonal=unit_diagonal,
                            conjugate=transpose_conjugate,
                            block_nnz=block_nnz,
                            max_segments=max_segments,
                            diag_eps=diag_eps,
                            launch_groups=launch_groups,
                            block_nnz_use=block_nnz_use,
                            max_segments_use=max_segments_use,
                            data_ri_in=complex_kernel_data_ri,
                        )
                    )
                elif solve_kind == "transpose_cw":
                    cols.append(
                        vec_complex(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            transpose_diag_in,
                            transpose_indegree_init,
                            bj,
                            n_rows,
                            lower=lower_eff,
                            unit_diagonal=unit_diagonal,
                            conjugate=transpose_conjugate,
                            block_nnz=block_nnz,
                            max_segments=max_segments,
                            diag_eps=diag_eps,
                            block_nnz_use=block_nnz_use,
                            max_segments_use=max_segments_use,
                            worker_count=worker_count_use,
                            matrix_stats=matrix_stats_use,
                            data_ri_in=complex_kernel_data_ri,
                            diag_ri_in=complex_transpose_diag_ri,
                        )
                    )
                elif solve_kind == "csr_cw":
                    cols.append(
                        vec_complex(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            cw_diag_in,
                            bj,
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
                            data_ri_in=complex_kernel_data_ri,
                            diag_ri_in=complex_cw_diag_ri,
                        )
                    )
                else:
                    cols.append(
                        vec_complex(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            bj,
                            n_rows,
                            lower=lower_eff,
                            unit_diagonal=unit_diagonal,
                            block_nnz=block_nnz,
                            max_segments=max_segments,
                            diag_eps=diag_eps,
                            levels=launch_groups,
                            block_nnz_use=block_nnz_use,
                            max_segments_use=max_segments_use,
                            data_ri_in=complex_kernel_data_ri,
                        )
                    )
            else:
                if solve_kind == "transpose_push":
                    cols.append(
                        vec_real(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            bj,
                            n_rows,
                            lower=lower_eff,
                            unit_diagonal=unit_diagonal,
                            block_nnz=block_nnz,
                            max_segments=max_segments,
                            diag_eps=diag_eps,
                            launch_groups=launch_groups,
                            block_nnz_use=block_nnz_use,
                            max_segments_use=max_segments_use,
                        )
                    )
                elif solve_kind == "transpose_cw":
                    cols.append(
                        vec_real(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            transpose_diag_in,
                            transpose_indegree_init,
                            bj,
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
                        )
                    )
                elif solve_kind == "csr_cw":
                    cols.append(
                        vec_real(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            cw_diag_in,
                            bj,
                            n_rows,
                            lower=lower_eff,
                            block_nnz=block_nnz,
                            max_segments=max_segments,
                            diag_eps=diag_eps,
                            block_nnz_use=block_nnz_use,
                            max_segments_use=max_segments_use,
                            worker_count=worker_count_use,
                            matrix_stats=matrix_stats_use,
                        )
                    )
                else:
                    cols.append(
                        vec_real(
                            data_in,
                            kernel_indices,
                            kernel_indptr,
                            bj,
                            n_rows,
                            lower=lower_eff,
                            unit_diagonal=unit_diagonal,
                            block_nnz=block_nnz,
                            max_segments=max_segments,
                            diag_eps=diag_eps,
                            levels=launch_groups,
                            block_nnz_use=block_nnz_use,
                            max_segments_use=max_segments_use,
                        )
                    )
        x = torch.stack(cols, dim=1)
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


def _analyze_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsv_csr_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsv_csr_runtime(
        data,
        indices,
        indptr,
        b,
        shape,
        lower,
        transpose,
        unit_diagonal,
    )
    if return_time:
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0


def flagsparse_spsv_coo(
    data,
    row,
    col,
    b,
    shape,
    lower=True,
    unit_diagonal=False,
    transpose=False,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
):
    """COO SpSV by canonicalizing COO into CSR, then reusing CSR SpSV."""
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
    data_csr, indices_csr, indptr_csr = _coo_to_csr_sorted_unique(
        data, row64, col64, n_rows, n_cols
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
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
    )
