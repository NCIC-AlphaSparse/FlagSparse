"""
Diagnostics for CSR SpMM opt_alg2 protected four-way comparisons.

Usage:
    python tests/diagnose_spmm_opt_alg2.py path/to/matrix.mtx --dense-cols 32 --out-dir diag_out
    python tests/diagnose_spmm_opt_alg2.py path/to/mtx_dir --dense-cols 32 --out-dir diag_out --no-cusparse
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from test_spmm_opt import load_mtx_to_csr_torch
from test_spmm_opt_alg2 import (
    SUMMARY_FIELDS,
    _resolve_input_paths,
    run_one_case,
)


BUCKET_FIELDS = [
    "matrix",
    "bucket_label",
    "row_count",
    "row_nnz_min",
    "row_nnz_mean",
    "row_nnz_max",
    "block_k",
    "block_n",
    "num_warps",
    "num_stages",
    "batch_rows",
    "segments",
]

LAUNCH_FIELDS = [
    "matrix",
    "device_name",
    "sm_count",
    "warp_size",
    "dtype",
    "dense_cols",
    "bucket_label",
    "grid_m",
    "grid_n",
    "chosen_block_k",
    "chosen_block_n",
    "chosen_num_warps",
    "chosen_num_stages",
]

WORST_ROW_FIELDS = [
    "matrix",
    "row",
    "row_nnz",
    "bucket_label",
    "opt_vs_torch_row_err",
    "opt_vs_cusparse_row_err",
    "opt_alg2_vs_torch_row_err",
    "opt_alg2_vs_cusparse_row_err",
    "opt_alg2_sym_vs_torch_row_err",
    "opt_alg2_sym_vs_cusparse_row_err",
]


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def _fmt_speed(value):
    return "N/A" if value is None else f"{value:.2f}x"


def _bucket_label_map(prepared):
    label_map = {}
    for bucket in prepared.opt_buckets:
        for row in bucket["rows"].to(torch.int64).cpu().tolist():
            label_map[row] = bucket["label"]
    return label_map


def _build_bucket_rows(matrix_name, prepared, meta):
    rows = []
    for bucket, launch in zip(prepared.opt_buckets, meta["launch_configs"]):
        bucket_rows = bucket["rows"].to(torch.int64)
        if bucket_rows.numel() == 0:
            continue
        row_nnz = prepared.row_lengths.index_select(0, bucket_rows.to(prepared.row_lengths.device)).to(torch.float64)
        rows.append(
            {
                "matrix": matrix_name,
                "bucket_label": launch["bucket_label"],
                "row_count": int(bucket_rows.numel()),
                "row_nnz_min": int(row_nnz.min().item()),
                "row_nnz_mean": float(row_nnz.mean().item()),
                "row_nnz_max": int(row_nnz.max().item()),
                "block_k": launch["block_k"],
                "block_n": launch["block_n"],
                "num_warps": launch["num_warps"],
                "num_stages": launch["num_stages"],
                "batch_rows": launch["batch_rows"],
                "segments": launch["segments"],
            }
        )
    return rows


def _build_launch_rows(matrix_name, dtype_name, dense_cols, meta):
    rows = []
    for launch in meta["launch_configs"]:
        rows.append(
            {
                "matrix": matrix_name,
                "device_name": meta["device_name"],
                "sm_count": meta["sm_count"],
                "warp_size": meta["warp_size"],
                "dtype": dtype_name,
                "dense_cols": dense_cols,
                "bucket_label": launch["bucket_label"],
                "grid_m": launch.get("grid_m"),
                "grid_n": launch.get("grid_n"),
                "chosen_block_k": launch["block_k"],
                "chosen_block_n": launch["block_n"],
                "chosen_num_warps": launch["num_warps"],
                "chosen_num_stages": launch["num_stages"],
            }
        )
    return rows


def _build_worst_rows(matrix_name, prepared, profiles, top_rows):
    label_map = _bucket_label_map(prepared)
    alg2_vs_torch = profiles["opt_alg2_vs_torch"]["row_max_ratio"]
    if alg2_vs_torch is None or alg2_vs_torch.numel() == 0:
        return []
    order = torch.argsort(alg2_vs_torch, descending=True)
    if top_rows is not None and top_rows > 0:
        order = order[:top_rows]
    opt_vs_torch = profiles["opt_vs_torch"]["row_max_ratio"]
    opt_vs_cusparse = profiles["opt_vs_cusparse"]["row_max_ratio"]
    alg2_vs_cusparse = profiles["opt_alg2_vs_cusparse"]["row_max_ratio"]
    alg2_sym_vs_torch = profiles["opt_alg2_sym_vs_torch"]["row_max_ratio"]
    alg2_sym_vs_cusparse = profiles["opt_alg2_sym_vs_cusparse"]["row_max_ratio"]
    row_lengths = prepared.row_lengths.to(torch.int64)
    rows = []
    for row_id in order.to(torch.int64).cpu().tolist():
        rows.append(
            {
                "matrix": matrix_name,
                "row": row_id,
                "row_nnz": int(row_lengths[row_id].item()),
                "bucket_label": label_map.get(row_id, "unassigned"),
                "opt_vs_torch_row_err": float(opt_vs_torch[row_id].item()) if opt_vs_torch is not None else None,
                "opt_vs_cusparse_row_err": float(opt_vs_cusparse[row_id].item()) if opt_vs_cusparse is not None else None,
                "opt_alg2_vs_torch_row_err": float(alg2_vs_torch[row_id].item()),
                "opt_alg2_vs_cusparse_row_err": (
                    float(alg2_vs_cusparse[row_id].item()) if alg2_vs_cusparse is not None else None
                ),
                "opt_alg2_sym_vs_torch_row_err": (
                    float(alg2_sym_vs_torch[row_id].item()) if alg2_sym_vs_torch is not None else None
                ),
                "opt_alg2_sym_vs_cusparse_row_err": (
                    float(alg2_sym_vs_cusparse[row_id].item()) if alg2_sym_vs_cusparse is not None else None
                ),
            }
        )
    return rows


def _dtype_suffix(dtype_name):
    return "f32" if dtype_name == "float32" else "f64"


def _run_diagnose_for_dtype(args, paths, dtype_name, with_cusparse):
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    suffix = _dtype_suffix(dtype_name)
    out_dir = f"{args.out_dir}_{suffix}"
    index_dtype = torch.int32
    device = torch.device("cuda")

    summary_rows = []
    bucket_rows = []
    launch_rows = []
    bucket_rows_alg2s = []
    launch_rows_alg2s = []
    worst_rows = []
    for path in paths:
        data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
        indices = indices.to(index_dtype)
        result = run_one_case(
            os.path.basename(path),
            data,
            indices,
            indptr,
            shape,
            dtype,
            index_dtype,
            args.dense_cols,
            args.warmup,
            args.iters,
            args.seed,
            with_cusparse,
            return_details=True,
        )
        matrix_name = result["summary"]["matrix"]
        summary_rows.append(result["summary"])
        bucket_rows.extend(_build_bucket_rows(matrix_name, result["prepared_alg2"], result["alg2_meta"]))
        launch_rows.extend(_build_launch_rows(matrix_name, dtype_name, args.dense_cols, result["alg2_meta"]))
        bucket_rows_alg2s.extend(_build_bucket_rows(matrix_name, result["prepared_alg2_sym"], result["alg2_sym_meta"]))
        launch_rows_alg2s.extend(_build_launch_rows(matrix_name, dtype_name, args.dense_cols, result["alg2_sym_meta"]))
        worst_rows.extend(_build_worst_rows(matrix_name, result["prepared_alg2"], result["profiles"], args.top_rows))
        print(
            f"[{suffix}] "
            f"{matrix_name}: matrix_status={result['summary']['matrix_status']}  "
            f"opt_alg2_total_ms={result['summary']['opt_alg2_op_total_ms']:.4f}  "
            f"symbolic_ms={result['summary']['opt_alg2_symbolic_ms']:.4f}  "
            f"compute_ms={result['summary']['opt_alg2_compute_ms']:.4f}  "
            f"alg2s_total_ms={result['summary']['opt_alg2_sym_total_ms']:.4f}  "
            f"alg2s_symbolic_ms={result['summary']['opt_alg2_sym_symbolic_ms']:.4f}  "
            f"alg2s_compute_ms={result['summary']['opt_alg2_sym_compute_ms']:.4f}  "
            f"Base/Alg2={result['summary']['base_vs_opt_alg2_speedup']:.2f}x  "
            f"Torch/Alg2={result['summary']['torch_vs_opt_alg2_speedup']:.2f}x  "
            f"CU/Alg2={_fmt_speed(result['summary']['cusparse_vs_opt_alg2_speedup'])}  "
            f"Base/Alg2S={result['summary']['base_vs_opt_alg2_sym_speedup']:.2f}x  "
            f"Torch/Alg2S={result['summary']['torch_vs_opt_alg2_sym_speedup']:.2f}x  "
            f"CU/Alg2S={_fmt_speed(result['summary']['cusparse_vs_opt_alg2_sym_speedup'])}  "
            f"opt_alg2_vs_torch_err={result['summary']['opt_alg2_vs_torch_err']}"
            f"  opt_alg2_sym_vs_torch_err={result['summary']['opt_alg2_sym_vs_torch_err']}"
        )

    _write_csv(os.path.join(out_dir, f"summary_{suffix}.csv"), summary_rows, SUMMARY_FIELDS)
    _write_csv(os.path.join(out_dir, f"bucket_stats_{suffix}.csv"), bucket_rows, BUCKET_FIELDS)
    _write_csv(os.path.join(out_dir, f"launch_stats_{suffix}.csv"), launch_rows, LAUNCH_FIELDS)
    _write_csv(os.path.join(out_dir, f"worst_rows_{suffix}.csv"), worst_rows, WORST_ROW_FIELDS)
    _write_csv(os.path.join(out_dir, f"bucket_stats_alg2s_{suffix}.csv"), bucket_rows_alg2s, BUCKET_FIELDS)
    _write_csv(os.path.join(out_dir, f"launch_stats_alg2s_{suffix}.csv"), launch_rows_alg2s, LAUNCH_FIELDS)
    print(f"Wrote {dtype_name} diagnostics to {os.path.abspath(out_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose CSR SpMM opt_alg2 with protected comparisons.")
    parser.add_argument("input_path", help=".mtx file or directory")
    parser.add_argument("--out-dir", required=True, help="Base directory name for dtype-suffixed diagnostics")
    parser.add_argument("--dtype", default="all", choices=["all", "float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cusparse", action="store_true", help="Disable cuSPARSE reference timing")
    parser.add_argument("--with-cusparse", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--top-rows", type=int, default=512, help="Top rows by opt_alg2_vs_torch error")
    args = parser.parse_args()

    paths = _resolve_input_paths([args.input_path])
    if not paths:
        raise ValueError(f"No .mtx files found from input: {args.input_path}")

    dtype_names = ["float32", "float64"] if args.dtype == "all" else [args.dtype]
    with_cusparse = not args.no_cusparse
    for dtype_name in dtype_names:
        _run_diagnose_for_dtype(args, paths, dtype_name, with_cusparse)


if __name__ == "__main__":
    main()
