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

import argparse
import csv
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as ast

DEFAULT_CASES = [
    (32_768, 1_024),
    (131_072, 4_096),
    (524_288, 16_384),
    (1_048_576, 65_536),
]
DEFAULT_VALUE_DTYPES = "float16,bfloat16,float32,float64,complex64,complex128"
DEFAULT_INDEX_DTYPES = "int32,int64"
WARMUP = 20
ITERS = 200
KERNEL_GRAPH_BATCH = 100


def _fmt_ms(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _fmt_speedup(value):
    if value is None:
        return "N/A"
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


def _fmt_err(value):
    if value is None:
        return "N/A"
    return f"{value:.2e}"


def _parse_value_dtypes(raw):
    allowed = {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }
    tokens = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        raise ValueError("value dtypes list is empty")
    invalid = [tok for tok in tokens if tok not in allowed]
    if invalid:
        raise ValueError(f"unsupported value dtypes: {invalid}")
    return tokens


def _parse_index_dtypes(raw):
    mapping = {"int32": torch.int32, "int64": torch.int64}
    tokens = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not tokens:
        raise ValueError("index dtypes list is empty")
    invalid = [tok for tok in tokens if tok not in mapping]
    if invalid:
        raise ValueError(f"unsupported index dtypes: {invalid}")
    return [(tok, mapping[tok]) for tok in tokens]


def _parse_cases(raw):
    if raw is None or not str(raw).strip():
        return list(DEFAULT_CASES)
    pairs = []
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid case '{item}', expected dense:nnz")
        left, right = item.split(":", 1)
        dense_size = int(left)
        nnz = int(right)
        if dense_size <= 0 or nnz <= 0:
            raise ValueError(f"case values must be positive: {item}")
        pairs.append((dense_size, nnz))
    if not pairs:
        raise ValueError("case list is empty")
    return pairs


def _ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(path, rows, fieldnames):
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _to_real_imag(value):
    if hasattr(value, "is_complex") and value.is_complex():
        return float(value.real.item()), float(value.imag.item())
    if getattr(value, "ndim", 0) == 1 and int(value.numel()) == 2:
        return float(value[0].item()), float(value[1].item())
    scalar = float(value.item())
    return scalar, 0.0


def _collect_samples(case_id, expected, flagsparse_out, limit):
    rows = []
    if expected is None or flagsparse_out is None:
        return rows
    max_items = min(int(limit), int(expected.shape[0]), int(flagsparse_out.shape[0]))
    for pos in range(max_items):
        exp_val = expected[pos]
        fs_val = flagsparse_out[pos]
        exp_real, exp_imag = _to_real_imag(exp_val)
        fs_real, fs_imag = _to_real_imag(fs_val)
        abs_error = float(torch.max(torch.abs(fs_val - exp_val)).item())
        rows.append(
            {
                "case_id": case_id,
                "pos": pos,
                "expected_real": exp_real,
                "expected_imag": exp_imag,
                "flagsparse_real": fs_real,
                "flagsparse_imag": fs_imag,
                "abs_error": abs_error,
            }
        )
    return rows


def _check_dtype_supported(value_dtype_req):
    if value_dtype_req in ("bfloat16",) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 not supported on this GPU")


def _resolve_value_dtype(value_dtype_req):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    return mapping[value_dtype_req]


def _is_supported_gather_combo(index_dtype):
    # Required gather coverage is the full 6 value dtypes x 2 index dtypes matrix.
    return index_dtype in (torch.int32, torch.int64)


def _status_from_result(verification):
    triton_ok = verification.get("triton_match_pytorch")
    cusparse_ok = verification.get("cusparse_match_pytorch")
    overall_ok = bool(triton_ok) and (cusparse_ok is None or bool(cusparse_ok))
    return "PASS" if overall_ok else "FAIL"


def _print_header():
    print("-" * 196)
    print(
        f"{'ValueReq':>14} {'ValueEff':>18} {'Index':>6} {'Dense':>10} {'NNZ':>10} "
        f"{'IFB':>4} {'FS(ms)':>10} {'PT(ms)':>10} {'CS(ms)':>10} "
        f"{'FS/PT':>8} {'FS/CS':>8} {'Status':>6} {'Err(FS)':>12} {'Err(CS)':>12}"
    )
    print("-" * 196)


def _print_row(row):
    print(
        f"{row['value_dtype_req']:>14} {row['value_dtype_compute']:>18} {row['index_dtype']:>6} "
        f"{row['dense_size']:>10,d} {row['nnz']:>10,d} {str(row['index_fallback_applied']):>4} "
        f"{_fmt_ms(row['triton_ms']):>10} {_fmt_ms(row['pytorch_ms']):>10} {_fmt_ms(row['cusparse_ms']):>10} "
        f"{_fmt_speedup(row['triton_speedup_vs_pytorch']):>8} {_fmt_speedup(row['triton_speedup_vs_cusparse']):>8} "
        f"{row['status']:>6} {_fmt_err(row['triton_max_error']):>12} {_fmt_err(row['cusparse_max_error']):>12}"
    )


def run_cli(args):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return

    value_dtype_tokens = _parse_value_dtypes(args.value_dtypes)
    index_dtype_pairs = _parse_index_dtypes(args.index_dtypes)
    run_cusparse = not args.no_cusparse
    work_cases = _parse_cases(args.cases)

    print("=" * 180)
    print("FLAGSPARSE GATHER BENCHMARK/VALIDATION")
    print("=" * 180)
    print(f"FlagSparse source: {Path(ast.__file__).resolve()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Warmup: {args.warmup} | Iterations: {args.iters} | "
        f"Kernel graph batch: {KERNEL_GRAPH_BATCH} | "
        f"index_fallback_policy: {args.index_fallback_policy}"
    )
    print()
    _print_header()

    summary_rows = []
    sample_rows = []
    total_cases = 0
    failed_cases = 0

    for value_dtype in value_dtype_tokens:
        for index_name, index_dtype in index_dtype_pairs:
            if not _is_supported_gather_combo(index_dtype):
                continue
            for dense_size, nnz in work_cases:
                dense_size = int(dense_size)
                nnz = int(nnz)
                case_id = (
                    f"{value_dtype}|{index_name}|dense={dense_size}|nnz={nnz}|"
                    f"ifb_policy={args.index_fallback_policy}"
                )
                total_cases += 1
                try:
                    _check_dtype_supported(value_dtype)
                    value_dtype_t = _resolve_value_dtype(value_dtype)
                    result = ast.benchmark_gather_case(
                        index_dtype=index_dtype,
                        dense_size=dense_size,
                        nnz=nnz,
                        value_dtype=value_dtype_t,
                        warmup=args.warmup,
                        iters=args.iters,
                        run_cusparse=run_cusparse,
                    )
                    perf = result["performance"]
                    verify = result["verification"]
                    params = result["parameters"]
                    backend = result["backend_status"]
                    timing_method = perf.get("kernel_timing_method")
                    graph_batch = params.get("kernel_graph_batch")
                    if timing_method != "cuda_graph_event_amortized_device_estimate":
                        raise RuntimeError(
                            "loaded benchmark_gather_case does not provide CUDA Graph timing; "
                            f"flagsparse was loaded from {Path(ast.__file__).resolve()}"
                        )
                    if graph_batch != KERNEL_GRAPH_BATCH:
                        raise RuntimeError(
                            "unexpected gather graph batch: "
                            f"expected {KERNEL_GRAPH_BATCH}, got {graph_batch}"
                        )
                    status = _status_from_result(verify)
                    if status != "PASS":
                        failed_cases += 1
                    row = {
                        "case_id": case_id,
                        "gpu": torch.cuda.get_device_name(0),
                        "value_dtype_req": value_dtype,
                        "value_dtype_compute": str(params.get("value_dtype")).replace(
                            "torch.", ""
                        ),
                        "index_dtype": str(params.get("index_dtype")).replace(
                            "torch.", ""
                        ),
                        "dense_size": int(params.get("dense_size")),
                        "nnz": int(params.get("nnz")),
                        "mode": "gather_triton",
                        "index_fallback_policy": args.index_fallback_policy,
                        "index_fallback_applied": False,
                        "triton_ms": perf.get("triton_ms"),
                        "pytorch_ms": perf.get("pytorch_ms"),
                        "cusparse_ms": perf.get("cusparse_ms"),
                        "triton_speedup_vs_pytorch": perf.get(
                            "triton_speedup_vs_pytorch"
                        ),
                        "triton_speedup_vs_cusparse": perf.get(
                            "triton_speedup_vs_cusparse"
                        ),
                        "kernel_timing_method": timing_method,
                        "kernel_graph_batch": graph_batch,
                        "triton_match_pytorch": verify.get("triton_match_pytorch"),
                        "cusparse_match_pytorch": verify.get("cusparse_match_pytorch"),
                        "triton_max_error": verify.get("triton_max_error"),
                        "cusparse_max_error": verify.get("cusparse_max_error"),
                        "cusparse_unavailable_reason": backend.get(
                            "cusparse_unavailable_reason"
                        ),
                        "index_fallback_reason": None,
                        "status": status,
                    }
                    summary_rows.append(row)
                    _print_row(row)

                    if args.csv_samples:
                        sample_rows.extend(
                            _collect_samples(
                                case_id,
                                result["samples"].get("pytorch"),
                                result["samples"].get("triton"),
                                args.sample_limit,
                            )
                        )
                except Exception as exc:
                    failed_cases += 1
                    error_text = f"{exc.__class__.__name__}: {exc}"
                    print(f"\nERROR [{case_id}]: {error_text}")
                    row = {
                        "case_id": case_id,
                        "gpu": torch.cuda.get_device_name(0),
                        "value_dtype_req": value_dtype,
                        "value_dtype_compute": "N/A",
                        "index_dtype": index_name,
                        "dense_size": dense_size,
                        "nnz": nnz,
                        "mode": "gather_triton",
                        "index_fallback_policy": args.index_fallback_policy,
                        "index_fallback_applied": False,
                        "triton_ms": None,
                        "pytorch_ms": None,
                        "cusparse_ms": None,
                        "triton_speedup_vs_pytorch": None,
                        "triton_speedup_vs_cusparse": None,
                        "kernel_timing_method": "cuda_graph_event_amortized_device_estimate",
                        "kernel_graph_batch": KERNEL_GRAPH_BATCH,
                        "triton_match_pytorch": None,
                        "cusparse_match_pytorch": None,
                        "triton_max_error": None,
                        "cusparse_max_error": None,
                        "cusparse_unavailable_reason": error_text,
                        "index_fallback_reason": error_text,
                        "status": "ERROR",
                    }
                    summary_rows.append(row)
                    _print_row(row)

    print("-" * 196)
    print(f"Total cases: {total_cases}")
    print(f"Failed cases: {failed_cases}")
    print(f"Passed cases: {total_cases - failed_cases}")

    if args.csv_summary:
        summary_fields = [
            "case_id",
            "gpu",
            "value_dtype_req",
            "value_dtype_compute",
            "index_dtype",
            "dense_size",
            "nnz",
            "mode",
            "index_fallback_policy",
            "index_fallback_applied",
            "triton_ms",
            "pytorch_ms",
            "cusparse_ms",
            "triton_speedup_vs_pytorch",
            "triton_speedup_vs_cusparse",
            "kernel_timing_method",
            "kernel_graph_batch",
            "triton_match_pytorch",
            "cusparse_match_pytorch",
            "triton_max_error",
            "cusparse_max_error",
            "cusparse_unavailable_reason",
            "index_fallback_reason",
            "status",
        ]
        _write_csv(args.csv_summary, summary_rows, summary_fields)
        print(f"Wrote summary CSV: {args.csv_summary}")

    if args.csv_samples:
        sample_fields = [
            "case_id",
            "pos",
            "expected_real",
            "expected_imag",
            "flagsparse_real",
            "flagsparse_imag",
            "abs_error",
        ]
        _write_csv(args.csv_samples, sample_rows, sample_fields)
        print(f"Wrote samples CSV: {args.csv_samples}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Gather benchmark/validation aligned with scatter-style CLI and CSV export."
    )
    parser.add_argument("--value-dtypes", default=DEFAULT_VALUE_DTYPES)
    parser.add_argument("--index-dtypes", default=DEFAULT_INDEX_DTYPES)
    parser.add_argument(
        "--cases",
        default=",".join(f"{dense}:{nnz}" for dense, nnz in DEFAULT_CASES),
        help="Comma-separated dense:nnz pairs, e.g. 32768:1024,131072:4096",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument(
        "--index-fallback-policy", choices=["auto", "strict"], default="auto"
    )
    parser.add_argument("--csv-summary", default=None)
    parser.add_argument("--csv-samples", default=None)
    parser.add_argument("--sample-limit", type=int, default=32)
    return parser


if __name__ == "__main__":
    cli_parser = build_parser()
    run_cli(cli_parser.parse_args())
