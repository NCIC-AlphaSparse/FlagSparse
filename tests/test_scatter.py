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

import torch

import sys
from pathlib import Path

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
DEFAULT_VALUE_DTYPES = "float16,float32,float64"
DEFAULT_INDEX_DTYPES = "int32,int64"
WARMUP = 20
ITERS = 200


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


def _parse_bool_token(raw, name):
    token = str(raw).strip().lower()
    if token in ("1", "true", "yes", "y", "on"):
        return True
    if token in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"{name} must be true/false, got: {raw}")


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
        if dense_size < 0 or nnz < 0:
            raise ValueError(f"case values must be non-negative: {item}")
        pairs.append((dense_size, nnz))
    if not pairs:
        raise ValueError("case list is empty")
    return pairs


def _iter_reset_output_modes(raw):
    token = str(raw).strip().lower()
    if token == "both":
        return [True, False]
    if token == "true":
        return [True]
    if token == "false":
        return [False]
    raise ValueError("reset-output must be one of: true, false, both")


def _status_from_result(verification):
    triton_ok = verification.get("triton_match_pytorch")
    cusparse_ok = verification.get("cusparse_match_pytorch")
    overall_ok = bool(triton_ok) and (cusparse_ok is None or bool(cusparse_ok))
    return "PASS" if overall_ok else "FAIL"


def _to_real_imag(value):
    if hasattr(value, "is_complex") and value.is_complex():
        return float(value.real.item()), float(value.imag.item())
    scalar = float(value.item())
    return scalar, 0.0


def _collect_samples(case_id, expected, triton, limit):
    rows = []
    if expected is None or triton is None:
        return rows
    max_items = min(int(limit), int(expected.numel()), int(triton.numel()))
    for pos in range(max_items):
        exp_val = expected[pos]
        tri_val = triton[pos]
        exp_real, exp_imag = _to_real_imag(exp_val)
        tri_real, tri_imag = _to_real_imag(tri_val)
        abs_error = float(torch.abs(tri_val - exp_val).item())
        rows.append(
            {
                "case_id": case_id,
                "pos": pos,
                "expected_real": exp_real,
                "expected_imag": exp_imag,
                "triton_real": tri_real,
                "triton_imag": tri_imag,
                "abs_error": abs_error,
            }
        )
    return rows


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


def _print_header():
    print("-" * 196)
    print(
        f"{'ValueReq':>10} {'ValueEff':>10} {'Index':>6} {'Dense':>10} {'NNZ':>10} "
        f"{'Reset':>6} {'FB':>4} {'IFB':>4} {'PT(ms)':>10} {'FS(ms)':>10} {'CS(ms)':>10} "
        f"{'FS/PT':>8} {'FS/CS':>8} {'Status':>6} {'Err(FS)':>12} {'Err(CS)':>12}"
    )
    print("-" * 196)


def _print_row(row):
    print(
        f"{row['value_dtype_req']:>10} {row['value_dtype_compute']:>10} {row['index_dtype']:>6} "
        f"{row['dense_size']:>10,d} {row['nnz']:>10,d} "
        f"{str(row['reset_output']):>6} {str(row['fallback_applied']):>4} "
        f"{str(row['index_fallback_applied']):>4} "
        f"{_fmt_ms(row['pytorch_ms']):>10} {_fmt_ms(row['triton_ms']):>10} {_fmt_ms(row['cusparse_ms']):>10} "
        f"{_fmt_speedup(row['triton_speedup_vs_pytorch']):>8} {_fmt_speedup(row['triton_speedup_vs_cusparse']):>8} "
        f"{row['status']:>6} {_fmt_err(row['triton_max_error']):>12} {_fmt_err(row['cusparse_max_error']):>12}"
    )


def run_cli(args):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return

    value_dtype_tokens = _parse_value_dtypes(args.value_dtypes)
    index_dtype_pairs = _parse_index_dtypes(args.index_dtypes)
    cases = _parse_cases(args.cases)
    reset_modes = _iter_reset_output_modes(args.reset_output)
    unique_indices = _parse_bool_token(args.unique_indices, "unique-indices")
    run_cusparse = not args.no_cusparse

    print("=" * 180)
    print("FLAGSPARSE SCATTER BENCHMARK/VALIDATION")
    print("=" * 180)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Warmup: {args.warmup} | Iterations: {args.iters} | "
        f"dtype_policy: {args.dtype_policy} | index_fallback_policy: {args.index_fallback_policy} | "
        f"unique_indices: {unique_indices}"
    )
    print()
    _print_header()

    summary_rows = []
    sample_rows = []
    total_cases = 0
    failed_cases = 0

    for value_dtype in value_dtype_tokens:
        for index_name, index_dtype in index_dtype_pairs:
            for reset_output in reset_modes:
                for dense_size, nnz in cases:
                    total_cases += 1
                    case_id = (
                        f"{value_dtype}|{index_name}|dense={dense_size}|nnz={nnz}|"
                        f"reset={str(reset_output).lower()}|unique={str(unique_indices).lower()}|"
                        f"policy={args.dtype_policy}|ifb_policy={args.index_fallback_policy}"
                    )
                    try:
                        result = ast.benchmark_scatter_case(
                            dense_size=dense_size,
                            nnz=nnz,
                            value_dtype=value_dtype,
                            index_dtype=index_dtype,
                            warmup=args.warmup,
                            iters=args.iters,
                            run_cusparse=run_cusparse,
                            unique_indices=unique_indices,
                            reset_output=reset_output,
                            dtype_policy=args.dtype_policy,
                            index_fallback_policy=args.index_fallback_policy,
                        )
                        perf = result["performance"]
                        verify = result["verification"]
                        params = result["parameters"]
                        backend = result["backend_status"]
                        status = _status_from_result(verify)
                        if status != "PASS":
                            failed_cases += 1
                        row = {
                            "case_id": case_id,
                            "gpu": torch.cuda.get_device_name(0),
                            "value_dtype_req": params.get("value_dtype"),
                            "value_dtype_compute": str(
                                params.get("effective_value_dtype")
                            ),
                            "index_dtype": str(params.get("index_dtype")).replace(
                                "torch.", ""
                            ),
                            "dense_size": int(params.get("dense_size")),
                            "nnz": int(params.get("nnz")),
                            "unique_indices": bool(params.get("unique_indices")),
                            "reset_output": bool(params.get("reset_output")),
                            "dtype_policy": params.get("dtype_policy"),
                            "fallback_applied": bool(params.get("fallback_applied")),
                            "index_fallback_applied": bool(
                                backend.get("index_fallback_applied")
                            ),
                            "triton_ms": perf.get("triton_ms"),
                            "pytorch_ms": perf.get("pytorch_ms"),
                            "cusparse_ms": perf.get("cusparse_ms"),
                            "triton_speedup_vs_pytorch": perf.get(
                                "triton_speedup_vs_pytorch"
                            ),
                            "triton_speedup_vs_cusparse": perf.get(
                                "triton_speedup_vs_cusparse"
                            ),
                            "triton_match_pytorch": verify.get("triton_match_pytorch"),
                            "cusparse_match_pytorch": verify.get(
                                "cusparse_match_pytorch"
                            ),
                            "triton_max_error": verify.get("triton_max_error"),
                            "cusparse_max_error": verify.get("cusparse_max_error"),
                            "cusparse_unavailable_reason": backend.get(
                                "cusparse_unavailable_reason"
                            ),
                            "fallback_reason": backend.get("fallback_reason"),
                            "index_fallback_reason": backend.get(
                                "index_fallback_reason"
                            ),
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
                        row = {
                            "case_id": case_id,
                            "gpu": torch.cuda.get_device_name(0),
                            "value_dtype_req": value_dtype,
                            "value_dtype_compute": "N/A",
                            "index_dtype": index_name,
                            "dense_size": dense_size,
                            "nnz": nnz,
                            "unique_indices": unique_indices,
                            "reset_output": reset_output,
                            "dtype_policy": args.dtype_policy,
                            "fallback_applied": False,
                            "index_fallback_applied": False,
                            "triton_ms": None,
                            "pytorch_ms": None,
                            "cusparse_ms": None,
                            "triton_speedup_vs_pytorch": None,
                            "triton_speedup_vs_cusparse": None,
                            "triton_match_pytorch": None,
                            "cusparse_match_pytorch": None,
                            "triton_max_error": None,
                            "cusparse_max_error": None,
                            "cusparse_unavailable_reason": str(exc),
                            "fallback_reason": None,
                            "index_fallback_reason": str(exc),
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
            "unique_indices",
            "reset_output",
            "dtype_policy",
            "fallback_applied",
            "index_fallback_applied",
            "triton_ms",
            "pytorch_ms",
            "cusparse_ms",
            "triton_speedup_vs_pytorch",
            "triton_speedup_vs_cusparse",
            "triton_match_pytorch",
            "cusparse_match_pytorch",
            "triton_max_error",
            "cusparse_max_error",
            "cusparse_unavailable_reason",
            "fallback_reason",
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
            "triton_real",
            "triton_imag",
            "abs_error",
        ]
        _write_csv(args.csv_samples, sample_rows, sample_fields)
        print(f"Wrote samples CSV: {args.csv_samples}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scatter benchmark/validation with dtype fallback and CSV export."
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
    parser.add_argument("--unique-indices", default="true")
    parser.add_argument(
        "--reset-output", choices=["true", "false", "both"], default="true"
    )
    parser.add_argument("--dtype-policy", choices=["auto", "strict"], default="auto")
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
