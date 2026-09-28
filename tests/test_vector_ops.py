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

"""Benchmark the sparse-vector operators: ``--op axpby`` or ``--op spvv``.

Synthetic cases (dense size, nnz) like the gather/scatter benchmarks, since these
operators take a sparse *vector*, not a matrix file. Each row carries:

  triton_ms    FlagSparse
  cusparse_ms  native cusparseAxpby / cusparseSpVV (CUDA only; empty elsewhere,
               with the reason in cusparse_reason)
  pytorch_ms   the same expression in PyTorch (index_add_ / index_select + sum)

and the speedup columns the unified runner reads (vendor first, then PyTorch).
The timed FlagSparse call passes ``validate=False``: the index range check is a
host sync that neither cuSPARSE nor the PyTorch expression performs. The accuracy
check runs once with validation on.
Accuracy is checked against a CPU float64 / complex128 golden: ``status`` is PASS
or FAIL per row, ERROR when FlagSparse raised.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from benchmark_utils import accelerator_device  # noqa: E402

import cusparse_generic_baseline as cusparse_baseline  # noqa: E402
import flagsparse as fs  # noqa: E402
from flagsparse.sparse_operations._common import _benchmark_cuda_op  # noqa: E402

DEFAULT_CASES = "32768:1024,131072:4096,524288:16384,1048576:65536"
# (value dtype, op) per operator; the q4 list's variants come first.
DEFAULT_DTYPES = {
    "spvv": "float16,int8,complex64,float32",
    "axpby": "float16,float32,complex64",
}
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "int8": torch.int8,
}
_TOL = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-4, torch.complex64: 1e-4}
_ALPHA, _BETA = 1.5, 0.5
# dtype -> the type token of the variant names (docs/NEW_OPERATORS_CUSPARSE_12_5.md):
# spvv writes its compute type, so half and int8 inputs name two types.
_SPVV_TAGS = {"float16": "f16f32", "bfloat16": "bf16f32", "int8": "i8i32"}
_TAGS = {"float16": "f16", "bfloat16": "bf16", "float32": "f32", "float64": "f64",
         "complex64": "c32", "complex128": "c64", "int8": "i8"}
FIELDS = [
    "variant",
    "case_id",
    "op",
    "value_dtype",
    "compute_dtype",
    "index_dtype",
    "dense_size",
    "nnz",
    "triton_ms",
    "cusparse_ms",
    "pytorch_ms",
    "triton_speedup_vs_cusparse",
    "triton_speedup_vs_pytorch",
    "triton_max_error",
    "cusparse_max_error",
    "cusparse_reason",
    "pytorch_reason",
    "status",
    "error",
]


def _compute_dtype(dtype):
    return {torch.float16: torch.float32, torch.bfloat16: torch.float32, torch.int8: torch.int32}.get(dtype, dtype)


def _wide(t):
    t = t if torch.is_tensor(t) else torch.tensor(t)
    return t.detach().cpu().to(torch.complex128 if t.is_complex() else torch.float64)


def _rand(n, dtype, gen):
    if dtype == torch.int8:
        return torch.randint(-4, 5, (n,), dtype=dtype, generator=gen)
    if dtype.is_complex:
        return torch.randn(n, dtype=dtype, generator=gen)
    return torch.randn(n, generator=gen).to(dtype)


def _error(got, ref):
    return float((_wide(got) - ref).abs().max() / max(1.0, float(ref.abs().max())))


def _passes(dtype, err):
    if dtype == torch.int8:
        return err == 0.0
    return err <= _TOL.get(dtype, 1e-10)


def _ratio(base, ours):
    return None if base is None or not ours else base / ours


def _run_case(op, dtype, spvv_op, dense_size, nnz, warmup, iters, gen):
    dev = accelerator_device()
    compute = _compute_dtype(dtype)
    idx_cpu = torch.randperm(dense_size, generator=gen)[:nnz].to(torch.int32)
    x_cpu, y_cpu = _rand(nnz, dtype, gen), _rand(dense_size, dtype, gen)
    idx, x = idx_cpu.to(dev), x_cpu.to(dev)
    name = str(dtype).replace("torch.", "")
    tag = (_SPVV_TAGS.get(name) if op == "spvv" else None) or _TAGS[name]
    row = {
        "variant": f"{op}_{tag}_int" + (f"_{spvv_op}" if op == "spvv" else ""),
        "case_id": f"{op}|{str(dtype).replace('torch.', '')}|dense={dense_size}|nnz={nnz}"
        + (f"|{spvv_op}" if op == "spvv" else ""),
        "op": spvv_op if op == "spvv" else "non",
        "value_dtype": str(dtype).replace("torch.", ""),
        "compute_dtype": str(compute).replace("torch.", ""),
        "index_dtype": "int32",
        "dense_size": dense_size,
        "nnz": nnz,
    }
    if op == "spvv":
        xr = _wide(x_cpu).conj() if spvv_op == "conj" else _wide(x_cpu)
        ref = (xr * _wide(y_cpu)[idx_cpu.long()]).sum().reshape(1)
        y = y_cpu.to(dev)
        ours = lambda: fs.flagsparse_spvv(x, idx, y, op=spvv_op, validate=False)  # noqa: E731
        first = ours()
        err = _error(first.reshape(1), ref)

        def torch_op():
            xx = x.conj() if spvv_op == "conj" else x
            return (xx.to(compute) * y.index_select(0, idx.long()).to(compute)).sum()
    else:
        ref = _wide(y_cpu) * _BETA
        ref[idx_cpu.long()] += _ALPHA * _wide(x_cpu)
        y_check = y_cpu.to(dev)
        fs.flagsparse_axpby(x, idx, y_check, alpha=_ALPHA, beta=_BETA)
        err = _error(y_check, ref)
        # The timed loops update y in place; beta < 1 keeps it bounded.
        y = y_cpu.to(dev)
        ours = lambda: fs.flagsparse_axpby(x, idx, y, alpha=_ALPHA, beta=_BETA, validate=False)  # noqa: E731
        y_torch = y_cpu.to(dev).to(compute)

        def torch_op():
            y_torch.mul_(_BETA).index_add_(0, idx.long(), x.to(compute) * _ALPHA)
            return y_torch

    row["triton_max_error"] = err
    _, row["triton_ms"] = _benchmark_cuda_op(ours, warmup, iters)
    try:
        _, row["pytorch_ms"] = _benchmark_cuda_op(torch_op, warmup, iters)
    except Exception as exc:  # e.g. a backend without complex index/sum kernels
        row["pytorch_reason"] = f"{type(exc).__name__}: {exc}"

    reason = cusparse_baseline.skip_reason()
    if reason is None:
        try:
            if op == "spvv":
                plan = cusparse_baseline.prepare_spvv(x, idx, y, spvv_op, compute)
                row["cusparse_max_error"] = _error(torch.tensor(plan.run()).reshape(1), ref)
            else:
                y_cu = y_cpu.to(dev)
                check = cusparse_baseline.prepare_axpby(x, idx, y_cu, _ALPHA, _BETA, compute)
                check.run()
                row["cusparse_max_error"] = _error(y_cu, ref)
                check.close()
                plan = cusparse_baseline.prepare_axpby(x, idx, y_cpu.to(dev), _ALPHA, _BETA, compute)
            _, row["cusparse_ms"] = _benchmark_cuda_op(plan.run, warmup, iters)
            plan.close()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    row["cusparse_reason"] = reason
    row["triton_speedup_vs_cusparse"] = _ratio(row.get("cusparse_ms"), row["triton_ms"])
    row["triton_speedup_vs_pytorch"] = _ratio(row.get("pytorch_ms"), row["triton_ms"])
    row["status"] = "PASS" if _passes(dtype, err) else "FAIL"
    return row


def _parse_cases(raw):
    cases = []
    for item in str(raw).split(","):
        dense, nnz = (int(v) for v in item.split(":"))
        if not 0 < nnz <= dense:
            raise ValueError(f"case {item}: need 0 < nnz <= dense")
        cases.append((dense, nnz))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--op", choices=("spvv", "axpby"), required=True)
    parser.add_argument("--csv-summary", required=True, help="output CSV path")
    parser.add_argument("--value-dtypes", default=None)
    parser.add_argument("--cases", default=DEFAULT_CASES, help="dense:nnz,...")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dtypes = [t.strip() for t in (args.value_dtypes or DEFAULT_DTYPES[args.op]).split(",") if t.strip()]
    unknown = [t for t in dtypes if t not in _DTYPES]
    if unknown:
        parser.error(f"unsupported value dtypes: {unknown}")
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for name in dtypes:
        dtype = _DTYPES[name]
        ops = ("conj",) if args.op == "spvv" and dtype.is_complex else ("non",)
        for spvv_op in ops:
            for dense, nnz in _parse_cases(args.cases):
                try:
                    row = _run_case(args.op, dtype, spvv_op, dense, nnz, args.warmup, args.iters, gen)
                except Exception as exc:
                    row = {
                        "case_id": f"{args.op}|{name}|dense={dense}|nnz={nnz}",
                        "op": spvv_op,
                        "value_dtype": name,
                        "dense_size": dense,
                        "nnz": nnz,
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(
                    f"{row['case_id']:48s} {row.get('status'):5s} "
                    f"ours={row.get('triton_ms')} cusparse={row.get('cusparse_ms')} "
                    f"torch={row.get('pytorch_ms')}"
                )
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_summary)) or ".", exist_ok=True)
    with open(args.csv_summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})
    return 1 if any(r.get("status") == "ERROR" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
