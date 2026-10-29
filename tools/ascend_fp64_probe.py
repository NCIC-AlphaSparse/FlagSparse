#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Time raw-COO preparation apart from native prepared FP64 SpMM, and ask whether
the sort is needed at all.

Supersedes the separate `ascend_fp64_prepare_bench.py`; the archived results in
`results_ascend_baseline_fix_20261029T065946Z/COMMANDS.md` name that old path.

Background. The delivery table reported spmm_coo_f64 at 0.529x against the CANN
baseline while the same comparison in a dedicated bench read 0.992x. Two things
differed, both measured here:

  1. Timing boundary. `tests/ascend_sparse_baseline.py` records the CANN adapter's
     setup in a separate `cann_prepare_ms` and times only `plan.run()`, while the
     delivery number timed the whole public one-shot call. Public COO prepare
     sorts the coordinates, so that sort landed inside the timed region.
  2. Input order. `benchmark_ascend.py` builds COO from `mmread(...).tocoo()`,
     the raw Matrix Market order. All ten delivery matrices are unsorted.

So this probe prepares once, records `prepare_ms` with the sort time broken out,
and times only the prepared run against CANN with the two alternating on the same
card. It instruments `_sort_coo_lex_inplace` to prove the sort runs exactly once
in prepare, checks the prepared coordinates really are lexicographically ordered,
and makes `torch.argsort` raise during steady-state timing so a future change that
moves sorting back into the run fails loudly instead of quietly costing speedup.

The remaining open question is whether the sort is needed at all: TensorFlow's
SparseTensorDenseMatMul does not require ordered indices, and nothing has ever
tested CANN's on unsorted input. `shuffled_max_error` answers it by calling
`run_fp64_spmm_once(..., sort_rows=False)` on randomly permuted COO coordinates
and comparing against the CPU FP64 golden.

Reading the result:
  speedup ~= 1 and shuffled_status OK   -> the sort is removable; flipping the
                                           `sort_rows` default would let the
                                           public one-shot reach the same ratio
  speedup ~= 1 and shuffled_status FAIL -> the sort is required for correctness,
                                           so the public one-shot must keep
                                           paying it and only the comparison
                                           boundary can move
  input_row_sorted all True             -> the argsort was never the cause; the
                                           prepare/run split says what is

What it cannot tell: whether dropping the sort is safe beyond these inputs. The
ten delivery matrices are deduplicated and get one random permutation each, and
the GE operator is separately known to leave an nnz=0 output uninitialised. It
also cannot say whether CANN does preprocessing of its own internally. Finally,
the delivery entrypoint times FlagSparse, PyTorch and CANN sequentially rather
than alternating, and on ASIC_680ks that read 0.726x/0.710x against the ~1.0 here
for reasons nobody has explained -- do not quote these means as a delivery
aggregate.

Usage (Ascend box, repo root; the output directory must not already exist):
    ASCEND_RT_VISIBLE_DEVICES=1 python3 tools/ascend_fp64_probe.py --output run1
    ... --formats coo --matrices ASIC_680ks,wave --width 32

Writes command.json, comparison.csv, variant_summary.csv, raw/*.json and
complete.json. Exit status 0 when every requested case was measured.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
import torch  # noqa: E402

MATRICES = (
    "ASIC_680ks,GL7d14,NACA0015,amazon0601,auto,cage12,cfd2,filter3D,roadNet-TX,wave"
)


def _require_npu():
    """Import torch_npu here, not at module scope, so --help works anywhere."""
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit(f"this probe needs torch_npu on an Ascend box: {exc}")
    if not getattr(torch, "npu", None) or not torch.npu.is_available():
        raise SystemExit("no Ascend NPU is available to this process")
    return torch_npu


def measured(call):
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    result = call()
    torch.npu.synchronize()
    return result, (time.perf_counter_ns() - started) / 1e6


def validate(result, golden):
    actual = result.detach().cpu()
    torch.testing.assert_close(actual, golden, rtol=1e-12, atol=1e-12)
    difference = (actual - golden).abs()
    return float(difference.max()) if difference.numel() else 0.0


def shuffled_unsorted_error(native, arrays, dense, shape, golden):
    """Run the GE operator on permuted coordinates with sorting switched off.

    Returns (status, max_error). A clean result here is what would justify
    flipping the `sort_rows` default; anything else means the sort is load
    bearing and the 840 ms it costs on a COO prepare cannot simply be dropped.
    """
    values, rows, columns = arrays
    order = torch.randperm(values.numel(), device=values.device)
    try:
        result = native.run_fp64_spmm_once(
            values.index_select(0, order),
            rows.index_select(0, order),
            columns.index_select(0, order),
            dense,
            shape,
            sort_rows=False,
        )
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}", None
    if result is None:
        return "SKIPPED: native path declined", None
    error = (result.detach().cpu() - golden).abs()
    error = float(error.max()) if error.numel() else 0.0
    return ("OK" if error < 1e-9 else "FAIL"), error


def fixed_output_checks(run, prepared, dense, output, golden, matrix):
    """Three invariants a reused output buffer depends on.

    residue   writing garbage into C first must not survive the call. The
              operator is assumed to overwrite rather than accumulate -- the
              internal path allocates with torch.empty and still matched the
              golden exactly, which only holds if it overwrites -- and sharing
              one C across calls turns that assumption load bearing.
    empty_row a row with no stored entries must come back zero, not whatever the
              previous call left there. This operator is separately known to
              leave an all-nnz=0 output untouched, so the hole is real.
    b_update  changing B between calls must change the result, i.e. the plan
              bound a pointer and not a snapshot of the values.
    """
    results = {}

    output.fill_(float("nan"))
    got = run(prepared, dense, out=output)
    error = (got.detach().cpu() - golden).abs()
    results["residue"] = float(error.max()) if error.numel() else 0.0

    empty = torch.from_numpy(np.asarray((matrix.getnnz(axis=1) == 0).nonzero()[0]))
    if empty.numel():
        output.fill_(float("nan"))
        got = run(prepared, dense, out=output)
        rows = got.detach().cpu().index_select(0, empty)
        results["empty_row_max_abs"] = float(rows.abs().max())
    else:
        results["empty_row_max_abs"] = None

    scaled = dense * 2.0
    got = run(prepared, scaled, out=output)
    error = (got.detach().cpu() - golden * 2.0).abs()
    results["b_update"] = float(error.max()) if error.numel() else 0.0
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "tests/data")
    parser.add_argument("--matrices", default=MATRICES)
    parser.add_argument("--formats", default="coo,csr")
    parser.add_argument("--width", type=int, default=32, help="dense columns (N)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args(argv)
    if args.width < 1 or args.iters < 1 or args.warmup < 0:
        parser.error("width/iters must be positive and warmup nonnegative")
    formats = args.formats.split(",")
    if not set(formats) <= {"coo", "csr"}:
        parser.error("formats must contain only coo and csr")

    torch_npu = _require_npu()
    import flagsparse as fs
    from flagsparse.sparse_operations import _ascend_fp64_spmm as native

    from tests import ascend_sparse_baseline as cann

    coo_module = importlib.import_module("flagsparse.sparse_operations.spmm_coo")

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "raw").mkdir()
    torch.npu.set_device(0)
    os.environ["FLAGSPARSE_ASCEND_FP64_SPMM"] = "native"
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/flagsparse/sparse_operations/spmm_coo.py",
        ROOT / "src/flagsparse/sparse_operations/spmm_csr.py",
        ROOT / "src/flagsparse/sparse_operations/_ascend_fp64_spmm.py",
        ROOT / "tests/ascend_sparse_baseline.py",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    (args.output / "command.json").write_text(
        json.dumps(
            dict(
                argv=sys.argv,
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                source_sha256=hashes,
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                timing="same-card alternating prepared/CANN; synchronized wall arithmetic mean",
                coo_order="raw Matrix Market; sorting inside public prepare, not before timing",
                csr_order="CSR row storage; conversion/upload excluded from prepare and run",
                shuffled="run_fp64_spmm_once(sort_rows=False) on permuted COO coordinates",
            ),
            indent=2,
        )
        + "\n"
    )
    records = []
    for name in args.matrices.split(","):
        raw = scipy.io.mmread(args.input / f"{name}.mtx").tocoo()
        matrix = raw.tocsr().astype(np.float64)
        matrix.sum_duplicates()
        rhs = torch.randn(
            (matrix.shape[1], args.width),
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(20261028),
        )
        golden = torch.from_numpy(np.asarray(matrix @ rhs.numpy()))
        for fmt in formats:
            print(f"START {name} {fmt}", flush=True)
            host_arrays = (
                (
                    raw.data.astype(np.float64),
                    raw.row.astype(np.int32),
                    raw.col.astype(np.int32),
                )
                if fmt == "coo"
                else (
                    matrix.data,
                    matrix.indices.astype(np.int32),
                    matrix.indptr.astype(np.int32),
                )
            )
            uploaded, upload_ms = measured(
                lambda: (
                    tuple(
                        torch.from_numpy(np.ascontiguousarray(value)).npu()
                        for value in host_arrays
                    ),
                    rhs.npu(),
                )
            )
            arrays, dense = uploaded
            sort_times = []
            original_sort = coo_module._sort_coo_lex_inplace

            def timed_sort(*operands, **kwargs):
                result, elapsed = measured(lambda: original_sort(*operands, **kwargs))
                sort_times.append(elapsed)
                return result

            prepared = baseline = None
            try:
                with patch.object(coo_module, "_sort_coo_lex_inplace", timed_sort):
                    prepared, prepare_ms = measured(
                        lambda: getattr(fs, f"prepare_spmm_{fmt}_route")(
                            *arrays, matrix.shape
                        )
                    )
                if fmt == "coo":
                    assert len(sort_times) == 1
                    assert bool((prepared.row[1:] >= prepared.row[:-1]).all().item())
                    rows_cpu = prepared.row.cpu().numpy()
                    cols_cpu = prepared.col.cpu().numpy()
                    same_row = rows_cpu[1:] == rows_cpu[:-1]
                    assert np.all(cols_cpu[1:][same_row] >= cols_cpu[:-1][same_row])
                assert prepared.ascend_fp64_plan is not None
                run = getattr(fs, f"flagsparse_spmm_{fmt}_run")
                first, first_run_ms = measured(
                    lambda: run(prepared, dense, return_meta=True)
                )
                result, meta = first
                assert meta["implementation"] == "cann_sparse_tensor_dense_matmul_fp64"
                max_error = validate(result, golden)
                output = torch.empty_like(result)
                baseline, cann_prepare_ms = measured(
                    lambda: cann.prepare_spmm(fmt, arrays, matrix.shape, dense, output)
                )
                _, cann_first_run_ms = measured(baseline.run)
                cann_error = validate(output, golden)
                # Both sides write into the SAME fixed C, so neither draws a
                # fresh address between calls: this operator runs 222 ms or 155
                # ms on one matrix depending only on where B and C sit, and the
                # CANN plan binds its output once, so a per-call allocation on
                # our side alone was half of the old 0.72x.
                operations = dict(
                    prepared=lambda: run(prepared, dense, out=output),
                    cann=baseline.run,
                )
                samples = {key: [] for key in operations}
                order = []
                with patch.object(
                    torch,
                    "argsort",
                    side_effect=AssertionError(
                        "torch.argsort must not run in the timed prepared path"
                    ),
                ):
                    for iteration in range(args.warmup + args.iters):
                        keys = (
                            ("prepared", "cann")
                            if iteration % 2 == 0
                            else ("cann", "prepared")
                        )
                        for key in keys:
                            result, duration = measured(operations[key])
                            if iteration >= args.warmup:
                                samples[key].append(duration)
                                order.append(key)
                    max_error = max(
                        max_error, validate(operations["prepared"](), golden)
                    )
                    cann_error = max(cann_error, validate(operations["cann"](), golden))
                checks = fixed_output_checks(
                    run, prepared, dense, output, golden, matrix
                )
                # Outside the argsort guard: this call deliberately hands the
                # operator unsorted coordinates.
                if fmt == "coo":
                    shuffled_status, shuffled_error = shuffled_unsorted_error(
                        native, arrays, dense, matrix.shape, golden
                    )
                else:
                    shuffled_status, shuffled_error = (
                        "N/A: CSR has no coordinate order",
                        None,
                    )
                run_ms = statistics.mean(samples["prepared"])
                cann_ms = statistics.mean(samples["cann"])
                row_sorted = True
                if fmt == "coo":
                    row_sorted = bool(np.all(np.diff(raw.row) >= 0))
                record = dict(
                    matrix=name,
                    variant=f"spmm_{fmt}_f64_int_non_non_row",
                    status="PASS",
                    input_row_sorted=row_sorted,
                    upload_ms=upload_ms,
                    prepare_ms=prepare_ms,
                    sort_ms=sum(sort_times),
                    sort_calls_prepare=len(sort_times),
                    torch_argsort_calls_run=0,
                    first_run_ms=first_run_ms,
                    cann_prepare_ms=cann_prepare_ms,
                    cann_adapter_prepare_ms=baseline.prepare_ms,
                    cann_first_run_ms=cann_first_run_ms,
                    run_ms=run_ms,
                    cann_ms=cann_ms,
                    speedup=cann_ms / run_ms,
                    max_error=max_error,
                    cann_max_error=cann_error,
                    shuffled_status=shuffled_status,
                    shuffled_max_error=shuffled_error,
                    fixed_out_residue=checks["residue"],
                    fixed_out_empty_row=checks["empty_row_max_abs"],
                    fixed_out_b_update=checks["b_update"],
                    implementation=meta["implementation"],
                )
                records.append(record)
                (args.output / "raw" / f"{name}__{fmt}.json").write_text(
                    json.dumps(dict(record, samples=samples, order=order), indent=2)
                    + "\n"
                )
                with (args.output / "comparison.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                    writer.writerows(records)
                print(json.dumps(record), flush=True)
            finally:
                if baseline is not None:
                    baseline.close()
                if prepared is not None and prepared.ascend_fp64_plan is not None:
                    prepared.ascend_fp64_plan.close()
                operations = {}
                prepared = baseline = arrays = dense = uploaded = None
                output = result = first = None
                torch.npu.empty_cache()
    summary = []
    for fmt in formats:
        selected = [
            row
            for row in records
            if row["variant"] == f"spmm_{fmt}_f64_int_non_non_row"
        ]
        summary.append(
            dict(
                variant=selected[0]["variant"],
                matrices=len(selected),
                speedup_mean=statistics.mean(row["speedup"] for row in selected),
                prepare_ms_mean=statistics.mean(row["prepare_ms"] for row in selected),
                sort_ms_mean=statistics.mean(row["sort_ms"] for row in selected),
                run_ms_mean=statistics.mean(row["run_ms"] for row in selected),
                cann_ms_mean=statistics.mean(row["cann_ms"] for row in selected),
                max_error=max(row["max_error"] for row in selected),
                shuffled=sorted({row["shuffled_status"] for row in selected}),
            )
        )
    with (args.output / "variant_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    assert hashes == {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    (args.output / "complete.json").write_text(
        json.dumps(
            dict(valid_cases=len(records), source_unchanged=True, summary=summary),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)
    print(
        "\nspeedup_mean is CANN run / prepared run, both excluding preparation."
        "\nshuffled says whether the GE operator was correct on unsorted"
        " coordinates, i.e. whether the sort in prepare could be dropped.",
        flush=True,
    )
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
