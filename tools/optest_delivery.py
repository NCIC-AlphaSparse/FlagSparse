# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Schedule delivery parents with optest and report their individual variants."""

import json
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from run_flagsparse_pytest import (
    ASCEND_ACCURACY_OPS,
    ASCEND_PER_MATRIX_PERFORMANCE_OPS,
    TIMEOUT_RETURN_CODE,
    _delivery_results,
    _flag_gems_dtype,
    _flaggems_perf_data,
    _operator_summary,
    run_accuracy,
)
from tools.delivery_variants import (
    load_delivery_matrices,
    load_delivery_variants,
    select_delivery_matrices,
)


def _read_parent_results(output_dir, parents, delivery_dir):
    for op, parent in parents.items():
        accuracy = dict(parent["accuracy"])
        if accuracy_file := accuracy.get("data_file"):
            accuracy["result_path"] = str(output_dir / accuracy_file)
        performance = parent["performance"]
        if (phase_file := delivery_dir / f"{op}.json").is_file():
            performance = json.loads(phase_file.read_text(encoding="utf-8"))
            performance["data_file"] = parent["performance"].get("data_file")
        yield {"operator": op, "accuracy": accuracy, "performance": performance}


def _write_delivery_summary(optest, variants, delivery_dir):
    output_dir = optest.CFG.output_dir
    summary_path = output_dir / "summary.json"
    parent_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parents = parent_summary["result"]
    (output_dir / "parent_summary.json").write_text(
        json.dumps(parent_summary, indent=2), encoding="utf-8"
    )
    results = {}
    projected = _delivery_results(
        list(_read_parent_results(output_dir, parents, delivery_dir)), variants=variants
    )
    for variant in projected:
        entry = _operator_summary(variant)
        for phase in ("accuracy", "performance"):
            parent_phase = parents.get(variant["variant"]["operator"], {}).get(
                phase, {}
            )
            if entry[phase]["status"] == "NotFound" and parent_phase.get("status") in {
                "Timeout",
                "Error",
            }:
                entry[phase]["status"] = parent_phase["status"]
                entry[phase]["exit_code"] = parent_phase["exit_code"]
            if reason := variant[phase].get("reason"):
                entry[phase]["reason"] = reason
        performance = variant["performance"]
        entry["performance"].setdefault("data", {})
        if records := performance.get("records"):
            # Keep missing baselines null and give each delivered case its shape.
            selected_dtypes = performance.get("data", {})
            entry["performance"]["data"] = {
                _flag_gems_dtype(dtype): data
                for dtype, data in _flaggems_perf_data(records).items()
                if _flag_gems_dtype(dtype) in selected_dtypes
            }
        results[variant["operator"]] = entry

    summary = dict(parent_summary, result=results, total_ops=len(results))
    if scaled_baseline := json.loads(
        os.environ.get("FLAGSPARSE_OPTEST_SCALED_BASELINE", "null")
    ):
        summary["scaled_baseline"] = scaled_baseline
    for phase in ("accuracy", "performance"):
        summary[f"{phase}_counts"] = optest.count_op_statuses(results, phase)
    pending = summary_path.with_suffix(".tmp")
    pending.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pending.replace(summary_path)
    print(
        f"Delivery report: {len(results)} variants from {len(parents)} parent operators",
        flush=True,
    )

    failed = []
    for op, entry in results.items():
        accuracy, performance = entry["accuracy"], entry["performance"]
        measured = any(
            case.get("gems") is not None and case["gems"] > 0
            for dtype in performance["data"].values()
            for case in dtype["details"].values()
        )
        if (
            accuracy["status"] != "Passed"
            or not accuracy["passed"]
            or accuracy["exit_code"]
            or performance["status"] != "Passed"
            or performance["exit_code"]
            or not measured
        ):
            failed.append(op)
    if failed:
        print(
            f"Incomplete delivery results ({len(failed)}): {', '.join(failed)}",
            flush=True,
        )
    return int(bool(failed))


def _ascend_accuracy_adapter(optest):
    """Run native Ascend probes with Optest's scheduling and reporting."""
    default_accuracy = optest.run_accuracy_q

    def accuracy(gpu_id, op):
        if op not in ASCEND_ACCURACY_OPS:
            return default_accuracy(gpu_id, op)

        op_dir = optest.CFG.output_dir / op
        op_dir.mkdir(parents=True, exist_ok=True)
        phase = run_accuracy(
            project_root=optest.ROOT,
            op=op,
            gpu_id=gpu_id,
            marker=optest.op_marker(op),
            mode="quick" if optest.CFG.quick else "normal",
            op_dir=op_dir,
            extra_pytest_args=[],
            timeout=optest.CFG.timeout,
            delivery_only=True,
        )
        result_file = op_dir / "accuracy_result.json"
        timed_out = (
            phase["returncode"] == TIMEOUT_RETURN_CODE or phase["status"] == "TIMEOUT"
        )
        if timed_out or not result_file.is_file():
            result = {
                "status": "Timeout" if timed_out else "Error",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": int(not timed_out),
            }
            data_file = None
        else:
            result = optest.parse_accuracy_data(result_file)
            data_file = str(result_file.relative_to(optest.CFG.output_dir))
        return dict(
            result,
            exit_code=optest.TIMEOUT if timed_out else phase["returncode"],
            duration=phase["duration"],
            data_file=data_file,
        )

    return patch.object(optest, "run_accuracy_q", accuracy)


def _ascend_benchmark_timeout_adapter(optest):
    """Apply Optest's timeout per matrix for Ascend's matrix-isolated benchmarks."""
    run_cmd = optest.run_cmd

    def run_with_matrix_timeout(op, cmd, **kwargs):
        if (
            kwargs.get("flavor") == "performance"
            and op in ASCEND_PER_MATRIX_PERFORMANCE_OPS
        ):
            matrix_dir = Path(os.environ["FLAGSPARSE_OPTEST_BENCHMARK_INPUT"])
            matrix_count = sum(1 for _ in matrix_dir.glob("*.mtx"))
            timeout = kwargs["timeout"]
            if matrix_count and timeout > 0:
                kwargs["env"] = dict(
                    kwargs["env"], FLAGSPARSE_OPTEST_MATRIX_TIMEOUT=str(timeout)
                )
                kwargs["timeout"] = matrix_count * (timeout + 30) + 120
        return run_cmd(op, cmd, **kwargs)

    return patch.object(optest, "run_cmd", run_with_matrix_timeout)


def run_delivery(optest, benchmark_input):
    """Use optest's selectors for variants and its subprocesses for parents."""
    selected = []
    select_ops = optest.get_ops_to_test

    with TemporaryDirectory(prefix="flagsparse-optest-delivery-") as temp:
        delivery_dir = Path(temp)

        def select_parent_ops():
            root = optest.ROOT
            manifest = root / "conf" / "operators.yaml"
            variants = {
                variant["id"]: variant for variant in load_delivery_variants(manifest)
            }
            catalog = {op["id"]: op for op in optest.get_ops_from_inventory()}
            delivery_catalog = [
                dict(catalog[variant["operator"]], id=variant["id"])
                for variant in variants.values()
            ]
            with patch.object(
                optest, "get_ops_from_inventory", return_value=delivery_catalog
            ):
                ids = select_ops()
            unknown = set(ids) - variants.keys()
            if unknown:
                raise optest.OperatorSelectionError(
                    f"Unknown delivery variant IDs: {', '.join(sorted(unknown))}"
                )
            selected.extend(variants[op] for op in ids)
            parents = list(dict.fromkeys(variant["operator"] for variant in selected))
            optest.CFG.all_op_ids = set(catalog)
            optest.CFG.skip_cpu_tests = [
                op for op in parents if "NoCPU" in catalog[op]["labels"]
            ]

            source = root / (benchmark_input or "tests/data")
            if not source.exists():
                raise optest.OperatorSelectionError(
                    f"Benchmark input does not exist: {source}"
                )
            input_path, note = select_delivery_matrices(
                source, delivery_dir / "matrices", load_delivery_matrices(manifest)
            )
            print(
                f"Scheduling {len(selected)} delivery variants through {len(parents)} parents",
                flush=True,
            )
            print(
                f"Delivery matrix input: {input_path}" + (f" ({note})" if note else ""),
                flush=True,
            )
            os.environ.update(
                FLAGSPARSE_OPTEST_DELIVERY_DIR=str(delivery_dir),
                FLAGSPARSE_OPTEST_BENCHMARK_INPUT=str(input_path),
            )
            if (
                ascend
                and input_path.is_dir()
                and set(parents) & ASCEND_PER_MATRIX_PERFORMANCE_OPS
            ):
                print(
                    f"Ascend matrix-isolated benchmarks: {optest.CFG.timeout}s per matrix",
                    flush=True,
                )
            return parents

        ascend = os.environ.get("FLAGSPARSE_BACKEND") == "ascend"
        with (
            patch.dict(os.environ),
            patch.object(optest, "get_ops_to_test", select_parent_ops),
            _ascend_accuracy_adapter(optest) if ascend else nullcontext(),
            _ascend_benchmark_timeout_adapter(optest) if ascend else nullcontext(),
        ):
            status = optest.main()
        if status:
            return status
        return _write_delivery_summary(optest, selected, delivery_dir)
