#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Emit summary.json in the SAME schema FlagSparse's Python runner emits.

WHY IDENTICAL AND NOT MERELY SIMILAR. The two repos test the same kernels through
two front ends, and the whole point of running both is to compare them. A schema
that differs by a key name or a status spelling forces whoever reads them to
write a converter, and converters silently drop what they do not recognise. So
this matches run_flagsparse_pytest.py's `_flaggems_summary` exactly:

    {"timestamp", "env", "result"}          -- three top-level keys, in this order
    env:      architecture, os_name, os_release, python, torch, flagtree,
              triton, flag_gems
    result:   {operator: {customized, accuracy, performance, labels}}
    accuracy: total, skipped, failed, passed, details, status, duration,
              exit_code, data_file
    perf:     duration, exit_code, data, status, data_file, test_case

ONE DIFFERENCE FROM THE PYTHON SIDE, deliberate: there, accuracy and performance
are two independent pytest runs and each names its own data_file. Here both come
from one benchmark run -- a row is timed only after its result has been checked
against the host fp64 oracle -- so both phases point at the same
<op>_benchmark.json. The numbers are real; what does not exist is a separate
accuracy artifact, and this says so rather than naming one.
    data:     {dtype: {result, details: {shape: {base, gems, speedup}}, speedup}}

Status strings come from that runner's STATUS_TO_FLAGGEMS ("Passed"/"Failed"/
"Skipped"/...), NOT from the C tests' own lowercase vocabulary.

THE MAPPING THAT MATTERS. FlagGems' perf schema is keyed dtype -> shape. Here a
"shape" is a corpus matrix (plus the dense width where the operator sweeps one),
and `base`/`gems` are the vendor and our median in ms -- which is what the Python
side puts there too, so the two files' numbers are comparable cell by cell.

Usage:
    python3 tools/write_summary.py --bench-dir ./bench --out ./bench
"""

import argparse
import datetime as _dt
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.delivery_variants import load_delivery_variants  # noqa: E402

# Verbatim from run_flagsparse_pytest.py -- if that table changes this must too.
STATUS_TO_FLAGGEMS = {
    "PASS": "Passed",
    "FAIL": "Failed",
    "SKIP": "Skipped",
    "TIMEOUT": "Timeout",
    "NO_TESTS": "NotFound",
    "CRASH": "Error",
    "NOT_CONFIGURED": "NotFound",
    "Passed": "Passed",
    "Failed": "Failed",
    "Skipped": "Skipped",
    "Timeout": "Timeout",
    "NotFound": "NotFound",
    "Error": "Error",
    "NoBaseline": "NoBaseline",
}

# The C tests tag dtypes by component width (c32 = complex<float>); FlagGems
# spells them fp32/fp64 and has no complex alias, so complex keeps the C spelling
# rather than being mangled into a real dtype it is not.
DTYPE_ALIASES = {"f32": "fp32", "f64": "fp64", "f16": "fp16", "bf16": "bf16"}

# The delivery list (算子列表注册修改.xlsx, "新算子列表") names each VARIANT as an
# operator: `spmv_csr_f32_int_non`, not `spmv`. Keying `result` that way is what
# makes the 40 rows appear in a FlagGems-shaped summary and HTML without changing
# either format -- one entry per variant, exactly as the list is written.
#
# The trailing parts encode what the delivery list fixes: `int` (index type,
# covering i32/i64 through the descriptor), then the operation directions. Every
# delivery variant is the non-transposed, row-major case, so those are constants
# here -- when trans/conj/col are implemented they become real dimensions and
# this mapping grows.
DELIVERY_BY_OPERATOR_DTYPE = {}


def variant_name(row):
    """The delivery-list name for a measured row, e.g. spmv_csr_f32_int_non."""
    op = str(row.get("operator") or row.get("_op", ""))
    dtype = str(row.get("dtype", ""))
    matches = DELIVERY_BY_OPERATOR_DTYPE.get((op, dtype), [])
    return matches[0] if len(matches) == 1 else f"__undeclared__{op}_{dtype}"


def flaggems_status(s):
    return STATUS_TO_FLAGGEMS.get(str(s or ""), str(s or "Unknown"))


def flag_gems_dtype(d):
    return DTYPE_ALIASES.get(str(d), str(d))


def env_info(rows):
    """The strict FlagGems env block, filled from this machine and the JSONs."""
    backend = rows[0].get("_backend", "") if rows else ""
    # The adaptor's arch string (e.g. "120"). Used as the device-name fallback
    # below: on a backend with no nvidia-smi it is the only device fact the JSONs
    # carry, and an empty device_name reads as "no device" rather than "unnamed".
    arch = rows[0].get("_arch", "") if rows else ""
    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}

    # The C API has no torch; device facts come from the adaptor, which is what
    # the `env` block of every benchmark JSON already carries.
    device_name = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            device_name = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

    if not device_name and arch:
        device_name = f"{backend} arch {arch}"

    return {
        "architecture": platform.machine(),
        "os_name": str(os_release.get("ID") or platform.system()).lower(),
        "os_release": str(os_release.get("VERSION_ID") or platform.release()),
        "python": platform.python_version(),
        "torch": {
            # Deliberately blank: this front end does not go through torch. A
            # fabricated version here would make the two summaries look like they
            # ran the same stack when they did not.
            "version": "",
            "cuda_available": backend == "cuda",
            "device_name": device_name,
            "device_count": 1 if device_name else 0,
        },
        "flagtree": None,
        "triton": {"version": "", "has_config": False},
        "flag_gems": {
            "version": "",
            "vendor": backend or "cpu",
            "device": backend or "cpu",
        },
    }


def shape_key(row):
    """The `shape` a metrics cell is filed under: the matrix, plus the dense
    width when the operator sweeps one, so n=8 and n=128 do not collide."""
    parts = [str(row.get("matrix", "?"))]
    for k in ("n", "k"):
        if row.get(k):
            parts.append(f"{k}={int(row[k])}")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-dir",
        type=pathlib.Path,
        default=pathlib.Path("capi_results"),
        help="directory of *_benchmark.json (default: capi_results/)",
    )
    # Defaults to capi_results/ beside pytest_results/, which is where the
    # Python runner writes. Two reasons it is a separate directory rather than a
    # shared one: the two summaries are the same FILENAME with different
    # granularity (25 operators there, the delivery variants here), so sharing a
    # directory means one silently overwrites the other; and CI uploads whole
    # directories, so separate names keep both artifacts.
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("capi_results"),
        help="output directory (default: capi_results/)",
    )
    ap.add_argument(
        "--no-html",
        action="store_true",
        help="skip result.html (the JSON is still written)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="include retained variants; the default is the "
        "registered delivery variants only (conf/operators.yaml)",
    )
    args = ap.parse_args()

    files = sorted(args.bench_dir.glob("*_benchmark.json"))
    if not files:
        sys.exit(f"no *_benchmark.json under {args.bench_dir}")

    delivery_variants = load_delivery_variants()
    expected_variants = {variant["id"]: variant for variant in delivery_variants}
    DELIVERY_BY_OPERATOR_DTYPE.clear()
    for variant in delivery_variants:
        DELIVERY_BY_OPERATOR_DTYPE.setdefault(
            (variant["operator"], variant["dtype"]), []
        ).append(variant["id"])

    all_rows, result = [], {}
    for path in files:
        doc = json.loads(path.read_text())
        op = doc.get("operator", path.stem.replace("_benchmark", ""))
        env = doc.get("env", {})
        rows = doc.get("result", [])
        for r in rows:
            r["_backend"] = env.get("backend", "")
            r["_arch"] = str(env.get("arch", ""))
            # The benchmark family, used to pick the delivery-list suffix.
            r["_op"] = op
        # Delivery scope. Rows from a build before the `reporting` tag existed
        # have no such key; treating those as delivery would silently fold the
        # retained variants into the headline numbers, so they are EXCLUDED and
        # counted, and the run says so rather than reporting a wrong total.
        if not args.all:
            untagged = [r for r in rows if "reporting" not in r]
            if untagged:
                print(
                    f"  {path.name}: {len(untagged)} rows carry no `reporting` "
                    f"tag (built before it existed) -- excluded; rerun that "
                    f"operator, or pass --all"
                )
            rows = [r for r in rows if r.get("reporting") == "delivery"]
        all_rows.extend(rows)
        if not rows:
            continue

        # Group by variant: one summary entry per delivery-list name.
        by_variant = {}
        for r in rows:
            by_variant.setdefault(variant_name(r), []).append(r)

        # ---- accuracy artifact, indexed by variant so each entry cites the
        # rows that decided it. From the operator's own file when the
        # sweep wrote one, so the phase names a file that really holds the
        # ratios. Falls back to the benchmark rows (same numbers, same run) for
        # a JSON produced before that artifact existed.
        acc_path = path.with_name(path.name.replace("_benchmark", "_accuracy"))
        acc_rows, acc_file = rows, path.name
        if acc_path.exists():
            acc_rows = json.loads(acc_path.read_text()).get("result", [])
            acc_file = acc_path.name
        # One entry per variant. The accuracy rows are grouped the same way, so
        # each entry's counts come from the rows that actually decided it rather
        # than from an operator-wide total that would hide which variant failed.
        acc_by_variant = {}
        for r in acc_rows:
            acc_by_variant.setdefault(variant_name(r), []).append(r)

        for vname, vrows in by_variant.items():
            if vname not in expected_variants:
                continue
            va = acc_by_variant.get(vname, vrows)
            # pass_relaxed counts as passed: spec 6.3.1 calls it a PASS, and it
            # is only awarded when the vendor missed the strict tolerance too.
            v_passed = [r for r in va if r.get("accuracy") in ("pass", "pass_relaxed")]
            v_failed = [r for r in va if r.get("accuracy") == "fail"]
            v_skipped = [
                r
                for r in va
                if str(r.get("status", "")).startswith("skipped")
                or r.get("status") == "not_supported"
            ]

            dt = flag_gems_dtype(vrows[0].get("dtype", "?"))
            det, sp, no_base = {}, [], []
            for r in vrows:
                if r.get("status") != "ok":
                    continue
                det[shape_key(r)] = {
                    "base": float(r.get("baseline_ms") or 0.0),
                    "gems": float(r.get("median_ms") or 0.0),
                    "speedup": float(r.get("speedup") or 0.0),
                }
                if r.get("speedup"):
                    sp.append(float(r["speedup"]))
                elif r.get("baseline_status") != "ok" and r.get("accuracy") in (
                    "pass",
                    "pass_relaxed",
                ):
                    no_base.append(r)

            # NoBaseline: the kernel ran and passed on EVERY measured row, only the
            # vendor library had nothing to compare against (e.g. muSPARSE has no
            # fp16 gather/scatter). This used to fall through to "Skipped", which
            # reads as "not run" -- the MUSA f16 rows had 30 real timings each.
            ok_rows = [r for r in vrows if r.get("status") == "ok"]
            no_baseline = not sp and bool(ok_rows) and len(no_base) == len(ok_rows)
            perf_status = (
                "Passed" if sp else ("NoBaseline" if no_baseline else "Skipped")
            )
            dt_data = {
                "result": "Passed"
                if sp
                else ("NoBaseline" if no_baseline else "Unknown"),
                "details": det,
                "speedup": sum(sp) / len(sp) if sp else 0.0,
            }
            if no_baseline:
                dt_data["reason"] = sorted(
                    {
                        str(r.get("baseline_detail") or r.get("baseline_status"))
                        for r in no_base
                    }
                )

            v_details = {}
            if v_failed:
                v_details["failed"] = [
                    {
                        "name": r.get("name"),
                        "error_ratio": r.get("error_ratio"),
                        "detail": r.get("detail"),
                    }
                    for r in v_failed
                ]

            acc_status = "Failed" if v_failed else ("Passed" if v_passed else "Skipped")
            result[vname] = {
                "customized": True,
                "accuracy": {
                    "total": len(va),
                    "skipped": len(v_skipped),
                    "failed": len(v_failed),
                    "passed": len(v_passed),
                    "details": v_details,
                    "status": flaggems_status(acc_status),
                    "exit_code": 1 if v_failed else 0,
                    "duration": 0.0,
                    "data_file": acc_file,
                },
                "performance": {
                    "duration": 0.0,
                    "exit_code": 0,
                    "data_file": path.name,
                    "data": {dt: dt_data},
                    "status": flaggems_status(perf_status),
                    "test_case": "matrix",
                },
                "labels": ["flagsparse", "c_api"],
            }

    for vname, variant in expected_variants.items():
        if vname in result:
            continue
        result[vname] = {
            "customized": True,
            "accuracy": {
                "total": 0,
                "skipped": 0,
                "failed": 0,
                "passed": 0,
                "details": {"reason": "delivery variant was not measured"},
                "status": "NotFound",
                "exit_code": 0,
                "duration": 0.0,
                "data_file": "",
            },
            "performance": {
                "duration": 0.0,
                "exit_code": 0,
                "data_file": "",
                "data": {},
                "status": "NotFound",
                "test_case": "Unknown",
            },
            "labels": [
                "flagsparse",
                "c_api",
                "delivery",
                variant["format"],
                variant["dtype"],
            ],
        }

    summary = {
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": env_info(all_rows),
        "result": dict(sorted(result.items())),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  ({len(result)} variants, {len(all_rows)} rows)")

    # The HTML describes the same run, so it is rendered here rather than by a
    # second process that re-reads the JSON -- one command, no way for the two
    # to drift apart. A rendering failure must not cost the JSON, which is the
    # machine-readable artifact and the one CI collects.
    if not args.no_html:
        try:
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            import write_html

            html_path, n = write_html.render(summary, args.out / "result.html")
            print(f"wrote {html_path}  ({n} variants)")
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            print(f"  result.html not written: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
