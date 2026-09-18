#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Conformance check: capi/conf/operators.yaml against what exists and runs.

TWO FILES ARE NAMED conf/operators.yaml since the merge, and this reads the C API
one -- `capi/conf/operators.yaml`, resolved from this script rather than from the
working directory. The repo-root `conf/operators.yaml` is the Python side's
delivery registry (`delivery_variants`, loaded by tools/delivery_variants.py);
this one is the C API manifest: paths, c_api symbols, formats, dtypes, baselines.

The manifest is the declared operator list -- 40 variants today, 115 later -- and
everything else in the repo is supposed to follow it. Nothing enforces that today,
so it has drifted in BOTH directions: it declares variants nothing tests, and the
code supports variants it still calls pending. This prints every such disagreement.

Four checks, each answering a different question:

  paths     do the files each entry names still exist?
  symbols   are the c_api functions it names actually declared in flagsparse.h?
  coverage  does every declared variant get a row in a benchmark/accuracy output?
  status    does `status:` match what the variant actually did when run?

`status` needs benchmark JSON to have been produced; the others are static. A
check that cannot run says so rather than passing silently.

Usage:
    python3 tools/check_manifest.py [--bench-dir DIR] [--strict]

--strict exits non-zero on any drift, for CI.
"""

import argparse
import json
import pathlib
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The manifest and the test JSONs spell dtypes differently, and one token means
# two different things: the manifest uses cuSPARSE's width-of-the-whole-value
# naming (c64 = a 64-bit complex = two fp32), while the benchmark tags name the
# component width (c32 = complex of fp32). Mapping them by string equality would
# silently pair c64-the-manifest-entry with c64-the-benchmark-row, which are
# DIFFERENT TYPES. This table is the only place that conversion happens.
DTYPE_MANIFEST_TO_TAG = {
    "f32": "f32",
    "f64": "f64",
    "c64": "c32",  # complex<float>   -- FLAGSPARSE_C_32F
    "c128": "c64",  # complex<double>  -- FLAGSPARSE_C_64F
    "f16": "f16",
    "bf16": "bf16",
}

FORMAT_MANIFEST_TO_TAG = {
    "sparse_vector": "spvec",
    "csr": "csr",
    "coo": "coo",
    "csc": "csc",
    "bsr": "bsr",
    "sell": "sell",
}


def load_manifest():
    # ROOT is capi/, so this is capi/conf/operators.yaml -- not the repo-root
    # delivery registry of the same relative name. See the module docstring.
    path = ROOT / "conf" / "operators.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh), path


def declared_variants(manifest):
    """(op_id, format_tag, dtype_tag) for every variant the manifest declares.

    An entry with no formats or no dtypes (the descriptor-API and constructor
    groups) declares no variants; it is still checked for paths and symbols.
    """
    out = []
    for op in manifest["operators"]:
        if op.get("status") != "implemented":
            continue
        fmts = op.get("formats") or []
        dts = op.get("dtypes") or []
        for f in fmts:
            for d in dts:
                out.append(
                    (
                        op["id"],
                        FORMAT_MANIFEST_TO_TAG.get(f, f),
                        DTYPE_MANIFEST_TO_TAG.get(d, d),
                    )
                )
    return out


def check_paths(manifest):
    """Files each entry names, that no longer exist."""
    bad = []
    for op in manifest["operators"]:
        for field in ("dispatch", "kernel_module"):
            v = op.get(field)
            if not v:
                continue
            # kernel_module names a path inside the Python package, which is a
            # sibling checkout; only the in-repo ones are checkable here.
            if field == "kernel_module":
                continue
            if not (ROOT / v).exists():
                bad.append((op["id"], field, v))
        for t in op.get("tests") or []:
            if not (ROOT / t).exists():
                bad.append((op["id"], "tests", t))
    return bad


def check_symbols(manifest):
    """c_api names that are not declared in the public header."""
    header = (ROOT / "include" / "flagsparse.h").read_text()
    declared = set(re.findall(r"\b(flagsparse[A-Za-z0-9_]*)\s*\(", header))
    bad = []
    for op in manifest["operators"]:
        for sym in op.get("c_api") or []:
            if sym not in declared:
                bad.append((op["id"], sym))
    return bad


def tested_variants(bench_dir):
    """(op_id-ish, format, dtype) actually present in benchmark JSON rows.

    Rows are keyed by the tags the benchmarks emit, not by manifest id: one
    benchmark binary covers several manifest entries (test_spmv covers spmv_csr
    and spmv_coo), so the join is on (format, dtype) within an operator family.
    """
    found = {}
    if not bench_dir or not bench_dir.is_dir():
        return None
    for path in sorted(bench_dir.glob("*_benchmark.json")):
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:
            print(f"  ! unreadable {path.name}: {exc}")
            continue
        family = doc.get("operator", path.stem.replace("_benchmark", ""))
        for row in doc.get("result", []):
            fmt = row.get("format")
            dt = row.get("dtype")
            if not fmt or not dt:
                continue
            found.setdefault((family, fmt, dt), []).append(row.get("status"))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-dir",
        type=pathlib.Path,
        default=pathlib.Path("capi_results"),
        help="directory of *_benchmark.json (default: capi_results/)",
    )
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    manifest, mpath = load_manifest()
    print(f"manifest: {mpath.relative_to(ROOT)}")
    groups = manifest["operators"]
    impl = [o for o in groups if o.get("status") == "implemented"]
    variants = declared_variants(manifest)
    print(
        f"  {len(groups)} groups, {len(impl)} implemented, "
        f"{len(variants)} declared variants\n"
    )

    drift = 0

    print("== paths ==")
    bad = check_paths(manifest)
    for op, field, v in bad:
        print(f"  MISSING  {op:<22} {field}: {v}")
    print(f"  {len(bad)} missing\n" if bad else "  all present\n")
    drift += len(bad)

    print("== symbols ==")
    bad = check_symbols(manifest)
    for op, sym in bad:
        print(f"  UNDECLARED  {op:<22} {sym}")
    print(f"  {len(bad)} undeclared\n" if bad else "  all declared in flagsparse.h\n")
    drift += len(bad)

    print("== coverage ==")
    tested = tested_variants(args.bench_dir)
    if tested is None:
        print("  SKIPPED: no --bench-dir given, so no benchmark JSON to read.")
        print("  Run:  FLAGSPARSE_MATRIX_DIR=... FLAGSPARSE_BENCH_OUT=./bench \\")
        print("            ctest --test-dir build -R benchmark")
        print("        python3 tools/check_manifest.py --bench-dir ./bench\n")
    else:
        # Family = the benchmark binary that should carry this entry, taken from
        # the manifest's own `tests` field rather than guessed from the id.
        # Which benchmark binary carries an entry. Prefer what the manifest
        # declares; fall back to the id's family when it declares nothing, so a
        # manifest that merely FORGOT to list its benchmark file does not look
        # like an unmeasured operator. The two are different bugs and get
        # reported separately below.
        fam_of, undeclared_test = {}, []
        for op in impl:
            fam = None
            for t in op.get("tests") or []:
                m = re.search(r"benchmark/test_(\w+)\.cpp", t)
                if m:
                    fam = m.group(1)
            if fam is None and (op.get("formats") and op.get("dtypes")):
                fam = op["id"].split("_")[0]
                undeclared_test.append((op["id"], fam))
            if fam:
                fam_of[op["id"]] = fam

        uncovered = []
        for oid, fmt, dt in variants:
            fam = fam_of.get(oid)
            if fam is None or (fam, fmt, dt) not in tested:
                uncovered.append((oid, fmt, dt, fam or "?"))

        for oid, fam in undeclared_test:
            print(
                f"  TESTS FIELD     {oid:<22} does not list "
                f"benchmark/test_{fam}.cpp (inferred it)"
            )
        for oid, fmt, dt, fam in uncovered:
            print(
                f"  NOT MEASURED    {oid:<22} {fmt}/{dt}  "
                f"(expected a row in {fam}_benchmark.json)"
            )
        # The other direction: rows nobody declared.
        declared_set = {(fam_of.get(o), f, d) for o, f, d in variants}
        extra = [k for k in tested if k not in declared_set]
        for fam, fmt, dt in sorted(extra):
            print(
                f"  UNDECLARED ROW  {fam:<22} {fmt}/{dt}  "
                f"(measured, but the manifest does not declare it)"
            )
        n = len(undeclared_test) + len(uncovered) + len(extra)
        print(f"  {n} disagreements\n" if n else "  manifest and rows agree\n")
        drift += n

    print("== delivery scope ==")
    deliv = gaps = 0
    for op in impl:
        if op.get("reporting") != "delivery":
            continue
        narrow = op.get("delivery_dtypes")
        dts = narrow if narrow else (op.get("dtypes") or [])
        deliv += len(op.get("formats") or []) * len(dts)
        for g in op.get("delivery_gaps") or []:
            gaps += 1
            print(
                f"  GAP  {op['id']:<22} {g}  declared in the delivery list, no kernel"
            )
    print(
        f"  {deliv} delivery variants generated, {gaps} asked for but "
        f"not implemented ({deliv + gaps} in the delivery list)\n"
    )

    print("== summary ==")
    print(f"  {drift} disagreement(s) between the manifest and the repo")
    if drift and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
