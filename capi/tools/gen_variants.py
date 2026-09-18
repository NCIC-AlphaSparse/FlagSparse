#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate ctest's variant registry from conf/operators.yaml.

THE POINT. Before this, the list of variants the benchmarks swept was the product
of hardcoded tables in eight .cpp files, and the manifest was a document nobody
read at build time. The two disagreed by 22 variants and nothing noticed. Now the
manifest is the list: growing it from 40 to 115 is an edit to the YAML plus a
rebuild, and a variant the manifest declares but no benchmark implements shows up
as a row saying so instead of being silently absent.

Run by CMake; the output is a generated header, never edited by hand.
"""

import argparse
import pathlib
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required to generate the variant registry: pip install pyyaml")

# The manifest names dtypes by the width of the whole value (cuSPARSE's
# convention: c64 is a 64-bit complex, i.e. two fp32). The C enum and the JSON
# tags name the component. Keeping the conversion in ONE table is what stops a
# manifest `c64` from being paired with a benchmark row tagged `c64`, which is a
# different type.
DTYPES = {
    "f32": ("FLAGSPARSE_R_32F", "f32"),
    "f64": ("FLAGSPARSE_R_64F", "f64"),
    "c64": ("FLAGSPARSE_C_32F", "c32"),
    "c128": ("FLAGSPARSE_C_64F", "c64"),
    "f16": ("FLAGSPARSE_R_16F", "f16"),
    "bf16": ("FLAGSPARSE_R_16BF", "bf16"),
}

FORMATS = {
    "sparse_vector": "spvec",
    "csr": "csr",
    "coo": "coo",
    "csc": "csc",
    "bsr": "bsr",
    "sell": "sell",
}


def family_of(op_id):
    """Which benchmark binary owns this entry. Taken from the manifest's own
    `tests:` field, falling back to the id's first token."""
    return op_id.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    doc = yaml.safe_load(args.manifest.read_text())
    rows = []
    for op in doc["operators"]:
        if op.get("status") != "implemented":
            continue
        fmts = op.get("formats") or []
        dts = op.get("dtypes") or []
        fam = None
        for t in op.get("tests") or []:
            if "benchmark/test_" in t:
                fam = t.split("benchmark/test_")[1].replace(".cpp", "")
        fam = fam or family_of(op["id"])
        # Delivery scope. `delivery_dtypes` narrows an operator that is in the
        # list for only some of its dtypes; absent means all of them.
        reporting = op.get("reporting", "retained")
        narrow = op.get("delivery_dtypes")
        for f in fmts:
            ftag = FORMATS.get(f)
            if ftag is None:
                continue
            for d in dts:
                ent = DTYPES.get(d)
                if ent is None:
                    continue
                enum, dtag = ent
                scope = reporting
                if reporting == "delivery" and narrow and d not in narrow:
                    scope = "retained"
                rows.append((op["id"], fam, ftag, dtag, enum, scope))

    lines = [
        "// GENERATED from conf/operators.yaml by tools/gen_variants.py -- DO NOT EDIT.",
        "//",
        "// One entry per variant the manifest declares implemented. A benchmark",
        "// iterates the entries whose `family` is its own; a family/format pair it",
        "// has no code path for still produces a row, carrying the reason, so the",
        "// gap between what is declared and what is measured stays visible.",
        "",
        "#ifndef FLAGSPARSE_CTEST_VARIANTS_INC",
        "#define FLAGSPARSE_CTEST_VARIANTS_INC",
        "",
        "namespace fstest::registry {",
        "",
        "struct Variant {",
        '    const char* op;       // manifest id, e.g. "spmv_csc"',
        '    const char* family;   // benchmark binary, e.g. "spmv"',
        '    const char* format;   // row tag, e.g. "csc"',
        '    const char* dtype;    // row tag, e.g. "c32"',
        "    flagsparseDataType_t dt;",
        '    const char* reporting;  // "delivery" or "retained"',
        "};",
        "",
        "inline constexpr Variant kVariants[] = {",
    ]
    for op, fam, ftag, dtag, enum, scope in rows:
        lines.append(f'    {{"{op}", "{fam}", "{ftag}", "{dtag}", {enum}, "{scope}"}},')
    lines += [
        "};",
        "",
        f"inline constexpr int kVariantCount = {len(rows)};",
        "",
        "}  // namespace fstest::registry",
        "",
        "#endif  // FLAGSPARSE_CTEST_VARIANTS_INC",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    fams, scopes = {}, {}
    for _, fam, _, _, _, scope in rows:
        fams[fam] = fams.get(fam, 0) + 1
        scopes[scope] = scopes.get(scope, 0) + 1
    print(f"variants: {len(rows)} from {args.manifest.name} -> {args.out.name}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(scopes.items())))
    for f, n in sorted(fams.items()):
        print(f"  {f:<10}{n}")


if __name__ == "__main__":
    main()
