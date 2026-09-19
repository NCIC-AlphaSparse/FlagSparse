"""Offline compile of the production CSR JIT functions; no Torch/GPU execution.

This checks compiler legality, not numerical correctness or device support.
Example: python tools/ci/check_spmv_csr_compile.py --backend cuda --arch 80
"""

import argparse
import ast
import itertools
import json
import os
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "hip"), default="cuda")
    parser.add_argument("--arch", default="80")
    parser.add_argument("--subgroup-width", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("build/spmv_csr_compile.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "build/triton-csr-cache"))
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    path = root / "src/flagsparse/sparse_operations/_spmv_csr_kernels.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Load exactly the production JIT definitions; exclude Torch orchestration.
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    ]
    module = types.ModuleType("_spmv_csr_offline_kernels")
    module.__dict__.update(triton=triton, tl=tl, __file__=str(path))
    sys.modules[module.__name__] = module
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    width = args.subgroup_width or (32 if args.backend == "cuda" else 64)
    target = GPUTarget(
        args.backend, int(args.arch) if args.backend == "cuda" else args.arch, width
    )
    records = []

    def check(name, pointers, scalars, constants, warps=4):
        fn = getattr(module, name)
        signature = dict(
            pointers,
            **{key: "i64" for key in scalars},
            **{key: "constexpr" for key in constants},
        )
        assert set(signature) == set(fn.arg_names), (name, signature, fn.arg_names)
        kernel = triton.compile(
            ASTSource(fn, signature, constexprs=constants),
            target=target,
            options={"num_warps": warps, "enable_fp_fusion": False},
        )
        records.append(
            dict(
                kernel=name,
                signature=signature,
                constants=constants,
                num_warps=warps,
                shared_bytes=kernel.metadata.shared,
                status="COMPILED",
            )
        )
        print(f"compiled {name} {pointers} {constants}", flush=True)

    # Complex pointers expose native real components, never a Triton complex type.
    value_modes = (
        ("fp16", False, tl.float32),
        ("bf16", False, tl.float32),
        ("fp32", False, tl.float64),
        ("fp64", False, tl.float64),
        ("fp32", True, tl.float32),
        ("fp64", True, tl.float64),
    )
    for (value, complex_input, acc), col, ptr, indexed in itertools.product(
        value_modes, ("i32", "i64"), ("i32", "i64"), (False, True)
    ):
        pointers = dict(
            A="*" + value,
            CI="*" + col,
            RP="*" + ptr,
            X="*" + value,
            Y="*" + value,
            ROWS="*i64" if indexed else "*" + ptr,
        )
        check(
            "row_tile_kernel",
            pointers,
            ("N",),
            dict(
                INDEXED=indexed,
                R=2 * width // 8,
                V=8,
                STAGES=1,
                COMPLEX=complex_input,
                ACC=acc,
            ),
            2,
        )
        check(
            "row_vector_kernel",
            pointers,
            ("N",),
            dict(INDEXED=indexed, B=128, STAGES=1, COMPLEX=complex_input, ACC=acc),
            2,
        )
        if indexed and col == "i32" and (complex_input or value in ("fp16", "bf16")):
            for batch in (1, 4):
                check(
                    "bucket_rows_kernel",
                    pointers,
                    ("N",),
                    dict(
                        BATCH=batch, B=128, MAX_SEGS=3, COMPLEX=complex_input, ACC=acc
                    ),
                    2,
                )
    for ptr, adaptive in itertools.product(("i32", "i64"), (False, True)):
        check(
            "classify_kernel",
            dict(RP="*" + ptr, SHORT="*i64", MID="*i64", COUNTS="*i64"),
            ("M",),
            dict(S=32, L=1024, T=1024, ADAPTIVE=adaptive, B=256),
        )
    check(
        "compact_rows_kernel",
        {name: "*i64" for name in ("SHORT", "MID", "SP", "MP", "SR", "MR")},
        ("M",),
        dict(B=256),
    )
    for ptr in ("i32", "i64"):
        check(
            "descriptors_kernel",
            dict(
                PREFIX="*i64", RP="*" + ptr, ROWS="*i64", STARTS="*i64", LENGTHS="*i64"
            ),
            ("TOTAL", "M"),
            dict(T=1024, B=256),
        )
    for (value, complex_input, acc), col in itertools.product(
        value_modes, ("i32", "i64")
    ):
        check(
            "segment_kernel",
            dict(
                A="*" + value,
                CI="*" + col,
                X="*" + value,
                STARTS="*i64",
                LENGTHS="*i64",
                PARTIAL="*fp64" if acc == tl.float64 else "*fp32",
            ),
            (),
            dict(B=128, STAGES=1, COMPLEX=complex_input, ACC=acc),
            2,
        )
    for block, acc, complex_input in itertools.product(
        (2, 256), ("fp32", "fp64"), (False, True)
    ):
        check(
            "reduce_level_kernel",
            dict(
                PARTIAL="*" + acc,
                PREV_PREFIX="*i64",
                NEXT_PREFIX="*i64",
                OUTPUT="*" + acc,
            ),
            ("M",),
            dict(B=block, COMPLEX=complex_input),
        )
    for (value, complex_input, acc), empty, partial in itertools.product(
        value_modes, (False, True), (False, True)
    ):
        check(
            "finish_kernel",
            dict(
                PARTIAL="*fp64" if acc == tl.float64 else "*fp32",
                PREFIX="*i64",
                COUNTS="*i64",
                Y="*" + value,
            ),
            ("M",),
            dict(WRITE_EMPTY=empty, HAS_PARTIAL=partial, B=256, COMPLEX=complex_input),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                triton=triton.__version__,
                target=vars(target),
                numerical_validation=False,
                records=records,
            ),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"{len(records)} variants compiled; report: {args.output}")


if __name__ == "__main__":
    main()
