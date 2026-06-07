#!/usr/bin/env python3
"""Export the FlagSparse sparse-operator support matrix to CSV.

The script is intentionally static: it does not import torch/triton/cupy or
flagsparse, and it never launches kernels. It reads source files and reports the
public support declared by source constants plus API-specific rules where the
runtime support is narrower than a shared ``SUPPORTED_*`` set.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = (
    "operator",
    "format",
    "index_dtype",
    "value_dtype",
    "op",
    "route",
    "status",
)

DEFAULT_VALUE_DTYPES = ("float16", "bfloat16", "float32", "float64", "complex64", "complex128")
DEFAULT_INDEX_DTYPES = ("int32", "int64")
NA = "N/A"
VALUE_DTYPE_ORDER = {
    "float16": 0,
    "bfloat16": 1,
    "float32": 2,
    "float64": 3,
    "complex64": 4,
    "complex128": 5,
}
INDEX_DTYPE_ORDER = {"int32": 0, "int64": 1}
OP_ORDER = {"non": 0, "trans": 1, "conj": 2}


@dataclass(frozen=True)
class ApiSpec:
    operator: str
    api: str
    module: str
    fmt: str
    route: str
    value_const: str | None = None
    index_const: str | None = None
    values: tuple[str, ...] | None = None
    indices: tuple[str, ...] | None = None
    ops: tuple[str, ...] | str | None = None
    notes: str = ""


class SourceModule:
    """Small AST-backed view of one sparse operation module."""

    def __init__(self, path: Path, shared: dict[str, Any] | None = None):
        self.path = path
        self.name = path.stem
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assignments: dict[str, ast.AST] = {}
        self.functions: set[str] = set()
        self.shared = shared or {}
        self._collect()

    def _collect(self) -> None:
        for node in self.tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.assignments[target.id] = node.value
            elif isinstance(node, ast.FunctionDef):
                self.functions.add(node.name)

    def get(self, name: str) -> Any:
        if name in self.assignments:
            return self._eval(self.assignments[name])
        return self.shared.get(name)

    def _eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Tuple):
            return tuple(self._flatten(el) for el in node.elts)
        if isinstance(node, ast.List):
            return tuple(self._flatten(el) for el in node.elts)
        if isinstance(node, ast.Set):
            return tuple(self._flatten(el) for el in node.elts)
        if isinstance(node, ast.Dict):
            return {self._eval(k): self._eval(v) for k, v in zip(node.keys, node.values) if k is not None}
        if isinstance(node, ast.Starred):
            return self._eval(node.value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch":
            return node.attr
        if isinstance(node, ast.Name):
            if node.id in self.assignments:
                return self._eval(self.assignments[node.id])
            if node.id in self.shared:
                return self.shared[node.id]
            return node.id
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.IfExp):
            return self._eval(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._as_tuple(self._eval(node.left))
            right = self._as_tuple(self._eval(node.right))
            return left + right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._eval(node.operand)
            return -value if isinstance(value, (int, float)) else value
        return None

    def _eval_call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            if node.func.id == "tuple" and node.args:
                return self._as_tuple(self._eval(node.args[0]))
        return None

    def _flatten(self, node: ast.AST) -> Any:
        value = self._eval(node)
        if isinstance(node, ast.Starred):
            return value
        return value

    @staticmethod
    def _as_tuple(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)


def normalize_dtype_values(value: Any) -> tuple[str, ...]:
    raw = flatten(value)
    dtypes: list[str] = []
    for item in raw:
        if item is None:
            continue
        token = str(item).replace("torch.", "")
        if token not in dtypes:
            dtypes.append(token)
    return tuple(dtypes)


def flatten(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list, set)):
        out: list[Any] = []
        for item in value:
            out.extend(flatten(item))
        return tuple(out)
    return (value,)


def discover_modules(src_root: Path) -> dict[str, SourceModule]:
    common = SourceModule(src_root / "_common.py")
    common_values = normalize_dtype_values(common.get("SUPPORTED_VALUE_DTYPES"))
    if "complex64" not in common_values:
        # _common.py builds this list with append/extend after initial assignment.
        # Static expression evaluation intentionally avoids executing that code,
        # so keep this fallback aligned with the declared global support set.
        common_values = DEFAULT_VALUE_DTYPES
    shared = {
        "SUPPORTED_VALUE_DTYPES": common_values,
        "SUPPORTED_INDEX_DTYPES": normalize_dtype_values(common.get("SUPPORTED_INDEX_DTYPES")),
    }
    modules = {"_common": common}
    for path in sorted(src_root.glob("*.py")):
        if path.name == "_common.py":
            continue
        modules[path.stem] = SourceModule(path, shared=shared)
    return modules


def collect_public_apis(*init_paths: Path) -> set[str]:
    values: set[str] = set()
    for init_path in init_paths:
        if not init_path.exists():
            continue
        init_module = SourceModule(init_path)
        values.update(normalize_dtype_values(init_module.get("__all__")))
    return values


def op_names(module: SourceModule, const_name: str) -> tuple[str, ...]:
    names = module.get(const_name)
    if isinstance(names, dict):
        values = [str(v) for _, v in sorted(names.items(), key=lambda item: item[0])]
        if values:
            return tuple(values)
    return ("non", "trans", "conj")


def normalize_op_label(op: Any) -> str:
    token = str(op).strip().lower().replace("-", "_")
    token = token.replace(" ", "_")
    if token in ("n/a", "na", "none"):
        return NA
    if token in ("n", "non", "non_trans", "non_transpose"):
        return "non"
    if token in ("t", "trans", "transpose"):
        return "trans"
    if token in ("c", "conj", "conj_trans", "conj_transpose", "conjugate_transpose"):
        return "conj"
    return token


def registry(modules: dict[str, SourceModule]) -> tuple[ApiSpec, ...]:
    spmv_ops = (
        op_names(modules["spmv_csr"], "SPMV_OP_NAMES")
        if "spmv_csr" in modules
        else ("non", "trans", "conj")
    )
    spmv_coo_ops = (
        op_names(modules["spmv_coo"], "SPMV_COO_OP_NAMES")
        if "spmv_coo" in modules
        else ("non", "trans", "conj")
    )
    spmm_values = (
        normalize_dtype_values(modules["spmm_csr"].get("SUPPORTED_SPMM_VALUE_DTYPES"))
        if "spmm_csr" in modules
        else None
    )
    spmm_ops = (
        op_names(modules["spmm_csr"], "SPMM_OP_NAMES")
        if "spmm_csr" in modules
        else ("non", "trans", "conj")
    )
    return (
        ApiSpec(
            "gather",
            "flagsparse_gather",
            "gather_scatter",
            "index",
            "triton",
            value_const="SUPPORTED_VALUE_DTYPES",
            index_const="SUPPORTED_INDEX_DTYPES",
        ),
        ApiSpec(
            "scatter",
            "flagsparse_scatter",
            "gather_scatter",
            "index",
            "triton",
            value_const="SUPPORTED_SCATTER_VALUE_DTYPES",
            index_const="SUPPORTED_INDEX_DTYPES",
        ),
        ApiSpec("spmv", "flagsparse_spmv_csr", "spmv_csr", "CSR", "triton", value_const="SUPPORTED_SPMV_VALUE_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=spmv_ops, notes="op supports non/trans/conj; conj on real dtypes is transpose-equivalent"),
        ApiSpec("spmv", "flagsparse_spmv_coo", "spmv_coo", "COO", "triton", value_const="SUPPORTED_SPMV_COO_VALUE_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=spmv_coo_ops, notes="COO path stores canonical row/col tensors and supports non/trans/conj"),
        ApiSpec("spmv", "flagsparse_spmv_coo_tocsr", "spmv_csr", "COO->CSR", "triton", value_const="SUPPORTED_SPMV_VALUE_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=("non",), notes="COO input is converted to CSR before compute"),
        ApiSpec("spmm", "flagsparse_spmm_csr", "spmm_csr", "CSR", "triton", value_const="SUPPORTED_SPMM_VALUE_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=spmm_ops, notes="op supports non/trans/conj; conj on real dtypes is transpose-equivalent"),
        ApiSpec("spmm", "flagsparse_spmm_csr_opt", "spmm_csr", "CSR", "triton_opt", values=("float32", "float64"), index_const="SUPPORTED_INDEX_DTYPES", ops=("non",), notes="bucketed opt path only supports float32/float64"),
        ApiSpec("spmm", "flagsparse_spmm_csr_opt_alg2", "spmm_csr_opt_alg2", "CSR", "triton_opt_alg2", value_const="SUPPORTED_SPMM_OPT_ALG2_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=("non",), notes="hardware-aware alg2 opt path only supports float32/float64"),
        ApiSpec("spmm", "flagsparse_spmm_coo", "spmm_coo", "COO", "triton", values=spmm_values, index_const="SUPPORTED_INDEX_DTYPES", ops=("non",), notes="COO SpMM reuses CSR SpMM dtype declaration"),
        ApiSpec("spgemm", "flagsparse_spgemm_csr", "spgemm_csr", "CSR", "triton", value_const="SUPPORTED_SPGEMM_VALUE_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=("non",)),
        ApiSpec("sddmm", "flagsparse_sddmm_csr", "sddmm_csr", "CSR", "triton", value_const="SUPPORTED_SDDMM_VALUE_DTYPES", index_const="SUPPORTED_INDEX_DTYPES", ops=("non",)),
        ApiSpec("spsv", "flagsparse_spsv_csr", "spsv", "CSR", "triton", value_const="SUPPORTED_SPSV_VALUE_DTYPES", index_const="SUPPORTED_SPSV_INDEX_DTYPES", ops=("NON_TRANS", "TRANS"), notes="TRANS support is narrower than NON_TRANS; see combo constants"),
        ApiSpec("spsv", "flagsparse_spsv_coo", "spsv", "COO", "triton", value_const="SUPPORTED_SPSV_VALUE_DTYPES", index_const="SUPPORTED_SPSV_INDEX_DTYPES", ops=("NON_TRANS", "TRANS"), notes="TRANS support is narrower than NON_TRANS; see combo constants"),
        ApiSpec("spsm", "flagsparse_spsm_csr", "spsm", "CSR", "triton", value_const="SUPPORTED_SPSM_VALUE_DTYPES", index_const="SUPPORTED_SPSM_INDEX_DTYPES", values=("float32", "float64", "complex64", "complex128"), indices=("int32",), ops=("NON_TRANS",), notes="opA/opB must both be NON_TRANS; row-major dense layout only"),
        ApiSpec("spsm", "flagsparse_spsm_coo", "spsm", "COO", "triton", value_const="SUPPORTED_SPSM_VALUE_DTYPES", index_const="SUPPORTED_SPSM_INDEX_DTYPES", values=("float32", "float64", "complex64", "complex128"), indices=("int32",), ops=("NON_TRANS",), notes="opA/opB must both be NON_TRANS; row-major dense layout only"),
    )


def rows_for_spec(spec: ApiSpec, modules: dict[str, SourceModule], public_apis: set[str], src_root: Path) -> list[dict[str, str]]:
    module = modules.get(spec.module)
    notes = [spec.notes] if spec.notes else []
    status = "SUPPORTED"
    if module is None:
        return [row(spec, NA, NA, NA, "PARTIAL")]

    values = spec.values
    if values is None and spec.value_const:
        values = normalize_dtype_values(module.get(spec.value_const))
    indices = spec.indices
    if indices is None and spec.index_const:
        indices = normalize_dtype_values(module.get(spec.index_const))
    if not values:
        values = (NA,)
        status = "PARTIAL"
        notes.append(f"value dtype constant {spec.value_const or '<none>'} not found")
    if not indices:
        indices = (NA,)
        status = "PARTIAL"
        notes.append(f"index dtype constant {spec.index_const or '<none>'} not found")
    if spec.api not in module.functions and spec.api not in public_apis:
        status = "PARTIAL"
        notes.append(f"api {spec.api} not found in source exports/functions")

    ops = spec.ops
    if ops is None:
        ops_tuple = ("non",)
    elif isinstance(ops, str):
        ops_tuple = (ops,)
    else:
        ops_tuple = ops

    return [
        row(spec, value_dtype, index_dtype, op, status)
        for value_dtype in values
        for index_dtype in indices
        for op in ops_tuple
    ]


def row(spec: ApiSpec, value_dtype: str, index_dtype: str, op: str, status: str) -> dict[str, str]:
    return {
        "operator": spec.operator,
        "format": spec.fmt,
        "index_dtype": str(index_dtype),
        "value_dtype": str(value_dtype),
        "op": normalize_op_label(op),
        "route": spec.route,
        "status": status,
    }


def discovered_unmapped_rows(modules: dict[str, SourceModule], specs: Iterable[ApiSpec], src_root: Path) -> list[dict[str, str]]:
    mapped = {(spec.module, spec.value_const) for spec in specs if spec.value_const}
    mapped |= {(spec.module, spec.index_const) for spec in specs if spec.index_const}
    out: list[dict[str, str]] = []
    for module_name, module in sorted(modules.items()):
        if module_name in {"_common", "__init__", "benchmarks"}:
            continue
        for const_name in sorted(module.assignments):
            if not (const_name.startswith("SUPPORTED_") and const_name.endswith("_DTYPES")):
                continue
            if (module_name, const_name) in mapped:
                continue
            values = normalize_dtype_values(module.get(const_name))
            dtype_kind = "index_dtype" if "INDEX" in const_name else "value_dtype"
            for dtype in values or (NA,):
                data = {
                    "operator": module_name,
                    "format": NA,
                    "value_dtype": dtype if dtype_kind == "value_dtype" else NA,
                    "index_dtype": dtype if dtype_kind == "index_dtype" else NA,
                    "op": NA,
                    "route": NA,
                    "status": "DISCOVERED_UNMAPPED",
                }
                out.append(data)
    return out


def _ordered_value(mapping: dict[str, int], value: str) -> tuple[int, str]:
    return (mapping.get(value, len(mapping)), value)


def _sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda item: (
            item["operator"],
            item["format"],
            _ordered_value(INDEX_DTYPE_ORDER, item["index_dtype"]),
            _ordered_value(VALUE_DTYPE_ORDER, item["value_dtype"]),
            _ordered_value(OP_ORDER, item["op"]),
            item["route"],
            item["status"],
        ),
    )


def build_rows(src_root: Path) -> list[dict[str, str]]:
    modules = discover_modules(src_root)
    public_apis = collect_public_apis(src_root / "__init__.py", src_root.parent / "__init__.py")
    specs = registry(modules)
    rows: list[dict[str, str]] = []
    for spec in specs:
        rows.extend(rows_for_spec(spec, modules, public_apis, src_root))
    rows.extend(discovered_unmapped_rows(modules, specs, src_root))
    return _sort_rows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--src-root",
        type=Path,
        default=script_dir / "src" / "flagsparse" / "sparse_operations",
        help="Path to src/flagsparse/sparse_operations (default: project-local path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "ops_support.csv",
        help="CSV output path (default: ./ops_support.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    output = args.output.resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"sparse operations source root not found: {src_root}")

    rows = build_rows(src_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} support rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
