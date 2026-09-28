#!/usr/bin/env python3
"""Bundle an H800 pytest results directory into ONE file that ships with the repo.

The scaled baseline (``tools/baseline_bound.py``, ``run_flagsparse_pytest.py
--h800-reference``) needs the H800 cuSPARSE times.  Those used to live in a gitignored
``pytest_results_*`` directory that had to be located by hand; this tool freezes the part
that matters into ``conf/h800_reference.json`` (plus a human summary,
``docs/H800_REFERENCE.md``).

Kept per operator (everything else -- timing breakdowns, error norms, reason text -- is
dropped, it is not needed to scale a baseline):

* the columns that identify a case (matrix / case_id, dtype, index_dtype, op, alg, k, ...),
  stored as the exact strings the CSV had so keys match row for row;
* every ``*_ms`` wall-time column and every ``*speedup*`` column, as numbers;
* the ``status`` columns (a row whose vendor call failed must not become a baseline).

Usage::

    python3 tools/h800_reference.py pytest_results_sparse_202609221038      # regenerate
    python3 tools/h800_reference.py DIR --out conf/h800_reference.json --doc docs/H800_REFERENCE.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if (
    str(ROOT) not in sys.path
):  # run as a script, `tools` must be this repo's, not another package's
    sys.path.insert(0, str(ROOT))
DEFAULT_PATH = ROOT / "conf" / "h800_reference.json"
DEFAULT_DOC = ROOT / "docs" / "H800_REFERENCE.md"
SCHEMA = 1

# Columns that say WHICH case a row is.  Kept verbatim (as strings).
AXIS_COLUMNS = (
    "matrix",
    "case_id",
    "dense_size",
    "n_rows",
    "n_cols",
    "nnz",
    "m",
    "n",
    "dtype",
    "value_dtype",
    "value_dtype_req",
    "value_dtype_compute",
    "index_dtype",
    "indptr_dtype",
    "op",
    "opA",
    "transpose",
    "alg",
    "layout",
    "mode",
    "dense_cols",
    "k",
    "block_dim",
    "nnzb",
    "n_rhs",
    "format",
)
# `*_ms` columns that are breakdowns / host-side / conversion cost, not the timed operator.
_DROP_MS = re.compile(
    r"process_cpu|gpu_ms$|^gpu|prepare|^count_|^fill_|bucket_|scipy_cpu"
)
_STATUS_COLUMNS = ("status", "cu_status", "pt_status", "vendor_status")


def _is_time(name: str) -> bool:
    return (name == "ms" or name.endswith("_ms")) and not _DROP_MS.search(name)


def _is_number_column(name: str) -> bool:
    return _is_time(name) or "speedup" in name


def kept_columns(header):
    """The whitelisted subset of ``header``, in CSV order."""
    out = []
    for name in header:
        if name in AXIS_COLUMNS or name in _STATUS_COLUMNS or _is_number_column(name):
            out.append(name)
    return out


def _number(text):
    """CSV cell -> float rounded to 7 significant digits, or None when empty/non-numeric."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(f"{value:.7g}")


def _pack_cell(name, text):
    if _is_number_column(name):
        return _number(text)
    return text if text is not None else ""


def _unpack_cell(name, value):
    if value is None:
        return ""
    if _is_number_column(name):
        return repr(float(value))
    return str(value)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _variant_summary(summary):
    variants = {}
    for name, entry in (summary.get("result") or {}).items():
        acc = entry.get("accuracy") or {}
        perf = entry.get("performance") or {}
        speedups = {
            dtype: round(float(d["speedup"]), 4)
            for dtype, d in (perf.get("data") or {}).items()
            if isinstance(d, dict) and isinstance(d.get("speedup"), (int, float))
        }
        variants[name] = {
            "accuracy": acc.get("status"),
            "accuracy_passed": acc.get("passed"),
            "accuracy_total": acc.get("total"),
            "performance": perf.get("status"),
            "speedup": speedups,
        }
    return variants


def build(results_dir, when=None):
    """Read a pytest results directory and return the reference document (a dict)."""
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"{results_dir}: no summary.json -- not a runner results directory"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    env = summary.get("env") or {}
    torch_env = env.get("torch") or {}
    device = torch_env.get("device_name") or torch_env.get("device") or ""

    operators = {}
    commit = (env.get("git") or {}).get("commit")
    for op_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        csv_path = op_dir / "performance.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = kept_columns(reader.fieldnames or [])
            rows = []
            for row in reader:
                # the runner env has no git block; spmv_csr's CSV records the commit per row
                commit = commit or (row.get("commit") or "").strip() or None
                rows.append([_pack_cell(f, row.get(f)) for f in fields])
        if fields and rows:
            operators[op_dir.name] = {"fields": fields, "rows": rows}

    return {
        "schema": SCHEMA,
        "reference": {
            "device": device,
            "device_count": torch_env.get("device_count"),
            "peaks": "h800-sxm",
            "source_dir": results_dir.name,
            "generated": (when or date.today()).isoformat(),
            "run_timestamp": summary.get("timestamp"),
            "flagsparse_commit": commit,
            "torch": torch_env.get("version"),
            "triton": (env.get("triton") or {}).get("version"),
            "note": (
                "H800 wall times (ms). The vendor-library column (cusparse_ms / vendor_ms / "
                "*sparse_ms) is the baseline the scaled speedup divides into; see "
                "docs/H800_REFERENCE.md."
            ),
        },
        "variants": _variant_summary(summary),
        "operators": operators,
    }


def write(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # one row per line: diff-friendly and still compact
    lines = ["{"]
    lines.append(f'"schema": {doc["schema"]},')
    lines.append(f'"reference": {json.dumps(doc["reference"], ensure_ascii=False)},')
    lines.append('"variants": {')
    items = [
        f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
        for k, v in doc["variants"].items()
    ]
    lines.append(",\n".join(items))
    lines.append("},")
    lines.append('"operators": {')
    op_chunks = []
    for name, table in doc["operators"].items():
        body = ",\n".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            for r in table["rows"]
        )
        op_chunks.append(
            f'{json.dumps(name)}: {{"fields": {json.dumps(table["fields"])},\n"rows": [\n{body}\n]}}'
        )
    lines.append(",\n".join(op_chunks))
    lines.append("}")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    json.loads(text)  # never write a file the loader cannot read
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load (used by tools/baseline_bound.py and the runner)
# ---------------------------------------------------------------------------


class ReferenceTable:
    """One operator's rows, in the ``(fields, list-of-dict-of-str)`` shape a CSV read gives."""

    def __init__(self, source, op, fields, rows):
        self.source, self.op, self.fields, self._rows = source, op, list(fields), rows

    def __str__(self):
        return f"{self.source}::{self.op}"

    def read(self):
        rows = [
            {f: _unpack_cell(f, v) for f, v in zip(self.fields, raw)}
            for raw in self._rows
        ]
        return list(self.fields), rows


def load(path):
    """Parse and validate a bundled reference file; returns the document dict."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{path}: cannot read H800 reference ({exc})")
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("operators"), dict):
        raise SystemExit(
            f"{path}: not an H800 reference file (schema {doc.get('schema')!r})"
        )
    return doc


def is_reference_file(path):
    path = Path(path)
    return path.is_file() and path.suffix.lower() == ".json"


def tables(path):
    """{operator: ReferenceTable} for a bundled reference file."""
    doc = load(path)
    return {
        op: ReferenceTable(Path(path).name, op, t["fields"], t["rows"])
        for op, t in doc["operators"].items()
    }


def reference_info(path):
    """The ``reference`` block: device, commit, source dir, ..."""
    return dict(load(path).get("reference") or {})


# ---------------------------------------------------------------------------
# human summary
# ---------------------------------------------------------------------------


def _case_axes(fields):
    return [f for f in ("matrix", "case_id") if f in fields][:1]


def _usable_rows(op, table):
    """How many rows carry a vendor-library time (what the scaled baseline divides into)."""
    from tools import baseline_bound

    _, rows = ReferenceTable("", op, table["fields"], table["rows"]).read()
    return sum(1 for row in rows if baseline_bound._baseline_ms(row))


def _short_commit(commit):
    commit = commit or "?"
    sha, plus, tail = commit.partition("+")
    return (
        sha[:12] + plus + tail
    )  # keep the "+dirty" marker: that run was not a clean tree


def render_markdown(doc):
    ref = doc["reference"]
    out = [
        "# H800 参考结果（随仓库分发）",
        "",
        "> 本文件由 `tools/h800_reference.py` 从 H800 的一轮 runner 结果生成，**不要手改**；"
        "数据在 [`conf/h800_reference.json`](../conf/h800_reference.json)。",
        "",
        "## 来源",
        "",
        f"- 设备：{ref.get('device') or '?'} × {ref.get('device_count') or '?'}",
        f"- 原始结果目录：`{ref.get('source_dir')}`（被 git 忽略，不随仓库）",
        f"- FlagSparse commit：`{_short_commit(ref.get('flagsparse_commit'))}`；"
        f"torch {ref.get('torch') or '?'}；triton {ref.get('triton') or '?'}",
        f"- 生成日期：{ref.get('generated')}",
        "",
        "## 用途",
        "",
        "没有厂商 / PyTorch 基线的后端，用这里的 **H800 上 cuSPARSE 的时间**做基线：",
        "",
        "```",
        "h800_scaled_ms = T_cuSPARSE@H800 × P_H800 / P_目标卡      (P = 实测显存带宽)",
        "speedup        = h800_scaled_ms / T_FlagSparse@目标卡     (1.0 = 与折算后的 cuSPARSE 一样快，≥0.8 合格)",
        "```",
        "",
        "```bash",
        "# 不用再找目录：不带值即用本仓库自带的文件",
        "python3 run_flagsparse_pytest.py --h800-reference --vendor-card iluvatar-biv150 ...",
        "python3 tools/baseline_bound.py --vendor <目标卡结果目录> --vendor-card iluvatar-biv150",
        "```",
        "",
        "这是**估算**，不是厂商实测：H800 与目标卡的访存效率、缓存、launch 开销并不成比例。",
        "",
        "## 覆盖情况与限制",
        "",
        "- 只有带 cuSPARSE（或 vendor）时间的行才能做基线；下表“可作基线的行”就是这个数。",
        "  为 0 的算子（当时只有 CuPy 基线、没有对应格式的库）**没有折算基线**，行会保持 `N/A`。",
        "- FlagSparse commit 带 `+dirty` 表示那轮 H800 是在有未提交改动的工作树上跑的。",
        "",
        "## 各算子收录情况",
        "",
        "| 算子 | 行数 | 可作基线的行 | 保留的时间 / 加速比列 |",
        "|---|---:|---:|---|",
    ]
    for op, table in doc["operators"].items():
        fields = table["fields"]
        numeric = [f for f in fields if _is_number_column(f)]
        cols = ", ".join(f"`{c}`" for c in numeric)
        out.append(
            f"| `{op}` | {len(table['rows'])} | {_usable_rows(op, table)} | {cols} |"
        )
    out += [
        "",
        "## H800 上 20 个交付变体的结果",
        "",
        "| 变体 | 精度 | 性能 | 加速比（H800 vs cuSPARSE/PyTorch，runner 口径） |",
        "|---|---|---|---|",
    ]
    for name, v in doc["variants"].items():
        sp = (
            ", ".join(f"{d}: {x:.3f}" for d, x in (v.get("speedup") or {}).items())
            or "—"
        )
        acc = f"{v.get('accuracy')} ({v.get('accuracy_passed')}/{v.get('accuracy_total')})"
        out.append(f"| `{name}` | {acc} | {v.get('performance')} | {sp} |")
    out += [
        "",
        "## 重新生成",
        "",
        "```bash",
        "python3 tools/h800_reference.py <H800 结果目录>",
        "```",
        "",
        "只有在 H800 上重跑、且新结果确实要取代这份参考时才重新生成；生成后 diff 应当只有数值变化。",
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "results_dir", help="H800 runner results directory (with summary.json)"
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_PATH),
        help="output JSON (default conf/h800_reference.json)",
    )
    parser.add_argument(
        "--doc", default=str(DEFAULT_DOC), help="output markdown summary ('' to skip)"
    )
    args = parser.parse_args(argv)

    doc = build(args.results_dir)
    out = write(doc, args.out)
    rows = sum(len(t["rows"]) for t in doc["operators"].values())
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(doc['operators'])} operators, {rows} rows"
    )
    if args.doc:
        Path(args.doc).write_text(render_markdown(doc), encoding="utf-8")
        print(f"wrote {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
