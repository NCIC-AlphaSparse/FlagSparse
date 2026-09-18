#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Render capi_results/summary.json as result.html, in the FlagGems layout.

WHY A SEPARATE GENERATOR. run_flagsparse_pytest.py's write_result_html() consumes
the runner's in-memory `results` list, not summary.json, and its dtype columns are
a fixed tuple that has no fp64 and no complex -- half of what is measured here
would have nowhere to land. So the layout is reproduced (environment table, status
filter, sortable columns, one row per entry) while the columns follow the data.

ONE ROW PER VARIANT. `summary.json` is keyed by delivery-list name
(`spmv_csr_f32_int_non`), which is how 算子列表注册修改.xlsx names them, so the 40
rows fall out of the format rather than being forced into it.

Usage:
    python3 tools/write_html.py [--results-dir capi_results] [--out result.html]
"""

import argparse
import html
import json
import pathlib
import sys

# Every dtype the delivery list uses. Unlike the FlagGems tuple this includes
# fp64 and the two complex widths, because they are half of what is measured.
DTYPE_COLUMNS = ("fp16", "fp32", "fp64", "c32", "c64")

STATUS_CLASS = {
    "Passed": "ok",
    "Failed": "bad",
    "Skipped": "skip",
    "Timeout": "bad",
    "Error": "bad",
    "NotFound": "skip",
    "NoBaseline": "skip",
    "Unknown": "skip",
}

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     margin:24px;color:#1f2328;background:#fff}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#656d76;font-size:13px;margin-bottom:18px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px}
th,td{border:1px solid #d0d7de;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#f6f8fa;font-weight:600;white-space:nowrap}
tbody tr:nth-child(even){background:#fafbfc}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:#1a7f37;font-weight:600}
.bad{color:#cf222e;font-weight:600}
.skip{color:#9a6700}
.controls{margin-bottom:10px;font-size:13px}
.controls select,.controls input{padding:4px 6px;border:1px solid #d0d7de;border-radius:6px}
.sort-btn{border:none;background:none;cursor:pointer;color:#656d76;font-size:10px;padding:0 1px}
.sort-btn:hover{color:#0969da}
.stats{color:#656d76;font-size:12px;margin-bottom:6px}
.gain{background:#dafbe1}
.loss{background:#ffebe9}
details summary{cursor:pointer;color:#0969da}
details pre{background:#f6f8fa;padding:8px;overflow-x:auto;font-size:11px;margin:4px 0 0}
"""

JS = """
function applyFilter(){
  const st=document.getElementById('status-filter').value;
  const q=(document.getElementById('name-filter').value||'').toLowerCase();
  let shown=0;
  document.querySelectorAll('#rows tr').forEach(function(tr){
    const okStatus = st==='all' || tr.dataset.status===st;
    const okName = !q || tr.dataset.name.indexOf(q)>=0;
    const v = okStatus && okName;
    tr.style.display = v ? '' : 'none';
    if(v) shown++;
  });
  document.getElementById('visible-count').textContent=shown;
}
function sortTable(col,dir){
  const tb=document.getElementById('rows');
  const rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));
  if(dir==='reset'){rows.sort(function(a,b){return (+a.dataset.idx)-(+b.dataset.idx);});}
  else{
    rows.sort(function(a,b){
      const x=a.children[col].dataset.sort, y=b.children[col].dataset.sort;
      const nx=parseFloat(x), ny=parseFloat(y);
      // Blank cells sort last in both directions: a variant with no number is
      // not "the smallest", it is absent, and letting it head the ascending
      // list would read as a result.
      const bx=(x===''||isNaN(nx)), by=(y===''||isNaN(ny));
      if(bx&&by) return 0;
      if(bx) return 1;
      if(by) return -1;
      return dir==='asc' ? nx-ny : ny-nx;
    });
  }
  rows.forEach(function(r){tb.appendChild(r);});
}
"""


def esc(v):
    return html.escape(str(v), quote=True)


def fmt(v, nd=3):
    if v is None or v == "":
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return esc(v)
    return f"{f:.{nd}f}"


def speedup_cell(value):
    """A speedup cell, tinted by which side won. Blank when there is none --
    a missing baseline is not a 1.0."""
    if not value:
        return '<td class="num" data-sort=""></td>'
    cls = "gain" if value >= 1.0 else "loss"
    return f'<td class="num {cls}" data-sort="{value}">{fmt(value)}</td>'


def env_table(env):
    rows = []

    def add(k, v):
        rows.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")

    for k in ("architecture", "os_name", "os_release", "python"):
        add(k, env.get(k, ""))
    t = env.get("torch") or {}
    add("device", t.get("device_name") or "(unknown)")
    add("device_count", t.get("device_count", 0))
    fg = env.get("flag_gems") or {}
    add("vendor", fg.get("vendor", ""))
    add("backend", fg.get("device", ""))
    return "\n".join(rows)


def render(doc, out_path):
    """Write result.html for one summary document. Returns the path.

    Separate from main() so write_summary.py can call it directly: the two
    artifacts describe the same run, and making the HTML a second process that
    re-reads the JSON invites the two drifting out of step.
    """
    result = doc.get("result", {})

    body = []
    for i, (name, entry) in enumerate(sorted(result.items()), start=1):
        acc = entry.get("accuracy") or {}
        perf = entry.get("performance") or {}
        data = perf.get("data") or {}

        acc_status = acc.get("status", "Unknown")
        perf_status = perf.get("status", "Unknown")
        passed, failed = acc.get("passed", 0), acc.get("failed", 0)
        skipped = acc.get("skipped", 0)

        # Average speedup across whatever dtypes this variant has (one, by
        # construction -- the key names the dtype).
        sps = [e.get("speedup") or 0.0 for e in data.values() if e.get("speedup")]
        avg = sum(sps) / len(sps) if sps else 0.0

        per_dtype = {}
        for dt, e in data.items():
            per_dtype[dt] = e.get("speedup") or 0.0

        # Failure detail, collapsed. The list is the variant's own failing
        # matrices, which is what a reader needs to judge "is this the operator
        # or is this the matrix".
        det = acc.get("details") or {}
        detail_html = ""
        if det.get("failed"):
            items = "\n".join(
                esc(f"{d.get('name')}  error_ratio={d.get('error_ratio')}")
                for d in det["failed"]
            )
            detail_html = (
                f"<details><summary>{failed} failing</summary>"
                f"<pre>{items}</pre></details>"
            )

        cells = "".join(speedup_cell(per_dtype.get(dt)) for dt in DTYPE_COLUMNS)
        body.append(
            f'<tr data-idx="{i}" data-status="{esc(acc_status)}" '
            f'data-name="{esc(name.lower())}">'
            f'<td class="num" data-sort="{i}">{i}</td>'
            f'<td data-sort="{esc(name)}">{esc(name)}</td>'
            f'<td data-sort="{esc(acc_status)}" class="{STATUS_CLASS.get(acc_status, "skip")}">'
            f"{esc(acc_status)}{detail_html}</td>"
            f'<td data-sort="{passed}">{passed}/{failed}/{skipped}</td>'
            f'<td data-sort="{esc(perf_status)}" class="{STATUS_CLASS.get(perf_status, "skip")}">'
            f"{esc(perf_status)}</td>"
            f"{speedup_cell(avg)}"
            f"{cells}"
            "</tr>"
        )

    dtype_headers = "".join(
        f"<th>{dt}<br>"
        f'<button class="sort-btn" onclick="sortTable({6 + j},\'asc\')">&#9650;</button>'
        f'<button class="sort-btn" onclick="sortTable({6 + j},\'desc\')">&#9660;</button>'
        f"</th>"
        for j, dt in enumerate(DTYPE_COLUMNS)
    )

    doc_html = f"""<!doctype html>
<meta charset="utf-8">
<title>FlagSparse C API — {len(result)} variants</title>
<style>{CSS}</style>
<h1>FlagSparse C API 测试结果</h1>
<div class="sub">{esc(doc.get("timestamp", ""))} &nbsp;|&nbsp; 基线 cuSPARSE &nbsp;|&nbsp;
已登记交付变体 {len(result)} 个（conf/operators.yaml delivery_variants）</div>

<table>
<thead><tr><th>Env</th><th>Setting</th></tr></thead>
<tbody>{env_table(doc.get("env") or {})}</tbody>
</table>

<div class="controls">
  状态
  <select id="status-filter" onchange="applyFilter()">
    <option value="all">全部</option><option value="Passed">Passed</option>
    <option value="Failed">Failed</option><option value="Skipped">Skipped</option>
  </select>
  &nbsp; 名称 <input id="name-filter" oninput="applyFilter()" placeholder="spmv_csr…">
</div>
<div class="stats">共 {len(result)} 个变体，显示 <span id="visible-count">{len(result)}</span></div>

<table>
<thead><tr>
<th>No.<button class="sort-btn" onclick="sortTable(0,'reset')">&#8634;</button></th>
<th>ID</th>
<th>AccRes</th>
<th>AccStat(pass/fail/skip)</th>
<th>PerfRes</th>
<th>OPAverageSpeedUp
<button class="sort-btn" onclick="sortTable(5,'asc')">&#9650;</button>
<button class="sort-btn" onclick="sortTable(5,'desc')">&#9660;</button></th>
{dtype_headers}
</tr></thead>
<tbody id="rows">
{chr(10).join(body)}
</tbody>
</table>
<script>{JS}</script>
"""
    out_path.write_text(doc_html, encoding="utf-8")
    return out_path, len(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir", type=pathlib.Path, default=pathlib.Path("capi_results")
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="default: <results-dir>/result.html",
    )
    args = ap.parse_args()

    path = args.results_dir / "summary.json"
    if not path.exists():
        sys.exit(f"{path} not found -- run tools/write_summary.py first")
    out, n = render(
        json.loads(path.read_text()), args.out or (args.results_dir / "result.html")
    )
    print(f"wrote {out}  ({n} variants)")


if __name__ == "__main__":
    main()
