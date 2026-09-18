#!/usr/bin/env python3
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

"""Find the single test that wedges the GPU, for a pytest marker that times out.

Motivation: on an MTT S5000 the `spgemm_csr` and `spmm_coo` accuracy markers hit a
300s hard timeout with **zero tests reported** (exit -100), while the same markers
take ~4s on CUDA. A whole-marker timeout says nothing about which case is stuck, and
a run summary showing "0 failed" hides it entirely.

Three phases, each narrowing the blame:

  1. collect  -- can pytest even enumerate the tests? An import-time or
                 module-scope hang shows up here and nowhere else.
  2. whole    -- does the marker hang when run as one process? Confirms the
                 timeout is reproducible before spending time on per-test runs.
  3. per-test -- every node id in its OWN process under a hard kill timeout.
                 One process per case is not optional: once a kernel faults, the
                 vendor runtime is poisoned and every later result in the same
                 process is garbage.

Usage (on the card, from the repo root)::

    export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads
    python tools/bisect_hang.py --marker spmm_coo
    python tools/bisect_hang.py --marker spgemm_csr --timeout 90 --json hang.json
    python tools/bisect_hang.py --marker spmm_coo --stop-after-first-hang

Every subprocess is killed with SIGKILL, never SIGTERM: a thread blocked in the
vendor driver does not handle signals, so a soft timeout leaves the process alive
and the next one queues behind it.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

KILL = ["timeout", "-s", "KILL"]


def run(cmd, timeout, env=None):
    """Run a command under a hard kill timeout. Returns (status, out, err, secs)."""
    started = time.time()
    full = KILL + [str(timeout)] + cmd
    try:
        proc = subprocess.run(
            full, capture_output=True, text=True, env=env or os.environ.copy()
        )
    except FileNotFoundError as exc:  # `timeout` missing is a setup error, not a hang
        return "ERROR", "", f"{type(exc).__name__}: {exc}", 0.0
    secs = time.time() - started
    # Expiry codes differ by observer, and getting this wrong relabels every wedge
    # as a plain failure -- which points the whole diagnosis the wrong way.
    # `timeout -s KILL` re-raises the signal on itself to report it, so a SHELL sees
    # 137 (128+9) while Python's subprocess sees the negative signal number, -9.
    # 124 is GNU timeout's code when it is not re-raising. -11 (SIGSEGV) is a crash,
    # not a hang, and must stay out of this set.
    if proc.returncode in (124, 137, -9, -15):
        return "HANG", proc.stdout, proc.stderr, secs
    if proc.returncode == 0:
        return "OK", proc.stdout, proc.stderr, secs
    return "FAIL", proc.stdout, proc.stderr, secs


def tail(text, n=12):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-n:]


def collect(marker, mode, timeout):
    """Enumerate node ids for a marker.

    ``-o addopts=`` clears the ini's ``addopts = -v``: with -v in effect
    ``--collect-only`` prints a ``<Function ...>`` tree instead of node ids, and the
    parse below silently yields nothing -- which reads exactly like "no tests".
    """
    cmd = [
        sys.executable, "-m", "pytest", "tests/pytest",
        "-m", marker, "--mode", mode,
        "--collect-only", "-q", "-p", "no:cacheprovider",
        "-o", "addopts=",
    ]
    status, out, err, secs = run(cmd, timeout)
    ids = [
        l.strip() for l in (out or "").splitlines()
        if "::" in l and not l.lstrip().startswith(("=", "<"))
    ]
    return status, ids, out, err, secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--marker", required=True, help="pytest marker that times out")
    # MUST match pytest's own default (conftest: default="normal"). A tool that
    # defaults differently bisects a different test set than the run that failed:
    # quick drops every shape but the smallest, so a hang that only appears at
    # 64-16-96 is invisible and the bisect "finds" an unrelated flake instead.
    ap.add_argument("--mode", default="normal", help="--mode passed to pytest")
    ap.add_argument("--timeout", type=float, default=60,
                    help="per-test hard kill timeout, seconds (default 60)")
    ap.add_argument("--collect-timeout", type=float, default=120)
    ap.add_argument("--whole-timeout", type=float, default=0,
                    help="timeout for the whole-marker run; 0 = 4x per-test timeout")
    ap.add_argument("--skip-whole", action="store_true",
                    help="skip phase 2 when the hang is already known to reproduce")
    ap.add_argument("--stop-after-first-hang", action="store_true",
                    help="stop phase 3 at the first wedge (the minimal repro)")
    ap.add_argument("--json", default="", help="write full results here")
    args = ap.parse_args(argv)

    print(f"marker        : {args.marker}")
    print(f"mode          : {args.mode}")
    print(f"per-test kill : {args.timeout:.0f}s")
    try:
        import flagsparse.sparse_operations._common as C
        print(f"backend       : {C._backend_name()} / {C._accel_device_type()}")
        print(f"fallback      : {C._accel_fallback_reason()}")
    except Exception as exc:
        print(f"backend       : probe failed: {type(exc).__name__}: {exc}")
    print()

    results = {}

    # --- phase 1: collection ------------------------------------------------
    print("=" * 78)
    print("phase 1: collection")
    print("=" * 78)
    status, ids, out, err, secs = collect(args.marker, args.mode, args.collect_timeout)
    results["collect"] = {"status": status, "count": len(ids), "seconds": round(secs, 1)}
    print(f"  {status} in {secs:.1f}s, {len(ids)} test(s)")
    if status == "HANG":
        print("\n  Collection itself wedged -- the hang is at import or module scope,")
        print("  before any test body runs. Nothing below can narrow it further;")
        print("  bisect the module's top-level statements instead.")
        for line in tail(err):
            print(f"    | {line}")
        return 2
    if status != "OK" or not ids:
        print(f"  Could not enumerate tests (status={status}, {len(ids)} parsed).")
        print("  If pytest reported tests but none were parsed, the output format")
        print("  changed -- check ini addopts/verbosity before believing this.")
        for line in tail(out) + tail(err):
            print(f"    | {line}")
        return 2

    # --- phase 2: whole marker ---------------------------------------------
    if not args.skip_whole:
        whole_timeout = args.whole_timeout or args.timeout * 4
        print()
        print("=" * 78)
        print(f"phase 2: whole marker in one process (kill after {whole_timeout:.0f}s)")
        print("=" * 78)
        cmd = [
            sys.executable, "-m", "pytest", "tests/pytest",
            "-m", args.marker, "--mode", args.mode, "-q", "-p", "no:cacheprovider",
            "-o", "addopts=",
        ]
        status, out, err, secs = run(cmd, whole_timeout)
        results["whole"] = {"status": status, "seconds": round(secs, 1)}
        print(f"  {status} in {secs:.1f}s")
        for line in tail(out, 4):
            print(f"    | {line}")
        if status == "OK":
            print("\n  The marker completed. The original timeout did not reproduce --")
            print("  do NOT conclude it is fixed: check whether this run used the same")
            print("  files, mode and GPU as the one that timed out.")

    # --- phase 3: one process per test -------------------------------------
    print()
    print("=" * 78)
    print(f"phase 3: {len(ids)} test(s), one process each")
    print("=" * 78)
    width = min(max((len(i) for i in ids), default=20), 92)
    hangs, fails = [], []
    for i, node in enumerate(ids, 1):
        cmd = [
            sys.executable, "-m", "pytest", node,
            "--mode", args.mode, "-q", "-p", "no:cacheprovider",
            "-o", "addopts=",
        ]
        status, out, err, secs = run(cmd, args.timeout)
        results[node] = {"status": status, "seconds": round(secs, 1),
                         "stderr": (err or "")[-1500:]}
        mark = {"OK": "ok", "FAIL": "FAIL", "HANG": "HANG", "ERROR": "ERR"}[status]
        print(f"  [{i:>3}/{len(ids)}] {node[:width]:<{width}} {mark:>5} {secs:>6.1f}s")
        if status == "HANG":
            hangs.append(node)
            if args.stop_after_first_hang:
                print("\n  --stop-after-first-hang: stopping at the first wedge.")
                break
        elif status in ("FAIL", "ERROR"):
            fails.append(node)

    # --- verdict ------------------------------------------------------------
    print()
    print("=" * 78)
    print("verdict")
    print("=" * 78)
    if hangs:
        print(f"  {len(hangs)} test(s) wedged. Minimal repro:")
        print(f"    {hangs[0]}")
        print()
        print("  Reproduce it alone, with the driver reporting synchronously:")
        print(f"    CUDA_LAUNCH_BLOCKING=1 timeout -s KILL 120 \\")
        print(f"      {shlex.quote(sys.executable)} -m pytest {shlex.quote(hangs[0])} --mode {args.mode} -q")
        err = (results[hangs[0]].get("stderr") or "").strip()
        if err:
            print("\n  stderr before the kill (read the vendor's wording literally):")
            for line in tail(err):
                print(f"    | {line}")
        else:
            print("\n  Nothing on stderr before the kill -- consistent with a wedge")
            print("  inside the driver rather than an error the runtime reported.")
    elif fails:
        print(f"  No test wedged, but {len(fails)} failed:")
        for n in fails[:10]:
            print(f"    {n}")
        print("\n  A marker that times out as a whole but has no individually hanging")
        print("  test usually means the cost is cumulative (per-test compile, leaked")
        print("  memory) rather than one stuck case. Compare phase 2's wall time with")
        print("  the sum of phase 3's.")
    else:
        print("  Every test passed in its own process.")
        total = sum(v["seconds"] for k, v in results.items() if k not in ("collect", "whole"))
        print(f"  Sum of per-test wall time: {total:.1f}s")
        print("  If the whole-marker run still times out, the cost is cumulative --")
        print("  state that leaks across tests in one process, not a single wedge.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n  wrote {args.json}")
    return 1 if (hangs or fails) else 0


if __name__ == "__main__":
    sys.exit(main())
