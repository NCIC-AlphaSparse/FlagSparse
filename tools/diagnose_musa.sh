#!/usr/bin/env bash
# Two diagnostics for the MUSA run, in one pass. Read-only apart from its own logs.
#
#   1. capability probe -- settles what the card supports per dtype, independent of
#      how any test happens to be written. Includes the advanced-indexing checks
#      added after the "IndexMusa" complex gap showed up inside the library.
#   2. hang bisect -- spgemm_csr and spmm_coo hit a 300s timeout with ZERO tests
#      reported, while the same markers take ~4s on CUDA. This narrows each to the
#      individual test that wedges, one process per case.
#
# Usage, from the repo root on the card:
#   export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads
#   bash tools/diagnose_musa.sh [outdir]
set -u
OUT="${1:-musa_diag_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
echo "输出目录: $OUT"

echo
echo "############ 1/3  能力探针（含新加的 indexing 检查）############"
timeout -s KILL 1800 python3 tools/probe_accel_capabilities.py \
    --json "$OUT/musa_caps.json" 2>&1 | tee "$OUT/probe.log"
echo "  -> $OUT/probe.log , $OUT/musa_caps.json"

# The two markers are run in SEPARATE invocations on purpose: once a kernel wedges,
# the vendor runtime is poisoned and anything sharing the process reports garbage.
for MARK in spgemm_csr spmm_coo; do
  echo
  echo "############ 卡死定位: $MARK ############"
  timeout -s KILL 2400 python3 tools/bisect_hang.py \
      --marker "$MARK" --timeout 60 --json "$OUT/hang_$MARK.json" \
      2>&1 | tee "$OUT/hang_$MARK.log"
  echo "  -> $OUT/hang_$MARK.log"
done

echo
echo "============================================================"
echo "打包回传:  tar czf $OUT.tgz $OUT"
echo "============================================================"
ls -la "$OUT"
