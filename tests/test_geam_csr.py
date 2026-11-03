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

"""Benchmark flagsparse_geam_csr, C = A + B in CSR, on .mtx matrices.

B is A^T for a square matrix (the symmetrisation csrgeam2 is typically used
for), otherwise a second value set on A's pattern. Rows come from
tests/q4_variant_bench.py's ``geam`` kind -- one per (variant, matrix), tagged
with the delivery variant id -- with three timings:

  triton_ms    flagsparse_geam_csr (validate=False)
  cusparse_ms  cusparse<t>csrgeam2: bufferSizeExt, Nnz, then the sum, with C
               allocated per call as FlagSparse does (CUDA only)
  pytorch_ms   torch.sparse CSR addition

Each result is checked against SciPy in float64 / complex128.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import q4_variant_bench  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help=".mtx files or directories")
    parser.add_argument("--csv", required=True, help="output CSV path")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args(argv)
    paths = q4_variant_bench.matrix_paths(args.inputs)
    if not paths:
        parser.error("no .mtx files found")
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.csv).write_text("", encoding="utf-8")
    q4_variant_bench.run_and_append("geam_csr", paths, args.csv, args.warmup, args.iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
