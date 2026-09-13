"""Compatibility forwarding entry point; use tests/test_spmv_csr.py."""

import sys
from pathlib import Path

__test__ = False
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_spmv_csr import main, load_mtx_to_csr_torch

if __name__ == "__main__":
    raise SystemExit(main())
