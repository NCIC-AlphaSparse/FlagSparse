"""Run SpSM synthetic benchmark. From project root: python benchmark/benchmark_spsm.py [--n 512 --rhs 1024]."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from tests.test_spsm import main

if __name__ == "__main__":
    main()
