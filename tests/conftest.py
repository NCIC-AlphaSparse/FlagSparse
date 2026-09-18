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

"""Make this directory importable the way running a script here already does.

The benchmark scripts are named test_*.py, so `pytest tests/` imports them even
though their work is behind a CLI. Run as scripts they get this directory on
sys.path for free and `import benchmark_utils` resolves; imported by pytest they
do not, and collection would fail on that import alone.
"""

import sys
from pathlib import Path

_TESTS_ROOT = str(Path(__file__).resolve().parent)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)
