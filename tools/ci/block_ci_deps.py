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

"""Reproduce the CI runner's missing packages locally, as a pytest plugin.

``tools/ci/requirements-ci.txt`` installs no numpy, scipy, torch or triton, so a
``tests/ci`` test that imports one has to guard it with ``pytest.importorskip``.
A developer box has all four, which makes a local ``pytest tests/ci`` run green
no matter how many of those guards are missing -- a module-level guard is easy to
remember, but an import written inside a test function body runs at call time and
slips past it. That is how CI broke on a numpy import more than once.

Run the suite the way CI sees it::

    make test-ci-offline

Everything should pass or skip, with one known exception:
``test_installed_wheel.py`` fails locally because no wheel is installed here; on
CI the wheel is installed and it passes.

``BLOCKED`` must stay disjoint from the CI requirements, or this plugin would
hide a package CI actually has. ``tests/ci/test_precommit_policy.py`` asserts
that, so adding a dependency to CI fails loudly here instead of making this tool
quietly lie.
"""

from __future__ import annotations

import sys

#: Imports available on a developer box but NOT installed by
#: ``tools/ci/requirements-ci.txt``.
BLOCKED = ("numpy", "scipy", "torch", "torch_npu", "triton")


def pytest_configure(config):
    """Make ``import <blocked>`` raise, the way it does on the CI runner."""
    for name in BLOCKED:
        prefix = name + "."
        for module in [
            key for key in list(sys.modules) if key == name or key.startswith(prefix)
        ]:
            del sys.modules[module]
        # A ``None`` entry is the documented way to make an import fail.
        sys.modules[name] = None
