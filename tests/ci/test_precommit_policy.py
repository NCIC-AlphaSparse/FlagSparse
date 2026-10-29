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

"""Static policy checks for repository pre-commit wiring."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_precommit_config_exists():
    path = PROJECT_ROOT / ".pre-commit-config.yaml"
    assert path.is_file()


def test_makefile_runs_precommit():
    text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "pre-commit run --all-files" in text


def _blocked_names():
    """``BLOCKED`` from tools/ci/block_ci_deps.py, read without importing it.

    Importing the plugin would blank those modules for the rest of the session.
    """
    import ast

    tree = ast.parse((PROJECT_ROOT / "tools/ci/block_ci_deps.py").read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BLOCKED"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("tools/ci/block_ci_deps.py defines no BLOCKED")


def test_offline_blocklist_is_disjoint_from_the_ci_requirements():
    """A package CI installs must not be blocked, or the plugin would lie.

    ``make test-ci-offline`` is only evidence about the runner while the two
    lists disagree about nothing: adding numpy to the CI deps has to fail here
    rather than leave the local run hiding a guard that is no longer needed.
    """
    requirements = (PROJECT_ROOT / "tools/ci/requirements-ci.txt").read_text("utf-8")
    installed = {
        line.split("#", 1)[0].strip().split(">")[0].split("=")[0].split("<")[0].strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert _blocked_names().isdisjoint(installed)


def test_makefile_offers_the_offline_ci_run():
    text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "test-ci-offline:" in text
    assert "-p block_ci_deps" in text
