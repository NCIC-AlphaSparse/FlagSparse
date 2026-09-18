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

"""CPU-only checks for README command examples and workflow references."""

from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
TEXT = README.read_text(encoding="utf-8")


def test_readme_install_command_is_present():
    assert "pip install . --no-deps --no-build-isolation" in TEXT


def test_readme_cpu_smoke_examples_are_present():
    accuracy_runner = "python run_flagsparse_accuracy.py --mode quick --gpus 0"
    performance_runner = (
        "python run_flagsparse_performance.py --ops spmv_csr,spmm_csr "
        "--benchmark-input matrix --benchmark-warmup 5 --benchmark-iters 20"
    )
    both_phase_runner = (
        "python run_flagsparse_pytest.py --phase both --mode quick "
        "--gpus 0,1 --benchmark-input matrix --results-dir pytest_results"
    )
    for snippet in [
        "pytest tests/pytest --mode quick",
        accuracy_runner,
        performance_runner,
        both_phase_runner,
        "python tests/test_spmv.py <dir_or_file.mtx>",
        "python tests/test_spmm.py --synthetic",
        "python tests/test_spgemm.py <dir_or_file.mtx> --input-mode auto",
        "python tests/test_spsm.py --synthetic --n 512 --rhs 1024",
    ]:
        assert snippet in TEXT


def test_readme_ci_cd_section_mentions_workflows():
    for snippet in [
        ".github/workflows/ci.yml",
        ".github/workflows/nightly-cpu.yml",
        ".github/workflows/release.yml",
        ".github/workflows/triton-smoke.yml",
        ".github/workflows/gpu-benchmark.yml",
    ]:
        assert snippet in TEXT


def test_readme_documents_standardized_command_policy():
    for snippet in [
        "documented invocation standard",
        ("CPU-only install, build, help-text, and smoke paths are checked in CI"),
        "`make ci` / `make check`",
        "`make format-check`, `make lint`, and `make lint-src`",
        "`make release-check` / `make release`",
        "`make triton-smoke` and `make triton-deps`",
        "CI / Build and smoke test",
    ]:
        assert snippet in TEXT
