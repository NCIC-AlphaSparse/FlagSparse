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

"""Static policy checks for GPU CI wiring.

These tests make the GPU workflow reviewable from CPU-only CI. The workflow
itself still requires a self-hosted CUDA runner to execute.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"
TOOLS_DIR = PROJECT_ROOT / "tools" / "ci"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_gpu_accuracy_workflow_is_self_hosted_and_manual():
    text = _read(WORKFLOWS_DIR / "gpu-ci.yml")
    assert "workflow_dispatch:" in text
    # Runner scale sets (ARC) must be referenced by name alone in `runs-on`;
    # combining it with labels like self-hosted/linux breaks job matching.
    assert "runs-on: test-flagsparse" in text
    assert "actions/setup-python" not in text
    assert "python3.12 --version" in text
    assert "$GITHUB_PATH" not in text
    assert (
        "pip install --upgrade -r tools/ci/requirements-triton-smoke.lock.txt"
        not in text
    )
    assert "python3.12 -m pip show build pytest torch" in text
    assert "python3.12 -m pip uninstall -y triton" in text
    assert "flagtree===0.6.0rc2" in text
    assert "mkdir -p pytest_results" in text
    assert "gpu_ci_metadata.json" in text


def test_gpu_accuracy_workflow_checks_cuda_before_tests():
    text = _read(WORKFLOWS_DIR / "gpu-ci.yml")
    assert "nvidia-smi" in text
    assert "tools/ci/check_gpu_environment.py" in text
    assert "--require-cuda" in text
    assert "run_flagsparse_accuracy.py" in text


def test_gpu_benchmark_workflow_uploads_artifacts():
    text = _read(WORKFLOWS_DIR / "gpu-benchmark.yml")
    assert "runs-on: test-flagsparse" in text
    assert (
        "pip install --upgrade -r tools/ci/requirements-triton-smoke.lock.txt"
        not in text
    )
    assert "python3.12 -m pip show build pytest torch" in text
    assert "python3.12 -m pip uninstall -y triton" in text
    assert "flagtree===0.6.0rc2" in text
    assert "mkdir -p benchmark_results" in text
    assert "run_flagsparse_performance.py" in text
    assert "matrix_dir:" in text
    assert "actions/upload-artifact@v7" in text


def test_gpu_dependency_bundle_leaves_triton_to_flagtree():
    text = _read(TOOLS_DIR / "requirements-triton-smoke.lock.txt")
    assert "torch==2.9.0" in text
    assert "triton==" not in text


def test_gpu_environment_check_can_run_without_gpu():
    text = _read(TOOLS_DIR / "check_gpu_environment.py")
    assert "--require-cuda" in text
    assert "torch.cuda.is_available" in text
    assert "nvidia-smi" in text
    assert "metadata" in text
