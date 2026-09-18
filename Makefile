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

PYTHON ?= python
DIST_DIR ?= dist
EXPECTED_VERSION ?= 1.0.0

.PHONY: help ci check ci-deps compile format-check lint lint-src pre-commit-check build install-wheel validate-wheel test-ci smoke triton-smoke triton-deps gpu-env-check gpu-benchmark release-check release clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make ci             - run the default CPU-only CI pipeline' \
		'  make check          - alias for make ci' \
		'  make smoke          - alias for the CPU smoke test stage' \
		'  make format-check   - verify CI Python files are ruff-formatted' \
		'  make lint           - run ruff lint checks for CI files' \
		'  make lint-src       - run critical ruff checks for package source' \
		'  make pre-commit-check - run configured pre-commit hooks' \
		'  make release        - alias for make release-check' \
		'  make release-check  - build, validate, and checksum release artifacts' \
		'  make triton-deps    - install the opt-in triton smoke dependency bundle' \
		'  make triton-smoke   - opt-in triton-dependent smoke tests' \
		'  make gpu-env-check  - validate CUDA visibility on a GPU runner' \
		'  make gpu-benchmark  - run the quick GPU benchmark suite on a CUDA machine' \
		'  make help           - show this list'

ci: ci-deps compile format-check lint lint-src pre-commit-check build install-wheel validate-wheel test-ci

check: ci

ci-deps:
	$(PYTHON) -m pip install --upgrade -r tools/ci/requirements-ci.lock.txt

triton-deps:
	$(PYTHON) -m pip install --upgrade -r tools/ci/requirements-triton-smoke.lock.txt

compile:
	$(PYTHON) -m compileall src tests tools

format-check:
	ruff format --check tests/ci tools/ci tools/run_backend_tests.py capi/tools

lint:
	ruff check tests/ci tools/ci tools/run_backend_tests.py capi/tools

lint-src:
	ruff check src --select E9,F63,F7,F82

pre-commit-check:
	pre-commit run --all-files

build:
	$(PYTHON) -m build

install-wheel:
	$(PYTHON) -m pip install --force-reinstall --no-deps $(DIST_DIR)/*.whl

validate-wheel:
	$(PYTHON) tools/ci/check_installed_wheel.py --expected-version $(EXPECTED_VERSION)

test-ci:
	pytest tests/ci -q

smoke: test-ci

triton-smoke:
	FLAGSPARSE_TRITON_SMOKE=1 pytest tests/ci -q

gpu-env-check:
	$(PYTHON) tools/ci/check_gpu_environment.py --require-cuda

gpu-benchmark: gpu-env-check
	$(PYTHON) run_flagsparse_performance.py --ops gather,scatter,spmv_csr,spmm_csr --results-dir benchmark_results

release-check: ci-deps compile format-check lint lint-src pre-commit-check build install-wheel validate-wheel test-ci
	$(PYTHON) -m twine check $(DIST_DIR)/*.whl $(DIST_DIR)/*.tar.gz
	$(PYTHON) tools/ci/check_release_artifacts.py $(DIST_DIR)
	$(PYTHON) tools/ci/write_release_checksums.py $(DIST_DIR)
	$(PYTHON) tools/ci/write_release_checksums.py --verify $(DIST_DIR)

release: release-check

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
