PYTHON ?= python
DIST_DIR ?= dist
EXPECTED_VERSION ?= 1.0.0

.PHONY: help ci check ci-deps compile lint build install-wheel validate-wheel test-ci smoke triton-smoke triton-deps release-check release clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make ci             - run the default CPU-only CI pipeline' \
		'  make check          - alias for make ci' \
		'  make smoke          - alias for the CPU smoke test stage' \
		'  make release        - alias for make release-check' \
		'  make release-check  - build, validate, and checksum release artifacts' \
		'  make triton-deps    - install the opt-in triton smoke dependency bundle' \
		'  make triton-smoke   - opt-in triton-dependent smoke tests' \
		'  make help           - show this list'

ci: ci-deps compile lint build install-wheel validate-wheel test-ci

check: ci

ci-deps:
	$(PYTHON) -m pip install --upgrade -r tools/ci/requirements-ci.lock.txt

triton-deps:
	$(PYTHON) -m pip install --upgrade -r tools/ci/requirements-triton-smoke.lock.txt

compile:
	$(PYTHON) -m compileall src tests tools

lint:
	ruff check tests/ci tools/ci

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

release-check: ci-deps build install-wheel validate-wheel test-ci
	$(PYTHON) -m twine check $(DIST_DIR)/*.whl $(DIST_DIR)/*.tar.gz
	$(PYTHON) tools/ci/check_release_artifacts.py $(DIST_DIR)
	$(PYTHON) tools/ci/write_release_checksums.py $(DIST_DIR)
	$(PYTHON) tools/ci/write_release_checksums.py --verify $(DIST_DIR)

release: release-check

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
