# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""run_flagsparse_delivery.py: one command line for the delivery run on every backend.

The per-backend commands in docs/debug/ had drifted apart (three of them passed
--ops instead of --delivery-only, which skipped the delivery narrowing). The
profiles that replace them are checked here against the runners they drive, so a
flag a runner does not have, an operator the registry does not deliver, or an
--h800-reference with no card to rescale to fails before anyone is on a vendor box.
"""

import importlib.util
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC_BACKENDS = ("cuda", "rocm", "metax", "mthreads", "iluvatar", "ascend")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry():
    return _load("_delivery_entry", ROOT / "run_flagsparse_delivery.py")


def _flags(script):
    out = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(re.findall(r"--[a-z0-9][a-z0-9-]*", out))


def _command(entry, backend):
    return entry.build_command(backend, entry.PROFILES[backend], "results_x", "0")


def test_every_documented_backend_has_a_profile():
    assert set(_entry().PROFILES) == set(DOC_BACKENDS)


def test_every_generated_flag_exists_in_the_runner_it_is_passed_to():
    entry = _entry()
    known = {
        "run_flagsparse_pytest.py": _flags("run_flagsparse_pytest.py"),
        "run_flagsparse_split_delivery.py": _flags("run_flagsparse_split_delivery.py"),
    }
    for backend in DOC_BACKENDS:
        cmd = _command(entry, backend)
        script = cmd[2]
        flags = {token.split("=", 1)[0] for token in cmd[3:] if token.startswith("--")}
        assert flags <= known[script], (backend, sorted(flags - known[script]))


def test_pytest_profiles_take_the_delivery_scope_never_an_ops_list():
    entry = _entry()
    for backend in DOC_BACKENDS:
        cmd = _command(entry, backend)
        assert "--ops" not in cmd, backend
        assert cmd[cmd.index("--benchmark-input") + 1] == "tests/data", backend
        if entry.PROFILES[backend].runner == "pytest":
            assert cmd[2] == "run_flagsparse_pytest.py", backend
            assert "--delivery-only" in cmd, backend
            assert cmd[cmd.index("--benchmark-warmup") + 1] == "5", backend
            assert cmd[cmd.index("--benchmark-iters") + 1] == "20", backend
    # MUSA's performance half is the C API's (torch.sparse has no matmul there).
    split = _command(entry, "mthreads")
    assert split[2] == "run_flagsparse_split_delivery.py"
    assert split[split.index("--backend") + 1] == "mthreads"


def test_benchmark_overrides_name_delivered_operators():
    # The runner rejects --op-benchmark-args for an operator --delivery-only did
    # not select, so a stale name would stop the whole run at startup.
    from tools.delivery_variants import load_delivery_variants

    delivered = {
        variant["operator"]
        for variant in load_delivery_variants(ROOT / "conf" / "operators.yaml")
    }
    for backend, profile in _entry().PROFILES.items():
        for value in profile.op_benchmark_args:
            op, args = value.split("=", 1)
            assert op in delivered, (backend, op)
            assert shlex.split(args), (backend, value)
        if profile.pytest_args:
            assert shlex.split(profile.pytest_args), backend


def test_h800_reference_only_where_a_card_is_known():
    # Without a card the runners exit at startup ("no card known for backend").
    bound = _load("_baseline_bound_for_entry", ROOT / "tools" / "baseline_bound.py")
    for backend, profile in _entry().PROFILES.items():
        if profile.h800_reference:
            assert backend in bound._BACKEND_CARDS, backend
    # CUDA's denominator is cuSPARSE itself; DCU leaves it off by choice.
    assert not _entry().PROFILES["cuda"].h800_reference
    assert not _entry().PROFILES["rocm"].h800_reference


def test_profiles_keep_the_retest_commands_backend_settings():
    profiles = _entry().PROFILES
    iluvatar = profiles["iluvatar"]
    assert iluvatar.env["FLAGSPARSE_BACKEND"] == "iluvatar"
    assert iluvatar.env["FLAGSPARSE_ILUVATAR_VENDOR"] == "cupy_cusparse"
    # BI-V150's fp64 H2D returns zeros: no fp64 case may be selected.
    assert "not (double or float64" in iluvatar.pytest_args
    assert all("float64" not in value for value in iluvatar.op_benchmark_args)
    assert profiles["ascend"].env == {
        "FLAGSPARSE_BACKEND": "ascend",
        "FLAGSPARSE_ASCEND_VENDOR": "torch",
    }
    assert profiles["metax"].op_benchmark_args == ("sddmm_csr=--no-cusparse",)
    assert profiles["metax"].timeout == 4500


def test_dry_run_prints_the_command_without_detecting_or_running(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "run_flagsparse_delivery.py",
            "--dry-run",
            "--backend",
            "iluvatar",
            "--results-dir",
            str(tmp_path / "new"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "backend=iluvatar runner=pytest" in out
    assert "--delivery-only" in out
    assert not (tmp_path / "new").exists()


def test_an_existing_results_dir_is_refused(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "run_flagsparse_delivery.py",
            "--dry-run",
            "--backend",
            "cuda",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "already exists" in proc.stderr
