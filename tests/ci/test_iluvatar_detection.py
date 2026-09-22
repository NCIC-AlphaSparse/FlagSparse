# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar CoreX is detected from its own signals, and never claims NVIDIA.

CoreX is CUDA-compatible: torch.version.cuda is set and there is no plugin or
namespace, so `_detect_iluvatar_runtime()` has three weaker signals -- the torch
build's `+corex` version tag, the CoreX SDK environment, and the device name.
These tests pin both directions: each signal ALONE selects Iluvatar, and a plain
NVIDIA box -- the machine these tests actually run on -- is left alone.

Why three rather than one: measured 2026-09-22 on a BI-V100 box, the vendor
container's torch reports a bare "2.10.0" with no `+corex` tag, and CUDA
initialization failed there (error 803, container CoreX newer than the host
driver), so the device name could not be read either. COREX_HOME is the signal
that survives both -- and every test here clears it, because a CI machine with
the SDK installed would otherwise see every case as Iluvatar.
"""

import pytest

pytest.importorskip("torch", reason="tests/ci runs on a CPU-only runner without torch")

from flagsparse.sparse_operations import _common  # noqa: E402


class _Props:
    def __init__(self, name):
        self.name = name


def _detect(monkeypatch, version, device_name):
    monkeypatch.delenv("FLAGSPARSE_BACKEND", raising=False)
    for env in ("COREX_HOME", "COREX_PATH"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(_common.torch, "__version__", version)
    monkeypatch.setattr(
        _common.torch.cuda, "is_available", lambda: device_name is not None
    )
    monkeypatch.setattr(
        _common.torch.cuda,
        "get_device_properties",
        lambda device=0: _Props(device_name),
    )
    return _common._detect_iluvatar_runtime()


def test_corex_torch_build_selects_iluvatar(monkeypatch):
    assert _detect(monkeypatch, "2.1.1+corex.4.1.2", None)


@pytest.mark.parametrize("name", ["Iluvatar BI-V150", "BI-V150", "Iluvatar MR-V100"])
def test_iluvatar_device_name_selects_iluvatar(monkeypatch, name):
    assert _detect(monkeypatch, "2.1.1", name)


@pytest.mark.parametrize(
    "name",
    ["NVIDIA A100-SXM4-80GB", "NVIDIA H800", "MetaX C550", "Tesla V100-PCIE-32GB"],
)
def test_other_cuda_compatible_devices_are_not_iluvatar(monkeypatch, name):
    assert not _detect(monkeypatch, "2.5.1+cu124", name)


@pytest.mark.parametrize("env", ["COREX_HOME", "COREX_PATH"])
def test_corex_sdk_environment_selects_iluvatar(monkeypatch, env):
    """The signal that survives a bare torch version and an unusable runtime."""
    monkeypatch.delenv("FLAGSPARSE_BACKEND", raising=False)
    monkeypatch.setenv(env, "/usr/local/corex")
    monkeypatch.setattr(_common.torch, "__version__", "2.10.0")
    monkeypatch.setattr(_common.torch.cuda, "is_available", lambda: False)
    assert _common._detect_iluvatar_runtime()


def test_bare_torch_version_alone_is_not_enough(monkeypatch):
    """A CoreX container's torch can report a plain "2.10.0" -- measured, BI-V100.

    Without the SDK environment or a readable device name there is nothing left
    to detect, and the backend must NOT be claimed on a guess: an explicit
    FLAGSPARSE_BACKEND=iluvatar is the documented way in.
    """
    assert not _detect(monkeypatch, "2.10.0", None)


def test_override_wins_in_both_directions(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_BACKEND", "iluvatar")
    assert _common._detect_iluvatar_runtime()
    monkeypatch.setenv("FLAGSPARSE_BACKEND", "cuda")
    monkeypatch.setattr(_common.torch, "__version__", "2.1.1+corex.4.1.2")
    monkeypatch.setenv("COREX_HOME", "/usr/local/corex")
    assert not _common._detect_iluvatar_runtime()
