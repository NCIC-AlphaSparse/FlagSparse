# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Both hip-python descriptor signatures, without a DCU or hip-python present.

hip-python 7.x returns the sparse descriptor; earlier releases write it through
an out-parameter. Callers keep their own descriptor unless a value comes back,
so a helper that returned a status here would replace the descriptor with it.
"""

import pytest

common = pytest.importorskip(
    "flagsparse.sparse_operations._common",
    reason="tests/ci runs on a CPU-only runner without torch",
)


class _Descriptor:
    pass


class _OutParamRef:
    """Stands in for the object a pre-7.x binding writes through."""

    def __init__(self):
        self.written = False


class _Binding7x:
    ARGC = {"csr": 10, "coo": 9, "csc": 10}

    @classmethod
    def _create(cls, kind, args):
        if len(args) != cls.ARGC[kind]:
            raise TypeError(f"expected {cls.ARGC[kind]} arguments, got {len(args)}")
        return 0, _Descriptor()

    @classmethod
    def hipsparseCreateCsr(cls, *args):
        return cls._create("csr", args)

    @classmethod
    def hipsparseCreateCoo(cls, *args):
        return cls._create("coo", args)

    @classmethod
    def hipsparseCreateCsc(cls, *args):
        return cls._create("csc", args)


class _BindingLegacy:
    ARGC = {"csr": 11, "coo": 10, "csc": 11}

    @classmethod
    def _create(cls, kind, args):
        if len(args) != cls.ARGC[kind]:
            raise TypeError(f"expected {cls.ARGC[kind]} arguments, got {len(args)}")
        args[0].written = True
        return 0

    @classmethod
    def hipsparseCreateCsr(cls, *args):
        return cls._create("csr", args)

    @classmethod
    def hipsparseCreateCoo(cls, *args):
        return cls._create("coo", args)

    @classmethod
    def hipsparseCreateCsc(cls, *args):
        return cls._create("csc", args)


def _call(kind, ref):
    args = (ref, 4, 4, 8, 10, 20, 30)
    tail = ("index_base", "value_type")
    if kind == "coo":
        return common._hipsparse_create_coo_descriptor(*args, "i32", *tail)
    creator = (
        common._hipsparse_create_csr_descriptor
        if kind == "csr"
        else common._hipsparse_create_csc_descriptor
    )
    return creator(*args, "i32", "i32", *tail)


@pytest.mark.parametrize("kind", ["csr", "csc", "coo"])
def test_seven_x_binding_returns_the_descriptor(monkeypatch, kind):
    monkeypatch.setattr(common, "hipsparse", _Binding7x)
    ref = _OutParamRef()
    descriptor = _call(kind, ref)
    assert isinstance(descriptor, _Descriptor)
    assert not ref.written


@pytest.mark.parametrize("kind", ["csr", "csc", "coo"])
def test_legacy_binding_keeps_the_out_parameter(monkeypatch, kind):
    # None is what makes callers keep their own descriptor: returning the status
    # instead would have them pass an int to hipsparseSpMM_bufferSize.
    monkeypatch.setattr(common, "hipsparse", _BindingLegacy)
    ref = _OutParamRef()
    assert _call(kind, ref) is None
    assert ref.written
