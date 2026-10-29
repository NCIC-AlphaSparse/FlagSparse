"""Device-independent dispatch and resource-lifetime checks for native FP64 SpMM."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip(
    "torch", reason="tests/ci runs on a CPU-only runner without torch"
)

native = pytest.importorskip("flagsparse.sparse_operations._ascend_fp64_spmm")


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.complex64]
)
def test_cpu_inputs_do_not_load_acl(monkeypatch, dtype):
    def fail():
        pytest.fail("CPU dispatch unexpectedly loaded ACL")

    monkeypatch.setattr(native, "_runtime", fail)
    values = torch.ones(1, dtype=dtype)
    indices = torch.zeros(1, dtype=torch.int32)
    assert native.prepare_fp64_spmm(values, indices, indices, (1, 1)) is None
    assert (
        native.run_fp64_spmm_once(
            values, indices, indices, values.reshape(1, 1), (1, 1)
        )
        is None
    )


def test_fp64_route_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_ASCEND_FP64_SPMM", "silently_lower_precision")
    with pytest.raises(ValueError, match="auto, native or scatter"):
        native._mode()


class _ACL:
    def __init__(self):
        self.next_handle = 0
        self.descriptors = set()
        self.buffers = set()
        self.attributes = set()
        self.op = SimpleNamespace(
            create_attr=self.create_attr,
            destroy_attr=self.attributes.remove,
            set_attr_bool=lambda *args: 0,
        )

    def _new(self, target):
        self.next_handle += 1
        target.add(self.next_handle)
        return self.next_handle

    def create_tensor_desc(self, *args):
        return self._new(self.descriptors)

    def destroy_tensor_desc(self, descriptor):
        self.descriptors.remove(descriptor)

    def create_data_buffer(self, *args):
        return self._new(self.buffers)

    def destroy_data_buffer(self, buffer):
        self.buffers.remove(buffer)

    def create_attr(self):
        return self._new(self.attributes)

    def get_recent_err_msg(self):
        return "injected native failure"


def _make_plan(monkeypatch, acl, execute):
    monkeypatch.setattr(native, "_runtime", lambda: (acl, object(), execute))
    values = torch.tensor([1.25], dtype=torch.float64)
    indices = torch.zeros(1, dtype=torch.int32)
    return native._NativeFP64Spmm(values, indices, indices, (1, 1))


def test_preparation_failure_releases_partially_created_descriptors(monkeypatch):
    acl = _ACL()
    original = acl.create_data_buffer
    calls = []

    def create_buffer(*args):
        calls.append(args)
        return 0 if len(calls) == 2 else original(*args)

    monkeypatch.setattr(acl, "create_data_buffer", create_buffer)
    with pytest.raises(RuntimeError, match="create_data_buffer failed"):
        _make_plan(monkeypatch, acl, lambda *args: 0)
    assert not acl.descriptors and not acl.buffers and not acl.attributes


@pytest.mark.parametrize("status", [0, 42])
def test_per_call_buffers_are_released_after_stream_completion(monkeypatch, status):
    acl = _ACL()
    events = []
    stream = SimpleNamespace(npu_stream=5, synchronize=lambda: events.append("sync"))
    namespace = SimpleNamespace(
        device=lambda device: nullcontext(), current_stream=lambda device: stream
    )
    monkeypatch.setattr(native.torch, "npu", namespace, raising=False)
    plan = _make_plan(monkeypatch, acl, lambda *args: status)
    fixed_descriptors = set(acl.descriptors)
    fixed_buffers = set(acl.buffers)
    if status:
        with pytest.raises(RuntimeError, match="injected native failure"):
            plan.run(torch.ones((1, 2), dtype=torch.float64))
    else:
        result = plan.run(torch.ones((1, 2), dtype=torch.float64))
        assert result.shape == (1, 2) and result.dtype == torch.float64
    assert events == ["sync"]
    assert acl.descriptors == fixed_descriptors and acl.buffers == fixed_buffers
    plan.close()
    plan.close()
    assert not acl.descriptors and not acl.buffers and not acl.attributes


def test_prepared_run_refuses_a_plan_for_the_other_orientation():
    """A plan built for A must never serve a prepared describing A^T.

    Both prepared paths materialize the transpose before building the plan, so
    this cannot happen today; on a square matrix a regression there would
    return A @ B for A^T @ B instead of raising, so the mismatch is rejected
    explicitly.
    """
    plan = SimpleNamespace(shape=(4, 7))
    prepared = SimpleNamespace(
        ascend_fp64_plan=plan,
        n_rows=7,
        n_cols=4,
        data=torch.ones(1, dtype=torch.float64),
    )
    dense = torch.ones((4, 2), dtype=torch.float64)

    assert native.run_fp64_spmm(prepared, dense, "row") is None


def test_prepared_run_accepts_a_matching_plan():
    calls = []
    plan = SimpleNamespace(
        shape=(7, 4), run=lambda dense: calls.append(dense) or torch.zeros((7, 2))
    )
    prepared = SimpleNamespace(
        ascend_fp64_plan=plan,
        n_rows=7,
        n_cols=4,
        data=torch.ones(1, dtype=torch.float64),
    )
    dense = torch.ones((4, 2), dtype=torch.float64)

    result = native.run_fp64_spmm(prepared, dense, "row")

    assert result is not None
    assert len(calls) == 1
    assert result[1]["execution_path"] == "native"
