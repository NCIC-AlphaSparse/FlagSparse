"""CSR route policy tests runnable with stdlib unittest, without GPU imports."""

import ast
import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/flagsparse/sparse_operations"
spec = importlib.util.spec_from_file_location(
    "_csr_policy_test_config", SOURCE / "_spmv_csr_config.py"
)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class ConfigPolicy(unittest.TestCase):
    def setUp(self):
        self.cuda = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, True, True, True)
        self.hip = policy.BackendCaps(
            "rocm", "gfx90a", "hip", 64, 1024, True, True, True
        )

    def test_registry_support(self):
        self.assertEqual(len(policy.list_algorithms()), 7)
        for alg in policy.NEW_ALGORITHMS:
            self.assertIn(alg, policy.list_algorithms("non", "torch.float32", "rocm"))
            self.assertIn(alg, policy.list_algorithms("trans", "float32", "cuda"))
            self.assertIn(alg, policy.list_algorithms("non", "complex64", "cuda"))
            for backend in ("metax", "mthreads", "ascend"):
                self.assertNotIn(alg, policy.list_algorithms(backend=backend))
        with self.assertRaises(ValueError):
            policy.algorithm_spec("auto")
        with self.assertRaises(ValueError):
            policy.normalize_alg("compare")

    def test_all_dtype_ops_and_precision_requirements(self):
        no_fp64 = policy.BackendCaps("cuda", "80", "cuda", 32, 1024, False, True, True)
        for alg in policy.ALGORITHMS:
            for dtype in policy.VALUE_DTYPES:
                for op in ("non", "trans", "conj"):
                    policy.validate_support(alg, op, dtype, "int32", "int64", self.cuda)
            spec = policy.algorithm_spec(alg)
            self.assertEqual(spec["compute_dtype_by_input"]["complex64"], "complex64")
            self.assertEqual(spec["compute_dtype_by_input"]["float16"], "float32")
        for alg in policy.NEW_ALGORITHMS:
            policy.resolve_config(alg, no_fp64)
            for dtype in ("float16", "bfloat16", "complex64"):
                policy.validate_support(alg, "conj", dtype, "int64", "int32", no_fp64)
            for dtype in ("float32", "float64", "complex128"):
                with self.assertRaisesRegex(NotImplementedError, "FP64"):
                    policy.validate_support(
                        alg, "trans", dtype, "int64", "int32", no_fp64
                    )

    def test_backend_profiles_and_mixed_indices(self):
        for caps, rows in ((self.cuda, 8), (self.hip, 16)):
            config, source = policy.resolve_config("row_adaptive_split", caps)
            self.assertEqual(config["row_tile"]["rows_per_program"], rows)
            self.assertEqual(config["row_split_reduce"]["reduce_block_size"], 256)
            self.assertIn(caps.backend, source)
            for col in ("int32", "int64"):
                for ptr in ("int32", "int64"):
                    policy.validate_support(
                        "row_tile", "non", "float64", col, ptr, caps
                    )

    def test_unknown_capabilities_fail_closed(self):
        with self.assertRaises(NotImplementedError):
            policy.validate_support(
                "row_vector",
                "non",
                "float32",
                "int32",
                "int64",
                policy.BackendCaps("cuda"),
            )
        for backend in ("metax", "mthreads", "ascend"):
            with self.assertRaises(NotImplementedError):
                policy.resolve_config("row_tile", policy.BackendCaps(backend))
            policy.validate_support(
                "legacy_rowpar",
                "conj",
                "complex128",
                "int64",
                "int32",
                policy.BackendCaps(backend),
            )

    def test_invalid_explicit_config(self):
        for bad in (
            {"bad": 1},
            {"row_tile": {"lanes_per_row": 3}},
            {"row_vector": {"num_warps": 64}},
            {"short_row_threshold": 1024},
            {"row_split_reduce": {"reduce_block_size": 1}},
            {"row_vector": {"block_nnz": True}},
            {"row_tile": {"lanes_per_row": 64}},
        ):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                policy.resolve_config("row_tile", self.cuda, bad)
        with self.assertRaises(ValueError):
            policy.resolve_config(
                "row_tile", self.hip, {"row_vector": {"num_warps": 16}}
            )

    def test_configuration_is_copied_and_fixed(self):
        override = {"row_vector": {"block_nnz": 64}}
        config, _ = policy.resolve_config("row_vector", self.cuda, override)
        override["row_vector"]["block_nnz"] = 256
        self.assertEqual(config["row_vector"]["block_nnz"], 64)
        policy.assert_route_match(
            "row_vector",
            config,
            config={"row_vector": {"block_nnz": 64}},
            caps=self.cuda,
        )
        with self.assertRaises(ValueError):
            policy.assert_route_match("row_vector", config, "row_tile")
        with self.assertRaises(ValueError):
            policy.assert_route_match(
                "row_vector", config, config=override, caps=self.cuda
            )

    def test_fallback_is_narrow(self):
        self.assertTrue(
            policy.is_index_compatibility_error(
                RuntimeError("unsupported int64 indices")
            )
        )
        for message in (
            "CUDA out of memory",
            "illegal memory access int64",
            "int64 launch failed",
            "unsupported fp64",
        ):
            self.assertFalse(policy.is_index_compatibility_error(RuntimeError(message)))

    def test_bad_builtin_profile_is_rejected_and_recorded(self):
        with patch.dict(
            policy.ARCH_PROFILES, {("cuda", "80"): {"row_vector": {"num_warps": 128}}}
        ):
            config, source, rejections = policy.resolve_config(
                "row_vector", self.cuda, return_rejections=True
            )
        self.assertEqual(config["row_vector"]["num_warps"], 2)
        self.assertEqual(source, "cuda:conservative-v1")
        self.assertEqual(rejections[0]["profile"], "cuda:80")
        limited = policy.BackendCaps("rocm", "test", "hip", 64, 64, True, True, True)
        config, source = policy.resolve_config("row_tile", limited)
        self.assertEqual(config["row_split_reduce"]["reduce_num_warps"], 1)
        self.assertEqual(source, "rocm:resource-conservative-v1")
        with self.assertRaises(ValueError):
            policy.resolve_config("row_tile", limited, {"row_vector": {"num_warps": 2}})


def runtime_namespace():
    """Execute the real Python orchestration functions with an instrumented backend.

    Only imports and JIT kernels are excluded; decisions/timing are production code.
    """
    names = {
        "flagsparse_spmv_csr_run",
        "flagsparse_spmv_csr",
        "_spmv_phase",
        "_execute_spmv_route",
        "_execute_spmv_route_with_fallback",
        "_spmv_execution_matrix",
    }
    tree = ast.parse((SOURCE / "spmv_csr.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    class Prepared:
        def __init__(self):
            self.alg = self.alg_requested = "row_adaptive_split"
            self.op, self.transpose, self.n_rows = 0, False, 3
            self.shape = (3, 4)
            self.backend_caps = policy.BackendCaps(
                "cuda", "80", "cuda", 32, 1024, True, True, True
            )
            self.config, self.config_source = policy.resolve_config(
                self.alg, self.backend_caps
            )
            self.config_rejections = []
            self.data = types.SimpleNamespace(
                dtype="float32", device="cuda", is_conj=lambda: False
            )
            self.kernel_indices = types.SimpleNamespace(dtype="int64")
            self.kernel_indptr = types.SimpleNamespace(dtype="int32")
            self._baseline_compute_dtype = "float64"
            self.index_fallback_policy = "auto"
            self.index_fallback_applied, self.index_fallback_reason = False, None

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 7.0

    package = types.ModuleType("_csr_policy_runtime")
    kernels = types.SimpleNamespace(
        build_plan=Mock(side_effect=lambda *args: object()),
        compute=Mock(side_effect=lambda p, x, y, *args: y),
    )
    package._spmv_csr_kernels = kernels
    sys.modules[package.__name__] = package
    env = {
        "__package__": package.__name__,
        "_csr_config": policy,
        "copy": copy,
        "deepcopy": deepcopy,
        "asdict": asdict,
        "PreparedCsrSpmv": Prepared,
        "SPMV_CSR_NEW_ALGORITHMS": policy.NEW_ALGORITHMS,
        "torch": types.SimpleNamespace(empty=Mock(return_value=object())),
        "_ACCEL": types.SimpleNamespace(Event=Event, synchronize=lambda: None),
        "_spmv_device_context": lambda device: nullcontext(),
        "_validate_spmv_x": lambda x, p: types.SimpleNamespace(
            resolve_conj=lambda: types.SimpleNamespace(contiguous=lambda: x)
        ),
        "_spmv_check_output": lambda *a: None,
        "get_spmv_csr_algorithm_spec": policy.algorithm_spec,
        "_normalize_spmv_op": lambda op, transpose=False: (
            {"non": 0, "trans": 1, "conj": 2}.get(op, op)
            if op is not None
            else int(transpose)
        ),
        "_spmv_op_transposes": lambda op: op in (1, 2),
        "_spmv_csr_default_backend": lambda: "rowpar",
        "_spmv_uses_int64_indices": lambda p: True,
        "_is_ascend_runtime": lambda: False,
    }
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(SOURCE / "spmv_csr.py"), "exec"
        ),
        env,
    )
    return env, Prepared(), kernels


class RuntimePolicy(unittest.TestCase):
    def test_each_run_builds_plan_and_diagnostics_are_separate(self):
        env, p, kernels = runtime_namespace()
        run = env["flagsparse_spmv_csr_run"]
        _, plain = run(p, object(), return_meta=True)
        self.assertEqual(kernels.build_plan.call_count, 1)
        _, detailed = run(p, object(), return_meta=True, timing=True)
        self.assertEqual(kernels.build_plan.call_count, 3)
        self.assertEqual(kernels.compute.call_count, 3)
        self.assertEqual(plain["ms"], detailed["ms"])
        self.assertEqual(
            detailed["ms"], detailed["process_cpu_ms"] + detailed["gpu_ms"]
        )
        self.assertNotEqual(
            detailed["ms"], detailed["process_gpu_ms"] + detailed["compute_ms"]
        )
        self.assertEqual(detailed["compute_dtype"], "float64")
        self.assertEqual(detailed["alg_resolved"], p.alg)

    def test_row_routes_have_no_plan_and_out_is_forwarded(self):
        env, p, kernels = runtime_namespace()
        p.alg = p.alg_requested = "row_tile"
        out = object()
        result, meta = env["flagsparse_spmv_csr_run"](
            p, object(), out=out, return_meta=True, timing=True
        )
        self.assertIs(result, out)
        kernels.build_plan.assert_not_called()
        self.assertEqual(meta["process_gpu_ms"], 0)

    def test_prepared_and_legacy_argument_conflicts(self):
        env, p, _ = runtime_namespace()
        for kw in (
            {"op": "trans"},
            {"alg": "row_tile"},
            {"config": {"short_row_threshold": 16}},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                env["flagsparse_spmv_csr_run"](p, object(), **kw)
        for kw in (
            {"alg": "row_tile", "use_opt": True},
            {"use_opt": False},
            {"transpose": True},
            {"shape": (9, 9)},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                env["flagsparse_spmv_csr"](prepared=p, x=object(), **kw)

    def test_unrelated_errors_are_never_retried(self):
        env, p, kernels = runtime_namespace()
        kernels.compute.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            env["flagsparse_spmv_csr_run"](p, object())
        self.assertEqual(kernels.compute.call_count, 1)


class IntegrationPolicy(unittest.TestCase):
    def test_cli_aliases_and_csv_fields(self):
        from tests.test_spmv_csr import FIELDS, parser

        args = parser().parse_args(
            [
                "--synthetic",
                "--dtype",
                "float64",
                "--index-dtype",
                "int64",
                "--csv",
                "result.csv",
                "--no-cusparse",
                "--timing",
            ]
        )
        self.assertEqual(args.dtypes, "float64")
        self.assertEqual(args.index_dtypes, "int64")
        self.assertEqual(args.csv_csr, "result.csv")
        self.assertTrue(args.no_vendor and args.timing)
        self.assertEqual(len(FIELDS), len(set(FIELDS)))
        for key in (
            "vendor_ms",
            "speedup_vs_vendor",
            "config",
            "alg_resolved",
            "ms",
            "gpu_ms",
        ):
            self.assertIn(key, FIELDS)
        self.assertNotIn("cu_ms", FIELDS)

    def test_single_accuracy_module_and_static_support(self):
        import ops_support

        paths = list((ROOT / "tests").rglob("test_spmv_csr_accuracy.py"))
        self.assertEqual(paths, [ROOT / "tests/pytest/test_spmv_csr_accuracy.py"])
        rows = ops_support.build_rows(SOURCE)
        for alg in policy.NEW_ALGORITHMS:
            cases = [row for row in rows if row["route"] == alg]
            self.assertEqual(len(cases), 36)
            self.assertEqual({row["op"] for row in cases}, {"non", "trans", "conj"})
            self.assertEqual({row["status"] for row in cases}, {"UNVERIFIED"})

    def test_runner_preserves_algorithm_identity_and_latency(self):
        import run_flagsparse_pytest as runner

        row = dict(
            matrix="same.mtx",
            dtype="float32",
            index_dtype="int32",
            indptr_dtype="int64",
            alg_resolved="row_tile",
            op="non",
            vendor_ms="3",
            ms="2",
            speedup_vs_vendor="1.5",
        )
        detail = runner._benchmark_json_detail(row, 0)
        self.assertEqual(detail["latency_base"], 3)
        self.assertEqual(detail["latency"], 2)
        self.assertEqual(detail["speedup"], 1.5)
        other = runner._row_shape(dict(row, alg_resolved="row_vector"), 1)
        self.assertNotEqual(detail["shape_detail"], other)


if __name__ == "__main__":
    unittest.main()
