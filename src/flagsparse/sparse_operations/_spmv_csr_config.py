"""CPU-only CSR SpMV algorithm contracts and backend launch profiles."""

from copy import deepcopy
from dataclasses import dataclass

IMPLEMENTATION_VERSION = 2
NEW_ALGORITHMS = ("row_tile", "row_vector", "row_split_reduce", "row_adaptive_split")
LEGACY_ALGORITHMS = ("legacy_rowpar", "legacy_segbin", "legacy_bucket_vector")
ALGORITHMS = LEGACY_ALGORITHMS + NEW_ALGORITHMS
BACKENDS = ("cuda", "rocm", "metax", "mthreads", "ascend")
VALUE_DTYPES = ("float16", "bfloat16", "float32", "float64", "complex64", "complex128")


@dataclass(frozen=True)
class BackendCaps:
    backend: str
    arch: str = "unknown"
    target: str = "unknown"
    subgroup_width: int = 0
    max_threads_per_block: int = 0
    fp64: bool = False
    int64: bool = False
    reduction: bool = False


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def set_known_backends(backends):
    """Synchronize legacy-route backend names with the shared runtime registry."""
    global BACKENDS
    normalized = tuple(dict.fromkeys(str(name) for name in backends if name))
    if not normalized:
        raise ValueError("CSR SpMV backend registry must not be empty")
    BACKENDS = normalized


def normalize_alg(alg):
    name = "auto" if alg is None else str(alg).strip().lower()
    if name == "base":
        name = "auto"
    if name not in ("auto",) + ALGORITHMS:
        raise ValueError(
            f"unknown CSR SpMV algorithm {name!r}; expected auto or {ALGORITHMS}"
        )
    return name


def compute_dtype(alg, dtype):
    name = dtype_name(dtype)
    if name in ("float16", "bfloat16"):
        return "float32"
    if name == "float32" and alg in NEW_ALGORITHMS + ("legacy_rowpar",):
        return "float64"
    return name


def algorithm_spec(alg):
    name = normalize_alg(alg)
    if name == "auto":
        raise ValueError("auto is a selection policy, not a concrete algorithm")
    new = name in NEW_ALGORITHMS
    return {
        "name": name,
        "implementation_version": IMPLEMENTATION_VERSION,
        "ops": ("non", "trans", "conj"),
        "value_dtypes": VALUE_DTYPES,
        "index_dtypes": (
            ("int32",) if name == "legacy_bucket_vector" else ("int32", "int64")
        ),
        "indptr_dtypes": ("int32", "int64"),
        "backends": ("cuda", "rocm") if new else BACKENDS,
        "runtime_process": True,
        "runtime_process_ops": (
            ("non", "trans", "conj")
            if name
            in ("row_split_reduce", "row_adaptive_split", "legacy_bucket_vector")
            else ("trans", "conj")
        ),
        "transpose_strategy": "per_run_csr_rebuild",
        "cross_call_plan_cache": False,
        "capability_requirements": ("int64", "reduction") if new else (),
        "compute_dtype": "input_dependent",
        "compute_dtype_by_input": {dt: compute_dtype(name, dt) for dt in VALUE_DTYPES},
    }


def list_algorithms(op=None, dtype=None, backend=None):
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    result = []
    for name in ALGORITHMS:
        spec = algorithm_spec(name)
        if op is not None and op not in spec["ops"]:
            continue
        if dtype is not None and dtype_name(dtype) not in spec["value_dtypes"]:
            continue
        if backend is not None and backend not in spec["backends"]:
            continue
        result.append(name)
    return tuple(result)


def validate_support(alg, op, dtype, indices_dtype, indptr_dtype, caps):
    spec = algorithm_spec(alg)
    for value, key in (
        (op, "ops"),
        (dtype_name(dtype), "value_dtypes"),
        (dtype_name(indices_dtype), "index_dtypes"),
        (dtype_name(indptr_dtype), "indptr_dtypes"),
        (caps.backend, "backends"),
    ):
        if value not in spec[key]:
            raise NotImplementedError(f"CSR SpMV {alg} does not support {key}={value}")
    if alg in NEW_ALGORITHMS and not (
        caps.int64
        and caps.reduction
        and caps.subgroup_width in (32, 64)
        and caps.max_threads_per_block > 0
    ):
        raise NotImplementedError(
            f"CSR SpMV {alg}: unverified capabilities for {caps.backend}/{caps.arch}"
        )
    if (
        alg in NEW_ALGORITHMS
        and compute_dtype(alg, dtype) in ("float64", "complex128")
        and not caps.fp64
    ):
        raise NotImplementedError(
            f"CSR SpMV {alg}: FP64 capability required for {dtype}"
        )


def _defaults(caps):
    return {
        "short_row_threshold": 32,
        "split_row_threshold": 1024,
        "process": {"block_size": 256, "num_warps": 4},
        "row_tile": {
            "rows_per_program": 2 * caps.subgroup_width // 8,
            "lanes_per_row": 8,
            "num_warps": 2,
            "loop_num_stages": 1,
        },
        "row_vector": {"block_nnz": 128, "num_warps": 2, "loop_num_stages": 1},
        "row_split_reduce": {
            "segment_nnz": 1024,
            "block_nnz": 128,
            "num_warps": 2,
            "reduce_block_size": 256,
            "reduce_num_warps": 4,
            "loop_num_stages": 1,
        },
    }


# Architecture-specific overrides may be added only with a recorded benchmark.
ARCH_PROFILES = {}


def _merge(base, overrides):
    if not isinstance(overrides, dict):
        raise TypeError("CSR SpMV config must be a dictionary")
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f"unknown CSR SpMV config field {key!r}")
        if isinstance(base[key], dict):
            _merge(base[key], value)
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"config {key} must be an integer")
            base[key] = value


def _validate_config(resolved, caps):
    if (
        resolved["short_row_threshold"] < 0
        or resolved["split_row_threshold"] <= resolved["short_row_threshold"]
    ):
        raise ValueError("require 0 <= short_row_threshold < split_row_threshold")
    for section in ("row_tile", "row_vector", "row_split_reduce", "process"):
        for key, value in resolved[section].items():
            if value <= 0:
                raise ValueError(f"{section}.{key} must be positive")
            if key != "loop_num_stages" and value & (value - 1):
                raise ValueError(f"{section}.{key} must be a power of two")
            if key.endswith("num_warps"):
                if (
                    value not in (1, 2, 4, 8, 16, 32)
                    or value * caps.subgroup_width > caps.max_threads_per_block
                ):
                    raise ValueError(f"{section}.{key} exceeds backend launch limits")
                if caps.backend == "rocm" and value > 8:
                    raise ValueError("ROCm CSR SpMV profile limits num_warps to 8")
    tile = resolved["row_tile"]
    if tile["lanes_per_row"] > caps.subgroup_width:
        raise ValueError("lanes_per_row exceeds subgroup width")
    if tile["rows_per_program"] * tile["lanes_per_row"] > 65536:
        raise ValueError("row_tile exceeds the supported tile size")
    split = resolved["row_split_reduce"]
    if split["reduce_block_size"] < 2:
        raise ValueError("reduce_block_size must be >= 2 to make reduction progress")
    if (
        max(
            split["block_nnz"],
            split["reduce_block_size"],
            resolved["row_vector"]["block_nnz"],
            resolved["process"]["block_size"],
        )
        > 65536
    ):
        raise ValueError("block size exceeds the supported tile size")


def resolve_config(alg, caps, config=None, *, return_rejections=False):
    if config is not None and not isinstance(config, dict):
        raise TypeError("CSR SpMV config must be a dictionary")
    if alg not in NEW_ALGORITHMS:
        if config:
            raise ValueError(
                "legacy CSR SpMV routes do not accept new-algorithm config"
            )
        return ({}, "legacy", []) if return_rejections else ({}, "legacy")
    if (
        caps.backend not in ("cuda", "rocm")
        or caps.subgroup_width not in (32, 64)
        or not (caps.int64 and caps.reduction and caps.max_threads_per_block > 0)
    ):
        raise NotImplementedError(
            f"no verified CSR SpMV profile for {caps.backend}/{caps.arch}"
        )
    candidates = []
    profile = ARCH_PROFILES.get((caps.backend, caps.arch))
    if profile is not None:
        candidates.append((f"{caps.backend}:{caps.arch}", profile))
    candidates.append((f"{caps.backend}:conservative-v1", {}))
    rejections = []
    resolved = None
    source = None
    for name, overrides in candidates:
        candidate = _defaults(caps)
        try:
            _merge(candidate, overrides)
            _validate_config(candidate, caps)
        except (ValueError, TypeError) as exc:
            rejections.append({"profile": name, "reason": str(exc)})
            continue
        resolved, source = candidate, name
        break
    if resolved is None:
        # Internal candidate may shrink launch groups to available resources.
        # Explicit user values below are always validated without modification.
        resolved = _defaults(caps)
        available = caps.max_threads_per_block // caps.subgroup_width
        warps = max((w for w in (1, 2, 4, 8) if w <= available), default=0)
        if not warps:
            raise NotImplementedError(f"no legal launch profile: {rejections}")
        for section in ("row_tile", "row_vector", "row_split_reduce", "process"):
            resolved[section]["num_warps"] = min(resolved[section]["num_warps"], warps)
        resolved["row_split_reduce"]["reduce_num_warps"] = min(4, warps)
        _validate_config(resolved, caps)
        source = f"{caps.backend}:resource-conservative-v1"
    if config is not None:
        _merge(resolved, config)
        _validate_config(resolved, caps)
        source = "explicit"
    result = (deepcopy(resolved), source)
    return (*result, rejections) if return_rejections else result


def assert_route_match(prepared_alg, prepared_config, alg=None, config=None, caps=None):
    if alg is not None and normalize_alg(alg) not in ("auto", prepared_alg):
        raise ValueError(f"alg does not match prepared.alg={prepared_alg}")
    if config is not None:
        candidate, _ = resolve_config(prepared_alg, caps, config)
        if candidate != prepared_config:
            raise ValueError("config does not match prepared.config")


def is_index_compatibility_error(exc):
    """Never reinterpret OOM, invalid memory access, or arbitrary failures as index errors."""
    text = str(exc).lower()
    return (
        isinstance(exc, (RuntimeError, TypeError, NotImplementedError))
        and any(
            token in text for token in ("int64", "i64", "64-bit index", "64 bit index")
        )
        and any(
            token in text
            for token in (
                "unsupported",
                "not supported",
                "cannot lower",
                "not implemented",
            )
        )
    )
