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

"""Run the pinned FlagOS scheduler with FlagSparse runtime metadata."""

import argparse
import os
import sys

_VENDOR_BACKENDS = {
    "nvidia": "cuda",
    "hygon": "rocm",
    "amd": "rocm",
    "metax": "metax",
    "mthreads": "mthreads",
    "ascend": "ascend",
    "kunlunxin": "xpu",
    "enflame": "gcu",
    "iluvatar": "iluvatar",
}


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--delivery", action="store_true", help="Test the delivery registry."
    )
    parser.add_argument(
        "--benchmark-input", help="Delivery MatrixMarket file or directory."
    )
    flags, args = parser.parse_known_args()
    if flags.benchmark_input and not flags.delivery:
        parser.error("--benchmark-input requires --delivery")
    sys.argv[1:] = args

    try:
        from flagos_optest import gpu_check, run_tests
    except ModuleNotFoundError as exc:
        if exc.name != "flagos_optest":
            raise
        raise SystemExit(
            "Install optest from tools/ci/requirements-optest.txt first."
        ) from exc

    if "-h" in args or "--help" in args:
        parser.print_help()
        return run_tests.main()

    vendor = os.environ.get("FLAGOS_GPU_VENDOR") or gpu_check.probe_vendor()
    backend = _VENDOR_BACKENDS.get(vendor)
    if backend is None:
        raise SystemExit(
            f"No FlagSparse backend for detected optest vendor {vendor!r}."
        )
    selected_backend = os.environ.setdefault("FLAGSPARSE_BACKEND", backend)
    if selected_backend != backend:
        raise SystemExit(
            f"Vendor {vendor} expects FLAGSPARSE_BACKEND={backend}, got {selected_backend!r}."
        )
    if backend == "xpu":
        os.environ.setdefault("FLAGTREE_BACKEND", "xpu")
        os.environ.setdefault("TRITON_BACKEND", "xpu")
        os.environ.setdefault("XPU_EVENT_KL3_ENABLE", "1")

    import flagsparse
    from flagsparse.sparse_operations import _common as runtime

    fallback = runtime._accel_fallback_reason()
    if fallback or not runtime._accel_module().is_available():
        raise SystemExit(
            fallback or f"The {vendor} accelerator is unavailable to PyTorch."
        )

    # This optest revision discovers libraries through a fixed registry.
    flagsparse.vendor_name = vendor
    flagsparse.device = runtime._accel_device_type()
    run_tests._KNOWN_LIBS[:] = [("flagsparse", "FlagSparse")]
    if flags.delivery:
        from tools.optest_delivery import run_delivery

        if not any(arg.split("=", 1)[0] == "--stages" for arg in args):
            sys.argv.extend(("--stages", "all"))
        return run_delivery(run_tests, flags.benchmark_input)
    return run_tests.main()


if __name__ == "__main__":
    raise SystemExit(main())
