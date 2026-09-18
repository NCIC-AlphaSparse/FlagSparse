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

# The vendor baseline, as a TABLE -- same shape as src/adaptor/CMakeLists.txt and
# for the same reason: adding a chip should be an entry here plus one 35-line
# prefix file, not an if-chain that grows a branch per vendor.
#
# A backend is in one of three states, and the state is a fact about the vendor,
# not a TODO:
#
#   WIRED       a generic (descriptor) sparse library exists and is named here.
#               CUDA -> cuSPARSE is the only one VERIFIED (compiled and measured
#               on the dev box). DCU -> hipSPARSE is written against the public
#               API and first compiled on a DCU box.
#
#   PROBED      named from the vendor's CUDA-mirroring convention but never
#               compiled here. MACA and MUSA are here. The names are checked at
#               configure time, so being wrong costs a blank speedup column and a
#               message, not a build.
#
#   SEAM        the vendor HAS sparse operators but no descriptor API, so the
#               prefix-table port does not apply. ASCEND is here: it gets its own
#               file whose entry points each name the CANN operator they would
#               bind to, and return `unavailable` with that name. More useful than
#               `none`, which says only "no baseline on this backend".
#
#   NONE        the vendor has no cuSPARSE-shaped sparse library to bind at all.
#               XPU is here; see baseline/none/baseline.cpp. Not waiting on a name.
#
# Every non-WIRED backend compiles baseline/none/baseline.cpp, which answers
# `unavailable` with the reason below. That is why a backend without a baseline
# produces rows with a blank speedup and an explanation rather than no rows.

set(FLAGSPARSE_BASELINE_REASON "" CACHE INTERNAL "")

# --------------------------------------------------------------- WIRED slots
# The toolkit root is where both the header and the library live. CUDA_HOME wins
# if set, so a box with several toolkits picks the one the rest of the build used.
set(FLAGSPARSE_CUDA_SPARSE_LIB    "cusparse")
set(FLAGSPARSE_CUDA_SPARSE_HEADER "cusparse.h")
set(FLAGSPARSE_CUDA_SPARSE_DIR    "cuda")
set(FLAGSPARSE_CUDA_SPARSE_HOME_VAR "CUDA_HOME")
set(FLAGSPARSE_CUDA_SPARSE_HOME_DEF "/usr/local/cuda")

# hipSPARSE ships inside ROCm; the header is namespaced, unlike cuSPARSE's.
set(FLAGSPARSE_ROCM_SPARSE_LIB    "hipsparse")
set(FLAGSPARSE_ROCM_SPARSE_HEADER "hipsparse/hipsparse.h")
set(FLAGSPARSE_ROCM_SPARSE_DIR    "rocm")
set(FLAGSPARSE_ROCM_SPARSE_HOME_VAR "ROCM_PATH")
set(FLAGSPARSE_ROCM_SPARSE_HOME_DEF "/opt/rocm")

# MetaX MACA and Moore Threads MUSA. Both toolkits expose CUDA-family runtime
# names; MACA remains a probed prefix slot, while MUSA has a native baseline
# implementation because SDK 4.3.5 adds stage arguments to its generic API.
#
# MACA's names are unverified and are probed before selection. MUSA's names and
# headers are verified on the development host; a missing SDK still falls back
# to baseline/none with a reason.
set(FLAGSPARSE_MACA_SPARSE_LIB    "mcsparse")
set(FLAGSPARSE_MACA_SPARSE_HEADER "mcsparse.h")
set(FLAGSPARSE_MACA_SPARSE_DIR    "maca")
set(FLAGSPARSE_MACA_SPARSE_HOME_VAR "MACA_PATH")
set(FLAGSPARSE_MACA_SPARSE_HOME_DEF "/opt/maca")

set(FLAGSPARSE_MUSA_SPARSE_LIB    "musparse")
set(FLAGSPARSE_MUSA_SPARSE_HEADER "musparse.h")
set(FLAGSPARSE_MUSA_SPARSE_DIR    "musa")
set(FLAGSPARSE_MUSA_SPARSE_HOME_VAR "MUSA_HOME")
set(FLAGSPARSE_MUSA_SPARSE_HOME_DEF "/usr/local/musa")

# ---------------------------------------------------------------- NONE slots
# Kunlunxin XPU. Two independent blockers, and the second outlives the first:
#   1. deps/libtriton_jit has no XPU backend (cmake/Backend*.cmake is CUDA, GCU,
#      HCU, MACA, MLU, MUSA, NPU), so the JIT bridge cannot compile or launch a
#      kernel there -- the C API has no numerator on XPU, never mind a baseline.
#      FlagTree's Triton does have an xpu target, so the PYTHON side of FlagSparse
#      is the reachable path today; the C API is not.
#   2. Kunlunxin's math library is XDNN, which exposes fixed sparse ops rather
#      than a cuSPARSE-shaped descriptor API. Even with the SDK present there is
#      no generic entry point to bind the eight functions in baseline.hpp to.
set(FLAGSPARSE_XPU_BASELINE_WHY
    "Kunlunxin XPU: XDNN exposes fixed sparse ops, not a cuSPARSE-shaped generic API, so there is no vendor baseline to bind; separately, deps/libtriton_jit has no XPU backend, so the C API does not run here at all (FlagTree Triton's xpu target makes the Python side the reachable path)")

# Ascend. CANN is NOT empty here -- it has sparse operators, reached through
# aclnn (aclnnSparse*) and through torch_npu. What it does not have is a
# DESCRIPTOR API: there is no create-matrix/bufferSize/execute triple to bind the
# eight entry points in baseline.hpp to, so a baseline here means writing an
# op-by-op adaptor with a different shape, not filling in a prefix table. That
# work is real and has not been done; this string says so rather than implying
# the vendor has nothing.
# Ascend compiles its own seam file rather than `none` -- see the SEAM note above.
# No library to link: every entry point returns unavailable until the op-by-op
# adaptor is written, so this slot needs a source but not a find_library.
set(FLAGSPARSE_ASCEND_SPARSE_DIR "ascend")

set(FLAGSPARSE_ASCEND_BASELINE_WHY
    "Ascend CANN has sparse ops via aclnn/torch_npu but no cuSPARSE-shaped descriptor API, so the prefix-table port does not apply; an op-by-op baseline adaptor has not been written -- see benchmark/benchmark_ascend_probe.py and docs/ASCEND.md")

# ------------------------------------------------------------------ resolve
# BACKEND is the top-level cache variable (capi/CMakeLists.txt:23); default CUDA.
# NOT FLAGSPARSE_BACKEND -- that name is the Python-side env var (values like
# "mthreads"), a different variable with different spellings. Reading it here
# silently no-ops: it is always empty, so _bl_backend always fell through to
# CUDA regardless of -DBACKEND=..., and every non-CUDA build got "CUDA -> none"
# instead of probing its own vendor library.
string(TOUPPER "${BACKEND}" _bl_backend)
if(NOT _bl_backend)
    set(_bl_backend "CUDA")
endif()

set(_bl_lib    "${FLAGSPARSE_${_bl_backend}_SPARSE_LIB}")
set(_bl_header "${FLAGSPARSE_${_bl_backend}_SPARSE_HEADER}")
set(_bl_dir    "${FLAGSPARSE_${_bl_backend}_SPARSE_DIR}")

add_library(flagsparse_ctest_baseline STATIC)
target_include_directories(flagsparse_ctest_baseline
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
    PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Where the vendor toolkit lives. The env var wins over the default, matching
# src/adaptor/CMakeLists.txt so the two cannot pick different toolkits.
set(_bl_home "$ENV{${FLAGSPARSE_${_bl_backend}_SPARSE_HOME_VAR}}")
if(NOT _bl_home)
    set(_bl_home "${FLAGSPARSE_${_bl_backend}_SPARSE_HOME_DEF}")
endif()

if(_bl_lib AND _bl_dir AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_bl_dir}/baseline.cpp")
    # The header has to be findable before we commit to this slot: a missing
    # toolkit should fall through to `none` with a reason, not fail the build of
    # a test suite that would otherwise run perfectly well without a baseline.
    find_path(_bl_inc "${_bl_header}"
              HINTS "${_bl_home}/include" "${_bl_home}/targets/x86_64-linux/include"
              NO_DEFAULT_PATH)
    find_library(_bl_lib_path "${_bl_lib}"
                 HINTS "${_bl_home}/lib" "${_bl_home}/lib64"
                       "${_bl_home}/targets/x86_64-linux/lib"
                 NO_DEFAULT_PATH)
endif()

# A source-only slot (ASCEND): a prefix file with nothing to link against.
if(NOT _bl_lib AND _bl_dir AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_bl_dir}/baseline.cpp")
    target_sources(flagsparse_ctest_baseline PRIVATE "${_bl_dir}/baseline.cpp")
    message(STATUS "ctest baseline: ${_bl_backend} -> ${_bl_dir} (no vendor library to link)")
    message(STATUS "  ${FLAGSPARSE_${_bl_backend}_BASELINE_WHY}")
elseif(_bl_lib AND _bl_dir AND _bl_inc AND _bl_lib_path)
    target_sources(flagsparse_ctest_baseline PRIVATE "${_bl_dir}/baseline.cpp")
    target_include_directories(flagsparse_ctest_baseline PRIVATE "${_bl_inc}")
    target_link_libraries(flagsparse_ctest_baseline PRIVATE "${_bl_lib_path}")
    message(STATUS "ctest baseline: ${_bl_backend} -> ${_bl_lib_path}")
else()
    set(_bl_why "${FLAGSPARSE_${_bl_backend}_BASELINE_WHY}")
    if(NOT _bl_why AND _bl_lib)
        # The slot IS wired; the toolkit just is not installed here. Say that,
        # rather than implying the vendor has no sparse library.
        set(_bl_why "${_bl_lib} not found under ${_bl_home} (set ${FLAGSPARSE_${_bl_backend}_SPARSE_HOME_VAR})")
    elseif(NOT _bl_why)
        set(_bl_why "no vendor sparse library is wired for backend ${_bl_backend}")
    endif()
    target_sources(flagsparse_ctest_baseline PRIVATE "none/baseline.cpp")
    target_compile_definitions(flagsparse_ctest_baseline
        PRIVATE "FLAGSPARSE_BASELINE_REASON=\"${_bl_why}\"")
    message(STATUS "ctest baseline: ${_bl_backend} -> none")
    message(STATUS "  reason: ${_bl_why}")
endif()
