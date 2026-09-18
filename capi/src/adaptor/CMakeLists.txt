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

# One adaptor per backend, as a TABLE rather than an if-chain: adding a chip is
# an entry below plus backend/<name>/adaptor.cpp -- and no change to
# include/flagsparse.h, which is the point of the abstraction.
#
# Every vendor here ships a driver API that mirrors CUDA's, so the adaptor source
# is _template.inc plus a symbol prefix (see backend/musa/adaptor.cpp: 25 lines).
# What differs per vendor is only where its toolkit lives and what the two
# libraries are called, which is exactly what the table carries.
#
# RESERVED SLOTS. XPU (Kunlunxin), GCU (Enflame) and MLU (Cambricon, the generic
# reserve) are declared but NOT configured: their library names are not guessed
# here.  A wrong library name fails at link time with a missing symbol that names
# neither the vendor nor the mistake, which is worse than not having the entry --
# so configuring one of these stops with a message naming the three things to
# supply.  Filling a slot in is: set its FLAGSPARSE_<NAME>_* values below, drop
# in backend/<name>/adaptor.cpp, done.

# Toolkit locations. These are NOT guesses: every value below is read off the
# matching cmake/Backend<NAME>.cmake in deps/libtriton_jit, which has to find the
# same SDK to build the JIT bridge. Taking them from there means the two cannot
# drift, and it is why the reserved slots carry real paths instead of TODOs.
set(FLAGSPARSE_MUSA_HOME_VAR  "MUSA_HOME")
set(FLAGSPARSE_MUSA_HOME_DEF  "/usr/local/musa")
set(FLAGSPARSE_MUSA_RUNTIME   "musart")
set(FLAGSPARSE_MUSA_DRIVER    "musa")
set(FLAGSPARSE_MUSA_HEADER    "musa.h")

# MetaX MACA -- BackendMACA.cmake: header mcr/mc_runtime.h, library mcruntime.
set(FLAGSPARSE_MACA_HOME_VAR  "MACA_PATH")
set(FLAGSPARSE_MACA_HOME_DEF  "/opt/maca")
set(FLAGSPARSE_MACA_RUNTIME   "mcruntime")
set(FLAGSPARSE_MACA_DRIVER    "mcruntime")
set(FLAGSPARSE_MACA_HEADER    "mcr/mc_runtime.h")

# Enflame GCU -- BackendGCU.cmake: TOPS_ROOT, header tops_runtime_api.h, lib topsrt.
set(FLAGSPARSE_GCU_HOME_VAR   "TOPS_ROOT")
set(FLAGSPARSE_GCU_HOME_DEF   "/opt/tops")
set(FLAGSPARSE_GCU_RUNTIME    "topsrt")
set(FLAGSPARSE_GCU_DRIVER     "topsrt")
set(FLAGSPARSE_GCU_HEADER     "tops_runtime_api.h")

# Cambricon MLU (the generic reserve) -- BackendMLU.cmake: NEUWARE_HOME, cn_api.h.
set(FLAGSPARSE_MLU_HOME_VAR   "NEUWARE_HOME")
set(FLAGSPARSE_MLU_HOME_DEF   "/usr/local/neuware")
set(FLAGSPARSE_MLU_RUNTIME    "cnrt")
set(FLAGSPARSE_MLU_DRIVER     "cndrv")
set(FLAGSPARSE_MLU_HEADER     "cn_api.h")

# Ascend NPU -- BackendNPU.cmake: ASCEND_TOOLKIT_HOME. Its driver API is ACL, not
# a CUDA mirror, so its adaptor.cpp will not come from _template.inc.
set(FLAGSPARSE_NPU_HOME_VAR   "ASCEND_TOOLKIT_HOME")
set(FLAGSPARSE_NPU_HOME_DEF   "/usr/local/Ascend/ascend-toolkit/latest")
set(FLAGSPARSE_NPU_RUNTIME    "ascendcl")
set(FLAGSPARSE_NPU_DRIVER     "ascendcl")
set(FLAGSPARSE_NPU_HEADER     "acl/acl.h")

# Hygon HCU -- BackendHCU.cmake uses find_package(hip), so it does not fit the
# two-library shape above; its slot keeps the DTK root for the message only.
set(FLAGSPARSE_HCU_HOME_VAR   "ROCM_PATH")
set(FLAGSPARSE_HCU_HOME_DEF   "/opt/dtk")
set(FLAGSPARSE_HCU_RUNTIME    "amdhip64")
set(FLAGSPARSE_HCU_DRIVER     "amdhip64")
set(FLAGSPARSE_HCU_HEADER     "hip/hip_runtime_api.h")

# Iluvatar IX -- libtriton_jit accepts it but ships no Backend module for it.
set(FLAGSPARSE_IX_HOME_VAR    "COREX_HOME")
set(FLAGSPARSE_IX_HOME_DEF    "/usr/local/corex")
set(FLAGSPARSE_IX_RUNTIME     "cudart")
set(FLAGSPARSE_IX_DRIVER      "cuda")
set(FLAGSPARSE_IX_HEADER      "cuda.h")

# Declared, located, but with no adaptor source yet.
set(FLAGSPARSE_RESERVED_BACKENDS "MACA" "NPU" "HCU" "GCU" "MLU" "IX")

# Kunlunxin XPU is a special case and gets its own message: the blocker is not
# here. deps/libtriton_jit accepts CUDA, IX, MUSA, MACA, MLU, NPU, GCU and HCU
# and has no XPU backend at all, so the JIT bridge -- which compiles and launches
# every kernel -- cannot target it. Writing an adaptor here would not help.
set(FLAGSPARSE_BLOCKED_BACKENDS "XPU")

# Configure a vendor whose driver API mirrors CUDA's.
function(flagsparse_vendor_adaptor name)
    set(home "${${FLAGSPARSE_${name}_HOME_VAR}}")
    if(NOT home)
        set(home "${FLAGSPARSE_${name}_HOME_DEF}")
    endif()
    string(TOLOWER "${name}" lower)
    set(src "${CMAKE_CURRENT_LIST_DIR}/backend/${lower}/adaptor.cpp")
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR
            "BACKEND=${name} has no adaptor source at ${src}. For a driver API "
            "that mirrors CUDA's, that file is _template.inc plus a symbol "
            "prefix -- see backend/musa/adaptor.cpp.")
    endif()
    # The same conditions libtriton_jit's own backend module checks; failing here
    # with the paths named beats a missing-symbol error at link time.
    find_library(${name}_RUNTIME_LIB ${FLAGSPARSE_${name}_RUNTIME}
                 PATHS "${home}/lib64" "${home}/lib")
    find_library(${name}_DRIVER_LIB ${FLAGSPARSE_${name}_DRIVER}
                 PATHS "${home}/lib64" "${home}/lib"
                       /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib)
    find_path(${name}_INCLUDE_DIR ${FLAGSPARSE_${name}_HEADER} PATHS "${home}/include")
    if(NOT (${name}_RUNTIME_LIB AND ${name}_DRIVER_LIB AND ${name}_INCLUDE_DIR))
        message(FATAL_ERROR
            "${name} toolkit incomplete under ${home}: "
            "${FLAGSPARSE_${name}_RUNTIME}=${${name}_RUNTIME_LIB} "
            "${FLAGSPARSE_${name}_DRIVER}=${${name}_DRIVER_LIB} "
            "include=${${name}_INCLUDE_DIR}")
    endif()
    target_sources(flagsparse PRIVATE ${src})
    target_include_directories(flagsparse PRIVATE ${${name}_INCLUDE_DIR})
    target_link_libraries(flagsparse PRIVATE ${${name}_RUNTIME_LIB} ${${name}_DRIVER_LIB})
    set(FLAGSPARSE_ADAPTOR_SRC "${src}" PARENT_SCOPE)
    set(FLAGSPARSE_ADAPTOR_LIBS "${${name}_RUNTIME_LIB};${${name}_DRIVER_LIB}" PARENT_SCOPE)
    set(FLAGSPARSE_ADAPTOR_INCLUDES "${${name}_INCLUDE_DIR}" PARENT_SCOPE)
endfunction()

if(BACKEND STREQUAL "CUDA")
    # GLOBAL: an imported target from find_package is scoped to the directory
    # that called it, and ctest/ needs CUDA::cuda_driver to build its own adaptor.
    find_package(CUDAToolkit REQUIRED GLOBAL)
    target_sources(flagsparse PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/adaptor.cpp)
    target_link_libraries(flagsparse PRIVATE CUDA::cuda_driver)
    # Published upward so the tests can compile their OWN copy. They must not
    # link the library's: the public ABI is flagsparse.h and nothing else, and
    # relaxing visibility to let a test in would widen it for everyone.
    set(FLAGSPARSE_ADAPTOR_SRC "${CMAKE_CURRENT_LIST_DIR}/backend/cuda/adaptor.cpp" PARENT_SCOPE)
    set(FLAGSPARSE_ADAPTOR_LIBS "CUDA::cuda_driver" PARENT_SCOPE)
    set(FLAGSPARSE_ADAPTOR_INCLUDES "" PARENT_SCOPE)
elseif(BACKEND STREQUAL "MUSA")
    flagsparse_vendor_adaptor(MUSA)
    set(FLAGSPARSE_ADAPTOR_SRC "${FLAGSPARSE_ADAPTOR_SRC}" PARENT_SCOPE)
    set(FLAGSPARSE_ADAPTOR_LIBS "${FLAGSPARSE_ADAPTOR_LIBS}" PARENT_SCOPE)
    set(FLAGSPARSE_ADAPTOR_INCLUDES "${FLAGSPARSE_ADAPTOR_INCLUDES}" PARENT_SCOPE)
elseif(BACKEND IN_LIST FLAGSPARSE_BLOCKED_BACKENDS)
    message(FATAL_ERROR
        "BACKEND=${BACKEND} cannot be built yet, and the blocker is upstream, "
        "not in this repository.\n"
        "deps/libtriton_jit supports CUDA, IX, MUSA, MACA, MLU, NPU, GCU and HCU "
        "-- it has no ${BACKEND} backend, and it is what compiles and launches "
        "every kernel. An adaptor here would have nothing to run on.\n"
        "FlagTree's Triton fork does carry an xpu backend, so the PYTHON side of "
        "FlagSparse can target Kunlunxin today (FLAGSPARSE_BACKEND=xpu); it is "
        "only the C wrapper that waits on libtriton_jit.")
elseif(BACKEND IN_LIST FLAGSPARSE_RESERVED_BACKENDS)
    string(TOLOWER "${BACKEND}" _lower)
    message(FATAL_ERROR
        "BACKEND=${BACKEND} is a reserved slot: the abstraction and the toolkit "
        "lookup are ready, the adaptor source is not written.\n"
        "One file: src/adaptor/backend/${_lower}/adaptor.cpp. For a driver API "
        "that mirrors CUDA's that is _template.inc plus a symbol prefix, 25 lines "
        "-- see backend/musa/adaptor.cpp. Toolkit expected under "
        "${FLAGSPARSE_${BACKEND}_HOME_DEF} (override with "
        "-D${FLAGSPARSE_${BACKEND}_HOME_VAR}=...).\n"
        "Then move ${BACKEND} out of FLAGSPARSE_RESERVED_BACKENDS into its own "
        "elseif calling flagsparse_vendor_adaptor(${BACKEND}).")
else()
    message(FATAL_ERROR
        "Unknown BACKEND=${BACKEND}. Known: CUDA (built), MUSA (built), and the "
        "reserved slots ${FLAGSPARSE_RESERVED_BACKENDS}.")
endif()
