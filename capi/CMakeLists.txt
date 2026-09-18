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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# BACKEND must be decided before project(): libtriton_jit switches its enabled
# languages on it.
# CUDA plus the domestic accelerators. XPU (Kunlunxin) and GCU (Enflame) are the
# two slots added alongside the four already named; MLU (Cambricon) doubles as
# the generic reserve. Which of these are actually BUILT is src/adaptor's
# business -- see the reserved-slot message there.
set(BACKEND "CUDA" CACHE STRING
    "Target backend: CUDA, MUSA, MACA, NPU, HCU, IX, GCU, MLU, XPU")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA MUSA MACA NPU HCU IX GCU MLU XPU)

# XPU is listed but cannot build: deps/libtriton_jit has no XPU backend. Catching
# it HERE matters -- add_subdirectory(libtriton_jit) comes first and would fail
# with "Invalid BACKEND: XPU", which names the dependency but not the reason.
if(BACKEND STREQUAL "XPU")
    message(FATAL_ERROR
        "BACKEND=XPU (Kunlunxin) cannot be built: deps/libtriton_jit supports "
        "CUDA, IX, MUSA, MACA, MLU, NPU, GCU and HCU, and it is what compiles "
        "and launches every kernel. FlagTree's Triton fork does have an xpu "
        "backend, so the Python side of FlagSparse can target Kunlunxin "
        "(FLAGSPARSE_BACKEND=xpu) -- only this C wrapper waits on libtriton_jit.")
endif()

project(FlagSparse LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(FLAGSPARSE_BUILD_TESTS "Build accuracy and benchmark ctest binaries" ON)
option(FLAGSPARSE_BUILD_CLI   "Build the command line tool"                 OFF)
option(FLAGSPARSE_INSTALL     "Install rules"                               ON)

# --------------------------------------------------------------- dependency
# libtriton_jit compiles and launches the Triton kernels. Normally the
# submodule; FLAGSPARSE_TRITON_JIT_DIR lets CI or a dev tree point elsewhere.
set(FLAGSPARSE_TRITON_JIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/libtriton_jit"
    CACHE PATH "libtriton_jit source directory")
if(NOT EXISTS "${FLAGSPARSE_TRITON_JIT_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "libtriton_jit not found at ${FLAGSPARSE_TRITON_JIT_DIR}.\n"
        "Run from the repository root: git submodule update --init --recursive\n"
        "or set FLAGSPARSE_TRITON_JIT_DIR to an existing libtriton_jit tree.")
endif()
set(TRITON_JIT_INSTALL OFF CACHE BOOL "" FORCE)
# libtriton_jit registers its own ctest cases under if(BUILD_TESTING). With
# EXCLUDE_FROM_ALL their binaries are never built, so those registrations would
# surface in `ctest --test-dir build` as "Not Run" failures that have nothing to
# do with FlagSparse. Turn testing off across the dependency, then back on for us.
set(_flagsparse_saved_build_testing ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${FLAGSPARSE_TRITON_JIT_DIR}" "${CMAKE_BINARY_DIR}/_triton_jit" EXCLUDE_FROM_ALL)
set(BUILD_TESTING ${_flagsparse_saved_build_testing} CACHE BOOL "" FORCE)

# ------------------------------------------------------------------ library
add_library(flagsparse SHARED
    src/api/context.cpp
    src/api/descriptors.cpp
    src/api/pending.cpp
    src/core/csr_meta.cpp
    src/core/dtype.cpp
    src/core/jit.cpp
    src/core/prologue.cpp
    src/ops/gather_scatter.cpp
    src/ops/sddmm.cpp
    src/ops/spgemm.cpp
    src/ops/spmm.cpp
    src/ops/spmv.cpp
    src/ops/spsm.cpp
    src/ops/spsv.cpp
)
add_library(FlagSparse::flagsparse ALIAS flagsparse)

target_include_directories(flagsparse
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Where the Triton kernel modules live. Baked in so an installed library finds
# them without the caller setting anything; FLAGSPARSE_CODEGEN_PATH overrides.
set(FLAGSPARSE_CODEGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/flagsparse_codegen"
    CACHE PATH "Directory holding the Triton kernel modules")
# The kernel modules RE-EXPORT from the flagsparse Python package rather than
# copying, so that package has to be importable from the embedded interpreter.
# In the combined source tree it is the top-level src/ directory; callers can
# still override this cache value for an installed or otherwise relocated tree.
set(FLAGSPARSE_PYTHON_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../src" CACHE PATH
    "src/ directory of the flagsparse Python package (holds flagsparse/)")

target_compile_definitions(flagsparse
    PRIVATE FLAGSPARSE_CODEGEN_DIR="${FLAGSPARSE_CODEGEN_DIR}"
            FLAGSPARSE_PYTHON_SRC_DEFAULT="${FLAGSPARSE_PYTHON_SRC}")

# Only the C API is exported: internal symbols stay hidden so the ABI is exactly
# what flagsparse.h declares.
set_target_properties(flagsparse PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

target_link_libraries(flagsparse PRIVATE TritonJIT::triton_jit)

# ------------------------------------------------------------------ backend
add_subdirectory(src/adaptor)

# -------------------------------------------------------------------- tools
option(FLAGSPARSE_BUILD_TOOLS "Build the cross-operator corpus sweep" ON)
if(FLAGSPARSE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# -------------------------------------------------------------------- tests
if(FLAGSPARSE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(ctest)
endif()

# ------------------------------------------------------------------ install
if(FLAGSPARSE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS flagsparse
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    # The public header is the ONLY one installed (spec §3.2).
    install(FILES include/flagsparse.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY flagsparse_codegen/
            DESTINATION ${CMAKE_INSTALL_DATADIR}/flagsparse/flagsparse_codegen
            FILES_MATCHING PATTERN "*.py")
endif()

message(STATUS "FlagSparse ${PROJECT_VERSION}  backend=${BACKEND}  tests=${FLAGSPARSE_BUILD_TESTS}")
