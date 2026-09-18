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

# libtriton_jit already pulls googletest in; reuse that target rather than
# fetching a second, possibly different, copy.
if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# The tests compile the backend adaptor themselves. libflagsparse.so exports
# only the C API (visibility=hidden), so linking against its internals is not
# possible by design -- and should not be made possible just to run tests.
# The variant registry, generated from conf/operators.yaml.
#
# This is what makes the manifest the operator list rather than a document beside
# it: the benchmarks iterate what is generated here, so declaring a variant in the
# YAML puts it in the sweep, and removing one takes it out. Growing 40 -> 115 is
# an edit to the YAML plus a rebuild.
# The generator needs an interpreter. Found HERE rather than at the pytest block
# below, because that one runs later and leaving Python_EXECUTABLE empty made the
# custom command a silent no-op: it printed "Generating..." and produced nothing,
# and the failure surfaced 200 lines later as a missing header.
find_package(Python COMPONENTS Interpreter QUIET)
if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR
        "No Python interpreter: ctest's variant registry is generated from "
        "conf/operators.yaml by tools/gen_variants.py and cannot be built without "
        "one. Install python3 with PyYAML, or pass -DPython_EXECUTABLE=...")
endif()

set(_fs_variants "${CMAKE_CURRENT_BINARY_DIR}/generated/variants.inc")
add_custom_command(
    OUTPUT "${_fs_variants}"
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gen_variants.py"
            --manifest "${CMAKE_SOURCE_DIR}/conf/operators.yaml"
            --out "${_fs_variants}"
    DEPENDS "${CMAKE_SOURCE_DIR}/conf/operators.yaml"
            "${CMAKE_SOURCE_DIR}/tools/gen_variants.py"
    COMMENT "Generating ctest variant registry from conf/operators.yaml"
    VERBATIM)
add_custom_target(flagsparse_variants DEPENDS "${_fs_variants}")

# The vendor baseline a speedup is measured against. Its own CMakeLists is the
# table of which backend has which sparse library; a backend without one compiles
# the `none` provider, which answers "unavailable" with a reason.
add_subdirectory(baseline)

# One profile owns the test policy for the backend selected at configure time.
# The Python selector and C API spelling intentionally differ for MACA, MUSA,
# DCU, and Ascend, so keep the mapping explicit.
set(_fs_ctest_profile_by_backend
    "CUDA=cuda"
    "MUSA=musa"
    "MACA=maca"
    "HCU=rocm"
    "NPU=ascend"
    "XPU=xpu")
set(_fs_ctest_profile "")
foreach(_entry IN LISTS _fs_ctest_profile_by_backend)
    string(REPLACE "=" ";" _entry_parts "${_entry}")
    list(GET _entry_parts 0 _cmake_backend)
    list(GET _entry_parts 1 _profile)
    if(BACKEND STREQUAL _cmake_backend)
        set(_fs_ctest_profile "${_profile}")
        break()
    endif()
endforeach()
if(NOT _fs_ctest_profile)
    message(FATAL_ERROR "No CTest backend profile is registered for BACKEND=${BACKEND}")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/backends/${_fs_ctest_profile}.cmake")
message(STATUS
    "ctest profile: ${FLAGSPARSE_CTEST_PROFILE} "
    "(selector=${FLAGSPARSE_CTEST_SELECTOR}, "
    "capi-buildable=${FLAGSPARSE_CTEST_CAPI_BUILDABLE})")

add_library(flagsparse_testsupport STATIC common.cpp corpus.cpp ${FLAGSPARSE_ADAPTOR_SRC})
add_dependencies(flagsparse_testsupport flagsparse_variants)
target_include_directories(flagsparse_testsupport
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}/generated
           ${CMAKE_SOURCE_DIR}/include
           ${CMAKE_SOURCE_DIR}/src
           ${FLAGSPARSE_ADAPTOR_INCLUDES})
target_link_libraries(flagsparse_testsupport
    PUBLIC flagsparse flagsparse_ctest_baseline ${FLAGSPARSE_ADAPTOR_LIBS})

# test_<operator>.cpp, one per API, in both directories (spec §6.2).
# ctest names are "<phase>.<operator>" so `ctest -R accuracy` and `ctest -R spmv`
# both select the way the spec documents.
# Per-test ctest timeout. 900s covers the synthetic cases; a 30-matrix corpus
# sweep over the full dtype grid does not, and the only way to raise it used to
# be editing this literal. Same problem the Python runner solved with --timeout.
set(FLAGSPARSE_CTEST_TIMEOUT "900" CACHE STRING
    "Per-test timeout in seconds for C API ctest cases")

function(flagsparse_add_test phase op)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/${phase}/test_${op}.cpp")
    if(NOT EXISTS "${src}")
        return()
    endif()
    set(target "test_${phase}_${op}")
    add_executable(${target} "${src}")
    add_dependencies(${target} flagsparse_variants)
    target_include_directories(${target}
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(${target} PRIVATE flagsparse_testsupport GTest::gtest_main)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "test_${op}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${phase}")
    add_test(NAME "${phase}.${op}" COMMAND ${target})
    set_tests_properties("${phase}.${op}" PROPERTIES
        TIMEOUT ${FLAGSPARSE_CTEST_TIMEOUT}
        LABELS "capi;${FLAGSPARSE_CTEST_PROFILE};${phase}")
endfunction()

# Every operator in the spec's list. Files that do not exist yet are skipped,
# so adding an operator is: drop in test_<op>.cpp, rebuild.
set(FLAGSPARSE_OPS spmv spmm spsv spsm spgemm sddmm gather scatter)
if(FLAGSPARSE_CTEST_CAPI_BUILDABLE)
    foreach(op ${FLAGSPARSE_OPS})
        flagsparse_add_test(accuracy ${op})
        flagsparse_add_test(benchmark ${op})
    endforeach()
else()
    message(STATUS
        "ctest: C API cases are disabled by the ${FLAGSPARSE_CTEST_PROFILE} "
        "profile until its adaptor is implemented and the profile is enabled")
endif()

# ---------------------------------------------------------------------------
# The Python operator suite, registered as ctest cases.
#
# These tests exercise the Triton kernels this library re-exports -- the same
# kernels, on the same device, through the other front end. Running them from
# here means ONE command covers both layers on whatever backend the build was
# configured for, which is the case that actually matters: a kernel that does not
# lower on an accelerator breaks the C API too, and finding that out from a
# separate repository's test run is finding it out late.
#
# They are registered, not duplicated: the files stay in this repository's
# top-level source tree, located through FLAGSPARSE_PYTHON_SRC (its src/).
#
#   ctest -R pytest        the Python suite only
#   ctest -R accuracy      the C API accuracy tests only
#   ctest -R spmv          both layers for one operator
# ---------------------------------------------------------------------------
option(FLAGSPARSE_CTEST_PYTEST "Register the Python operator suite as ctest cases" ON)

if(FLAGSPARSE_CTEST_PYTEST AND FLAGSPARSE_PYTHON_SRC)
    get_filename_component(_fs_py_root "${FLAGSPARSE_PYTHON_SRC}" DIRECTORY)
    set(_fs_pytest_dir "${_fs_py_root}/tests/pytest")
    if(EXISTS "${_fs_pytest_dir}")
        if(Python_Interpreter_FOUND)
            file(GLOB _fs_pytest_files "${_fs_pytest_dir}/test_*.py")
            foreach(_file ${_fs_pytest_files})
                get_filename_component(_stem "${_file}" NAME_WE)
                string(REGEX REPLACE "^test_" "" _case "${_stem}")
                string(REGEX REPLACE "_accuracy$" "" _case "${_case}")
                add_test(NAME "pytest.${_case}"
                         COMMAND "${Python_EXECUTABLE}" -m pytest -q "${_file}")
                # PYTHONPATH so the suite imports the SAME source tree the C
                # library's kernels are re-exported from -- an installed wheel
                # would silently test a different copy.
                set_tests_properties("pytest.${_case}" PROPERTIES
                    TIMEOUT 3600
                    ENVIRONMENT "PYTHONPATH=${FLAGSPARSE_PYTHON_SRC};FLAGSPARSE_BACKEND=${FLAGSPARSE_CTEST_SELECTOR}"
                    LABELS "python;${FLAGSPARSE_CTEST_PROFILE};accuracy")
            endforeach()
            list(LENGTH _fs_pytest_files _fs_pytest_count)
            message(STATUS
                "ctest: ${_fs_pytest_count} Python operator suites from ${_fs_pytest_dir}")
        else()
            message(STATUS "ctest: no Python interpreter; skipping the pytest suites")
        endif()
    else()
        message(STATUS
            "ctest: ${_fs_pytest_dir} not found; skipping the pytest suites "
            "(set FLAGSPARSE_PYTHON_SRC to a FlagSparse src/ directory)")
    endif()
endif()
