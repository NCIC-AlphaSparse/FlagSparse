# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Cross-operator tools. Unlike ctest/, these are not registered as tests: a
# corpus sweep takes minutes and depends on matrices that are not in the repo.

add_executable(matrix_sweep matrix_sweep.cpp)
# matrix_sweep.cpp calls flagsparse::adaptor::* instead of the CUDA runtime, so
# it needs the adaptor header (under src/) and the adaptor sources themselves --
# the same ones ctest's testsupport compiles, not a second copy.
target_sources(matrix_sweep PRIVATE ${FLAGSPARSE_ADAPTOR_SRC})
target_include_directories(matrix_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${FLAGSPARSE_ADAPTOR_INCLUDES})
target_link_libraries(matrix_sweep PRIVATE flagsparse ${FLAGSPARSE_ADAPTOR_LIBS})
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(matrix_sweep PRIVATE CUDA::cudart)
endif()
