# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


cmake_minimum_required(VERSION 3.15)

# _torcharrow is a shared library as it's a Python extension
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# To make the right CPython is built with on GitHub Actions,
# see https://github.com/actions/setup-python/issues/121#issuecomment-1014500503
set(Python_FIND_FRAMEWORK "LAST")

project(lib_velox_df)

set(VELOX_ROOT ${CMAKE_BINARY_DIR}/csrc/velox/velox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Be consistent with Velox CXX_FLAGS: https://github.com/facebookincubator/velox/blob/1bdf9fc2a0ba7e9351cc6d092aa5c0e4e0f78708/CMakeLists.txt#L125-L127
# velox/common/base/SimdUtil.h requires AVX
# TODO: only need to retain -mavx2?
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mavx -mf16c -mbmi2 -march=native")

# Setup CCache
# Based off on:
# https://github.com/pytorch/pytorch/blob/c03cae49fc70408a93f4de15a11a5a73997e9565/CMakeLists.txt#L302-L311
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
  message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
endif()

include_directories(velox)  # TODO: Do we need this? Or just ${VELOX_ROOT} is enough?
# Otherwise some targets, such as dwrf-proto-wrapper.cpp.o
# will complain cannot find files to be included
# Based off on:
# https://github.com/facebookexternal/presto_cpp/blob/f278b9b186d9753dc59f58f31bd09db1c89c3c2b/CMakeLists.txt#L104
include_directories(${VELOX_ROOT})

# Based on https://pybind11.readthedocs.io/en/stable/compiling.html#findpython-mode
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)

# Define our Python module:
pybind11_add_module(
  _torcharrow
  MODULE
  NO_EXTRAS # TODO: LTO crashes GCC9.2. File issues to pybind11
  register_bindings.cpp
  lib.cpp
  vector.h
  vector.cpp
  column.h
  column.cpp
  tensor_conversion.h
  tensor_conversion.cpp
  VariantToVector.h
  VariantToVector.cpp
)

# TODO(linb): Need this to match with velox, however pybind11 requires "hidden" visibility
# set_target_properties(_torcharrow PROPERTIES CXX_VISIBILITY_PRESET default)
find_package(Torch REQUIRED)
add_subdirectory(velox)
add_subdirectory(functions)


# Link with Velox:
target_link_libraries(_torcharrow PRIVATE
velox_vector
velox_memory
velox_buffer
velox_core
velox_functions_prestosql
velox_function_registry
velox_arrow_bridge
torcharrow_udfs
)

#target_compile_definitions(_torcharrow PRIVATE)

# Based off on
# https://github.com/pytorch/audio/blob/c69955c6dcff7840f69ba9be188b900e480c9633/torchaudio/csrc/CMakeLists.txt#L162-L166
install(
  TARGETS _torcharrow
  LIBRARY DESTINATION .
  )
