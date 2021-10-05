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


cmake_minimum_required(VERSION 3.15)

# _torcharrow is a shared library as it's a Python extension
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(lib_velox_df)

set(VELOX_ROOT ${CMAKE_BINARY_DIR}/csrc/velox/velox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# velox/common/base/SimdUtil.h requires AVX
# TODO: only need to retain -mavx2?
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mavx -mf16c -masm=intel -mlzcnt")

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
  column.cpp)

# TODO(linb): Need this to match with velox, however pybind11 requires "hidden" visibility
# set_target_properties(_torcharrow PROPERTIES CXX_VISIBILITY_PRESET default)

add_subdirectory(velox)
add_subdirectory(functions)

# Link with Velox:
target_link_libraries(_torcharrow PRIVATE
  velox_vector
  velox_memory
  velox_core
  velox_functions_prestosql
  velox_parse_expression
  velox_parse_parser
  velox_exec_test_util
  torcharrow_udfs)

#target_compile_definitions(_torcharrow PRIVATE)

# Based off on
# https://github.com/pytorch/audio/blob/c69955c6dcff7840f69ba9be188b900e480c9633/torchaudio/csrc/CMakeLists.txt#L162-L166
install(
  TARGETS _torcharrow
  LIBRARY DESTINATION .
)
