# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# add_library(
#   torcharrow_udfs OBJECT
#   functions.h
#   string_functions.h
#   numeric_functions.h
#   register_udf.cpp
#   # gpt2_bpe_tokenizer.h
#   # gpt2_bpe_tokenizer.cpp
#   # regex.h
#   # regex.cpp
# )

find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(../pybind11 ${CMAKE_BINARY_DIR}/csrc/pybind11_v2)

# Define our Python module:
pybind11_add_module(
  torcharrow_udfs
  OBJECT
  NO_EXTRAS # TODO: LTO crashes GCC9.2. File issues to pybind11
  functions.h
  string_functions.h
  numeric_functions.h
  register_udf.cpp
  gpt2_bpe_tokenizer_pybind.h
  gpt2_bpe_tokenizer_pybind.cpp
  regex.h
  regex.cpp
  
)

target_link_libraries(
  torcharrow_udfs PRIVATE
  velox_functions_string
  velox_functions_prestosql
)
