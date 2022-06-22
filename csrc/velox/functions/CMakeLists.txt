# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(
  TORCHARROW_UDF_SOURCES
  functions.h
  string_functions.h
  numeric_functions.h
  rec/bucketize.h
  rec/sigrid_hash.h
  rec/firstX.h
  rec/compute_score.h
  register_udf.cpp
  )

set(
  TORCHARROW_UDF_LINK_LIBRARIES
  velox_functions_string
  velox_functions_prestosql
  )
set(TORCHARROW_UDF_COMPILE_DEFINITIONS)

if (USE_TORCH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  list(
    APPEND
    TORCHARROW_UDF_SOURCES
    text/add_tokens.h
    text/bpe_tokenize.h
    text/gpt2_bpe_tokenizer.h
    text/gpt2_bpe_tokenizer.cpp
    text/regex.h
    text/regex.cpp
    text/vocab.h
    text/vocab.cpp
    text/vocab_ops.h
    )
  list(
    APPEND
    TORCHARROW_UDF_LINK_LIBRARIES
    "${TORCH_LIBRARIES}"
    )
  list(
    APPEND
    TORCHARROW_UDF_COMPILE_DEFINITIONS
    USE_TORCH
    )
endif()


add_library(
  torcharrow_udfs OBJECT
  ${TORCHARROW_UDF_SOURCES}
)

target_link_libraries(
  torcharrow_udfs
  ${TORCHARROW_UDF_LINK_LIBRARIES}
)

target_compile_definitions(
  torcharrow_udfs
  PRIVATE
  ${TORCHARROW_UDF_COMPILE_DEFINITIONS}
)
