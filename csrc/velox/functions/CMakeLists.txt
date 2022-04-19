# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_library(
  torcharrow_udfs OBJECT
  functions.h
  string_functions.h
  numeric_functions.h
  rec/bucketize.h
  rec/sigrid_hash.h
  register_udf.cpp
)

target_link_libraries(
  torcharrow_udfs
  velox_functions_string
  velox_functions_prestosql
)
