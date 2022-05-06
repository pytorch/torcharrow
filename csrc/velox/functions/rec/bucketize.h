/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <velox/functions/Macros.h>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {

template <typename T1>
inline void validateBordersSpec(const T1& borders) {
  VELOX_CHECK(borders.size() != 0, "Borders should not be empty.");
  for (auto i = 1; i < borders.size(); ++i) {
    VELOX_CHECK(
        !(borders[i] < borders[i - 1]),
        "Borders should have non-decreasing sequence.");
    if (i > 1 && (borders[i] == borders[i - 1])) {
      if (!(borders[i - 2] < borders[i])) {
        std::string err_detail = "";
        for (auto b : borders) {
          err_detail += std::to_string(b) + ", ";
        }
        VELOX_CHECK(
            (borders[i - 2] < borders[i]) || (borders[i - 2] == borders[i]),
            "Borders should not have more than 2 repeated values, got: loc {}, array: {}",
            i,
            err_detail);
      }
    }
  }
}

template <typename T1, typename T2>
inline int32_t computeBucketId(const T1& borders, const T2& val) {
  int32_t index =
      std::lower_bound(borders.begin(), borders.end(), val) - borders.begin();
  if (index >= borders.size() - 1) {
    return index;
  }
  return val < borders[index + 1] ? index : index + 1;
}

template <typename T>
struct bucketize {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput, typename TBordersInput>
  FOLLY_ALWAYS_INLINE void callNullFree(
      int32_t& result,
      const TInput& val,
      const TBordersInput& borders) {
    validateBordersSpec(borders);
    result = computeBucketId(borders, val);
  }

  template <typename TOutput, typename TInput, typename TBordersInput>
  FOLLY_ALWAYS_INLINE void callNullFree(
      TOutput& result,
      const TInput& values,
      const TBordersInput& borders) {
    validateBordersSpec(borders);
    for (const auto& val : values) {
      result.push_back(computeBucketId(borders, val));
    }
  }
};

} // namespace facebook::torcharrow::functions
