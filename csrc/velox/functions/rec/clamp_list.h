/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {

// TODO: remove this function once lambda expression is supported in
// TorchArrow
template <typename T>
struct ClampListFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TOutput, typename TInput, typename TElement>
  FOLLY_ALWAYS_INLINE void callNullFree(
      TOutput& result,
      const TInput& values,
      const TElement& lo,
      const TElement& hi) {
    VELOX_USER_CHECK_LE(lo, hi, "Lo > hi in clamp.");
    result.reserve(values.size());
    for (const auto& val : values) {
      result.push_back(std::clamp(val, lo, hi));
    }
  }
};

} // namespace facebook::torcharrow::functions
