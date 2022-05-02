/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <velox/functions/Macros.h>
#include <algorithm>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {

template <typename T>
struct firstX {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TOutput, typename TInput>
  FOLLY_ALWAYS_INLINE void
  callNullFree(TOutput& result, TInput& input, const int32_t& numToCopy) {
    int32_t actualNumToCopy = std::min(numToCopy, input.size());
    for (auto i = 0; i < actualNumToCopy; i++) {
      result.push_back(input[i]);
    }
  }
};

} // namespace facebook::torcharrow::functions
