/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {
template <typename T>
struct add_tokens {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TInput, typename TOutput, typename TTokens>
  FOLLY_ALWAYS_INLINE void call(
      TOutput& output,
      const TInput& input,
      const TTokens& tokens,
      bool begin = true) {
    output.reserve(input.size() + tokens.size());
    if (begin) {
      output.add_items(tokens);
      output.add_items(input);
    } else {
      output.add_items(input);
      output.add_items(tokens);
    }
  }
};
} // namespace facebook::torcharrow::functions
