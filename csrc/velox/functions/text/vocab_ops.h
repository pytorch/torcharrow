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
#include "vocab.h"

namespace facebook::torcharrow::functions {
template <typename T>
struct lookup_indices {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      velox::exec::ArrayWriter<int64_t>& result,
      const arg_type<std::shared_ptr<Vocab>>& vocab,
      const arg_type<velox::Array<velox::Varchar>>& tokens) {
    for (const auto& token : tokens) {
      std::string token_str = token.value().str();
      result.push_back(vocab->__getitem__(token_str));
    }
    return true;
  }
};
} // namespace facebook::torcharrow::functions
