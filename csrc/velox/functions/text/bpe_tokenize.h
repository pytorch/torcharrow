/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gpt2_bpe_tokenizer.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {

template <typename T>
struct bpe_tokenize {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      velox::exec::ArrayWriter<velox::Varchar>& result,
      const arg_type<std::shared_ptr<GPT2BPEEncoder>>& bpe_encoder,
      const arg_type<velox::Varchar>& text) {
    std::vector<int64_t> int_result = bpe_encoder->Encode(text.str());
    std::vector<std::string> str_result;
    str_result.reserve(int_result.size());
    std::transform(std::begin(int_result),
               std::end(int_result), 
               std::back_inserter(str_result),
               [](int64_t val) { return std::to_string(val); } 
              );

    result.copy_from(str_result);
    return true;
  }
};

} // namespace facebook::torcharrow::functions
