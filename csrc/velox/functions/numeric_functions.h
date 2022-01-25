// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "velox/functions/Udf.h"

namespace facebook::torcharrow::functions {

VELOX_UDF_BEGIN(torcharrow_log)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& input) {
  result = std::log(input);
  return true;
}
VELOX_UDF_END();

} // namespace facebook::torcharrow::functions
