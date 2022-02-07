// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "velox/functions/Udf.h"
#include <limits>

namespace facebook::torcharrow::functions {

VELOX_UDF_BEGIN(torcharrow_log)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& input) {
  result = std::log(input);
  return true;
}
VELOX_UDF_END();

VELOX_UDF_BEGIN(torcharrow_floordiv_int)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& a, const TInput& b) {
  // Same as velox divide, when inputs are integers and b is 0, return error
  // instead of inf because int(inf) will overflow.
  if (b == 0) {
    VELOX_ARITHMETIC_ERROR("division by zero");
  }
  // promote to float type to correctly compute floor divide for negative
  // integers. e.g: -3 / 2 is -1, but floor(float(-3) / 2) is -2.
  result = std::floor(float(a) / b);
  return true;
}
VELOX_UDF_END();

VELOX_UDF_BEGIN(torcharrow_floordiv)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& a, const TInput& b) {
  result = std::floor(a / b);
  return true;
}
VELOX_UDF_END();

// Torcharrow_floormod generates same result as Python mod, which is different from C++.
// Reference: "floored division" in https://en.wikipedia.org/wiki/Modulo_operation
VELOX_UDF_BEGIN(torcharrow_floormod_int)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& a, const TInput& b) {
  // Same as velox modulus, when inputs are integers and b is 0, return error
  // instead of inf because int(inf) will overflow.
  if (b == 0) {
    VELOX_ARITHMETIC_ERROR("Cannot divide by 0");
  }
  // promote to float type to correctly compute floor divide for negative
  // integers. e.g: -3 / 2 is -1, but floor(float(-3) / 2) is -2.
  result = a - std::floor(float(a) / b) * b;
  return true;
}
VELOX_UDF_END();

VELOX_UDF_BEGIN(torcharrow_floormod)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& a, const TInput& b) {
  result = a - std::floor(a / b) * b;
  return true;
}
VELOX_UDF_END();

VELOX_UDF_BEGIN(torcharrow_pow_int)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& a, const TInput& b) {
  if (b < 0) {
    VELOX_UNSUPPORTED("Integers to negative integer powers are not allowed")
  }
  auto tmp = std::pow(a, b);
  if (tmp > std::numeric_limits<int64_t>::max() || tmp < std::numeric_limits<int64_t>::min()) {
    VELOX_UNSUPPORTED("Inf is outside the range of representable values of type int64")
  }
  result = tmp;
  return true;
}
VELOX_UDF_END();

// Velox prestosql pow always returns double, but TorchArrow wants to return float
// when inputs are float. Ref: https://fburl.com/code/ytzqx8sg
VELOX_UDF_BEGIN(torcharrow_pow)
template <typename TOutput, typename TInput = TOutput>
FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& a, const TInput& b) {
  result = std::pow(a, b);
  return true;
}
VELOX_UDF_END();

} // namespace facebook::torcharrow::functions
