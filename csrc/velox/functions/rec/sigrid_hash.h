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

using uint128_t = unsigned __int128;

inline std::tuple<uint64_t, int> computeMultiperAndShift(
    int64_t divisor,
    int precision) {
  constexpr int N = 64;
  int l = ceil(std::log2(divisor));
  uint128_t low = (static_cast<uint128_t>(1) << (N + l)) / divisor;
  uint128_t high = ((static_cast<uint128_t>(1) << (N + l)) +
                    ((static_cast<uint128_t>(1) << (N + l - precision)))) /
      divisor;
  while (low / 2 < high / 2 && l > 0) {
    low = low / 2;
    high = high / 2;
    --l;
  }
  return std::make_tuple((uint64_t)high, l);
}

template <typename TInput>
inline int64_t computeSigridHash(
    const TInput& input,
    const int64_t salt,
    const int64_t maxValue,
    const uint64_t multiplier,
    const int shift) {
  if (maxValue == 1) {
    return 0;
  }

  int64_t hashed =
      folly::hash::hash_combine(salt, folly::hash::twang_mix64(input));
  if (maxValue > 1) {
    int64_t sign = hashed >> (64 - 1);
    int64_t q = sign ^
        ((static_cast<uint128_t>(multiplier) * (sign ^ hashed)) >>
         (64 + shift));

    int64_t output = hashed - q * maxValue;
    return output;
  }
  return hashed;
}

template <typename TExecParams>
struct sigrid_hash {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  // By default we use precision = 63, to computed signed division quotient
  // rouded towards negative infinity.
  const static int kPrecision = 63;

  template <typename TOutput, typename TInput>
  FOLLY_ALWAYS_INLINE void callNullFree(
      TOutput& result,
      const TInput& values,
      const int64_t& salt,
      const int64_t& maxValue) {
    VELOX_USER_CHECK_GT(maxValue, 0, "maxValue must be larger than 0.");

    uint64_t multiplier_;
    int shift_;
    std::tie(multiplier_, shift_) =
        computeMultiperAndShift(maxValue, kPrecision);

    if constexpr (std::is_same<TInput, int64_t>::value) {
      result = computeSigridHash(values, salt, maxValue, multiplier_, shift_);
    } else {
      for (const auto& val : values) {
        result.push_back(
            computeSigridHash(val, salt, maxValue, multiplier_, shift_));
      }
    }
  }
};

} // namespace facebook::torcharrow::functions
