/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <math.h>
#include <cmath>
#include <limits>
#include "velox/functions/Udf.h"

namespace facebook::torcharrow::functions {

template <typename T>
struct torcharrow_log {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& input) {
    result = std::log(input);
    return true;
  }
};

template <typename T>
struct torcharrow_floordiv_int {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
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
};

template <typename T>
struct torcharrow_floordiv {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
    result = std::floor(a / b);
    return true;
  }
};

// Torcharrow_floormod generates same result as Python mod, which is different
// from C++. Reference: "floored division" in
// https://en.wikipedia.org/wiki/Modulo_operation
template <typename T>
struct torcharrow_floormod_int {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
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
};

template <typename T>
struct torcharrow_not {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& input) {
    result = !input;
    return true;
  }
};

template <typename T>
struct torcharrow_not_int {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool call(TOutput& result, const TInput& input) {
    result = ~input;
    return true;
  }
};

template <typename T>
struct torcharrow_floormod {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
    result = a - std::floor(a / b) * b;
    return true;
  }
};

template <typename T>
struct torcharrow_pow_int {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
    if (b < 0) {
      VELOX_UNSUPPORTED("Integers to negative integer powers are not allowed")
    }
    auto tmp = std::pow(a, b);
    if (tmp > std::numeric_limits<int64_t>::max() ||
        tmp < std::numeric_limits<int64_t>::min()) {
      VELOX_UNSUPPORTED(
          "Inf is outside the range of representable values of type int64")
    }
    result = tmp;
    return true;
  }
};

// Velox prestosql pow always returns double, but TorchArrow wants to return
// float when inputs are float. Ref: https://fburl.com/code/ytzqx8sg
template <typename T>
struct torcharrow_pow {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
    result = std::pow(a, b);
    return true;
  }
};

// Velox round doesn't accept boolean input nor negative decimals, and round .5
// is different from Pytorch/Pandas. E.g Velox round(2.5) is 3, but we want
// round(2.5) to 2. Ref: https://fburl.com/code/emfs990p
// This implemenation is similar to Pandas/Numpy here:
// https://github.com/numpy/numpy/blob/7b2f20b406d27364c812f7a81a9c901afbd3600c/numpy/core/src/multiarray/calculation.c#L588
template <typename T>
struct torcharrow_round {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& number, const int64_t decimals = 0) {
    if (!std::isfinite(number)) {
      result = number;
    }

    if (decimals == 0) {
      result = rint(number);
    } else if (decimals > 0) {
      TInput factor = std::pow(10, decimals);
      result = rint(number * factor) / factor;
    } else {
      TInput factor = std::pow(10, -decimals);
      result = rint(double(number) / factor) * factor;
    }
    return true;
  }
};

template <typename T>
struct torcharrow_bitwiseand {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
    result = a & b;
    return true;
  }
};

template <typename T>
struct torcharrow_bitwiseor {
  template <typename TOutput, typename TInput = TOutput>
  FOLLY_ALWAYS_INLINE bool
  call(TOutput& result, const TInput& a, const TInput& b) {
    result = a | b;
    return true;
  }
};

} // namespace facebook::torcharrow::functions
