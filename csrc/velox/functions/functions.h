/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <velox/functions/Registerer.h>
#include "numeric_functions.h"
#include "string_functions.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/Re2Functions.h"

namespace facebook::torcharrow::functions {

inline void registerTorchArrowFunctions() {
  velox::registerFunction<torcharrow_isalpha, bool, velox::Varchar>(
      {"torcharrow_isalpha"});
  velox::registerFunction<torcharrow_isalnum, bool, velox::Varchar>(
      {"torcharrow_isalnum"});
  velox::registerFunction<torcharrow_isdigit, bool, velox::Varchar>(
      {"torcharrow_isdigit"});
  velox::registerFunction<torcharrow_isdecimal, bool, velox::Varchar>(
      {"torcharrow_isdecimal"});
  velox::registerFunction<torcharrow_islower, bool, velox::Varchar>(
      {"torcharrow_islower"});
  velox::registerFunction<torcharrow_isupper, bool, velox::Varchar>(
      {"torcharrow_isupper"});
  velox::registerFunction<torcharrow_isspace, bool, velox::Varchar>(
      {"torcharrow_isspace"});
  velox::registerFunction<torcharrow_istitle, bool, velox::Varchar>(
      {"torcharrow_istitle"});
  velox::registerFunction<torcharrow_isnumeric, bool, velox::Varchar>(
      {"torcharrow_isnumeric"});

  // Natural logarithm
  velox::registerFunction<torcharrow_log, float, float>({"torcharrow_log"});
  velox::registerFunction<torcharrow_log, double, double>({"torcharrow_log"});
  velox::registerFunction<torcharrow_log, float, bool>({"torcharrow_log"});
  // TODO: support type promotion in TorchArrow-Velox backend so registering
  // less overloads.
  velox::registerFunction<torcharrow_log, float, int8_t>({"torcharrow_log"});
  velox::registerFunction<torcharrow_log, float, int16_t>({"torcharrow_log"});
  velox::registerFunction<torcharrow_log, float, int32_t>({"torcharrow_log"});
  velox::registerFunction<torcharrow_log, float, int64_t>({"torcharrow_log"});

  // Floor divide
  velox::registerFunction<torcharrow_floordiv, float, float, float>(
      {"torcharrow_floordiv"});
  velox::registerFunction<torcharrow_floordiv, double, double, double>(
      {"torcharrow_floordiv"});
  velox::registerFunction<torcharrow_floordiv_int, int8_t, int8_t, int8_t>(
      {"torcharrow_floordiv"});
  velox::registerFunction<torcharrow_floordiv_int, int16_t, int16_t, int16_t>(
      {"torcharrow_floordiv"});
  velox::registerFunction<torcharrow_floordiv_int, int32_t, int32_t, int32_t>(
      {"torcharrow_floordiv"});
  velox::registerFunction<torcharrow_floordiv_int, int64_t, int64_t, int64_t>(
      {"torcharrow_floordiv"});

  // Floor mod
  velox::registerFunction<torcharrow_floormod, float, float, float>(
      {"torcharrow_floormod"});
  velox::registerFunction<torcharrow_floormod, double, double, double>(
      {"torcharrow_floormod"});
  velox::registerFunction<torcharrow_floormod_int, int8_t, int8_t, int8_t>(
      {"torcharrow_floormod"});
  velox::registerFunction<torcharrow_floormod_int, int16_t, int16_t, int16_t>(
      {"torcharrow_floormod"});
  velox::registerFunction<torcharrow_floormod_int, int32_t, int32_t, int32_t>(
      {"torcharrow_floormod"});
  velox::registerFunction<torcharrow_floormod_int, int64_t, int64_t, int64_t>(
      {"torcharrow_floormod"});

  // Pow
  velox::registerFunction<torcharrow_pow, float, float, float>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow, double, double, double>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int8_t, int8_t, int8_t>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int16_t, int16_t, int16_t>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int32_t, int32_t, int32_t>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int64_t, int64_t, int64_t>(
      {"torcharrow_pow"});

  // Bitwise
  // only supporting int and bool, not float or double
  velox::registerFunction<torcharrow_bitwiseand, bool, bool, bool>(
      {"torcharrow_bitwiseand"});
  velox::registerFunction<torcharrow_bitwiseand, int8_t, int8_t, int8_t>(
      {"torcharrow_bitwiseand"});
  velox::registerFunction<torcharrow_bitwiseand, int16_t, int16_t, int16_t>(
      {"torcharrow_bitwiseand"});
  velox::registerFunction<torcharrow_bitwiseand, int32_t, int32_t, int32_t>(
      {"torcharrow_bitwiseand"});
  velox::registerFunction<torcharrow_bitwiseand, int64_t, int64_t, int64_t>(
      {"torcharrow_bitwiseand"});

  velox::registerFunction<torcharrow_bitwiseor, bool, bool, bool>(
      {"torcharrow_bitwiseor"});
  velox::registerFunction<torcharrow_bitwiseor, int8_t, int8_t, int8_t>(
      {"torcharrow_bitwiseor"});
  velox::registerFunction<torcharrow_bitwiseor, int16_t, int16_t, int16_t>(
      {"torcharrow_bitwiseor"});
  velox::registerFunction<torcharrow_bitwiseor, int32_t, int32_t, int32_t>(
      {"torcharrow_bitwiseor"});
  velox::registerFunction<torcharrow_bitwiseor, int64_t, int64_t, int64_t>(
      {"torcharrow_bitwiseor"});


  // Round
  velox::registerFunction<torcharrow_round, float, float>({"torcharrow_round"});
  velox::registerFunction<torcharrow_round, double, double>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int8_t, int8_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int16_t, int16_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int32_t, int32_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int64_t, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, float, bool>({"torcharrow_round"});
  velox::registerFunction<torcharrow_round, float, float, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, double, double, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int8_t, int8_t, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int16_t, int16_t, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int32_t, int32_t, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, int64_t, int64_t, int64_t>(
      {"torcharrow_round"});
  velox::registerFunction<torcharrow_round, float, bool, int64_t>(
      {"torcharrow_round"});

  // Invert/Not
  velox::registerFunction<torcharrow_not, bool, bool>({"torcharrow_not"});
  velox::registerFunction<torcharrow_not_int, int8_t, int8_t>({"torcharrow_not"});
  velox::registerFunction<torcharrow_not_int, int16_t, int16_t>({"torcharrow_not"});
  velox::registerFunction<torcharrow_not_int, int32_t, int32_t>({"torcharrow_not"});
  velox::registerFunction<torcharrow_not_int, int64_t, int64_t>({"torcharrow_not"});

  // TODO: consider to refactor registration code with helper functions
  // to save some lines, like https://fburl.com/code/dk6zi7t3

  velox::exec::registerStatefulVectorFunction(
      "match_re",
      velox::functions::re2MatchSignatures(),
      velox::functions::makeRe2Match);
}

void registerUserDefinedFunctions();

} // namespace facebook::torcharrow::functions
