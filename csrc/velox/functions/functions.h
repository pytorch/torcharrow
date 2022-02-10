// Copyright (c) Facebook, Inc. and its affiliates.
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
  // TODO: Fix inconsistent udf alias isdecimal, isupper, isnumeric.
  velox::registerFunction<torcharrow_isdecimal, bool, velox::Varchar>(
      {"isdecimal"});
  velox::registerFunction<torcharrow_islower, bool, velox::Varchar>(
      {"torcharrow_islower"});
  velox::registerFunction<torcharrow_isupper, bool, velox::Varchar>(
      {"isupper"});
  velox::registerFunction<torcharrow_isspace, bool, velox::Varchar>(
      {"torcharrow_isspace"});
  velox::registerFunction<torcharrow_istitle, bool, velox::Varchar>(
      {"torcharrow_istitle"});
  velox::registerFunction<torcharrow_isnumeric, bool, velox::Varchar>(
      {"isnumeric"});

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
  velox::registerFunction<torcharrow_pow_int, int64_t, int8_t, int8_t>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int64_t, int16_t, int16_t>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int64_t, int32_t, int32_t>(
      {"torcharrow_pow"});
  velox::registerFunction<torcharrow_pow_int, int64_t, int64_t, int64_t>(
      {"torcharrow_pow"});

  velox::exec::registerStatefulVectorFunction(
      "match_re",
      velox::functions::re2MatchSignatures(),
      velox::functions::makeRe2Match);
}

void registerUserDefinedFunctions();

} // namespace facebook::torcharrow::functions
