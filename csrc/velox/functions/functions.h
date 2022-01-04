// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include "numeric_functions.h"
#include "string_functions.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/Re2Functions.h"

namespace facebook::torcharrow::functions {

inline void registerTorchArrowFunctions() {
  velox::registerFunction<udf_torcharrow_isalpha, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isalnum, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isinteger, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isdecimal, bool, velox::Varchar>(
      {"isdecimal"});
  velox::registerFunction<udf_torcharrow_islower, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isupper, bool, velox::Varchar>(
      {"isupper"});
  velox::registerFunction<udf_torcharrow_isspace, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_istitle, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isnumeric, bool, velox::Varchar>(
      {"isnumeric"});

  velox::registerFunction<udf_torcharrow_log, float, float>({"torcharrow_log"});
  velox::registerFunction<udf_torcharrow_log, double, double>(
      {"torcharrow_log"});
  velox::registerFunction<udf_torcharrow_log, float, bool>({"torcharrow_log"});
  velox::registerFunction<udf_torcharrow_log, float, int8_t>(
      {"torcharrow_log"});
  velox::registerFunction<udf_torcharrow_log, float, int16_t>(
      {"torcharrow_log"});
  velox::registerFunction<udf_torcharrow_log, float, int32_t>(
      {"torcharrow_log"});
  velox::registerFunction<udf_torcharrow_log, float, int64_t>(
      {"torcharrow_log"});

  velox::exec::registerStatefulVectorFunction(
      "match_re",
      velox::functions::re2MatchSignatures(),
      velox::functions::makeRe2Match);
}

void registerUserDefinedFunctions();

} // namespace facebook::torcharrow::functions
