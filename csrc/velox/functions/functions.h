// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include "string_functions.h"
#include "velox/exec/tests/utils/FunctionUtils.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/Re2Functions.h"
#include "velox/parse/Expressions.h"

namespace facebook::torcharrow::functions {

inline void registerTorchArrowFunctions() {
  velox::registerFunction<udf_torcharrow_isalpha, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isalnum, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isinteger, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isdecimal, bool, velox::Varchar>({"isdecimal"});
  velox::registerFunction<udf_torcharrow_islower, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isupper, bool, velox::Varchar>({"isupper"});
  velox::registerFunction<udf_torcharrow_isspace, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_istitle, bool, velox::Varchar>();
  velox::registerFunction<udf_torcharrow_isnumeric, bool, velox::Varchar>(
      {"isnumeric"});

  velox::exec::registerStatefulVectorFunction(
      "match_re",
      velox::functions::re2MatchSignatures(),
      velox::functions::makeRe2Match);
}

inline void initializeTorchArrowTypeResolver() {
  velox::exec::test::registerTypeResolver();
}

void registerUserDefinedFunctions();

} // namespace facebook::torcharrow::functions
