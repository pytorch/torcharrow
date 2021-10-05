/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
