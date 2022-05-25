/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <velox/common/base/VeloxException.h>
#include <velox/vector/SimpleVector.h>
#include <velox/vector/tests/VectorTestBase.h>

#include "pytorch/torcharrow/csrc/velox/functions/functions.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox {
namespace {

using namespace facebook::velox::test;

class firstXTest : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    torcharrow::functions::registerTorchArrowFunctions();
    parse::registerTypeResolver();
  }
};

TEST_F(firstXTest, basicUsage) {
  std::vector<std::vector<int64_t>> a = {{1, 2}, {3, 4}, {5, 6}};
  std::vector<int32_t> b(a.size(), 1);

  auto c0 = makeArrayVector(a);
  auto c1 = makeFlatVector(b);
  auto c = evaluate<ArrayVector>("firstx(c0, c1)", makeRowVector({c0, c1}));

  std::vector<std::vector<int64_t>> expected = {{1}, {3}, {5}};
  assertEqualVectors(makeArrayVector(expected), c);
}

} // namespace
} // namespace facebook::velox
