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
#include <velox/vector/tests/VectorMaker.h>

#include "pytorch/torcharrow/csrc/velox/functions/functions.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

namespace facebook::velox {
namespace {

using namespace facebook::velox::test;

class SigridHashTest : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    torcharrow::functions::registerTorchArrowFunctions();
  }

  template <typename T, typename T1 = T, typename TExpected = T>
  void assertArrayExpression(
      const std::string& expression,
      const std::vector<T>& arg0,
      const std::vector<T1>& arg1,
      const std::vector<TExpected>& expected) {
    auto vector0 = makeFlatVector(arg0);
    auto vector1 = makeArrayVector(arg1);

    auto result = evaluate<SimpleVector<TExpected>>(
        expression, makeRowVector({vector0, vector1}));

    assertEqualVectors(
        velox::test::VectorTestBase::makeFlatVector(expected), result);
  }
};

// Below test values are derived from some examples from running
// koski.function.sigrid_hash

TEST_F(SigridHashTest, Val) {
  std::vector<int64_t> a = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int64_t> salts(a.size(), 0);
  std::vector<int64_t> maxValues(a.size(), 100);

  auto vector0 = makeFlatVector(a);
  auto vector1 = makeFlatVector(salts);
  auto vector2 = makeFlatVector(maxValues);

  auto result = evaluate<SimpleVector<int64_t>>(
      "sigrid_hash(c0,c1,c2)", makeRowVector({vector0, vector1, vector2}));

  std::vector<int64_t> expected = {
      60, 54, 54, 9, 4, 91, 11, 67, 79, 2, 25, 92, 98, 83, 66, 2};
  assertEqualVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(SigridHashTest, ValList) {
  std::vector<std::vector<int64_t>> a = {{1}, {14, 15, 16}};
  std::vector<int64_t> salts(a.size(), 0);
  std::vector<int64_t> maxValues(a.size(), 100);

  auto vector0 = makeArrayVector(a);
  auto vector1 = makeFlatVector(salts);
  auto vector2 = makeFlatVector(maxValues);

  auto result = evaluate<ArrayVector>(
      "sigrid_hash(c0,c1,c2)", makeRowVector({vector0, vector1, vector2}));

  std::vector<std::vector<int64_t>> expected = {{60}, {83, 66, 2}};
  assertEqualVectors(
      velox::test::VectorTestBase::makeArrayVector(expected), result);
}

} // namespace
} // namespace facebook::velox
