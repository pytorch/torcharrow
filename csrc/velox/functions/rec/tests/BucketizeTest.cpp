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

namespace facebook::velox {
namespace {

  using namespace facebook::velox::test;
  
class BucketizeTest : public functions::test::FunctionBaseTest {
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

TEST_F(BucketizeTest, IntInputFloatBorders) {
  const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
  std::vector<std::vector<float>> allBorders(
      a.size(), {0.0f, 2.0f, 3.0f, 5.0f, 500.0f, 1000.0f});
  assertArrayExpression<int32_t, std::vector<float>, int32_t>(
      "bucketize(c0, c1)", a, allBorders, {1, 1, 2, 3, 3, 4});
}

TEST_F(BucketizeTest, FloatInputFloatBorders) {
  const std::vector<float> a = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<float> borders{0.0f, 2.0f, 3.0f, 5.0f, 500.0f, 1000.0f};
  std::vector<std::vector<float>> allBorders(a.size(), borders);
  assertArrayExpression<float, std::vector<float>, int32_t>(
      "bucketize(c0, c1)", a, allBorders, {1, 2, 3, 3, 4, 4});
}

TEST_F(BucketizeTest, IntInputIntBorders) {
  const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6};
  std::vector<std::vector<float>> allBorders(a.size(), {0, 2, 3, 5, 500, 1000});
  assertArrayExpression<int32_t, std::vector<float>, int32_t>(
      "bucketize(c0, c1)", a, allBorders, {1, 1, 2, 3, 3, 4});
}

TEST_F(BucketizeTest, FloatInputIntBorders) {
  const std::vector<float> a = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<float> borders{0, 2, 3, 5, 500, 1000};
  std::vector<std::vector<float>> allBorders(a.size(), borders);
  assertArrayExpression<float, std::vector<float>, int32_t>(
      "bucketize(c0, c1)", a, allBorders, {1, 2, 3, 3, 4, 4});
}

TEST_F(BucketizeTest, DecreasingBorder) {
  const std::vector<float> a = {1.5f};
  std::vector<std::vector<float>> allBorders(a.size(), {3.0f, 2.0f, 1.0f});
  // TODO: maybe use ASSERT_THROW_KEEP_AS_E to check error msg.
  EXPECT_THROW(
      (assertArrayExpression<float, std::vector<float>, int32_t>(
          "bucketize(c0, c1)", a, allBorders, {})),
      velox::VeloxException);
}

TEST_F(BucketizeTest, RepeatedBorderValuesOk) {
  const std::vector<float> a = {1.5f};
  std::vector<std::vector<float>> allBorders(a.size(), {1, 2, 2, 3, 3});
  assertArrayExpression<float, std::vector<float>, int32_t>(
      "bucketize(c0, c1)", a, allBorders, {1});
}

TEST_F(BucketizeTest, FloatIdListInput) {
  std::vector<std::vector<float>> a = {{1.5f, 2.5f, 150.0f}, {10.0f, 200.0f}};

  std::vector<float> borders{1.0f, 10.0f, 100.0f};
  std::vector<std::vector<float>> allBorders(a.size(), borders);

  std::vector<std::vector<int32_t>> expected = {{1, 1, 3}, {1, 3}};

  auto vector0 = makeArrayVector(a);
  auto vector1 = makeArrayVector(allBorders);

  auto result = evaluate<velox::ArrayVector>(
      "bucketize(c0, c1)", makeRowVector({vector0, vector1}));

  assertEqualVectors(
      velox::test::VectorTestBase::makeArrayVector(expected), result);
}

TEST_F(BucketizeTest, IntIdListInput) {
  std::vector<std::vector<int32_t>> a = {{1, 2, 150}, {10, 200}};

  std::vector<float> borders{1.0f, 10.0f, 100.0f};
  std::vector<std::vector<float>> allBorders(a.size(), borders);

  std::vector<std::vector<int32_t>> expected = {{0, 1, 3}, {1, 3}};

  auto vector0 = makeArrayVector(a);
  auto vector1 = makeArrayVector(allBorders);

  auto result = evaluate<velox::ArrayVector>(
      "bucketize(c0, c1)", makeRowVector({vector0, vector1}));

  assertEqualVectors(
      velox::test::VectorTestBase::makeArrayVector(expected), result);
}

} // namespace
} // namespace facebook::velox
