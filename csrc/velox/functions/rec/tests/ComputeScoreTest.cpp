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
#include "velox/parse/TypeResolver.h"

namespace facebook::velox {
namespace {

using namespace facebook::velox::test;

template <typename TExpected, typename TResult>
// TODO maybe up stream this to VectorTestBase
void assertFloatEqVectors(TExpected expected, TResult result) {
  ASSERT_EQ(expected->size(), result->size());
  ASSERT_EQ(expected->typeKind(), result->typeKind());
  for (size_t i = 0; i < result->size(); i++) {
    EXPECT_FLOAT_EQ(expected->valueAt(i), result->valueAt(i));
  }
}

class ComputeScoreTest : public functions::test::FunctionBaseTest {
 protected:
  std::vector<std::vector<int64_t>> inputIdsVec;
  std::vector<std::vector<float>> inputIdScoresVec;
  std::vector<std::vector<int64_t>> matchingIdsVec;
  std::vector<std::vector<float>> matchingIdScoresVec;
  static void SetUpTestCase() {
    torcharrow::functions::registerTorchArrowFunctions();
    parse::registerTypeResolver();
  }

  virtual void SetUp() override {
    inputIdsVec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 10, 11, 12}};
    inputIdScoresVec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 10, 11, 12}};
    matchingIdsVec = {{1, 2, 3}, {0, 10}, {7, 10, 10}, {10, 10, 11, 13}};
    matchingIdScoresVec = {{1, 2, 3}, {0, 10}, {7, 10, 10}, {10, 10, 11, 13}};
  }
};

TEST_F(ComputeScoreTest, HasIdOverlap) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto result = evaluate<SimpleVector<float>>(
      "has_id_overlap(c0,c1)", makeRowVector({inputIds, matchingIds}));

  std::vector<float> expected = {1.0, 0.0, 1.0, 1.0};
  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, IdOverlapCount) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto result = evaluate<SimpleVector<float>>(
      "id_overlap_count(c0,c1)", makeRowVector({inputIds, matchingIds}));

  std::vector<float> expected = {3.0, 0.0, 1.0, 3.0};
  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, GetMaxCount) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto result = evaluate<SimpleVector<float>>(
      "get_max_count(c0,c1)", makeRowVector({inputIds, matchingIds}));

  std::vector<float> expected = {3.0, 0.0, 1.0, 3.0};
  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, GetJaccardSimilarity) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto result = evaluate<SimpleVector<float>>(
      "get_jaccard_similarity(c0,c1)", makeRowVector({inputIds, matchingIds}));

  std::vector<float> expected = {3.0 / 3.0, 0 / 5.0, 1.0 / 5.0, 3.0 / 5.0};
  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, GetCosineSimilarity) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto inputIdScores = makeArrayVector(inputIdScoresVec);
  auto matchingIdScores = makeArrayVector(matchingIdScoresVec);
  auto result = evaluate<SimpleVector<float>>(
      "get_cosine_similarity(c0,c1,c2,c3)",
      makeRowVector({inputIds, inputIdScores, matchingIds, matchingIdScores}));

  std::vector<float> expected = {
      static_cast<float>(
          (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) /
          std::sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) /
          std::sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0)),
      static_cast<float>(0.0),
      static_cast<float>(
          7.0 * 7.0 / std::sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0) /
          std::sqrt(7.0 * 7.0 + 20.0 * 20.0)),
      static_cast<float>(
          (20.0 * 20.0 + 11 * 11) / std::sqrt(20 * 20 + 11 * 11 + 12 * 12) /
          std::sqrt(20 * 20 + 11 * 11 + 13 * 13))};

  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, GetScoreSum) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto inputIdScores = makeArrayVector(inputIdScoresVec);
  auto matchingIdScores = makeArrayVector(matchingIdScoresVec);
  auto result = evaluate<SimpleVector<float>>(
      "get_score_sum(c0,c1,c2,c3)",
      makeRowVector({inputIds, inputIdScores, matchingIds, matchingIdScores}));

  std::vector<float> expected = {
      static_cast<float>(1.0 + 2.0 + 3.0),
      static_cast<float>(0.0),
      static_cast<float>(7.0),
      static_cast<float>(10.0 + 10.0 + 11.0)};

  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, GetScoreMin) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto matchingIdScores = makeArrayVector(matchingIdScoresVec);

  auto result = evaluate<SimpleVector<float>>(
      "get_score_min(c0,c1,c2)",
      makeRowVector({inputIds, matchingIds, matchingIdScores}));

  std::vector<float> expected = {
      static_cast<float>(1.0),
      static_cast<float>(0.0),
      static_cast<float>(7.0),
      static_cast<float>(10.0)};

  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

TEST_F(ComputeScoreTest, GetScoreMax) {
  auto inputIds = makeArrayVector(inputIdsVec);
  auto matchingIds = makeArrayVector(matchingIdsVec);
  auto matchingIdScores = makeArrayVector(matchingIdScoresVec);
  auto result = evaluate<SimpleVector<float>>(
      "get_score_max(c0,c1,c2)",
      makeRowVector({inputIds, matchingIds, matchingIdScores}));

  std::vector<float> expected = {
      static_cast<float>(3.0),
      static_cast<float>(0.0),
      static_cast<float>(7.0),
      static_cast<float>(11.0)};

  assertFloatEqVectors(
      velox::test::VectorTestBase::makeFlatVector(expected), result);
}

} // namespace
} // namespace facebook::velox
