// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <optional>

#include <gmock/gmock.h>

#include <velox/common/base/VeloxException.h>
#include <velox/vector/SimpleVector.h>
#include "pytorch/torcharrow/csrc/velox/functions/functions.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

namespace facebook::velox {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr float kInfF = std::numeric_limits<float>::infinity();
constexpr float kNanF = std::numeric_limits<float>::quiet_NaN();

MATCHER(IsNan, "is NaN") {
  return arg && std::isnan(*arg);
}

class FunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    torcharrow::functions::registerTorchArrowFunctions();
  }

  template <typename T, typename T1 = T, typename TExpected = T>
  void assertExpression(
      const std::string& expression,
      const std::vector<T>& arg0,
      const std::vector<T1>& arg1,
      const std::vector<TExpected>& expected) {
    auto vector0 = makeFlatVector(arg0);
    auto vector1 = makeFlatVector(arg1);

    auto result = evaluate<SimpleVector<TExpected>>(
        expression, makeRowVector({vector0, vector1}));
    for (int32_t i = 0; i < arg0.size(); ++i) {
      if (std::isnan(expected[i])) {
        ASSERT_TRUE(std::isnan(result->valueAt(i))) << "at " << i;
      } else {
        ASSERT_EQ(result->valueAt(i), expected[i]) << "at " << i;
      }
    }
  }

  template <typename T, typename TExpected = T>
  void assertUnaryExpression(
      const std::string& expression,
      const std::vector<T>& arg0,
      const std::vector<TExpected>& expected) {
    auto vector0 = makeFlatVector(arg0);

    auto result =
        evaluate<SimpleVector<TExpected>>(expression, makeRowVector({vector0}));
    for (int32_t i = 0; i < arg0.size(); ++i) {
      if (std::isnan(expected[i])) {
        ASSERT_TRUE(std::isnan(result->valueAt(i))) << "at " << i;
      } else {
        ASSERT_EQ(result->valueAt(i), expected[i]) << "at " << i;
      }
    }
  }

  template <typename T, typename U = T, typename V = T>
  void assertError(
      const std::string& expression,
      const std::vector<T>& arg0,
      const std::vector<U>& arg1,
      const std::string& errorMessage) {
    auto vector0 = makeFlatVector(arg0);
    auto vector1 = makeFlatVector(arg1);

    try {
      evaluate<SimpleVector<V>>(expression, makeRowVector({vector0, vector1}));
      ASSERT_TRUE(false) << "Expected an error";
    } catch (const std::exception& e) {
      ASSERT_TRUE(
          std::string(e.what()).find(errorMessage) != std::string::npos);
    }
  }
};

TEST_F(FunctionsTest, floordiv) {
  assertExpression<int32_t>(
      "torcharrow_floordiv(c0, c1)",
      {10, 11, -1, -34},
      {2, 2, 2, 10},
      {5, 5, -1, -4});
  assertExpression<int64_t>(
      "torcharrow_floordiv(c0, c1)",
      {10, 11, -1, -34},
      {2, 2, 2, 10},
      {5, 5, -1, -4});

  assertError<int32_t>(
      "torcharrow_floordiv(c0, c1)", {10}, {0}, "division by zero");
  assertError<int32_t>(
      "torcharrow_floordiv(c0, c1)", {0}, {0}, "division by zero");

  assertExpression<float>(
      "torcharrow_floordiv(c0, c1)",
      {10.5, -3.0, 1.0, 0.0},
      {2, 2, 0, 0},
      {5.0, -2.0, kInfF, kNanF});
  assertExpression<double>(
      "torcharrow_floordiv(c0, c1)",
      {10.5, -3.0, 1.0, 0.0},
      {2, 2, 0, 0},
      {5.0, -2.0, kInf, kNan});
}

TEST_F(FunctionsTest, floormod) {
  assertExpression<int32_t>(
      "torcharrow_floormod(c0, c1)",
      {13, -13, 13, -13},
      {3, 3, -3, -3},
      {1, 2, -2, -1});
  assertExpression<int64_t>(
      "torcharrow_floormod(c0, c1)",
      {13, -13, 13, -13},
      {3, 3, -3, -3},
      {1, 2, -2, -1});

  assertError<int32_t>(
      "torcharrow_floormod(c0, c1)", {10}, {0}, "Cannot divide by 0");
  assertError<int32_t>(
      "torcharrow_floormod(c0, c1)", {0}, {0}, "Cannot divide by 0");

  assertExpression<float>(
      "torcharrow_floormod(c0, c1)",
      {13.0, -13.0, 13.0, -13.0, 1.0, 0},
      {3.0, 3.0, -3.0, -3.0, 0, 0},
      {1.0, 2.0, -2.0, -1.0, kNanF, kNanF});
  assertExpression<double>(
      "torcharrow_floormod(c0, c1)",
      {13.0, -13.0, 13.0, -13.0, 1.0, 0},
      {3.0, 3.0, -3.0, -3.0, 0, 0},
      {1.0, 2.0, -2.0, -1.0, kNan, kNan});
}

TEST_F(FunctionsTest, pow) {
  std::vector<float> baseFloat = {
      0, 0, 0, -1, -1, -1, -9, 9.1, 10.1, 11.1, -11.1, 0, kInf, kInf};
  std::vector<float> exponentFloat = {
      0, 1, -1, 0, 1, -1, -3.3, 123456.432, -99.9, 0, 100000, kInf, 0, kInf};
  std::vector<float> expectedFloat;
  for (size_t i = 0; i < baseFloat.size(); i++) {
    expectedFloat.emplace_back(pow(baseFloat[i], exponentFloat[i]));
  }
  assertExpression<float>(
      "torcharrow_pow(c0, c1)", baseFloat, exponentFloat, expectedFloat);

  std::vector<double> baseDouble = {
      0, 0, 0, -1, -1, -1, -9, 9.1, 10.1, 11.1, -11.1, 0, kInf, kInf};
  std::vector<double> exponentDouble = {
      0, 1, -1, 0, 1, -1, -3.3, 123456.432, -99.9, 0, 100000, kInf, 0, kInf};
  std::vector<double> expectedDouble;
  expectedDouble.reserve(baseDouble.size());
  for (size_t i = 0; i < baseDouble.size(); i++) {
    expectedDouble.emplace_back(pow(baseDouble[i], exponentDouble[i]));
  }
  assertExpression<double>(
      "torcharrow_pow(c0, c1)", baseDouble, exponentDouble, expectedDouble);

  std::vector<int64_t> baseInt = {9, -9, 9, -9, 0};
  std::vector<int64_t> exponentInt = {3, 3, 0, 0, 0};
  std::vector<int64_t> expectedInt;
  expectedInt.reserve(baseInt.size());
  for (size_t i = 0; i < baseInt.size(); i++) {
    expectedInt.emplace_back(pow(baseInt[i], exponentInt[i]));
  }
  assertExpression<int64_t>(
      "torcharrow_pow(c0, c1)", baseInt, exponentInt, expectedInt);

  assertError<int32_t>(
      "torcharrow_pow(c0, c1)",
      {2},
      {-2},
      "Integers to negative integer powers are not allowed");
  assertError<int64_t>(
      "torcharrow_pow(c0, c1)",
      {9},
      {123456},
      "Inf is outside the range of representable values of type int64");
}

TEST_F(FunctionsTest, round) {
  std::vector<double> doubles = {
      1.3, 2.3, 1.5, 2.5, 1.8, 2.8, -1.3, -2.3, -1.5, -2.5, -1.8, -2.8};
  std::vector<double> expectedDoubles = {
      1.0, 2.0, 2.0, 2.0, 2.0, 3.0, -1.0, -2.0, -2.0, -2.0, -2.0, -3.0};
  assertUnaryExpression<double>(
      "torcharrow_round(c0)", doubles, expectedDoubles);

  std::vector<float> floats = {
      1.3, 2.3, 1.5, 2.5, 1.8, 2.8, -1.3, -2.3, -1.5, -2.5, -1.8, -2.8};
  std::vector<float> expectedFloats = {
      1.0, 2.0, 2.0, 2.0, 2.0, 3.0, -1.0, -2.0, -2.0, -2.0, -2.0, -3.0};
  assertUnaryExpression<float>("torcharrow_round(c0)", floats, expectedFloats);

  std::vector<int32_t> ints = {1, 2, -1, -2};
  std::vector<int32_t> expectedInts = {1, 2, -1, -2};
  assertUnaryExpression<int32_t>("torcharrow_round(c0)", ints, expectedInts);

  std::vector<bool> bools = {true, false};
  std::vector<float> expectedBools = {1.0, 0.0};
  assertUnaryExpression<bool, float>(
      "torcharrow_round(c0)", bools, expectedBools);

  std::vector<float> floats2 = {
      1.21,
      1.31,
      1.25,
      1.35,
      1.27,
      1.37,
      -1.21,
      -1.31,
      -1.25,
      -1.35,
      -1.27,
      -1.37};
  std::vector<float> expectedFloats2 = {
      1.2, 1.3, 1.2, 1.4, 1.3, 1.4, -1.2, -1.3, -1.2, -1.4, -1.3, -1.4};
  std::vector<int64_t> decimals2 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  assertExpression<float, int64_t>(
      "torcharrow_round(c0, c1)", floats2, decimals2, expectedFloats2);

  std::vector<float> floats3 = {
      12.1,
      14.9,
      15.0,
      15.1,
      25.0,
      25.1,
      -12.1,
      -14.9,
      -15.0,
      -15.1,
      -25.0,
      -25.1};
  std::vector<float> expectedFloats3 = {
      10.0,
      10.0,
      20.0,
      20.0,
      20.0,
      30.0,
      -10.0,
      -10.0,
      -20.0,
      -20.0,
      -20.0,
      -30.0};
  std::vector<int64_t> decimals3 = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  assertExpression<float, int64_t>(
      "torcharrow_round(c0, c1)", floats3, decimals3, expectedFloats3);

  std::vector<int32_t> ints2 = {4, 15, 25, 123, -4, -15, -25, -123};
  std::vector<int32_t> expectedInts2 = {0, 20, 20, 120, 0, -20, -20, -120};
  assertExpression<int32_t>("torcharrow_round(c0, c1)", ints2, decimals3, expectedInts2);

  std::vector<float> limits = {NAN, INFINITY, -INFINITY};
  assertUnaryExpression<float>("torcharrow_round(c0)", limits, limits);
}

} // namespace
} // namespace facebook::velox
