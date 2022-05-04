/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <algorithm>

#include <velox/functions/Macros.h>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {

template <typename T>
inline folly::F14FastMap<int64_t, float> idToScoreSumMap(T& ids) {
  folly::F14FastMap<int64_t, float> map;
  for (const auto id : ids) {
    ++map[id];
  }
  return map;
}

template <typename TIds, typename TScores>
inline folly::F14FastMap<int64_t, float> idToScoreSumMap(
    TIds& ids,
    TScores& scores) {
  VELOX_CHECK(
      ids.size() == scores.size(),
      "ids and scores should have the same len, got ids: {}, scores: {}",
      ids.size(),
      scores.size());

  folly::F14FastMap<int64_t, float> map;
  for (int i = 0; i < ids.size(); i++) {
    map[ids[i]] += scores[i];
  }
  return map;
}

inline float getScoreNorm(const folly::F14FastMap<int64_t, float>& scoreMap) {
  float innerProduct = 0;
  for (const auto v : scoreMap) {
    innerProduct += v.second * v.second;
  }
  VELOX_CHECK(innerProduct != 0, "inner product should not be zero");
  return std::sqrt(innerProduct);
}

// TODO: check if F14FastMap will be faster then sorting with vectors
template <typename TIds, typename TMatchingIds>
inline float getOverlapCount(
    const TIds& inputIds,
    const TMatchingIds& matchingIds) {
  std::vector<int64_t> ids1(inputIds.begin(), inputIds.end());
  std::vector<int64_t> ids2(matchingIds.begin(), matchingIds.end());

  std::sort(ids1.begin(), ids1.end());
  std::sort(ids2.begin(), ids2.end());

  std::vector<int64_t> intersection;
  std::set_intersection(
      ids1.begin(),
      ids1.end(),
      ids2.begin(),
      ids2.end(),
      std::back_inserter(intersection));

  return intersection.size();
}

template <typename TExecParams>
struct hasIdOverlap {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  folly::F14FastSet<int64_t> ids_;

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds) {
    ids_.clear();

    std::copy(
        inputIds.begin(), inputIds.end(), std::inserter(ids_, ids_.begin()));

    for (const auto id : matchingIds) {
      if (ids_.find(id) != ids_.end()) {
        result = 1.0;
        return;
      }
    }
    result = 0.0;
  }
};

template <typename TExecParams>
struct idOverlapCount {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds) {
    result = getOverlapCount(inputIds, matchingIds);
  }
};

template <typename TExecParams>
struct getMaxCount {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds) {
    const auto& map0 = idToScoreSumMap(inputIds);
    const auto& map1 = idToScoreSumMap(matchingIds);

    result = 0.0;
    for (auto it = map0.begin(); it != map0.end(); ++it) {
      auto match = map1.find(it->first);
      if (match != map1.end()) {
        result += std::max(it->second, match->second);
      }
    }
  }
};

template <typename TExecParams>
struct getJaccardSimilarity {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds) {
    const auto count = getOverlapCount(inputIds, matchingIds);
    result = count / (inputIds.size() + matchingIds.size() - count);
  }
};

template <typename TExecParams>
struct getCosineSimilarity {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<float>>& inputScores,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds,
      const null_free_arg_type<velox::Array<float>>& matchingScores) {
    const auto& map0 = idToScoreSumMap(inputIds, inputScores);
    const auto& map1 = idToScoreSumMap(matchingIds, matchingScores);

    float sum = 0.0;
    for (auto it = map0.begin(); it != map0.end(); ++it) {
      auto match = map1.find(it->first);
      if (match != map1.end()) {
        sum += it->second * match->second;
      }
    }
    result = sum / getScoreNorm(map0) / getScoreNorm(map1);
  }
};

template <typename TExecParams>
struct getScoreSum {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<float>>& inputScores,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds,
      const null_free_arg_type<velox::Array<float>>& matchingScores) {
    const auto& map0 = idToScoreSumMap(inputIds, inputScores);
    const auto& map1 = idToScoreSumMap(matchingIds, matchingScores);

    float sum = 0.0;
    for (auto it = map0.begin(); it != map0.end(); ++it) {
      auto match = map1.find(it->first);
      if (match != map1.end()) {
        sum += match->second;
      }
    }
    result = sum;
  }
};

template <typename TExecParams>
struct getScoreMin {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds,
      const null_free_arg_type<velox::Array<float>>& matchingScores) {
    VELOX_CHECK(
        matchingIds.size() == matchingScores.size(),
        "matching ids and scores should have the same len, got ids: {}, scores: {}",
        matchingIds.size(),
        matchingScores.size());

    folly::F14FastMap<int64_t, float> map;
    for (auto i = 0; i < matchingIds.size(); ++i) {
      auto [itr, inserted] = map.try_emplace(matchingIds[i], matchingScores[i]);
      if (!inserted) {
        itr->second = std::min(itr->second, matchingScores[i]);
      }
    }

    float min = std::numeric_limits<float>::max();
    bool found = false;
    for (const auto& id : inputIds) {
      auto match = map.find(id);
      if (match != map.end()) {
        min = std::min(min, match->second);
        found = true;
      }
    }
    result = found ? min : 0;
  }
};

template <typename TExecParams>
struct getScoreMax {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  FOLLY_ALWAYS_INLINE void callNullFree(
      float& result,
      const null_free_arg_type<velox::Array<int64_t>>& inputIds,
      const null_free_arg_type<velox::Array<int64_t>>& matchingIds,
      const null_free_arg_type<velox::Array<float>>& matchingScores) {
    VELOX_CHECK(
        matchingIds.size() == matchingScores.size(),
        "matching ids and scores should have the same len, got ids: {}, scores: {}",
        matchingIds.size(),
        matchingScores.size());

    folly::F14FastMap<int64_t, float> map;
    for (auto i = 0; i < matchingIds.size(); ++i) {
      auto [itr, inserted] = map.try_emplace(matchingIds[i], matchingScores[i]);
      if (!inserted) {
        itr->second = std::max(itr->second, matchingScores[i]);
      }
    }

    float min = std::numeric_limits<float>::lowest();
    bool found = false;
    for (const auto& id : inputIds) {
      auto match = map.find(id);
      if (match != map.end()) {
        min = std::max(min, match->second);
        found = true;
      }
    }
    result = found ? min : 0;
  }
};

} // namespace facebook::torcharrow::functions
