// Copyright(c) Meta Platforms, Inc.and affiliates.
#pragma once

#include "velox/vector/ComplexVector.h"

namespace facebook::torcharrow {

// Based on VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH
//
// This macro doesn't include bool since bool often need special handling (e.g.
// Vector::rawValues doesn't work for bool type)
//
// Similar to AT_DISPATCH_ALL_TYPES in PyTorch:
// https://github.com/pytorch/pytorch/blob/8bf3179f6e3fc9d468ff34a891c081590cd2412c/aten/src/ATen/Dispatch.h#L211-L214
#define VELOX_DYNAMIC_NUMERIC_TYPE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)      \
  [&]() {                                                                     \
    switch (typeKind) {                                                       \
      case ::facebook::velox::TypeKind::INTEGER: {                            \
        return TEMPLATE_FUNC<::facebook::velox::TypeKind::INTEGER>(           \
            __VA_ARGS__);                                                     \
      }                                                                       \
      case ::facebook::velox::TypeKind::TINYINT: {                            \
        return TEMPLATE_FUNC<::facebook::velox::TypeKind::TINYINT>(           \
            __VA_ARGS__);                                                     \
      }                                                                       \
      case ::facebook::velox::TypeKind::SMALLINT: {                           \
        return TEMPLATE_FUNC<::facebook::velox::TypeKind::SMALLINT>(          \
            __VA_ARGS__);                                                     \
      }                                                                       \
      case ::facebook::velox::TypeKind::BIGINT: {                             \
        return TEMPLATE_FUNC<::facebook::velox::TypeKind::BIGINT>(            \
            __VA_ARGS__);                                                     \
      }                                                                       \
      case ::facebook::velox::TypeKind::REAL: {                               \
        return TEMPLATE_FUNC<::facebook::velox::TypeKind::REAL>(__VA_ARGS__); \
      }                                                                       \
      case ::facebook::velox::TypeKind::DOUBLE: {                             \
        return TEMPLATE_FUNC<::facebook::velox::TypeKind::DOUBLE>(            \
            __VA_ARGS__);                                                     \
      }                                                                       \
      default:                                                                \
        VELOX_FAIL(                                                           \
            "not a numeric type! kind: {}", mapTypeKindToName(typeKind));     \
    }                                                                         \
  }()

// Populate packed dense features into Tensor.
// Packed dense features are represented in Velox output slice as struct with same numeric-typed fields
//
// Tensor data is expected to be pre-allocated with correct size
void populateDenseFeaturesNoPresence(
    facebook::velox::RowVectorPtr vector,
    // TODO: remove the offset/length parameter once
    // https://github.com/facebookresearch/torcharrow/issues/177 is fixed
    facebook::velox::vector_size_t offset,
    facebook::velox::vector_size_t length,
    uintptr_t dataTensorPtr);

} // namespace facebook::torcharrow
