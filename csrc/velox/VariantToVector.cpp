/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Forked from
// https://github.com/facebookincubator/velox/blob/18aff402c2b5a070ef7ec3f33cde017e90c8aa8a/velox/parse/VariantToVector.cpp
// since parse is not part of VELOX_MINIMAL

#include "velox/vector/VariantToVector.h"
#include "velox/buffer/StringViewBufferHolder.h"
#include "velox/type/Variant.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::torcharrow {
namespace {

using namespace facebook::velox;

template <TypeKind KIND>
ArrayVectorPtr variantArrayToVectorImpl(
    const std::vector<variant>& variantArray,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<KIND>::NativeType;

  // First generate internal arrayVector elements.
  const size_t variantArraySize = variantArray.size();

  // Allocate buffer and set all values to null by default.
  BufferPtr arrayElementsBuffer =
      AlignedBuffer::allocate<T>(variantArraySize, pool);
  BufferPtr nulls =
      AlignedBuffer::allocate<bool>(variantArraySize, pool, bits::kNullByte);

  // Create array elements internal flat vector.
  auto arrayElements = std::make_shared<FlatVector<T>>(
      pool,
      CppToType<T>::create(),
      nulls,
      variantArraySize,
      std::move(arrayElementsBuffer),
      std::vector<BufferPtr>());

  // Populate internal array elements (flat vector).
  for (vector_size_t i = 0; i < variantArraySize; i++) {
    if (!variantArray[i].isNull()) {
      // `getOwnedValue` copies the content to its internal buffers (in case of
      // string/StringView); no-op for other primitive types.
      arrayElements->set(i, T(variantArray[i].value<KIND>()));
    }
  }

  // Create ArrayVector around the FlatVector containing array elements.
  BufferPtr offsets = AlignedBuffer::allocate<vector_size_t>(1, pool, 0);
  BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(1, pool, 0);

  auto rawSizes = sizes->asMutable<vector_size_t>();
  rawSizes[0] = variantArraySize;

  return std::make_shared<ArrayVector>(
      pool,
      ARRAY(Type::create<KIND>()),
      BufferPtr(nullptr),
      1,
      offsets,
      sizes,
      arrayElements,
      0);
}
} // namespace

ArrayVectorPtr variantArrayToVector(
    const std::vector<variant>& variantArray,
    velox::memory::MemoryPool* pool) {
  if (variantArray.empty()) {
    return variantArrayToVectorImpl<TypeKind::UNKNOWN>(variantArray, pool);
  }

  const auto elementType = variantArray.front().inferType();
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      variantArrayToVectorImpl, elementType->kind(), variantArray, pool);
}

} // namespace facebook::torcharrow
