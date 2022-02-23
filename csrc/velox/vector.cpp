/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "vector.h"
#include <velox/common/memory/Memory.h>
#include <velox/type/Type.h>
#include <velox/vector/BaseVector.h>
#include <velox/vector/ComplexVector.h>
#include <iostream>

namespace facebook::torcharrow {

template <velox::TypeKind kind>
velox::VectorPtr
simpleVectorSlice(const velox::BaseVector& src, int start, int end) {
  using T = typename velox::TypeTraits<kind>::NativeType;
  auto newVector = velox::BaseVector::create(
      velox::CppToType<T>::create(), end - start, src.pool());
  newVector.get()->copy(&src, 0, start, end - start);
  return newVector;
}

velox::VectorPtr
arrayVectorSlice(const velox::ArrayVector& src, int start, int end) {
  auto length = end - start;
  std::shared_ptr<const velox::Type> elementType = src.type();
  auto result = velox::BaseVector::create(ARRAY(elementType), length, src.pool());
  auto ptr = result.get()->as<velox::ArrayVector>();
  if (length > 0) {
    ptr->setElements(vectorSlice(
        *src.elements(),
        src.offsetAt(start),
        src.offsetAt(end - 1) + src.sizeAt(end - 1)));
  }

  for (int i = 0; i < length; i++) {
    auto isNull = src.isNullAt(start + i);
    ptr->setNull(i, isNull);
    if (!isNull) {
      auto offset = src.offsetAt(start + i) - src.offsetAt(start);
      auto size = src.sizeAt(start + i);
      ptr->setOffsetAndSize(i, offset, size);
    }
  }

  return result;
}

velox::VectorPtr vectorSlice(const velox::BaseVector& src, int start, int end) {
  auto type = src.type();
  auto kind = type.get()->kind();
  switch (kind) {
    case velox::TypeKind::ARRAY: {
      return arrayVectorSlice(*src.as<velox::ArrayVector>(), start, end);
    }
    case velox::TypeKind::MAP: {
      throw "Not implemented yet.";
    }
    default:
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          simpleVectorSlice, kind, src, start, end);
  }
}

velox::VectorPtr reshape(
    velox::VectorPtr vec,
    std::function<velox::vector_size_t(velox::vector_size_t)> offsets,
    std::function<velox::vector_size_t(velox::vector_size_t)> lengths,
    velox::vector_size_t size) {
  auto result = velox::BaseVector::create(
      ARRAY(vec.get()->type()), size, vec.get()->pool());
  result.get()->as<velox::ArrayVector>()->setElements(vec);
  for (int i = 0; i < size; i++) {
    result.get()->as<velox::ArrayVector>()->setOffsetAndSize(
        i, offsets(i), lengths(i));
  }
  return result;
}

} // namespace facebook::torcharrow
