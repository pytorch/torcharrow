// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include <velox/type/Type.h>
#include <velox/vector/BaseVector.h>
#include <velox/vector/ComplexVector.h>

namespace facebook::torcharrow {

velox::VectorPtr vectorSlice(const velox::BaseVector& src, int start, int end);

template <velox::TypeKind kind>
velox::VectorPtr
simpleVectorSlice(const velox::BaseVector& src, int start, int end);

velox::VectorPtr
arrayVectorSlice(const velox::ArrayVector& src, int start, int end);

velox::VectorPtr reshape(
    velox::VectorPtr vec,
    std::function<velox::vector_size_t(velox::vector_size_t)> offsets,
    std::function<velox::vector_size_t(velox::vector_size_t)> lengths,
    velox::vector_size_t size);

} // namespace facebook::torcharrow
