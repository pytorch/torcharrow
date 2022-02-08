// Copyright(c) Meta Platforms, Inc.and affiliates.
#include "tensor_conversion.h"
#include <velox/common/base/Exceptions.h>
#include "velox/vector/FlatVector.h"

namespace facebook::torcharrow {

template <velox::TypeKind kind>
void doPopulateDenseFeaturesNoPresence(
    facebook::velox::RowVectorPtr vector,
    facebook::velox::vector_size_t offset,
    facebook::velox::vector_size_t length,
    uintptr_t dataTensorPtr) {
  using T = typename velox::TypeTraits<kind>::NativeType;

  auto* castedDataTensorPtr = reinterpret_cast<T*>(dataTensorPtr);
  const auto batchSize = length;

  for (auto i = 0; i < vector->childrenSize(); i++) {
    const auto* childFlatVector = vector->childAt(i)->asFlatVector<T>();
    std::copy(
        childFlatVector->rawValues() + offset,
        childFlatVector->rawValues() + offset + length,
        castedDataTensorPtr + i * batchSize);
  }
}

void populateDenseFeaturesNoPresence(
    facebook::velox::RowVectorPtr vector,
    facebook::velox::vector_size_t offset,
    facebook::velox::vector_size_t length,
    uintptr_t dataTensorPtr) {
  VELOX_DYNAMIC_NUMERIC_TYPE_DISPATCH(
      doPopulateDenseFeaturesNoPresence,
      vector->childAt(0)->typeKind(),
      vector,
      offset,
      length,
      dataTensorPtr);
}

} // namespace facebook::torcharrow
