/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Forked from
// https://github.com/facebookincubator/velox/blob/18aff402c2b5a070ef7ec3f33cde017e90c8aa8a/velox/parse/VariantToVector.h
// since parse is not part of VELOX_MINIMAL
#pragma once

#include "velox/vector/ComplexVector.h"

namespace facebook::torcharrow {

// Converts a sequence of values from a variant array to an ArrayVector. The
// output ArrayVector contains one single row, which contains the elements
// extracted from the input variant vector.
velox::ArrayVectorPtr variantArrayToVector(
    const std::vector<velox::variant>& variantArray,
    velox::memory::MemoryPool* pool);

} // namespace facebook::velox::core
