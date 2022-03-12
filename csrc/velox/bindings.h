/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>

#include "velox/type/Variant.h"

namespace facebook::torcharrow {

void declareUserDefinedBindings(pybind11::module& m);

// Returns true if successfully produced a `velox::variant` with Opaque type `out` from `obj`,
// false otherwise
bool userDefinedPyToOpaque(const pybind11::handle& obj, velox::variant& out);

} // namespace facebook::torcharrow
