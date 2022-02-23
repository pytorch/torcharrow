/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include "velox/type/Variant.h"

namespace py = pybind11;

namespace facebook::torcharrow {

void declareUserDefinedBindings(py::module& m) {}

bool userDefinedPyToVariant(const pybind11::handle& obj, velox::variant& out) {
  return false;
}

} // namespace facebook::torcharrow
