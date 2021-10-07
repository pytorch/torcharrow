// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once

#include <pybind11/pybind11.h>

#include "velox/type/Variant.h"

namespace facebook::torcharrow {

void declareUserDefinedBindings(pybind11::module& m);

// Returns true if successfully produced a `velox::variant` `out` from `obj`,
// false otherwise
bool userDefinedPyToVariant(const pybind11::handle& obj, velox::variant& out);

} // namespace facebook::torcharrow
