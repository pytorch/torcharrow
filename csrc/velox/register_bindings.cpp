// Copyright (c) Facebook, Inc. and its affiliates.
#include <pybind11/pybind11.h>
#include "velox/type/Variant.h"

namespace py = pybind11;

namespace facebook::torcharrow {

void declareUserDefinedBindings(py::module& m) {}

bool userDefinedPyToVariant(const pybind11::handle& obj, velox::variant& out) {
  return false;
}

} // namespace facebook::torcharrow
