/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pyvelox.h"


namespace facebook::pyvelox {
    namespace py = pybind11;

    template <velox::TypeKind kind>
    void declareType(py::module& m) {
        using I = typename velox::TypeTraits<kind>::ImplType;
        py::class_<I, velox::Type, std::shared_ptr<I>>(
                m,
                (std::string("VeloxType_") + velox::TypeTraits<kind>::name).c_str())
                .def(py::init());
    }

    void addVeloxBindings(py::module& m) {
        py::enum_<velox::TypeKind>(
                m,
                "TypeKind",
                py::module_local())
                .value("BOOLEAN", velox::TypeKind::BOOLEAN)
                .value("TINYINT", velox::TypeKind::TINYINT)
                .value("SMALLINT", velox::TypeKind::SMALLINT)
                .value("INTEGER", velox::TypeKind::INTEGER)
                .value("BIGINT", velox::TypeKind::BIGINT)
                .value("REAL", velox::TypeKind::REAL)
                .value("DOUBLE", velox::TypeKind::DOUBLE)
                .value("VARCHAR", velox::TypeKind::VARCHAR)
                .value("VARBINARY", velox::TypeKind::VARBINARY)
                .value("TIMESTAMP", velox::TypeKind::TIMESTAMP)
                .value("ARRAY", velox::TypeKind::ARRAY)
                .value("MAP", velox::TypeKind::MAP)
                .value("ROW", velox::TypeKind::ROW)
                .export_values();

        py::class_<velox::Type, std::shared_ptr<facebook::velox::Type>>(
                m,
                "VeloxType")
                .def("kind", &velox::Type::kind)
                .def("kind_name", &velox::Type::kindName);

        declareType<velox::TypeKind::BIGINT>(m);
        declareType<velox::TypeKind::BOOLEAN>(m);
        declareType<velox::TypeKind::TINYINT>(m);
        declareType<velox::TypeKind::SMALLINT>(m);
        declareType<velox::TypeKind::INTEGER>(m);
        declareType<velox::TypeKind::REAL>(m);
        declareType<velox::TypeKind::DOUBLE>(m);
        declareType<velox::TypeKind::VARCHAR>(m);
        declareType<velox::TypeKind::VARBINARY>(m);
        declareType<velox::TypeKind::TIMESTAMP>(m);

        using I = typename velox::TypeTraits<velox::TypeKind::ARRAY>::ImplType;
        py::class_<I, velox::Type, std::shared_ptr<I>>(
                m,
                "VeloxArrayType")
                .def(py::init<velox::TypePtr>())
                .def("element_type", &velox::ArrayType::elementType);

        using J = typename velox::FixedSizeArrayType;
        py::class_<J, velox::Type, std::shared_ptr<J>>(
                m, "VeloxFixedArrayType")
                .def(py::init<int, velox::TypePtr>())
                .def("element_type", &velox::FixedSizeArrayType::elementType)
                .def("fixed_width", &velox::FixedSizeArrayType::fixedElementsWidth);

        using M = typename velox::TypeTraits<velox::TypeKind::MAP>::ImplType;
        py::class_<M, velox::Type, std::shared_ptr<M>>(
                m,
                "VeloxMapType")
                .def(py::init<velox::TypePtr, velox::TypePtr>())
                .def("key_type", &velox::MapType::keyType)
                .def("value_type", &velox::MapType::valueType);

        using R = typename velox::TypeTraits<velox::TypeKind::ROW>::ImplType;

        py::class_<R, velox::Type, std::shared_ptr<R>>(
                m,
                "VeloxRowType")
                .def(py::init<
                        std::vector<std::string>&&,
                        std::vector<std::shared_ptr<const velox::Type>>&&>())
                .def("size", &R::size)
                .def("get_child_idx", &R::getChildIdx)
                .def("contains_child", &R::containsChild)
                .def("name_of", &R::nameOf)
                .def("child_at", &R::childAt);
    }

#ifdef CREATE_PYVELOX_MODULE
    PYBIND11_MODULE(_pyvelox, m) {
        m.doc() = R"pbdoc(
         PyVelox native code module
         -----------------------
        )pbdoc";

        addVeloxBindings(m);

        m.attr("__version__") = "dev";
    }
#endif
}

