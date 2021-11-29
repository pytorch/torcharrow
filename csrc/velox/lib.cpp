// Copyright (c) Facebook, Inc. and its affiliates.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <velox/common/memory/Memory.h>
#include <velox/type/Type.h>
#include <velox/vector/TypeAliases.h>
#include <iostream>
#include <memory>

#include "bindings.h"
#include "column.h"
#include "functions/functions.h" // @manual=//pytorch/torcharrow/csrc/velox/functions:torcharrow_functions
#include "velox/buffer/StringViewBufferHolder.h"
#include "velox/functions/prestosql/SimpleFunctions.h"
#include "velox/functions/prestosql/VectorFunctions.h"
#include "velox/type/Type.h"
#include "velox/vector/TypeAliases.h"
#include "velox/vector/arrow/Bridge.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<bool>);

namespace facebook::torcharrow {

//
// SimpleColumn (scalar types)
//

// Based on VectorMaker::flatVectorNullable in Velox
template <typename T>
velox::FlatVectorPtr<T> flatVectorFromPyList(const py::list& data) {
  // TODO
  // Consider using the pattern used in arrayVectorFromPyList for creating the
  // underlying FlatVector, which creates an empty vector using
  // BaseVector::create() and then calls set() and setNullCount() to let the
  // library handle internal buffer allocation, call the appropriate API for
  // specific types, and keep internal data in sync with each other

  velox::BufferPtr dataBuffer = velox::AlignedBuffer::allocate<T>(
      data.size(), TorchArrowGlobalStatic::rootMemoryPool());
  velox::BufferPtr nullBuffer = velox::AlignedBuffer::allocate<bool>(
      data.size(), TorchArrowGlobalStatic::rootMemoryPool());

  T* rawData = dataBuffer->asMutable<T>();
  uint64_t* rawNulls = nullBuffer->asMutable<uint64_t>();
  // For non-string types, stringArena is merely a lightweight proxy for
  // creating an empty std::vector<BufferPtr> to be passed to construct the
  // FlatVector
  velox::StringViewBufferHolder stringArena(
      TorchArrowGlobalStatic::rootMemoryPool());
  velox::vector_size_t nullCount = 0;

  for (py::size_t i = 0; i < data.size(); i++) {
    if (!data[i].is_none()) {
      // Using bitUtils for bool vectors.
      if constexpr (std::is_same<T, bool>::value) {
        velox::bits::setBit(rawData, i, data[i].cast<bool>());
      } else if constexpr (std::is_same<T, velox::StringView>::value) {
        // Two memcpy's happen here: pybind11::object casting to std::string and
        // StringViewBufferHolder copying data from the buffer in the
        // std::string onto the buffers it manages. We can teach
        // StringViewBufferHolder how to copy data from
        // pybind11::str/pybind11::object to skip one copy
        rawData[i] = stringArena.getOwnedValue(data[i].cast<std::string>());
      } else {
        rawData[i] = data[i].cast<T>();
      }
      velox::bits::setNull(rawNulls, i, false);
    } else {
      // Prevent null StringViews to point to garbage.
      if constexpr (std::is_same<T, velox::StringView>::value) {
        rawData[i] = T();
      }
      velox::bits::setNull(rawNulls, i, true);
      ++nullCount;
    }
  }

  auto flatVector = std::make_shared<velox::FlatVector<T>>(
      TorchArrowGlobalStatic::rootMemoryPool(),
      std::move(nullBuffer),
      data.size(),
      std::move(dataBuffer),
      stringArena.moveBuffers());
  flatVector->setNullCount(nullCount);

  return flatVector;
}

template <
    velox::TypeKind kind,
    typename D,
    typename T = typename velox::TypeTraits<kind>::NativeType>
py::class_<SimpleColumn<T>, BaseColumn> declareSimpleType(
    py::module& m,
    const D& decoder) {
  py::class_<SimpleColumn<T>, BaseColumn> result(
      m, (std::string("SimpleColumn") + velox::TypeTraits<kind>::name).c_str());
  result
      .def(
          "__getitem__",
          [&decoder](SimpleColumn<T>& self, int index) {
            return decoder(self.valueAt(index));
          })
      .def("append_null", &SimpleColumn<T>::appendNull)
      .def("slice", &SimpleColumn<T>::slice);

  py::class_<FlatColumn<T>, SimpleColumn<T>>(
      m, (std::string("FlatColumn") + velox::TypeTraits<kind>::name).c_str());

  py::class_<ConstantColumn<T>, SimpleColumn<T>>(
      m,
      (std::string("ConstantColumn") + velox::TypeTraits<kind>::name).c_str())
      .def("__getitem__", [&decoder](ConstantColumn<T>& self, int index) {
        return decoder(self.valueAt(index));
      });

  using I = typename velox::TypeTraits<kind>::ImplType;
  py::class_<I, velox::Type, std::shared_ptr<I>>(
      m,
      (std::string("VeloxType_") + velox::TypeTraits<kind>::name).c_str(),
      // TODO: Move the Koksi binding of Velox type to OSS
      py::module_local())
      .def(py::init());

  // Empty Column
  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<SimpleColumn<T>>();
  });
  // Column construction from Python list
  m.def(
      "Column",
      [](std::shared_ptr<I> type,
         py::list data) -> std::unique_ptr<SimpleColumn<T>> {
        return std::make_unique<SimpleColumn<T>>(flatVectorFromPyList<T>(data));
      });

  // Import Arrow data
  //
  // VARCHAR is not supported at the moment
  // https://github.com/facebookincubator/velox/blob/f420d3115eeb8ad782aa9979f47be03671ed02f4/velox/vector/arrow/Bridge.cpp#L128
  if constexpr (
      kind == velox::TypeKind::BOOLEAN || kind == velox::TypeKind::TINYINT ||
      kind == velox::TypeKind::SMALLINT || kind == velox::TypeKind::INTEGER ||
      kind == velox::TypeKind::BIGINT || kind == velox::TypeKind::REAL ||
      kind == velox::TypeKind::DOUBLE) {
    m.def(
        "_import_from_arrow",
        [](std::shared_ptr<I> type, uintptr_t ptrArray, uintptr_t ptrSchema) {
          auto column =
              std::make_unique<SimpleColumn<T>>(velox::importFromArrowAsOwner(
                  *reinterpret_cast<ArrowSchema*>(ptrSchema),
                  *reinterpret_cast<ArrowArray*>(ptrArray),
                  TorchArrowGlobalStatic::rootMemoryPool()));

          VELOX_CHECK(
              column->type()->kind() == kind,
              "Expected TypeKind is {} but Velox created {}",
              velox::TypeTraits<kind>::name,
              column->type()->kindName());

          return column;
        });
  }

  return result;
};

template <typename T>
py::class_<SimpleColumn<T>, BaseColumn>& declareComparisons(
    py::class_<SimpleColumn<T>, BaseColumn>& pyClass) {
  return pyClass
      .def(
          "eq",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Eq, OperatorType::Direct);
          })
      .def(
          "eq",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Eq, OperatorType::Direct);
          })
      .def(
          "neq",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Neq, OperatorType::Direct);
          })
      .def(
          "neq",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Neq, OperatorType::Direct);
          })
      .def(
          "lt",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Lt, OperatorType::Direct);
          })
      .def(
          "lt",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Lt, OperatorType::Direct);
          })
      .def(
          "gt",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Gt, OperatorType::Direct);
          })
      .def(
          "gt",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Gt, OperatorType::Direct);
          })
      .def(
          "lte",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Lte, OperatorType::Direct);
          })
      .def(
          "lte",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Lte, OperatorType::Direct);
          })
      .def(
          "gte",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Gte, OperatorType::Direct);
          })
      .def(
          "gte",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(b, BinaryOpCode::Gte, OperatorType::Direct);
          });
}

template <typename T>
py::class_<SimpleColumn<T>, BaseColumn>& declareBitwiseOperations(
    py::class_<SimpleColumn<T>, BaseColumn>& pyClass) {
  return pyClass
      .def(
          "bitwise_and",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseAnd, OperatorType::Direct);
          })
      .def(
          "bitwise_and",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseAnd, OperatorType::Direct);
          })
      .def(
          "bitwise_rand",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseAnd, OperatorType::Reverse);
          })
      .def(
          "bitwise_or",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseOr, OperatorType::Direct);
          })
      .def(
          "bitwise_or",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseOr, OperatorType::Direct);
          })
      .def(
          "bitwise_ror",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseOr, OperatorType::Reverse);
          })
      .def(
          "bitwise_xor",
          [](SimpleColumn<T>& a,
             const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseXor, OperatorType::Direct);
          })
      .def(
          "bitwise_xor",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseXor, OperatorType::Direct);
          })
      .def(
          "bitwise_rxor",
          [](SimpleColumn<T>& a,
             const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
            return a.callBinaryOp(
                b, BinaryOpCode::BitwiseXor, OperatorType::Reverse);
          });
}

template <
    velox::TypeKind kind,
    typename T = typename velox::TypeTraits<kind>::NativeType>
py::class_<SimpleColumn<T>, BaseColumn> declareNumericalType(py::module& m) {
  py::class_<SimpleColumn<T>, BaseColumn> pyClass =
      declareSimpleType<kind>(m, [](auto val) { return py::cast(val); })
          .def("neg", &SimpleColumn<T>::neg)
          .def("abs", &SimpleColumn<T>::abs)
          // Defining three methods for each binary operation: one for column *
          // column, one for column * scalar, and one for scalar * column. Note
          // that the scalar * column form is for reverse operations, which is
          // called by Python when the scalar does not know how to perform an
          // operation with a column operand. Technically we do not need to
          // declare the scalar * column reverse form for commutative operators
          // but we are doing that to make the APIs look more consistent
          .def(
              "add",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Plus, OperatorType::Direct);
              })
          .def(
              "add",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Plus, OperatorType::Direct);
              })
          .def(
              "radd",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Plus, OperatorType::Reverse);
              })
          .def(
              "sub",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Minus, OperatorType::Direct);
              })
          .def(
              "sub",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Minus, OperatorType::Direct);
              })
          .def(
              "rsub",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Minus, OperatorType::Reverse);
              })
          .def(
              "mul",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Multiply, OperatorType::Direct);
              })
          .def(
              "mul",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Multiply, OperatorType::Direct);
              })
          .def(
              "rmul",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Multiply, OperatorType::Reverse);
              })
          .def(
              "mod",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Modulus, OperatorType::Direct);
              })
          .def(
              "mod",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Modulus, OperatorType::Direct);
              })
          .def(
              "rmod",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Modulus, OperatorType::Reverse);
              });
  declareComparisons(pyClass);

  return pyClass;
}

template <
    velox::TypeKind kind,
    typename T = typename velox::TypeTraits<kind>::NativeType>
py::class_<SimpleColumn<T>, BaseColumn> declareIntegralType(py::module& m) {
  py::class_<SimpleColumn<T>, BaseColumn> pyClass =
      declareNumericalType<kind>(m)
          .def(
              "append",
              [](SimpleColumn<T>& self, py::int_ value) {
                self.append(py::cast<T>(value));
              })
          .def("invert", &SimpleColumn<T>::invert);
  declareBitwiseOperations(pyClass);

  return pyClass;
}

template <
    velox::TypeKind kind,
    typename T = typename velox::TypeTraits<kind>::NativeType>
py::class_<SimpleColumn<T>, BaseColumn> declareFloatingType(py::module& m) {
  return declareNumericalType<kind>(m)
      .def(
          "append",
          [](SimpleColumn<T>& self, py::float_ value) {
            self.append(py::cast<T>(value));
          })
      .def(
          "append",
          [](SimpleColumn<T>& self, py::int_ value) {
            self.append(py::cast<T>(value));
          })
      .def("ceil", &SimpleColumn<T>::ceil)
      .def("floor", &SimpleColumn<T>::floor)
      .def("round", &SimpleColumn<T>::round);
}

//
// ArrayColumn
//

// Based on VectorMaker::arrayVectorNullable in Velox
template <
    velox::TypeKind kind,
    typename T = typename velox::TypeTraits<kind>::NativeType>
velox::ArrayVectorPtr arrayVectorFromPyList(const py::list& data) {
  using velox::vector_size_t;

  // Prepare the arguments for creating ArrayVector
  velox::BufferPtr offsets = velox::AlignedBuffer::allocate<vector_size_t>(
      data.size(), TorchArrowGlobalStatic::rootMemoryPool());
  velox::BufferPtr sizes = velox::AlignedBuffer::allocate<vector_size_t>(
      data.size(), TorchArrowGlobalStatic::rootMemoryPool());
  velox::BufferPtr nulls = velox::AlignedBuffer::allocate<bool>(
      data.size(), TorchArrowGlobalStatic::rootMemoryPool());

  vector_size_t* rawOffsets = offsets->asMutable<vector_size_t>();
  vector_size_t* rawSizes = sizes->asMutable<vector_size_t>();
  uint64_t* rawNulls = nulls->asMutable<uint64_t>();

  vector_size_t numElements = 0;
  vector_size_t nullCount = 0;
  for (py::size_t i = 0; i < data.size(); i++) {
    if (!data[i].is_none()) {
      numElements += data[i].cast<py::list>().size();
      velox::bits::setNull(rawNulls, i, false);
    } else {
      ++nullCount;
      velox::bits::setNull(rawNulls, i, true);
    }
  }

  // Create the underlying flat vector
  std::shared_ptr<velox::FlatVector<T>> flatVector =
      std::dynamic_pointer_cast<velox::FlatVector<T>>(velox::BaseVector::create(
          velox::CppToType<T>::create(),
          numElements,
          TorchArrowGlobalStatic::rootMemoryPool()));
  uint64_t* elementRawNulls = flatVector->mutableRawNulls();

  vector_size_t currentIdx = 0;
  vector_size_t elementNullCount = 0;
  for (const auto& d : data) {
    if (d.is_none()) {
      *rawSizes++ = 0;
      //
      *rawOffsets++ = 0;
      continue;
    }

    py::list elementArray = d.cast<py::list>();
    *rawSizes++ = elementArray.size();
    *rawOffsets++ = currentIdx;

    for (auto element : elementArray) {
      if (!element.is_none()) {
        if constexpr (std::is_same<T, velox::StringView>::value) {
          flatVector->set(
              currentIdx, velox::StringView(element.cast<std::string>()));
        } else {
          flatVector->set(currentIdx, element.cast<T>());
        }
        // `set()` will set nulls[i] = false for us
      } else {
        velox::bits::setNull(elementRawNulls, currentIdx, true);
        ++elementNullCount;
      }
      ++currentIdx;
    }
  }
  flatVector->setNullCount(elementNullCount);

  return std::make_shared<velox::ArrayVector>(
      TorchArrowGlobalStatic::rootMemoryPool(),
      velox::ARRAY(velox::CppToType<T>::create()),
      nulls,
      data.size(),
      offsets,
      sizes,
      flatVector,
      nullCount);
}

velox::ArrayVectorPtr arrayVectorFromPyListByType(
    velox::TypePtr elementType,
    const py::list& data) {
  switch (elementType->kind()) {
    case velox::TypeKind::ARRAY: {
      VELOX_CHECK(
          false, "Element type of {} is not supported", elementType->kind());
      break;
    }
    case velox::TypeKind::MAP: {
      VELOX_CHECK(
          false, "Element type of {} is not supported", elementType->kind());
      break;
    }
    case velox::TypeKind::ROW: {
      VELOX_CHECK(
          false, "Element type of {} is not supported", elementType->kind());
      break;
    }
    default: {
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          arrayVectorFromPyList, elementType->kind(), data);
      break;
    }
  }
}

void declareArrayType(py::module& m) {
  py::class_<ArrayColumn, BaseColumn>(m, "ArrayColumn")
      .def("append", &ArrayColumn::appendElement)
      .def("append_null", &ArrayColumn::appendNull)
      .def("__getitem__", &ArrayColumn::valueAt)
      .def("elements", &ArrayColumn::elements)
      .def("slice", &ArrayColumn::slice);

  using I = typename velox::TypeTraits<velox::TypeKind::ARRAY>::ImplType;
  py::class_<I, velox::Type, std::shared_ptr<I>>(
      m,
      "VeloxArrayType",
      // TODO: Move the Koksi binding of Velox type to OSS
      py::module_local())
      .def(py::init<velox::TypePtr>())
      .def("element_type", &velox::ArrayType::elementType);

  // Empty Column
  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<ArrayColumn>(type);
  });
  // Column construction from Python list
  m.def(
      "Column",
      [](std::shared_ptr<I> type,
         py::list data) -> std::unique_ptr<ArrayColumn> {
        return std::make_unique<ArrayColumn>(
            arrayVectorFromPyListByType(type->elementType(), data));
      });
}

//
// MapColumn
//

void declareMapType(py::module& m) {
  py::class_<MapColumn, BaseColumn>(m, "MapColumn")
      .def("append", &MapColumn::appendElement)
      .def("append_null", &MapColumn::appendNull)
      .def("offset_at", &MapColumn::offsetAt)
      .def("size_at", &MapColumn::sizeAt)
      .def("__getitem__", &MapColumn::valueAt)
      .def("keys", &MapColumn::mapKeys)
      .def("values", &MapColumn::mapValues)
      .def("slice", &MapColumn::slice);

  using I = typename velox::TypeTraits<velox::TypeKind::MAP>::ImplType;
  py::class_<I, velox::Type, std::shared_ptr<I>>(
      m,
      "VeloxMapType",
      // TODO: Move the Koksi binding of Velox type to OSS
      py::module_local())
      .def(py::init<velox::TypePtr, velox::TypePtr>())
      .def("key_type", &velox::MapType::keyType)
      .def("value_type", &velox::MapType::valueType);

  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<MapColumn>(type);
  });
}

//
// RowColumn
//

void declareRowType(py::module& m) {
  py::class_<RowColumn, BaseColumn>(m, "RowColumn")
      .def("child_at", &RowColumn::childAt)
      .def("set_child", &RowColumn::setChild)
      .def("children_size", &RowColumn::childrenSize)
      .def("slice", &RowColumn::slice)
      .def("set_length", &RowColumn::setLength)
      .def("set_null_at", &RowColumn::setNullAt)
      .def("copy", &RowColumn::copy);

  using I = typename velox::TypeTraits<velox::TypeKind::ROW>::ImplType;
  py::class_<I, velox::Type, std::shared_ptr<I>>(
      m,
      "VeloxRowType",
      // TODO: Move the Koksi binding of Velox type to OSS
      py::module_local())
      .def(py::init<
           std::vector<std::string>&&,
           std::vector<std::shared_ptr<const velox::Type>>&&>())
      .def("size", &I::size)
      .def("get_child_idx", &I::getChildIdx)
      .def("contains_child", &I::containsChild)
      .def("name_of", &I::nameOf)
      .def("child_at", &I::childAt);
  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<RowColumn>(type);
  });
}

PYBIND11_MODULE(_torcharrow, m) {
  m.doc() = R"pbdoc(
        TorchArrow native code module
        -----------------------

        .. currentmodule:: torcharrow

        .. autosummary::
           :toctree: _generate

        velox::TypeKind
    )pbdoc";

  py::class_<BaseColumn>(m, "BaseColumn")
      .def("type", &BaseColumn::type)
      .def("is_null_at", &BaseColumn::isNullAt)
      .def("get_null_count", &BaseColumn::getNullCount)
      .def_property_readonly("offset", &BaseColumn::getOffset)
      .def_property_readonly("length", &BaseColumn::getLength)
      .def("__len__", &BaseColumn::getLength);

  py::enum_<velox::TypeKind>(
      m,
      "TypeKind", // TODO: Move the Koksi binding of Velox type to OSS
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

  py::class_<velox::Type, std::shared_ptr<velox::Type>>(
      m,
      "VeloxType",
      // TODO: Move the Koksi binding of Velox type to OSS
      py::module_local())
      .def("kind", &velox::Type::kind)
      .def("kind_name", &velox::Type::kindName);

  declareIntegralType<velox::TypeKind::BIGINT>(m);
  declareIntegralType<velox::TypeKind::INTEGER>(m);
  declareIntegralType<velox::TypeKind::SMALLINT>(m);
  declareIntegralType<velox::TypeKind::TINYINT>(m);

  auto boolColumnClass = declareSimpleType<velox::TypeKind::BOOLEAN>(
                             m, [](auto val) { return py::cast(val); })
                             .def(
                                 "append",
                                 [](SimpleColumn<bool>& self, py::bool_ value) {
                                   self.append(py::cast<bool>(value));
                                 })
                             .def(
                                 "append",
                                 [](SimpleColumn<bool>& self, py::int_ value) {
                                   self.append(py::cast<bool>(value));
                                 })
                             .def("invert", &SimpleColumn<bool>::invert);
  declareComparisons(boolColumnClass);
  declareBitwiseOperations(boolColumnClass);

  declareFloatingType<velox::TypeKind::REAL>(m);
  declareFloatingType<velox::TypeKind::DOUBLE>(m);

  declareSimpleType<velox::TypeKind::VARCHAR>(
      m,
      [](const velox::StringView& val) {
        return py::cast<py::str>(
            PyUnicode_DecodeUTF8(val.data(), val.size(), nullptr));
      })
      .def(
          "append",
          [](SimpleColumn<velox::StringView>& self, const std::string& value) {
            self.append(velox::StringView(value));
          })
      .def("lower", &SimpleColumn<velox::StringView>::lower)
      .def("upper", &SimpleColumn<velox::StringView>::upper)
      .def("isalpha", &SimpleColumn<velox::StringView>::isalpha)
      .def("isalnum", &SimpleColumn<velox::StringView>::isalnum)
      .def("isinteger", &SimpleColumn<velox::StringView>::isinteger)
      .def("islower", &SimpleColumn<velox::StringView>::islower);

  declareArrayType(m);
  declareMapType(m);
  declareRowType(m);

  // constant columns
  m.def("ConstantColumn", [](const py::handle& value, py::int_ size) {
    return BaseColumn::createConstantColumn(
        pyToVariant(value), py::cast<velox::vector_size_t>(size));
  });

  declareUserDefinedBindings(m);

  // generic UDF dispatch
  m.def("generic_udf_dispatch", &BaseColumn::genericUnaryUDF);
  m.def("generic_udf_dispatch", &BaseColumn::genericBinaryUDF);
  m.def("generic_udf_dispatch", &BaseColumn::genericTrinaryUDF);

  // factory UDF dispatch
  m.def("factory_udf_dispatch", &BaseColumn::factoryNullaryUDF);

  py::register_exception<NotAppendableException>(m, "NotAppendableException");

  // Register Velox UDFs
  // TODO: we may only need to register UDFs that TorchArrow required?
  velox::functions::registerFunctions();
  velox::functions::registerVectorFunctions();

  functions::registerTorchArrowFunctions();

  functions::registerUserDefinedFunctions();

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

} // namespace facebook::torcharrow
