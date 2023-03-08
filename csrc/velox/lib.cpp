/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <velox/common/memory/Memory.h>
#include <velox/type/Type.h>
#include <velox/vector/TypeAliases.h>
#include <memory>

#include "bindings.h"
#include "column.h"
#include "functions/functions.h" // @manual=//pytorch/torcharrow/csrc/velox/functions:torcharrow_functions
#include "tensor_conversion.h"
#include "vector.h"
#include "velox/buffer/StringViewBufferHolder.h"
#include "velox/common/base/Exceptions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/arrow/Abi.h"
#include "velox/vector/arrow/Bridge.h"
#include "pyvelox/pyvelox.h"

#ifdef USE_TORCH
#include <torch/csrc/utils/pybind.h> // @manual
#include "functions/text/gpt2_bpe_tokenizer.h" // @manual
#endif

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<bool>);

namespace facebook::torcharrow {
//
// SimpleColumn (scalar types)
//

// Based on VectorMaker::flatVectorNullable in Velox
template <typename T, typename PySequence>
velox::FlatVectorPtr<T> flatVectorFromPySequence(const PySequence& data) {
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
        velox::bits::setBit(rawData, i, data[i].template cast<bool>());
      } else if constexpr (std::is_same<T, velox::StringView>::value) {
        // Two memcpy's happen here: pybind11::object casting to std::string and
        // StringViewBufferHolder copying data from the buffer in the
        // std::string onto the buffers it manages. We can teach
        // StringViewBufferHolder how to copy data from
        // pybind11::str/pybind11::object to skip one copy
        rawData[i] =
            stringArena.getOwnedValue(data[i].template cast<std::string>());
      } else {
        rawData[i] = data[i].template cast<T>();
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

  // Empty Column
  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<SimpleColumn<T>>();
  });
  // Column construction from Python list
  m.def(
      "Column",
      [](std::shared_ptr<I> type,
         py::list data) -> std::unique_ptr<SimpleColumn<T>> {
        return std::make_unique<SimpleColumn<T>>(
            flatVectorFromPySequence<T>(data));
      });
  // Column construction from Python tuple
  m.def(
      "Column",
      [](std::shared_ptr<I> type,
         py::tuple data) -> std::unique_ptr<SimpleColumn<T>> {
        return std::make_unique<SimpleColumn<T>>(
            flatVectorFromPySequence<T>(data));
      });

  // Import/Export Arrow data
  //
  // VARCHAR is not supported at the moment
  // https://github.com/facebookincubator/velox/blob/f420d3115eeb8ad782aa9979f47be03671ed02f4/velox/vector/arrow/Bridge.cpp#L128
  if constexpr (
      kind == velox::TypeKind::BOOLEAN || kind == velox::TypeKind::TINYINT ||
      kind == velox::TypeKind::SMALLINT || kind == velox::TypeKind::INTEGER ||
      kind == velox::TypeKind::BIGINT || kind == velox::TypeKind::REAL ||
      kind == velox::TypeKind::DOUBLE) {
    // _torcharrow._import_from_arrow
    m.def(
        "_import_from_arrow",
        [](std::shared_ptr<I> type, uintptr_t ptrArray, uintptr_t ptrSchema) {
          ArrowArray* castedArray = reinterpret_cast<ArrowArray*>(ptrArray);
          ArrowSchema* castedSchema = reinterpret_cast<ArrowSchema*>(ptrSchema);
          VELOX_CHECK_NOT_NULL(castedArray);
          VELOX_CHECK_NOT_NULL(castedSchema);

          auto column =
              std::make_unique<SimpleColumn<T>>(velox::importFromArrowAsOwner(
                  *castedSchema,
                  *castedArray,
                  TorchArrowGlobalStatic::rootMemoryPool()));

          VELOX_CHECK(
              column->type()->kind() == kind,
              "Expected TypeKind is {} but Velox created {}",
              velox::TypeTraits<kind>::name,
              column->type()->kindName());

          return column;
        });

    // _torcharrow.SimpleColumn<Type>._export_to_arrow
    result.def(
        "_export_to_arrow", [](SimpleColumn<T>& self, uintptr_t ptrArray) {
          ArrowArray* castedArray = reinterpret_cast<ArrowArray*>(ptrArray);
          VELOX_CHECK_NOT_NULL(castedArray);

          self.exportToArrow(castedArray);
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
          .def("cast", &SimpleColumn<T>::cast)
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
              "truediv",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Divide, OperatorType::Direct);
              })
          .def(
              "truediv",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Divide, OperatorType::Direct);
              })
          .def(
              "rtruediv",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Divide, OperatorType::Reverse);
              })
          .def(
              "floordiv",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Floordiv, OperatorType::Direct);
              })
          .def(
              "floordiv",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Floordiv, OperatorType::Direct);
              })
          .def(
              "rfloordiv",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Floordiv, OperatorType::Reverse);
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
              })
          .def(
              "pow",
              [](SimpleColumn<T>& a,
                 const BaseColumn& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Pow, OperatorType::Direct);
              })
          .def(
              "pow",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Pow, OperatorType::Direct);
              })
          .def(
              "rpow",
              [](SimpleColumn<T>& a,
                 const pybind11::handle& b) -> std::unique_ptr<BaseColumn> {
                return a.callBinaryOp(
                    b, BinaryOpCode::Pow, OperatorType::Reverse);
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
              [](SimpleColumn<T>& self, T value) { self.append(value); })
          .def("invert", &SimpleColumn<T>::invert);
  declareBitwiseOperations(pyClass);

  return pyClass;
}

template <
    velox::TypeKind kind,
    typename T = typename velox::TypeTraits<kind>::NativeType>
py::class_<SimpleColumn<T>, BaseColumn> declareFloatingType(py::module& m) {
  return declareNumericalType<kind>(m)
      .def("append", [](SimpleColumn<T>& self, T value) { self.append(value); })
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
velox::ArrayVectorPtr arrayVectorFromPyList(
    int fixed_size,
    const py::list& data) {
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

  const auto typ = fixed_size == -1
      ? velox::ARRAY(velox::CppToType<T>::create())
      : velox::FIXED_SIZE_ARRAY(fixed_size, velox::CppToType<T>::create());

  return std::make_shared<velox::ArrayVector>(
      TorchArrowGlobalStatic::rootMemoryPool(),
      typ,
      nulls,
      data.size(),
      offsets,
      sizes,
      flatVector,
      nullCount);
}

velox::ArrayVectorPtr arrayVectorFromPyListByType(
    int fixed_size,
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
          arrayVectorFromPyList, elementType->kind(), fixed_size, data);
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
      .def("slice", &ArrayColumn::slice)
      .def("withElements", &ArrayColumn::withElements);

  using I = typename velox::TypeTraits<velox::TypeKind::ARRAY>::ImplType;

    using J = typename velox::FixedSizeArrayType;

  // Empty Column
  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<ArrayColumn>(type);
  });
  m.def("Column", [](std::shared_ptr<J> type) {
    return std::make_unique<ArrayColumn>(type);
  });

  // Column construction from Python list
  m.def(
      "Column",
      [](std::shared_ptr<I> type,
         py::list data) -> std::unique_ptr<ArrayColumn> {
        return std::make_unique<ArrayColumn>(
            arrayVectorFromPyListByType(-1, type->elementType(), data));
      });
  m.def(
      "Column",
      [](std::shared_ptr<J> type,
         py::list data) -> std::unique_ptr<ArrayColumn> {
        return std::make_unique<ArrayColumn>(arrayVectorFromPyListByType(
            type->fixedElementsWidth(), type->elementType(), data));
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
      .def("copy", &RowColumn::copy)
      .def("_export_to_arrow", [](RowColumn& self, uintptr_t ptrArray) {
        ArrowArray* castedArray = reinterpret_cast<ArrowArray*>(ptrArray);
        VELOX_CHECK_NOT_NULL(castedArray);

        self.exportToArrow(castedArray);
      });

  using I = typename velox::TypeTraits<velox::TypeKind::ROW>::ImplType;
  m.def("Column", [](std::shared_ptr<I> type) {
    return std::make_unique<RowColumn>(type);
  });
  // _torcharrow._import_from_arrow
  m.def(
      "_import_from_arrow",
      [](std::shared_ptr<I> type, uintptr_t ptrArray, uintptr_t ptrSchema) {
        ArrowArray* castedArray = reinterpret_cast<ArrowArray*>(ptrArray);
        ArrowSchema* castedSchema = reinterpret_cast<ArrowSchema*>(ptrSchema);
        VELOX_CHECK_NOT_NULL(castedArray);
        VELOX_CHECK_NOT_NULL(castedSchema);

        auto column = std::make_unique<RowColumn>(velox::importFromArrowAsOwner(
            *castedSchema,
            *castedArray,
            TorchArrowGlobalStatic::rootMemoryPool()));

        VELOX_CHECK(
            column->type()->kind() == velox::TypeKind::ROW,
            "Expected TypeKind is {} but Velox created {}",
            velox::TypeTraits<velox::TypeKind::ROW>::name,
            column->type()->kindName());

        return column;
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


  pyvelox::addVeloxBindings(m);
  declareIntegralType<velox::TypeKind::BIGINT>(m);
  declareIntegralType<velox::TypeKind::INTEGER>(m);
  declareIntegralType<velox::TypeKind::SMALLINT>(m);
  declareIntegralType<velox::TypeKind::TINYINT>(m);

  using BIGINTNativeType =
      velox::TypeTraits<velox::TypeKind::BIGINT>::NativeType;
  auto boolColumnClass =
      declareSimpleType<velox::TypeKind::BOOLEAN>(
          m, [](auto val) { return py::cast(val); })
          .def("cast", &SimpleColumn<bool>::cast)
          .def(
              "append",
              [](SimpleColumn<bool>& self, bool value) { self.append(value); },
              // explicitly disallow all conversions to bools; enabling
              // this allows `None` and floats to convert to bools
              py::arg("value").noconvert())
          .def(
              "append",
              [](SimpleColumn<bool>& self, BIGINTNativeType value) {
                self.append(static_cast<bool>(value));
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
          });

  declareArrayType(m);
  declareMapType(m);
  declareRowType(m);

  // constant columns
  m.def("ConstantColumn", [](const py::handle& value, py::int_ size) {
    return BaseColumn::createConstantColumn(
        pyToVariant(value), py::cast<velox::vector_size_t>(size));
  });
  m.def(
      "ConstantColumn",
      [](const py::handle& value,
         py::int_ size,
         std::shared_ptr<velox::Type> type) {
        return BaseColumn::createConstantColumn(
            pyToVariantTyped(value, type),
            py::cast<velox::vector_size_t>(size));
      });

  declareUserDefinedBindings(m);

  // generic UDF dispatch
  m.def("generic_udf_dispatch", &BaseColumn::genericUDF);

  // factory UDF dispatch (e.g. UDF without any parameters, length is required
  // for such UDF call)
  m.def("factory_udf_dispatch", &BaseColumn::factoryNullaryUDF);

  // Tensor conversion related binding
  m.def(
      "_populate_dense_features_nopresence",
      [](const RowColumn& column, uintptr_t dataTensorPtr) {
        populateDenseFeaturesNoPresence(
            std::dynamic_pointer_cast<velox::RowVector>(
                column.getUnderlyingVeloxVector()),
            column.getOffset(),
            column.getLength(),
            dataTensorPtr);
      });

  py::register_exception<NotAppendableException>(m, "NotAppendableException");

  // Register Velox UDFs
  // TODO: we may only need to register UDFs that TorchArrow required?
  velox::functions::prestosql::registerAllScalarFunctions();

  functions::registerTorchArrowFunctions();

  functions::registerUserDefinedFunctions();

  // Is library built with torch
  m.def("is_built_with_torch", []() {
#ifdef USE_TORCH
    return true;
#else
    return false;
#endif
  });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
#ifdef USE_TORCH
  py::class_<functions::Vocab, c10::intrusive_ptr<functions::Vocab>>(m, "Vocab")
      .def(py::init<functions::StringList, c10::optional<int64_t>>())
      .def_readonly("itos_", &functions::Vocab::itos_)
      .def_readonly("default_index_", &functions::Vocab::default_index_)
      .def(
          "__contains__",
          [](c10::intrusive_ptr<functions::Vocab>& self,
             const py::str& item) -> bool {
            Py_ssize_t length;
            const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
            return self->__contains__(c10::string_view{buffer, (size_t)length});
          })
      .def(
          "__getitem__",
          [](c10::intrusive_ptr<functions::Vocab>& self,
             const py::str& item) -> int64_t {
            Py_ssize_t length;
            const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
            return self->__getitem__(c10::string_view{buffer, (size_t)length});
          })
      .def("insert_token", &functions::Vocab::insert_token)
      .def("set_default_index", &functions::Vocab::set_default_index)
      .def("get_default_index", &functions::Vocab::get_default_index)
      .def("__len__", &functions::Vocab::__len__)
      .def("append_token", &functions::Vocab::append_token)
      .def("lookup_token", &functions::Vocab::lookup_token)
      .def("lookup_tokens", &functions::Vocab::lookup_tokens)
      .def(
          "lookup_indices",
          [](const c10::intrusive_ptr<functions::Vocab>& self,
             const py::list& items) {
            std::vector<int64_t> indices(items.size());
            int64_t counter = 0;
            for (const auto& item : items) {
              Py_ssize_t length;
              const char* buffer = PyUnicode_AsUTF8AndSize(item.ptr(), &length);
              indices[counter++] =
                  self->__getitem__(c10::string_view{buffer, (size_t)length});
            }
            return indices;
          })
      .def("get_stoi", &functions::Vocab::get_stoi)
      .def("get_itos", &functions::Vocab::get_itos)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<functions::Vocab>& self)
              -> functions::VocabStates {
            return functions::_serialize_vocab(self);
          },
          // __setstate__
          [](functions::VocabStates states)
              -> c10::intrusive_ptr<functions::Vocab> {
            return functions::_deserialize_vocab(states);
          }));

  // text operator
  py::class_<
      functions::GPT2BPEEncoder,
      c10::intrusive_ptr<functions::GPT2BPEEncoder>>(m, "GPT2BPEEncoder")
      .def(py::init<
           std::unordered_map<std::string, int64_t>,
           std::unordered_map<std::string, int64_t>,
           std::string,
           std::unordered_map<int64_t, std::string>,
           bool>())
      .def_property_readonly(
          "bpe_encoder_", &functions::GPT2BPEEncoder::GetBPEEncoder)
      .def_property_readonly(
          "bpe_merge_ranks_", &functions::GPT2BPEEncoder::GetBPEMergeRanks)
      .def_readonly("seperator_", &functions::GPT2BPEEncoder::seperator_)
      .def_property_readonly(
          "byte_encoder_", &functions::GPT2BPEEncoder::GetByteEncoder)
      .def("encode", &functions::GPT2BPEEncoder::Encode)
      .def(py::pickle(
          // __getstate__
          [](const c10::intrusive_ptr<functions::GPT2BPEEncoder>& self)
              -> functions::GPT2BPEEncoderStatesPybind {
            return functions::_serialize_gpt2_bpe_encoder_pybind(self);
          },
          // __setstate__
          [](functions::GPT2BPEEncoderStatesPybind states)
              -> c10::intrusive_ptr<functions::GPT2BPEEncoder> {
            return functions::_deserialize_gpt2_bpe_encoder_pybind(states);
          }));
#endif
}

} // namespace facebook::torcharrow
