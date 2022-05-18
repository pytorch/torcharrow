/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "column.h"
#include <memory>
#include "VariantToVector.h"
#include "bindings.h"

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/expression/Expr.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

#ifdef USE_TORCH
#include "functions/text/gpt2_bpe_tokenizer.h" // @manual
#include "functions/text/vocab.h" // @manual
#endif

namespace py = pybind11;

namespace facebook::torcharrow {

template <velox::TypeKind kind>
std::unique_ptr<BaseColumn> createSimpleColumn(
    velox::VectorPtr vec,
    velox::vector_size_t offset,
    velox::vector_size_t length) {
  using T = typename velox::TypeTraits<kind>::NativeType;
  return std::make_unique<SimpleColumn<T>>(
      SimpleColumn<T>(vec), offset, length);
}

std::unique_ptr<BaseColumn> createColumn(velox::VectorPtr vec) {
  return createColumn(vec, 0, vec.get()->size());
}

std::unique_ptr<BaseColumn> createColumn(
    velox::VectorPtr vec,
    velox::vector_size_t offset,
    velox::vector_size_t length) {
  auto type = vec.get()->type();
  auto kind = type.get()->kind();
  switch (kind) {
    case velox::TypeKind::ARRAY: {
      return std::make_unique<ArrayColumn>(ArrayColumn(vec), offset, length);
    }
    case velox::TypeKind::MAP: {
      return std::make_unique<MapColumn>(MapColumn(vec), offset, length);
    }
    case velox::TypeKind::ROW: {
      return std::make_unique<RowColumn>(RowColumn(vec), offset, length);
    }
    default:
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createSimpleColumn, kind, vec, offset, length);
  }
}

template <velox::TypeKind kind>
std::unique_ptr<BaseColumn> doCreateConstantColumn(
    velox::variant value,
    velox::vector_size_t size) {
  using T = typename velox::TypeTraits<kind>::NativeType;
  return std::make_unique<ConstantColumn<T>>(value, size);
}

std::unique_ptr<BaseColumn> BaseColumn::createConstantColumn(
    velox::variant value,
    velox::vector_size_t size) {
  // Note here we are doing the same type dispatch twice:
  //   1. first happens when dispatching to doCreateSimpleColumn
  //   2. second happens in constructor of ConstantColumn<T> when calling
  //      velox::BaseVector::createConstant
  //
  // The second dispatch is required because the method
  // velox::BaseVector::createConstant dispatch to (newConstant) is a not yet a
  // public Velox API. Otherwise, we can create the ConstantVector and wrap it
  // into the ConstantColumn in one dispatch.
  //
  // We can avoid the second dispatch either by making `newConstant` a public
  // Velox API, or have template method to create ConstantVector (which
  // essentially fork `newConstant` method).
  //
  // However, at some point we also want to revisit whether
  // SimpleColumn/ConstantColumn needs to be templated and thus we could
  // remove the first dispatch.
  auto kind = value.kind();

  switch (kind) {
    case velox::TypeKind::ARRAY: {
      return std::make_unique<ConstantColumn<velox::ComplexType>>(
          // Convert array variant value into a single-element ArrayVector
          variantArrayToVector(
              value.value<velox::TypeKind::ARRAY>(),
              TorchArrowGlobalStatic::rootMemoryPool()),
          0,
          size);
    }
    default:
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          doCreateConstantColumn, value.kind(), value, size);
  }
}

std::unique_ptr<BaseColumn> ArrayColumn::valueAt(velox::vector_size_t i) {
  velox::ArrayVector* arrayVectorPtr =
      _delegate.get()->as<velox::ArrayVector>();
  velox::VectorPtr elementVectorPtr = arrayVectorPtr->elements();
  auto elementOffset = arrayVectorPtr->offsetAt(this->_offset + i);
  auto elementLength = arrayVectorPtr->sizeAt(this->_offset + i);
  return createColumn(elementVectorPtr, elementOffset, elementLength);
}

std::unique_ptr<BaseColumn> ArrayColumn::elements() {
  velox::ArrayVector* arrayVectorPtr =
      _delegate.get()->as<velox::ArrayVector>();
  velox::VectorPtr elementVectorPtr = arrayVectorPtr->elements();
  auto elementOffset = arrayVectorPtr->offsetAt(this->_offset);
  int elementsTotalLength = 0;
  for (int i = 0; i < this->_length; i++) {
    elementsTotalLength += arrayVectorPtr->sizeAt(this->_offset + i);
  }
  return createColumn(elementVectorPtr, elementOffset, elementsTotalLength);
}

std::unique_ptr<ArrayColumn> ArrayColumn::withElements(
    const BaseColumn& newElements) {
  velox::ArrayVector* currentArrayVec =
      _delegate.get()->as<velox::ArrayVector>();
  velox::VectorPtr newElementVec = newElements.getUnderlyingVeloxVector();
  VELOX_CHECK(newElements.getLength() == newElementVec->size());

  if (this->_length == currentArrayVec->size()) {
    velox::VectorPtr newArrayVec = std::make_unique<velox::ArrayVector>(
        TorchArrowGlobalStatic::rootMemoryPool(),
        ARRAY(newElementVec->type()),
        currentArrayVec->nulls(),
        currentArrayVec->size(),
        currentArrayVec->offsets(),
        currentArrayVec->sizes(),
        newElementVec);
    return std::make_unique<ArrayColumn>(newArrayVec);
  }

  VELOX_CHECK(false, "Not supported yet");
}

std::unique_ptr<BaseColumn> MapColumn::valueAt(velox::vector_size_t i) {
  velox::TypePtr keyType = type()->as<velox::TypeKind::MAP>().keyType();
  velox::TypePtr valueType = type()->as<velox::TypeKind::MAP>().valueType();
  auto dataPtr = _delegate.get()->as<velox::MapVector>();
  auto keys = dataPtr->mapKeys();
  auto values = dataPtr->mapValues();
  auto start = dataPtr->offsetAt(_offset + i);
  auto end = dataPtr->offsetAt(_offset + i) + dataPtr->sizeAt(_offset + i);
  auto slicedKeys = vectorSlice(*keys.get(), start, end);
  auto slicedValues = vectorSlice(*values.get(), start, end);
  auto slicedResult = velox::BaseVector::create(type(), 1, pool_);
  slicedResult.get()->as<velox::MapVector>()->setKeysAndValues(
      slicedKeys, slicedValues);
  return createColumn(slicedResult);
}

// TODO: Model binary functions as UDF.
/* static */
OperatorHandle* BaseColumn::getOrCreateBinaryOperatorHandle(
    velox::TypePtr c0Type,
    velox::TypePtr c1Type,
    velox::TypeKind commonTypeKind,
    BinaryOpCode opCode) {
  // FIXME This is fragile as it assumes velox::TypeKind numbers numeric types
  // starting from 0 and has DOUBLE being the last one
  constexpr size_t nNumericType =
      static_cast<size_t>(velox::TypeKind::DOUBLE) + 1;
  constexpr size_t nOp = static_cast<size_t>(BinaryOpCode::NOpCode);
  // Indices are [c0TypeKind][c1TypeKind][commonTypeKind][opCode]
  //
  // For every combination of c0TypeKind and c1TypeKind, there are only two
  // possible commonTypeKind's, one produced by column-column type promotion and
  // the other produced by column-scalar promotion, so there are (nNumericType -
  // 2) * nNumericType * nNumericType * nOp entries that are unused (more than
  // half ot the table :( or a few tens of KB). We're trading the space for more
  // consoidated code. Feel free to squeeze the space out when the day comes
  static std::unique_ptr<OperatorHandle> ops[nNumericType][nNumericType]
                                            [nNumericType][nOp];

  size_t c0TypeId = static_cast<size_t>(c0Type->kind());
  size_t c1TypeId = static_cast<size_t>(c1Type->kind());
  size_t commonTypeId = static_cast<size_t>(commonTypeKind);
  size_t opCodeId = static_cast<size_t>(opCode);
  if (ops[c0TypeId][c1TypeId][commonTypeId][opCodeId] == nullptr) {
    std::shared_ptr<const velox::RowType> inputRowType =
        velox::ROW({"c0", "c1"}, {c0Type, c1Type});
    velox::TypePtr commonType =
        VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(kind2type, commonTypeKind);
    velox::TypePtr outputType = [opCode, &commonType]() -> velox::TypePtr {
      switch (opCode) {
        case BinaryOpCode::Eq:
        case BinaryOpCode::Neq:
        case BinaryOpCode::Lt:
        case BinaryOpCode::Gt:
        case BinaryOpCode::Lte:
        case BinaryOpCode::Gte:
          return velox::TypeFactory<velox::TypeKind::BOOLEAN>::create();
        default:
          return commonType;
      }
    }();

    auto exprSet = genBinaryExprSet(
        inputRowType,
        std::move(commonType),
        std::move(outputType),
        opCodeToFunctionName(opCode));

    ops[c0TypeId][c1TypeId][commonTypeId][opCodeId] =
        std::make_unique<OperatorHandle>(
            std::move(inputRowType), std::move(exprSet));
  }

  return ops[c0TypeId][c1TypeId][commonTypeId][opCodeId].get();
}

void BaseColumn::exportToArrow(ArrowArray* output) {
  if (getOffset() != 0 || getLength() < getUnderlyingVeloxVector()->size()) {
    // This is a slice. Make a copy of the slice and then export the
    // slice to Arrow
    velox::VectorPtr temp = vectorSlice(
        *getUnderlyingVeloxVector(), getOffset(), getOffset() + getLength());
    temp->setNullCount(getNullCount());
    velox::exportToArrow(temp, *output);
  } else {
    velox::exportToArrow(getUnderlyingVeloxVector(), *output);
  }
}

std::shared_ptr<velox::exec::ExprSet> BaseColumn::genBinaryExprSet(
    std::shared_ptr<const velox::RowType> inputRowType,
    std::shared_ptr<const velox::Type> commonType,
    std::shared_ptr<const velox::Type> outputType,
    const std::string& functionName) {
  if (!outputType) {
    outputType = commonType;
  }

  // Construct Typed Expression
  using InputExprList =
      std::vector<std::shared_ptr<const velox::core::ITypedExpr>>;

  InputExprList castedFieldAccessTypedExprs;
  for (int i = 0; i < 2; i++) {
    auto fieldAccessTypedExpr =
        std::make_shared<velox::core::FieldAccessTypedExpr>(
            inputRowType->childAt(i), inputRowType->nameOf(i));

    if (*inputRowType->childAt(i) == *commonType) {
      // no need to cast
      castedFieldAccessTypedExprs.push_back(fieldAccessTypedExpr);
    } else {
      // type promotion
      InputExprList fieldAccessTypedExprs{fieldAccessTypedExpr};
      castedFieldAccessTypedExprs.push_back(
          std::make_shared<velox::core::CastTypedExpr>(
              commonType, fieldAccessTypedExprs, false /* nullOnFailure */));
    }
  }

  InputExprList callTypedExprs{std::make_shared<velox::core::CallTypedExpr>(
      outputType, std::move(castedFieldAccessTypedExprs), functionName)};

  // Container for expressions that get evaluated together. Common
  // subexpression elimination and other cross-expression
  // optimizations take place within this set of expressions.
  return std::make_shared<velox::exec::ExprSet>(
      std::move(callTypedExprs), &TorchArrowGlobalStatic::execContext());
}

std::unique_ptr<BaseColumn> BaseColumn::genericUnaryUDF(
    const std::string& udfName,
    const BaseColumn& col1) {
  auto rowType = velox::ROW({"c0"}, {col1.getUnderlyingVeloxVector()->type()});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter =
        dispatchTable.insert({key, OperatorHandle::fromUDF(rowType, udfName)})
            .first;
  }
  return iter->second->call({col1.getUnderlyingVeloxVector()});
}

std::unique_ptr<BaseColumn> BaseColumn::genericBinaryUDF(
    const std::string& udfName,
    const BaseColumn& col1,
    const BaseColumn& col2) {
  auto rowType = velox::ROW(
      {"c0", "c1"},
      {col1.getUnderlyingVeloxVector()->type(),
       col2.getUnderlyingVeloxVector()->type()});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter =
        dispatchTable.insert({key, OperatorHandle::fromUDF(rowType, udfName)})
            .first;
  }
  return iter->second->call(
      {col1.getUnderlyingVeloxVector(), col2.getUnderlyingVeloxVector()});
}

std::unique_ptr<BaseColumn> BaseColumn::genericTrinaryUDF(
    const std::string& udfName,
    const BaseColumn& col1,
    const BaseColumn& col2,
    const BaseColumn& col3) {
  auto rowType = velox::ROW(
      {"c0", "c1", "c2"},
      {col1.getUnderlyingVeloxVector()->type(),
       col2.getUnderlyingVeloxVector()->type(),
       col3.getUnderlyingVeloxVector()->type()});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter =
        dispatchTable.insert({key, OperatorHandle::fromUDF(rowType, udfName)})
            .first;
  }
  return iter->second->call(
      {col1.getUnderlyingVeloxVector(),
       col2.getUnderlyingVeloxVector(),
       col3.getUnderlyingVeloxVector()});
}

std::unique_ptr<BaseColumn> BaseColumn::genericQuaternaryUDF(
    const std::string& udfName,
    const BaseColumn& col1,
    const BaseColumn& col2,
    const BaseColumn& col3,
    const BaseColumn& col4) {
  auto rowType = velox::ROW(
      {"c0", "c1", "c2", "c3"},
      {col1.getUnderlyingVeloxVector()->type(),
       col2.getUnderlyingVeloxVector()->type(),
       col3.getUnderlyingVeloxVector()->type(),
       col4.getUnderlyingVeloxVector()->type()});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter =
        dispatchTable.insert({key, OperatorHandle::fromUDF(rowType, udfName)})
            .first;
  }
  return iter->second->call(
      {col1.getUnderlyingVeloxVector(),
       col2.getUnderlyingVeloxVector(),
       col3.getUnderlyingVeloxVector(),
       col4.getUnderlyingVeloxVector()});
}

std::unique_ptr<BaseColumn> BaseColumn::factoryNullaryUDF(
    const std::string& udfName,
    int size) {
  auto rowType = velox::ROW({}, {});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter =
        dispatchTable.insert({key, OperatorHandle::fromUDF(rowType, udfName)})
            .first;
  }
  return iter->second->call(size);
}

std::unique_ptr<OperatorHandle> OperatorHandle::fromCall(
    velox::RowTypePtr inputRowType,
    velox::TypePtr outputType,
    const std::string& functionName) {
  // Construct Typed Expression
  using InputExprList =
      std::vector<std::shared_ptr<const velox::core::ITypedExpr>>;

  InputExprList fieldAccessTypedExprs;
  fieldAccessTypedExprs.reserve(inputRowType->size());
  for (int i = 0; i < inputRowType->size(); i++) {
    auto fieldAccessTypedExpr =
        std::make_shared<velox::core::FieldAccessTypedExpr>(
            inputRowType->childAt(i), inputRowType->nameOf(i));

    fieldAccessTypedExprs.push_back(fieldAccessTypedExpr);
  }

  InputExprList callTypedExprs{std::make_shared<velox::core::CallTypedExpr>(
      outputType, std::move(fieldAccessTypedExprs), functionName)};

  return std::make_unique<OperatorHandle>(
      inputRowType,
      std::make_shared<velox::exec::ExprSet>(
          std::move(callTypedExprs), &TorchArrowGlobalStatic::execContext()));
}

std::unique_ptr<OperatorHandle> OperatorHandle::fromUDF(
    velox::RowTypePtr inputRowType,
    const std::string& udfName) {
  if (udfName == "coalesce") {
    return OperatorHandle::fromCall(
        inputRowType, inputRowType->childAt(0), udfName);
  }

  velox::TypePtr outputType =
      velox::resolveFunction(udfName, inputRowType->children());
  if (outputType == nullptr) {
    throw std::runtime_error("Request for unknown Velox UDF: " + udfName);
  }
  return OperatorHandle::fromCall(inputRowType, outputType, udfName);
}

std::unique_ptr<OperatorHandle> OperatorHandle::fromCast(
    velox::RowTypePtr inputRowType,
    velox::TypePtr outputType) {
  using InputExprList =
      std::vector<std::shared_ptr<const velox::core::ITypedExpr>>;
  InputExprList fieldAccessTypedExprs{
      std::make_shared<velox::core::FieldAccessTypedExpr>(
          inputRowType->childAt(0), inputRowType->nameOf(0))};

  InputExprList castTypedExprs{std::make_shared<velox::core::CastTypedExpr>(
      outputType, std::move(fieldAccessTypedExprs), false /* nullOnFailure */)};

  return std::make_unique<OperatorHandle>(
      inputRowType,
      std::make_shared<velox::exec::ExprSet>(
          std::move(castTypedExprs), &TorchArrowGlobalStatic::execContext()));
}

std::unique_ptr<BaseColumn> OperatorHandle::call(
    velox::RowVectorPtr inputRows,
    velox::vector_size_t size) {
  velox::exec::EvalCtx evalCtx(
      &TorchArrowGlobalStatic::execContext(), exprSet_.get(), inputRows.get());
  velox::SelectivityVector select(size);
  std::vector<velox::VectorPtr> outputRows(1);
  exprSet_->eval(0, 1, true, select, &evalCtx, &outputRows);

  // TODO: This causes an extra type-based dispatch.
  // We can optimize it by template OperatorHandle by return type
  return createColumn(outputRows[0]);
}

std::unique_ptr<BaseColumn> OperatorHandle::call(velox::vector_size_t size) {
  auto inputRows = std::make_shared<velox::RowVector>(
      TorchArrowGlobalStatic::rootMemoryPool(),
      inputRowType_,
      velox::BufferPtr(nullptr),
      size,
      std::vector<velox::VectorPtr>() /* children */);
  return call(inputRows, size);
}

std::unique_ptr<BaseColumn> OperatorHandle::call(velox::VectorPtr a) {
  velox::RowVectorPtr inputRows = wrapRowVector({a}, inputRowType_);
  return call(inputRows, a->size());
}

std::unique_ptr<BaseColumn> OperatorHandle::call(
    velox::VectorPtr a,
    velox::VectorPtr b) {
  velox::RowVectorPtr inputRows = wrapRowVector({a, b}, inputRowType_);
  return call(inputRows, a->size());
}

std::unique_ptr<BaseColumn> OperatorHandle::call(
    const std::vector<velox::VectorPtr>& args) {
  velox::RowVectorPtr inputRows = wrapRowVector(args, inputRowType_);
  velox::exec::EvalCtx evalCtx(
      &TorchArrowGlobalStatic::execContext(), exprSet_.get(), inputRows.get());
  return call(inputRows, args[0]->size());
}

std::string opCodeToFunctionName(BinaryOpCode opCode) {
  switch (opCode) {
    case BinaryOpCode::Plus: {
      return "plus";
    } break;
    case BinaryOpCode::Minus: {
      return "minus";
    } break;
    case BinaryOpCode::Multiply: {
      return "multiply";
    } break;
    case BinaryOpCode::Divide: {
      return "divide";
    } break;
    case BinaryOpCode::Floordiv: {
      return "torcharrow_floordiv";
    } break;
    case BinaryOpCode::Modulus: {
      return "torcharrow_floormod";
    } break;
    case BinaryOpCode::Pow: {
      return "torcharrow_pow";
    } break;
    case BinaryOpCode::Eq: {
      return "eq";
    } break;
    case BinaryOpCode::Neq: {
      return "neq";
    } break;
    case BinaryOpCode::Lt: {
      return "lt";
    } break;
    case BinaryOpCode::Gt: {
      return "gt";
    } break;
    case BinaryOpCode::Lte: {
      return "lte";
    } break;
    case BinaryOpCode::Gte: {
      return "gte";
    } break;
    case BinaryOpCode::BitwiseAnd: {
      return "bitwise_and";
    } break;
    case BinaryOpCode::BitwiseOr: {
      return "bitwise_or";
    } break;
    case BinaryOpCode::BitwiseXor: {
      return "bitwise_xor";
    } break;
    default: {
      throw std::logic_error(
          "Unsupported BinaryOpCode: " +
          std::to_string(static_cast<int16_t>(opCode)));
    } break;
  }
}

velox::core::QueryCtx& TorchArrowGlobalStatic::queryContext() {
  static velox::core::QueryCtx queryContext;
  return queryContext;
}

velox::core::ExecCtx& TorchArrowGlobalStatic::execContext() {
  static auto pool = velox::memory::getDefaultScopedMemoryPool();
  static velox::core::ExecCtx execContext(
      pool.get(), &TorchArrowGlobalStatic::queryContext());
  return execContext;
}

// This method only supports a limited set Python object to velox::variant
// conversion to minimize code duplication.
// TODO: Open source some part of utility codes in Koski (PyVelox?)
velox::variant pyToVariant(const pybind11::handle& obj) {
  // Numpy compatibility
  //
  // This check comes before primitive values, because np.float64() will
  // evaluate as True for both of the following:
  //    py::isinstance<py::float_>(obj)
  //    py::isinstance<py::buffer>(obj)
  //
  // Thus, if we check primitive values first, pyToVariant(np.float64(1.0))
  //    will return a variant with REAL type, although
  //    the value is annoated as float64.
  //
  // py::buffer comes with explicit dtype information, so check this before
  // primitive values.
  if (py::isinstance<py::buffer>(obj)) {
    auto buf = py::cast<py::buffer>(obj);
    auto info = buf.request();
    // Only scalars are supported for now, proper ndarrays can be added later
    // We don't handle default cases in `switch` because fall through code
    // at the end of the function generates a good error message anyway.
    if (info.size == 1 && info.ndim == 0) {
      auto dtype = py::dtype(info);
      // https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
      switch (dtype.kind()) {
        case 'i': {
          switch (dtype.itemsize()) {
            case 1:
              return velox::variant::create<velox::TypeKind::TINYINT>(
                  py::array_t<int8_t>::ensure(obj).at());
            case 2:
              return velox::variant::create<velox::TypeKind::SMALLINT>(
                  py::array_t<int16_t>::ensure(obj).at());
            case 4:
              return velox::variant::create<velox::TypeKind::INTEGER>(
                  py::array_t<int32_t>::ensure(obj).at());
            case 8:
              return velox::variant::create<velox::TypeKind::BIGINT>(
                  py::array_t<int64_t>::ensure(obj).at());
          }
        } break;
        case 'f': {
          switch (dtype.itemsize()) {
            case 4:
              return velox::variant::create<velox::TypeKind::REAL>(
                  py::array_t<float>::ensure(obj).at());
            case 8:
              return velox::variant::create<velox::TypeKind::DOUBLE>(
                  py::array_t<double>::ensure(obj).at());
          }
        } break;
        case 'b':
          return velox::variant::create<velox::TypeKind::BOOLEAN>(
              py::array_t<bool>::ensure(obj).at());
      }
    }
  }

  // Base cases: primitives can be simply returned
  if (py::isinstance<py::bool_>(obj)) {
    return velox::variant::create<velox::TypeKind::BOOLEAN>(
        py::cast<bool>(obj));
  } else if (py::isinstance<py::int_>(obj)) {
    return velox::variant::create<velox::TypeKind::BIGINT>(py::cast<long>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    return velox::variant::create<velox::TypeKind::REAL>(py::cast<float>(obj));
  } else if (py::isinstance<py::str>(obj)) {
    return velox::variant::create<velox::TypeKind::VARCHAR>(
        py::cast<std::string>(obj));
  } else if (obj.is_none()) {
    return velox::variant();
  }

  // Check opaque types before containers because some of them might be iterable
  velox::variant out;
  if (userDefinedPyToOpaque(obj, out)) {
    return out;
  }

#ifdef USE_TORCH
  // TODO: The implementation below to handle GPT2BPEEncoder opaque type is a
  // hack. We want to use Opaque registration mechanism in the future
  if (py::isinstance<functions::GPT2BPEEncoder>(obj)) {
    return velox::variant::opaque(
        obj.cast<std::shared_ptr<functions::GPT2BPEEncoder>>());
  }

  // handle vocab opaque type
  if (py::isinstance<functions::Vocab>(obj)) {
    return velox::variant::opaque(
        obj.cast<std::shared_ptr<functions::Vocab>>());
  }

#endif

  // Recursively allocate lists
  if (py::isinstance<py::list>(obj)) {
    auto casted = py::cast<py::list>(obj);
    using List = typename velox::detail::VariantTypeTraits<
        velox::TypeKind::ARRAY>::stored_type;
    List variantList;
    for (const auto& item : casted) {
      variantList.push_back(pyToVariant(item));
    }
    return velox::variant::create<velox::TypeKind::ARRAY>(variantList);
  }

  VELOX_CHECK(
      false,
      "Unsupported Python type {}",
      py::str(py::type::of(obj)).cast<std::string>());
}

velox::variant pyToVariantTyped(
    const py::handle& obj,
    const std::shared_ptr<const velox::Type>& type) {
  auto typeKind = type->kind();
  // Handle nulls
  if (obj.is_none()) {
    return velox::variant::null(typeKind);
  }
  if (py::isinstance<velox::variant>(obj)) {
    return velox::variant(py::cast<velox::variant>(obj));
  }
  switch (typeKind) {
    // Base cases: primitives can be simply returned
    case velox::TypeKind::BOOLEAN:
      return velox::variant::create<velox::TypeKind::BOOLEAN>(
          py::cast<bool>(obj));
    case velox::TypeKind::TINYINT:
      return velox::variant::create<velox::TypeKind::TINYINT>(
          py::cast<int8_t>(obj));
    case velox::TypeKind::SMALLINT:
      return velox::variant::create<velox::TypeKind::SMALLINT>(
          py::cast<int16_t>(obj));
    case velox::TypeKind::INTEGER:
      return velox::variant::create<velox::TypeKind::INTEGER>(
          py::cast<int32_t>(obj));
    case velox::TypeKind::BIGINT:
      return velox::variant::create<velox::TypeKind::BIGINT>(
          py::cast<int64_t>(obj));
    case velox::TypeKind::REAL:
      return velox::variant::create<velox::TypeKind::REAL>(
          py::cast<float>(obj));
    case velox::TypeKind::DOUBLE:
      return velox::variant::create<velox::TypeKind::DOUBLE>(
          py::cast<double>(obj));
    default: {
      std::string typeName = py::str(obj.get_type());
      VELOX_CHECK(
          false,
          fmt::format(
              "Inputted type conversion from {} to {} not implemented",
              typeName,
              typeKind));
    }
  }
}

} // namespace facebook::torcharrow
