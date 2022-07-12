/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <memory>
#include <string>
#include <unordered_map>
#include "vector.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/arrow/Abi.h"
#include "velox/vector/arrow/Bridge.h"

// TODO: Move uses of static variables into .cpp. Static variables are local to
// the compilation units so every file that includes this header will have its
// own instance of the static variables, which in most cases is not what we want

namespace facebook::torcharrow {

class NotAppendableException : public std::exception {
 public:
  virtual const char* what() const throw() {
    return "Cannot append in a view";
  }
};

struct TorchArrowGlobalStatic {
  static velox::core::QueryCtx& queryContext();
  static velox::core::ExecCtx& execContext();

  static velox::memory::MemoryPool* rootMemoryPool() {
    static velox::memory::MemoryPool* const pool =
        &velox::memory::getProcessDefaultMemoryManager().getRoot();
    return pool;
  }
};

struct GenericUDFDispatchKey {
  std::string udfName;
  // TODO: use row type instead of string
  std::string typeSignature;

  GenericUDFDispatchKey(std::string udfName, std::string typeSignature)
      : udfName(std::move(udfName)), typeSignature(std::move(typeSignature)) {}
};

inline bool operator==(
    const GenericUDFDispatchKey& lhs,
    const GenericUDFDispatchKey& rhs) {
  return lhs.udfName == rhs.udfName && lhs.typeSignature == rhs.typeSignature;
}

velox::variant pyToVariant(const pybind11::handle& obj);
velox::variant pyToVariantTyped(
    const pybind11::handle& obj,
    const std::shared_ptr<const velox::Type>& type);

class BaseColumn;

class OperatorHandle {
 public:
  OperatorHandle(
      velox::RowTypePtr inputRowType,
      std::shared_ptr<velox::exec::ExprSet> exprSet)
      : inputRowType_(inputRowType), exprSet_(exprSet) {}

  // Create OperatorHandle based on the function name, input type
  // and explicitly provided output type.
  static std::unique_ptr<OperatorHandle> fromCall(
      velox::RowTypePtr inputRowType,
      velox::TypePtr outputType,
      const std::string& functionName);

  // Create OperatorHandle based on the function name and input type
  // Doesn't handle type promotion yet
  static std::unique_ptr<OperatorHandle> fromUDF(
      velox::RowTypePtr inputRowType,
      const std::string& udfName);

  // Create OperatorHandle based on input type and output type only.
  // Note that this creates a cast expression, not a call expression.
  static std::unique_ptr<OperatorHandle> fromCast(
      velox::RowTypePtr inputRowType,
      velox::TypePtr outputType);

  static velox::RowVectorPtr wrapRowVector(
      const std::vector<velox::VectorPtr>& children,
      std::shared_ptr<const velox::RowType> rowType) {
    return std::make_shared<velox::RowVector>(
        TorchArrowGlobalStatic::rootMemoryPool(),
        rowType,
        velox::BufferPtr(nullptr),
        children[0]->size(),
        children);
  }

  // Specialized invoke methods for common arities
  // Input type velox::VectorPtr (instead of BaseColumn) since it might be a
  // ConstantVector
  // TODO: Use Column once ConstantColumn is supported
  std::unique_ptr<BaseColumn> call(
      velox::RowVectorPtr inputRows,
      velox::vector_size_t size);

  std::unique_ptr<BaseColumn> call(velox::vector_size_t size);

  std::unique_ptr<BaseColumn> call(velox::VectorPtr a);

  std::unique_ptr<BaseColumn> call(velox::VectorPtr a, velox::VectorPtr b);

  std::unique_ptr<BaseColumn> call(const std::vector<velox::VectorPtr>& args);

 private:
  velox::RowTypePtr inputRowType_;
  std::shared_ptr<velox::exec::ExprSet> exprSet_;
};

class PromoteNumericTypeKind {
 public:
  static velox::TypeKind promoteColumnColumn(
      velox::TypeKind a,
      velox::TypeKind b) {
    return promote(a, b, PromoteStrategy::ColumnColumn);
  }

  // Assume a being a column and b being a scalar
  static velox::TypeKind promoteColumnScalar(
      velox::TypeKind a,
      velox::TypeKind b) {
    return promote(a, b, PromoteStrategy::ColumnScalar);
  }

 private:
  enum class PromoteStrategy {
    ColumnColumn,
    ColumnScalar,
  };

  static velox::TypeKind promote(
      velox::TypeKind a,
      velox::TypeKind b,
      PromoteStrategy promoteStrategy) {
    constexpr auto b1 = velox::TypeKind::BOOLEAN;
    constexpr auto i1 = velox::TypeKind::TINYINT;
    constexpr auto i2 = velox::TypeKind::SMALLINT;
    constexpr auto i4 = velox::TypeKind::INTEGER;
    constexpr auto i8 = velox::TypeKind::BIGINT;
    constexpr auto f4 = velox::TypeKind::REAL;
    constexpr auto f8 = velox::TypeKind::DOUBLE;
    constexpr auto num_numeric_types =
        static_cast<int>(velox::TypeKind::DOUBLE) + 1;

    VELOX_CHECK(
        static_cast<int>(a) < num_numeric_types &&
        static_cast<int>(b) < num_numeric_types);

    if (a == b) {
      return a;
    }

    switch (promoteStrategy) {
      case PromoteStrategy::ColumnColumn: {
        // Sliced from
        // https://github.com/pytorch/pytorch/blob/1c502d1f8ec861c31a08d580ae7b73b7fbebebed/c10/core/ScalarType.h#L402-L421
        static constexpr velox::TypeKind
            promoteTypesLookup[num_numeric_types][num_numeric_types] = {
                /*        b1  i1  i2  i4  i8  f4  f8*/
                /* b1 */ {b1, i1, i2, i4, i8, f4, f8},
                /* i1 */ {i1, i1, i2, i4, i8, f4, f8},
                /* i2 */ {i2, i2, i2, i4, i8, f4, f8},
                /* i4 */ {i4, i4, i4, i4, i8, f4, f8},
                /* i8 */ {i8, i8, i8, i8, i8, f4, f8},
                /* f4 */ {f4, f4, f4, f4, f4, f4, f8},
                /* f8 */ {f8, f8, f8, f8, f8, f8, f8},
            };
        return promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
      } break;
      case PromoteStrategy::ColumnScalar: {
        // TODO: Decide on how we want to handle column-scalar type promotion.
        // Current strategy is to always respect the type of the column for
        // int-int cases.
        static constexpr velox::TypeKind
            promoteTypesLookup[num_numeric_types][num_numeric_types] = {
                /*        b1  i1  i2  i4  i8  f4  f8*/
                /* b1 */ {b1, b1, b1, b1, b1, f4, f8},
                /* i1 */ {i1, i1, i1, i1, i1, f4, f8},
                /* i2 */ {i2, i2, i2, i2, i2, f4, f8},
                /* i4 */ {i4, i4, i4, i4, i4, f4, f8},
                /* i8 */ {i8, i8, i8, i8, i8, f4, f8},
                /* f4 */ {f4, f4, f4, f4, f4, f4, f8},
                /* f8 */ {f8, f8, f8, f8, f8, f8, f8},
            };
        return promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
      } break;
      default: {
        throw std::logic_error(
            "Unsupported promote: " +
            std::to_string(static_cast<int64_t>(promoteStrategy)));
      } break;
    }
  }
};

enum class BinaryOpCode : int16_t {
  Plus = 0,
  Minus,
  Multiply,
  Divide,
  Floordiv,
  Modulus,
  Pow,
  Eq,
  Neq,
  Lt,
  Gt,
  Lte,
  Gte,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  // Do not use NOpCode! This is used for counting elements in BinaryOpCode
  NOpCode,
};

std::string opCodeToFunctionName(BinaryOpCode opCode);

enum class OperatorType : int8_t {
  Direct = 0,
  Reverse,
  // Do not use NOpType! This is used for counting elements in OperatorType
  NOpType,
};

class BaseColumn {
  friend class ArrayColumn;
  friend class MapColumn;
  friend class RowColumn;
  friend class OperatorHandle;

 protected:
  velox::VectorPtr _delegate;
  velox::vector_size_t _offset;
  velox::vector_size_t _length;
  velox::vector_size_t _nullCount;

  void bumpLength() {
    _length++;
    _delegate.get()->resize(_offset + _length);
  }

  bool isAppendable() {
    return _offset + _length == _delegate.get()->size();
  }

  // TODO: move this method as static...
  velox::RowVectorPtr wrapRowVector(
      const std::vector<velox::VectorPtr>& children,
      std::shared_ptr<const velox::RowType> rowType) {
    return std::make_shared<velox::RowVector>(
        pool_,
        rowType,
        velox::BufferPtr(nullptr),
        children[0]->size(),
        children);
  }

  // TODO: Model binary functions as UDF.
  static OperatorHandle* getOrCreateBinaryOperatorHandle(
      velox::TypePtr c0Type,
      velox::TypePtr c1Type,
      velox::TypeKind commonTypeKind,
      BinaryOpCode opCode);

 private:
  velox::memory::MemoryPool* pool_ =
      &velox::memory::getProcessDefaultMemoryManager().getRoot();

  static velox::vector_size_t countNulls(
      velox::VectorPtr veloxVector,
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    VELOX_CHECK_LE(offset + length, veloxVector->size());
    return velox::BaseVector::countNulls(
        veloxVector->nulls(), offset, offset + length);
  }

 public:
  BaseColumn(
      const BaseColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : _delegate(other._delegate), _offset(offset), _length(length) {
    if (!_delegate->getNullCount().has_value() ||
        *_delegate->getNullCount() == 0) {
      _delegate->setNullCount(countNulls(_delegate, 0, _delegate->size()));
    }
    _nullCount = countNulls(_delegate, _offset, _length);
  }

  explicit BaseColumn(velox::TypePtr type)
      : _offset(0), _length(0), _nullCount(0) {
    _delegate = velox::BaseVector::create(type, 0, pool_);
  }

  explicit BaseColumn(velox::VectorPtr delegate)
      : _delegate(delegate), _offset(0), _length(delegate->size()) {
    if (!_delegate->getNullCount().has_value() ||
        *_delegate->getNullCount() == 0) {
      _delegate->setNullCount(countNulls(_delegate, 0, _delegate->size()));
    }
    _nullCount = *_delegate->getNullCount();
  }

  virtual ~BaseColumn() = default;

  velox::TypePtr type() const {
    return _delegate->type();
  }

  bool isNullAt(velox::vector_size_t idx) const {
    return _delegate->isNullAt(_offset + idx);
  }

  velox::vector_size_t getOffset() const {
    return _offset;
  }

  velox::vector_size_t getLength() const {
    return _length;
  }

  velox::vector_size_t getNullCount() const {
    return _nullCount;
  }

  velox::VectorPtr getUnderlyingVeloxVector() const {
    return _delegate;
  }

  void exportToArrow(ArrowArray* output);

  static std::shared_ptr<velox::exec::ExprSet> genBinaryExprSet(
      std::shared_ptr<const velox::RowType> inputRowType,
      std::shared_ptr<const velox::Type> commonType,
      std::shared_ptr<const velox::Type> outputType,
      const std::string& functionName);

  // From velox/type/velox::variant.h
  // TODO: refactor into some type utility class
  template <velox::TypeKind Kind>
  static const std::shared_ptr<const velox::Type> kind2type() {
    return velox::TypeFactory<Kind>::create();
  }

 public:
  // generic UDF
  static std::unique_ptr<BaseColumn> genericUDF(
      const std::string& udfName,
      const std::vector<BaseColumn>& cols);

  // factory UDF (e.g rand)
  static std::unique_ptr<BaseColumn> factoryNullaryUDF(
      const std::string& udfName,
      int size);

  // factory methods to create columns
  static std::unique_ptr<BaseColumn> createConstantColumn(
      velox::variant value,
      velox::vector_size_t size);
};

std::unique_ptr<BaseColumn> createColumn(velox::VectorPtr vec);

std::unique_ptr<BaseColumn> createColumn(
    velox::VectorPtr vec,
    velox::vector_size_t offset,
    velox::vector_size_t length);

template <typename T>
class SimpleColumn : public BaseColumn {
 public:
  SimpleColumn() : BaseColumn(velox::CppToType<T>::create()) {}
  explicit SimpleColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  SimpleColumn(
      const SimpleColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  T valueAt(int i) {
    return _delegate.get()->template as<velox::SimpleVector<T>>()->valueAt(
        _offset + i);
  }

  void append(const T& value) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto index = _delegate.get()->size();
    auto flatVector = _delegate->asFlatVector<T>();
    flatVector->resize(index + 1);
    flatVector->set(index, value);
    bumpLength();
  }

  void appendNull() {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto index = _delegate.get()->size();
    _delegate->resize(index + 1);
    _delegate->setNull(index, true);
    _nullCount++;
    _delegate->setNullCount(_nullCount);
    bumpLength();
  }

  std::unique_ptr<SimpleColumn<T>> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<SimpleColumn<T>>(*this, offset, length);
  }

  //
  // unary numeric column ops
  //

  // TODO: return SimpleColumn<T> instead?
  std::unique_ptr<BaseColumn> invert() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto opHandle = []() -> std::unique_ptr<OperatorHandle> {
      if constexpr (std::is_same_v<T, bool>) {
        return OperatorHandle::fromCall(
            inputRowType, velox::CppToType<T>::create(), "not");
      } else {
        return OperatorHandle::fromCall(
            inputRowType, velox::CppToType<T>::create(), "bitwise_not");
      }
    }();

    return opHandle->call(_delegate);
  }

  std::unique_ptr<BaseColumn> neg() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto opHandle = OperatorHandle::fromCall(
        inputRowType, velox::CppToType<T>::create(), "negate");
    return opHandle->call(_delegate);
  }

  std::unique_ptr<BaseColumn> abs() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto opHandle = OperatorHandle::fromCall(
        inputRowType, velox::CppToType<T>::create(), "abs");
    return opHandle->call(_delegate);
  }

  std::unique_ptr<BaseColumn> ceil() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto opHandle = OperatorHandle::fromCall(
        inputRowType, velox::CppToType<T>::create(), "ceil");
    return opHandle->call(_delegate);
  }

  std::unique_ptr<BaseColumn> floor() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto opHandle = OperatorHandle::fromCall(
        inputRowType, velox::CppToType<T>::create(), "floor");
    return opHandle->call(_delegate);
  }

  std::unique_ptr<BaseColumn> round() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto opHandle = OperatorHandle::fromCall(
        inputRowType, velox::CppToType<T>::create(), "torcharrow_round");
    return opHandle->call(_delegate);
  }

  //
  // unary cast
  //
  // Note that we accept the casted-to return type as a value parameter, and
  // we populate a static array of OperatorHandles, one for each casted-to
  // type we see. We accept the return type as a value at runtime instead of
  // as a template parameter at compile-time to avoid creating N^2 number of
  // cast functions.
  //
  std::unique_ptr<BaseColumn> cast(velox::TypeKind toKind) {
    constexpr auto num_numeric_types =
        static_cast<int>(velox::TypeKind::DOUBLE) + 1;
    static std::array<
        std::unique_ptr<OperatorHandle>,
        num_numeric_types> /* library-local */ opHandles;

    int id = static_cast<int>(toKind);
    if (opHandles[id] == nullptr) {
      const static auto inputRowType =
          velox::ROW({"c0"}, {velox::CppToType<T>::create()});
      velox::TypePtr toType =
          VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(kind2type, toKind);
      opHandles[id] = OperatorHandle::fromCast(inputRowType, toType);
    }

    return opHandles[id]->call(_delegate);
  }

  //
  // binary numeric column ops
  //

 private:
  std::unique_ptr<BaseColumn> dispatchBinaryOperation(
      velox::VectorPtr other,
      velox::TypeKind commonTypeKind,
      BinaryOpCode opCode,
      OperatorType opType) {
    if (opType == OperatorType::Direct) {
      // c0 is `this` and c1 is `other`
      OperatorHandle* opHandle = getOrCreateBinaryOperatorHandle(
          this->type(), other->type(), commonTypeKind, opCode);
      return opHandle->call(_delegate, other);
    } else if (opType == OperatorType::Reverse) {
      // c0 is `other` and c1 is `this`
      OperatorHandle* opHandle = getOrCreateBinaryOperatorHandle(
          other->type(), this->type(), commonTypeKind, opCode);
      return opHandle->call(other, _delegate);
    } else {
      VELOX_CHECK(
          false,
          "Unsupported OperatorType: {:d}",
          static_cast<int16_t>(opType));
    }
  }

 public:
  // Perform the operation with the other operand also being a column
  std::unique_ptr<BaseColumn> callBinaryOp(
      const BaseColumn& other,
      BinaryOpCode opCode,
      OperatorType opType /* Direct or Reverse */) {
    velox::TypeKind commonTypeKind =
        PromoteNumericTypeKind::promoteColumnColumn(
            this->type()->kind(), other.type()->kind());

    return dispatchBinaryOperation(
        other.getUnderlyingVeloxVector(), commonTypeKind, opCode, opType);
  }

  // Perform the operation with the other operand being a scalar
  std::unique_ptr<BaseColumn> callBinaryOp(
      const pybind11::handle& otherPyObj,
      BinaryOpCode opCode,
      OperatorType opType /* Direct or Reverse */) {
    velox::variant val = pyToVariant(otherPyObj);
    velox::VectorPtr other = velox::BaseVector::createConstant(
        val, _delegate->size(), TorchArrowGlobalStatic::rootMemoryPool());

    velox::TypeKind commonTypeKind =
        PromoteNumericTypeKind::promoteColumnScalar(
            this->type()->kind(), other->typeKind());

    return dispatchBinaryOperation(other, commonTypeKind, opCode, opType);
  }
};

template <typename T>
class FlatColumn : public SimpleColumn<T> {};

template <typename T>
class ConstantColumn : public SimpleColumn<T> {
 public:
  // For scalar type that ConstantVector can be created from variant
  ConstantColumn(velox::variant value, velox::vector_size_t size)
      : SimpleColumn<T>(velox::BaseVector::createConstant(
            value,
            size,
            TorchArrowGlobalStatic::rootMemoryPool())) {}

  // Create from a Vector position
  ConstantColumn(velox::VectorPtr vector, int index, velox::vector_size_t size)
      : SimpleColumn<T>(
            velox::BaseVector::wrapInConstant(size, index, vector)) {}
};

class ArrayColumn : public BaseColumn {
 public:
  explicit ArrayColumn(velox::TypePtr type) : BaseColumn(type) {}
  explicit ArrayColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  ArrayColumn(
      const ArrayColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  void appendElement(const BaseColumn& new_element) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<velox::ArrayVector>();
    auto elements = dataPtr->elements();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = elements.get()->size();
    auto new_size = new_element.getLength();
    elements.get()->append(new_element._delegate.get());
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    bumpLength();
  }

  void appendNull() {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<velox::ArrayVector>();
    auto elements = dataPtr->elements();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = elements.get()->size();
    auto new_size = 0;
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    dataPtr->setNull(new_index, true);
    _nullCount++;
    _delegate->setNullCount(_nullCount);
    bumpLength();
  }

  std::unique_ptr<BaseColumn> elements();

  std::unique_ptr<BaseColumn> valueAt(velox::vector_size_t i);

  std::unique_ptr<ArrayColumn> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<ArrayColumn>(*this, offset, length);
  }

  std::unique_ptr<ArrayColumn> withElements(const BaseColumn& newElements);
};

class MapColumn : public BaseColumn {
 public:
  explicit MapColumn(velox::TypePtr type) : BaseColumn(type) {}
  explicit MapColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  MapColumn(
      const MapColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  velox::vector_size_t offsetAt(velox::vector_size_t index) const {
    return _delegate.get()->as<velox::MapVector>()->offsetAt(_offset + index);
  }

  velox::vector_size_t sizeAt(velox::vector_size_t index) const {
    return _delegate.get()->as<velox::MapVector>()->sizeAt(_offset + index);
  }

  void appendElement(const BaseColumn& newKey, const BaseColumn& newValue) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<velox::MapVector>();

    auto keys = dataPtr->mapKeys();
    auto values = dataPtr->mapValues();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = keys.get()->size();
    auto new_size = newKey.getLength();
    keys.get()->append(newKey._delegate.get());
    values.get()->append(newValue._delegate.get());
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    bumpLength();
  }

  void appendNull() {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<velox::MapVector>();

    auto keys = dataPtr->mapKeys();
    auto values = dataPtr->mapValues();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = keys.get()->size();
    auto new_size = 0;
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    dataPtr->setNull(new_index, true);
    _nullCount++;
    _delegate->setNullCount(_nullCount);
    bumpLength();
  }

  std::unique_ptr<BaseColumn> valueAt(velox::vector_size_t i);

  std::unique_ptr<BaseColumn> mapKeys() {
    auto dataPtr = _delegate.get()->as<velox::MapVector>();
    auto keys = dataPtr->mapKeys();
    auto reshapedKeys = reshape(
        keys,
        std::bind(&velox::MapVector::offsetAt, *dataPtr, std::placeholders::_1),
        std::bind(&velox::MapVector::sizeAt, *dataPtr, std::placeholders::_1),
        dataPtr->size());
    return createColumn(reshapedKeys, _offset, _length);
  }

  std::unique_ptr<BaseColumn> mapValues() {
    auto dataPtr = _delegate.get()->as<velox::MapVector>();
    auto values = dataPtr->mapValues();
    auto reshapedValues = reshape(
        values,
        std::bind(&velox::MapVector::offsetAt, *dataPtr, std::placeholders::_1),
        std::bind(&velox::MapVector::sizeAt, *dataPtr, std::placeholders::_1),
        dataPtr->size());
    return createColumn(reshapedValues, _offset, _length);
  }

  std::unique_ptr<MapColumn> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<MapColumn>(*this, offset, length);
  }
};

class RowColumn : public BaseColumn {
 public:
  explicit RowColumn(velox::TypePtr type) : BaseColumn(type) {}
  explicit RowColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  RowColumn(
      const RowColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  std::unique_ptr<BaseColumn> childAt(velox::column_index_t index) {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    return createColumn(dataPtr->childAt(index), _offset, _length);
  }

  void setChild(velox::column_index_t index, const BaseColumn& new_child) {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    dataPtr->children()[index] = new_child._delegate;
  }

  size_t childrenSize() {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    return dataPtr->childrenSize();
  }

  std::unique_ptr<RowColumn> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<RowColumn>(*this, offset, length);
  }

  void setLength(velox::vector_size_t length) {
    _length = length;
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    dataPtr->resize(_offset + _length);
  }

  void setNullAt(velox::vector_size_t idx) {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    if (!isNullAt(idx)) {
      _nullCount++;
    }
    dataPtr->setNull(_offset + idx, true);
  }

  std::unique_ptr<BaseColumn> copy() {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    auto newVector =
        velox::RowVector::createEmpty(dataPtr->type(), dataPtr->pool());
    newVector.get()->resize(dataPtr->size());
    newVector.get()->copy(dataPtr, 0, 0, dataPtr->size());
    auto newColumn = createColumn(newVector, _offset, _length);
    return newColumn;
  }
};

} // namespace facebook::torcharrow

namespace std {
template <>
struct hash<::facebook::torcharrow::GenericUDFDispatchKey> {
  size_t operator()(
      const ::facebook::torcharrow::GenericUDFDispatchKey& x) const {
    return std::hash<std::string>()(x.udfName) ^
        (~std::hash<std::string>()(x.typeSignature));
  }
};
} // namespace std
