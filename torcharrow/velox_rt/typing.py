# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as ty

import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt

# ------------------------------------------------------------------------------


# pyre-fixme[11]: Annotation `VeloxType` is not defined as a type.
def get_velox_type(dtype: dt.DType) -> velox.VeloxType:
    underlying_dtype = dt.get_underlying_dtype(dtype)
    if underlying_dtype == dt.int64:
        return velox.VeloxType_BIGINT()
    elif underlying_dtype == dt.int32:
        return velox.VeloxType_INTEGER()
    elif underlying_dtype == dt.int16:
        return velox.VeloxType_SMALLINT()
    elif underlying_dtype == dt.int8:
        return velox.VeloxType_TINYINT()
    elif underlying_dtype == dt.float32:
        return velox.VeloxType_REAL()
    elif underlying_dtype == dt.float64:
        return velox.VeloxType_DOUBLE()
    elif underlying_dtype == dt.string:
        return velox.VeloxType_VARCHAR()
    elif underlying_dtype == dt.boolean:
        return velox.VeloxType_BOOLEAN()
    elif isinstance(underlying_dtype, dt.List):
        if underlying_dtype.fixed_size == -1:
            return velox.VeloxArrayType(get_velox_type(underlying_dtype.item_dtype))
        else:
            return velox.VeloxFixedArrayType(
                underlying_dtype.fixed_size, get_velox_type(underlying_dtype.item_dtype)
            )
    elif isinstance(underlying_dtype, dt.Map):
        return velox.VeloxMapType(
            get_velox_type(underlying_dtype.key_dtype),
            get_velox_type(underlying_dtype.item_dtype),
        )
    elif isinstance(underlying_dtype, dt.Struct):
        return velox.VeloxRowType(
            [f.name for f in underlying_dtype.fields],
            [get_velox_type(f.dtype) for f in underlying_dtype.fields],
        )
    else:
        raise NotImplementedError(str(underlying_dtype))


def dtype_of_velox_type(vtype: velox.VeloxType) -> dt.DType:
    if vtype.kind() == velox.TypeKind.BOOLEAN:
        return dt.Boolean(nullable=True)
    if vtype.kind() == velox.TypeKind.TINYINT:
        return dt.Int8(nullable=True)
    if vtype.kind() == velox.TypeKind.SMALLINT:
        return dt.Int16(nullable=True)
    if vtype.kind() == velox.TypeKind.INTEGER:
        return dt.Int32(nullable=True)
    if vtype.kind() == velox.TypeKind.BIGINT:
        return dt.Int64(nullable=True)
    if vtype.kind() == velox.TypeKind.REAL:
        return dt.Float32(nullable=True)
    if vtype.kind() == velox.TypeKind.DOUBLE:
        return dt.Float64(nullable=True)
    if vtype.kind() == velox.TypeKind.VARCHAR:
        return dt.String(nullable=True)
    if vtype.kind() == velox.TypeKind.ARRAY:
        if type(vtype) == velox.VeloxArrayType:
            return dt.List(
                item_dtype=dtype_of_velox_type(vtype.element_type()),
                nullable=True,
            )
        elif type(vtype) == velox.VeloxFixedArrayType:
            return dt.List(
                item_dtype=dtype_of_velox_type(vtype.element_type()),
                fixed_size=vtype.fixed_width(),
                nullable=True,
            )
        else:
            raise TypeError(f"Unknown array type {vtype}")
    if vtype.kind() == velox.TypeKind.MAP:
        # pyre-fixme[11]: Annotation `VeloxMapType` is not defined as a type.
        vtype = ty.cast(velox.VeloxMapType, vtype)
        return dt.Map(
            key_dtype=dtype_of_velox_type(vtype.key_type()),
            item_dtype=dtype_of_velox_type(vtype.value_type()),
            nullable=True,
        )
    if vtype.kind() == velox.TypeKind.ROW:
        # pyre-fixme[11]: Annotation `VeloxRowType` is not defined as a type.
        vtype = ty.cast(velox.VeloxRowType, vtype)
        fields = [
            dt.Field(
                name=vtype.name_of(i), dtype=dtype_of_velox_type(vtype.child_at(i))
            )
            for i in range(vtype.size())
        ]
        return dt.Struct(fields=fields, nullable=True)

    raise AssertionError(
        f"translation of Velox typekind {vtype.kind()} to dtype unsupported"
    )
