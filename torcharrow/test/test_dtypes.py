# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import unittest

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
from torcharrow.dtypes import (
    Field,
    Int64,
    List,
    Map,
    String,
    Struct,
    int64,
    is_list,
    is_string,
    is_numerical,
    is_map,
    is_int64,
    is_struct,
    string,
)
from torcharrow.velox_rt.typing import dtype_of_velox_type


class TestTypes(unittest.TestCase):
    def test_int64(self):
        self.assertEqual(str(int64), "int64")
        self.assertEqual(int64.name, "int64")
        self.assertEqual(int64.typecode, "l")
        self.assertEqual(int64.arraycode, "l")
        self.assertTrue(is_numerical(int64))

    def test_string(self):
        self.assertEqual(str(string), "string")
        self.assertEqual(string.typecode, "u")
        self.assertEqual(string.nullable, False)
        self.assertEqual(String(nullable=True).nullable, True)

    def test_list(self):
        self.assertEqual(str(List(Int64(nullable=True))), "List(Int64(nullable=True))")
        self.assertEqual(
            str(List(Int64(nullable=True)).item_dtype), "Int64(nullable=True)"
        )
        self.assertEqual(List(Int64(nullable=True)).typecode, "+l")

    def test_map(self):
        self.assertEqual(str(Map(int64, string)), "Map(int64, string)")
        self.assertEqual(Map(int64, string).typecode, "+m")

    def test_struct(self):
        self.assertEqual(
            str(Struct([Field("a", int64), Field("b", string)])),
            "Struct([Field('a', int64), Field('b', string)])",
        )
        self.assertEqual(Struct([Field("a", int64), Field("b", string)]).typecode, "+s")

    def test_serialization(self):
        # simple types
        for nullable in [True, False]:
            for dtype in [
                dt.Int8(nullable),
                dt.Int16(nullable),
                dt.Int32(nullable),
                dt.Int64(nullable),
                dt.Float32(nullable),
                dt.Float64(nullable),
                dt.String(nullable),
            ]:
                serialized_dtype = pickle.dumps(dtype)
                deserialized_dtype = pickle.loads(serialized_dtype)
                self.assertEqual(dtype, deserialized_dtype)

        # list/map
        for nullable in [True, False]:
            for dtype in [
                dt.List(dt.int64, nullable),
                dt.List(dt.List(dt.string, nullable)),
                dt.Map(dt.string, dt.Int64(nullable)),
                dt.Map(dt.string, dt.List(dt.Int64, nullable)),
            ]:
                serialized_dtype = pickle.dumps(dtype)
                deserialized_dtype = pickle.loads(serialized_dtype)
                self.assertEqual(dtype, deserialized_dtype)

        # nested struct
        dtype = dt.Struct(
            [
                dt.Field("label", dt.int8),
                dt.Field(
                    "dense_features",
                    dt.Struct(
                        [
                            dt.Field(int_name, dt.Int32(nullable=True))
                            for int_name in ["int_1", "int_2", "int_3"]
                        ]
                    ),
                ),
            ]
        )
        serialized_dtype = pickle.dumps(dtype)
        deserialized_dtype = pickle.loads(serialized_dtype)
        self.assertEqual(dtype, deserialized_dtype)

    def test_convert_velox_type_array(self):
        vType = velox.VeloxArrayType(velox.VeloxArrayType(velox.VeloxType_VARCHAR()))
        dType = dtype_of_velox_type(vType)
        self.assertTrue(is_list(dType))
        self.assertTrue(is_list(dType.item_dtype))
        self.assertTrue(is_string(dType.item_dtype.item_dtype))

    def test_convert_velox_type_map(self):
        vType = velox.VeloxMapType(velox.VeloxType_VARCHAR(), velox.VeloxType_BIGINT())
        dType = dtype_of_velox_type(vType)
        self.assertTrue(is_map(dType))
        self.assertTrue(is_string(dType.key_dtype))
        self.assertTrue(is_int64(dType.item_dtype))

    def test_convert_velox_type_row(self):
        vType = velox.VeloxRowType(
            ["c0", "c1"], [velox.VeloxType_VARCHAR(), velox.VeloxType_BIGINT()]
        )
        dType = dtype_of_velox_type(vType)
        self.assertTrue(is_struct(dType))
        self.assertEqual(
            dType.fields,
            [
                Field(name="c0", dtype=String(nullable=True)),
                Field(name="c1", dtype=Int64(nullable=True)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
