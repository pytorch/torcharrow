# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torcharrow._torcharrow as velox
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
    dtype_of_velox_type,
)


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
