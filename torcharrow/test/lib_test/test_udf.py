#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from typing import List, Any

# @manual=//pytorch/torcharrow/csrc/velox:_torcharrow
import torcharrow._torcharrow as ta


class TestSimpleColumns(unittest.TestCase):
    def assert_SimpleColumn(self, col: ta.BaseColumn, val: List[Any]):
        self.assertEqual(len(col), len(val))
        for i in range(len(val)):
            if val[i] is None:
                self.assertTrue(col.is_null_at(i))
            else:
                self.assertFalse(col.is_null_at(i))
                if isinstance(val[i], float):
                    self.assertAlmostEqual(col[i], val[i], places=6)
                else:
                    self.assertEqual(col[i], val[i])

    @staticmethod
    def construct_simple_column(velox_type, data: List[Any]):
        col = ta.Column(velox_type)
        for item in data:
            if item is None:
                col.append_null()
            else:
                col.append(item)
        return col

    def test_basic(self):
        # test some UDFs together
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)

        lcol = ta.generic_udf_dispatch("lower", col)
        self.assert_SimpleColumn(
            lcol, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"]
        )

        ucol = ta.generic_udf_dispatch("upper", col)
        self.assert_SimpleColumn(
            ucol, ["ABC", "ABC", "XYZ123", None, "XYZ", "123", "ÄÖÅ"]
        )

        lcol2 = ta.generic_udf_dispatch("lower", ucol)
        self.assert_SimpleColumn(
            lcol2, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"]
        )

        ucol2 = ta.generic_udf_dispatch("upper", lcol)
        self.assert_SimpleColumn(
            ucol2, ["ABC", "ABC", "XYZ123", None, "XYZ", "123", "ÄÖÅ"]
        )

        alpha = ta.generic_udf_dispatch("torcharrow_isalpha", col)
        self.assert_SimpleColumn(alpha, [True, True, False, None, True, False, True])

        digit = ta.generic_udf_dispatch("isdecimal", col)
        self.assert_SimpleColumn(digit, [False, False, False, None, False, True, False])

        islower = ta.generic_udf_dispatch("torcharrow_islower", col)
        self.assert_SimpleColumn(
            islower, [True, False, False, None, False, False, True]
        )

        isupper = ta.generic_udf_dispatch("isupper", col)
        self.assert_SimpleColumn(
            isupper, [False, True, True, None, False, False, False]
        )

        # substr, 3 parameters
        substr = ta.generic_udf_dispatch(
            "substr",
            col,
            ta.ConstantColumn(2, 7),  # start
            ta.ConstantColumn(2, 7),  # length
        )
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]

        self.assert_SimpleColumn(substr, ["bc", "BC", "YZ", None, "YZ", "23", "öå"])

        data2 = [1, 2, 3, None, 5, None, -7]
        col2 = self.construct_simple_column(ta.VeloxType_BIGINT(), data2)

        neg = ta.generic_udf_dispatch("negate", col2)
        self.assert_SimpleColumn(neg, [-1, -2, -3, None, -5, None, 7])

        data3 = ["\n", "a", "\t", "76", " ", None]
        col3 = self.construct_simple_column(ta.VeloxType_VARCHAR(), data3)
        isspace = ta.generic_udf_dispatch("torcharrow_isspace", col3)
        self.assert_SimpleColumn(isspace, [True, False, True, False, True, None])

    def test_regex(self):
        # test some regex UDF
        data = ["abc", "a1", "b2", "c3", "___d4___", None]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)

        match = ta.generic_udf_dispatch(
            "match_re", col, ta.ConstantColumn("[a-z]\\d", 6)
        )
        self.assert_SimpleColumn(match, [False, True, True, True, False, None])

        search = ta.generic_udf_dispatch(
            "regexp_like", col, ta.ConstantColumn("[a-z]\\d", 6)
        )
        self.assert_SimpleColumn(search, [False, True, True, True, True, None])

        data = ["d4e5", "a1", "b2", "c3", "___d4___f6"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        extract = ta.generic_udf_dispatch(
            "regexp_extract_all",
            col,
            ta.ConstantColumn("([a-z])\\d", 5),
        )
        expected = [["d4", "e5"], ["a1"], ["b2"], ["c3"], ["d4", "f6"]]
        self.assertEqual(len(extract), len(expected))
        for i in range(len(extract)):
            self.assert_SimpleColumn(extract[i], expected[i])

    def test_lower(self):
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("lower", col)
        self.assert_SimpleColumn(
            lcol, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"]
        )

    def test_istitle(self):
        # All False
        data = [
            "\n",
            "a",
            "76",
            " ",
            ",",
            "hello, and Welcome !",
            "ª",
            "AaBbCd",
            None,
        ]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        istitle = ta.generic_udf_dispatch("torcharrow_istitle", col)
        self.assert_SimpleColumn(
            istitle,
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                None,
            ],
        )

        # All True
        data = [
            "Hello, And Welcome To My World!",
            "A_B_C_D",
            "ᾬª_ _D",
            "ᾏᵄ ǲb ǈdd",
            "_ Aa _ Abc",
            "A1 B2",
            "A1B2",
        ]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        istitle = ta.generic_udf_dispatch("torcharrow_istitle", col)
        self.assert_SimpleColumn(istitle, [True, True, True, True, True, True, True])

    def test_isnumeric(self):
        # All False
        data = ["-1", "1.5", "+2", "abc", "AA", "VIII", "1/3", None]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("isnumeric", col)
        self.assert_SimpleColumn(
            lcol, [False, False, False, False, False, False, False, None]
        )

        # All True
        data = ["9876543210123456789", "ⅧⅪ", "ⅷ〩𐍁ᛯ", "᧖७𝟡௫６", "¼⑲⑹⓲➎㉏𐧯"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("isnumeric", col)
        self.assert_SimpleColumn(lcol, [True, True, True, True, True])

    def test_factory(self):
        col = ta.factory_udf_dispatch("rand", 42)
        self.assertEqual(col.type().kind(), ta.TypeKind.DOUBLE)
        self.assertEqual(42, len(col))
        for i in range(42):
            self.assertLessEqual(0.0, col[i])
            self.assertLess(col[i], 1.0)


if __name__ == "__main__":
    unittest.main()
