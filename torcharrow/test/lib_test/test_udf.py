#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, List

import torcharrow._torcharrow as ta


class TestSimpleColumns(unittest.TestCase):
    # pyre-fixme[11]: Annotation `BaseColumn` is not defined as a type.
    def assert_SimpleColumn(self, col: ta.BaseColumn, val: List[Any]) -> None:
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

    def test_basic(self) -> None:
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

        alnum = ta.generic_udf_dispatch("torcharrow_isalnum", col)
        self.assert_SimpleColumn(alnum, [True, True, True, None, True, True, True])

        digit = ta.generic_udf_dispatch("torcharrow_isdecimal", col)
        self.assert_SimpleColumn(digit, [False, False, False, None, False, True, False])

        islower = ta.generic_udf_dispatch("torcharrow_islower", col)
        self.assert_SimpleColumn(
            islower, [True, False, False, None, False, False, True]
        )

        isupper = ta.generic_udf_dispatch("torcharrow_isupper", col)
        self.assert_SimpleColumn(
            isupper, [False, True, True, None, False, False, False]
        )

        data2 = [1, 2, 3, None, 5, None, -7]
        col2 = self.construct_simple_column(ta.VeloxType_BIGINT(), data2)

        neg = ta.generic_udf_dispatch("negate", col2)
        self.assert_SimpleColumn(neg, [-1, -2, -3, None, -5, None, 7])

        data3 = ["\n", "a", "\t", "76", " ", None]
        col3 = self.construct_simple_column(ta.VeloxType_VARCHAR(), data3)
        isspace = ta.generic_udf_dispatch("torcharrow_isspace", col3)
        self.assert_SimpleColumn(isspace, [True, False, True, False, True, None])

        data4 = ["a b c", "d,e,f"]
        col4 = self.construct_simple_column(ta.VeloxType_VARCHAR(), data4)
        splits = ta.generic_udf_dispatch(
            "split",
            col4,
            ta.ConstantColumn(" ", len(data4)),
        )
        expected = [["a", "b", "c"], ["d,e,f"]]
        self.assertEqual(len(splits), len(expected))
        for i in range(len(splits)):
            self.assert_SimpleColumn(splits[i], expected[i])

    def test_coalesce(self) -> None:
        col1 = self.construct_simple_column(ta.VeloxType_BIGINT(), [1, 2, None, 3])
        result = ta.generic_udf_dispatch(
            "coalesce",
            col1,
            ta.ConstantColumn(42, len(col1)),
        )
        self.assert_SimpleColumn(result, [1, 2, 42, 3])
        self.assertTrue(isinstance(result.type(), ta.VeloxType_BIGINT))

        col2 = self.construct_simple_column(ta.VeloxType_INTEGER(), [1, 2, None, 3])
        result = ta.generic_udf_dispatch(
            "coalesce",
            col2,
            ta.ConstantColumn(42, len(col2), ta.VeloxType_INTEGER()),
        )
        self.assert_SimpleColumn(result, [1, 2, 42, 3])
        self.assertTrue(isinstance(result.type(), ta.VeloxType_INTEGER))

    def test_regex(self) -> None:
        # test some regex UDF
        data = ["abc", "a1", "b2", "c3", "___d4___", None]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)

        match = ta.generic_udf_dispatch(
            "match_re",
            col,
            ta.ConstantColumn("[a-z]\\d", 6),
        )
        self.assert_SimpleColumn(match, [False, True, True, True, False, None])

        search = ta.generic_udf_dispatch(
            "regexp_like",
            col,
            ta.ConstantColumn("[a-z]\\d", 6),
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

    def test_lower(self) -> None:
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("lower", col)
        self.assert_SimpleColumn(
            lcol, ["abc", "abc", "xyz123", None, "xyz", "123", "äöå"]
        )

    def test_istitle(self) -> None:
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

    def test_isnumeric(self) -> None:
        # All False
        data = ["-1", "1.5", "+2", "abc", "AA", "VIII", "1/3", None]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("torcharrow_isnumeric", col)
        self.assert_SimpleColumn(
            lcol, [False, False, False, False, False, False, False, None]
        )

        # All True
        data = ["9876543210123456789", "ⅧⅪ", "ⅷ〩𐍁ᛯ", "᧖७𝟡௫６", "¼⑲⑹⓲➎㉏𐧯"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)
        lcol = ta.generic_udf_dispatch("torcharrow_isnumeric", col)
        self.assert_SimpleColumn(lcol, [True, True, True, True, True])

    def test_trinary(self) -> None:
        data = ["abc", "ABC", "XYZ123", None, "xYZ", "123", "äöå"]
        col = self.construct_simple_column(ta.VeloxType_VARCHAR(), data)

        # substr, 3 parameters
        substr = ta.generic_udf_dispatch(
            "substr",
            col,
            ta.ConstantColumn(2, 7),  # start
            ta.ConstantColumn(2, 7),  # length
        )
        self.assert_SimpleColumn(substr, ["bc", "BC", "YZ", None, "YZ", "23", "öå"])

    def test_quaternary(self) -> None:
        # Based on https://github.com/facebookincubator/velox/blob/8d7fe84d2ce43df952e08ac1015d5bc7c6b868ab/velox/functions/prestosql/tests/ArithmeticTest.cpp#L353-L357
        x = self.construct_simple_column(ta.VeloxType_DOUBLE(), [3.14, 2, -1])
        bound1 = self.construct_simple_column(ta.VeloxType_DOUBLE(), [0, 0, 0])
        bound2 = self.construct_simple_column(ta.VeloxType_DOUBLE(), [4, 4, 3.2])
        bucketCount = self.construct_simple_column(ta.VeloxType_BIGINT(), [3, 3, 4])

        widthBucket = ta.generic_udf_dispatch(
            "width_bucket", x, bound1, bound2, bucketCount
        )
        self.assert_SimpleColumn(widthBucket, [3, 2, 0])

    def test_factory(self) -> None:
        col = ta.factory_udf_dispatch("rand", 42)
        self.assertEqual(col.type().kind(), ta.TypeKind.DOUBLE)
        self.assertEqual(42, len(col))
        for i in range(42):
            self.assertLessEqual(0.0, col[i])
            self.assertLess(col[i], 1.0)


if __name__ == "__main__":
    unittest.main()
