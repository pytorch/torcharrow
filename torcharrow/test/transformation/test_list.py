# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torcharrow as ta
import torcharrow.dtypes as dt

# TODO: add/migrate more list tests
class _TestListBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare input data as CPU dataframe
        cls.base_df1 = ta.dataframe(
            {
                "int_list": [[1, 2, None, 3], [4, None, 5], None],
                "str_list": [["a,b,c", "d,e"], [None, "g,h"], None],
                "struct_list": ta.column(
                    [[(1, "a"), (2, "b")], [(3, "c")], None],
                    dtype=dt.List(
                        dt.Struct(
                            [dt.Field("f1", dt.int64), dt.Field("f2", dt.string)]
                        ),
                        nullable=True,
                    ),
                ),
            }
        )
        cls.base_df2 = ta.dataframe(
            {
                "a": [[1, 2, None, 3], [4, None, 5], [1, 2, 3, 4, 5]],
            }
        )

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        raise unittest.SkipTest("abstract base test")

    def test_slice(self):
        col = type(self).df2["a"]

        # 0-indexed, and Python start:end semantic as built-in function
        self.assertEqual(list(col.list.slice(1, 3)), [[2, None], [None, 5], [2, 3]])
        self.assertEqual(list(col.list.slice(2, 4)), [[None, 3], [5], [3, 4]])

        # Only start
        self.assertEqual(
            list(col.list.slice(start=1)), [[2, None, 3], [None, 5], [2, 3, 4, 5]]
        )

        # Only stop
        self.assertEqual(list(col.list.slice(stop=2)), [[1, 2], [4, None], [1, 2]])

    def test_vmap(self):
        df1 = type(self).df1

        self.assertEqual(
            list(df1["int_list"].list.vmap(lambda col: col + 7)),
            [[8, 9, None, 10], [11, None, 12], None],
        )

        self.assertEqual(
            list(df1["str_list"].list.vmap(lambda col: col.str.split(","))),
            [[["a", "b", "c"], ["d", "e"]], [None, ["g", "h"]], None],
        )

        self.assertEqual(
            list(df1["struct_list"].list.vmap(lambda df: df["f2"] + "_suffix")),
            [["a_suffix", "b_suffix"], ["c_suffix"], None],
        )


class TestNumericOpsCpu(_TestListBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df1 = cls.base_df1.copy()
        cls.df2 = cls.base_df2.copy()


if __name__ == "__main__":
    unittest.main()
