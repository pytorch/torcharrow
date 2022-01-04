# Copyright (c) Meta Platforms, Inc. and affiliates.
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt

# TODO: add/migrate more list tests
class _TestListBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare input data as CPU dataframe
        cls.base_df = ta.DataFrame(
            {
                "int_list": [[1, 2, None, 3], [4, None, 5], None],
                "str_list": [["a,b,c", "d,e"], [None, "g,h"], None],
                "struct_list": ta.Column(
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

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        raise unittest.SkipTest("abstract base test")

    def test_vmap(self):
        df = type(self).df

        self.assertEqual(
            list(df["int_list"].list.vmap(lambda col: col + 7)),
            [[8, 9, None, 10], [11, None, 12], None],
        )

        self.assertEqual(
            list(df["str_list"].list.vmap(lambda col: col.str.split(","))),
            [[["a", "b", "c"], ["d", "e"]], [None, ["g", "h"]], None],
        )

        self.assertEqual(
            list(df["struct_list"].list.vmap(lambda df: df["f2"] + "_suffix")),
            [["a_suffix", "b_suffix"], ["c_suffix"], None],
        )


class TestNumericOpsCpu(_TestListBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df = cls.base_df.copy()


if __name__ == "__main__":
    unittest.main()
