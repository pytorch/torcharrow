# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import functional


class _TestFunctionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare input data as CPU dataframe
        cls.base_df1 = ta.dataframe(
            {
                "int64_list": [[11, 12, 13], [21, 22, 23, 24, 25, 26], [31, 32]],
                "int32_list": [[11, 12, 13], [21, 22, 23, 24, 25, 26], [31, 32]],
            },
            dtype=dt.Struct(
                [
                    dt.Field("int64_list", dt.List(dt.int64)),
                    dt.Field("int32_list", dt.List(dt.int32)),
                ]
            ),
        )

        cls.base_width_bucket_df = ta.dataframe(
            {
                "x": [3.14, 2, -1],
                "bound1": [0, 0, 0],
                "bound2": [4, 4, 3.2],
                "bucketCount": [3, 3, 4],
            },
            dtype=dt.Struct(
                [
                    dt.Field("x", dt.float64),
                    dt.Field("bound1", dt.float64),
                    dt.Field("bound2", dt.float64),
                    dt.Field("bucketCount", dt.int64),
                ]
            ),
        )

        cls.base_if_else_df = ta.dataframe(
            {
                "cond": [True, False, True, False],
                "x": [1, 2, 3, 4],
                "y": [10, 20, 30, 40],
            },
        )

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    def test_slice(self):
        self.assertEqual(
            list(functional.slice(type(self).df1["int64_list"], 2, 3)),
            [[12, 13], [22, 23, 24], [32]],
        )

    def test_intersect_constant_aray(self):
        self.assertEqual(
            list(
                functional.array_intersect(
                    type(self).df1["int64_list"], [12, 22, 23, 32]
                )
            ),
            [[12], [22, 23], [32]],
        )

        int32_list_intersect = functional.array_intersect(
            type(self).df1["int32_list"],
            [np.int32(12), np.int32(22), np.int32(23), np.int32(32)],
        )
        self.assertTrue(dt.is_list(int32_list_intersect.dtype))
        self.assertTrue(dt.is_int32(int32_list_intersect.dtype.item_dtype))
        self.assertEqual(
            list(int32_list_intersect),
            [[12], [22, 23], [32]],
        )

    def test_if_else(self):
        df = self.if_else_df
        self.assertEqual(
            list(functional.if_else(df["cond"], df["x"], df["y"])), [1, 20, 3, 40]
        )

    def test_width_bucket(self):
        df = self.width_bucket_df

        self.assertEqual(
            list(
                functional.width_bucket(
                    df["x"],
                    df["bound1"],
                    df["bound2"],
                    df["bucketCount"],
                )
            ),
            [3, 2, 0],
        )


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df1 = cls.base_df1.copy()
        cls.if_else_df = cls.base_if_else_df.copy()
        cls.width_bucket_df = cls.base_width_bucket_df.copy()


if __name__ == "__main__":
    unittest.main()
