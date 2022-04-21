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
        cls.base_df_int_float = ta.dataframe(
            {
                "a": [1, 2, 3, 5, 8, 10, 11],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.int32),
                ]
            ),
        )

        cls.base_df_int_int = ta.dataframe(
            {
                "a": [1, 2, 3, 5, 8, 10, 11],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.int32),
                ]
            ),
        )

        cls.base_df_float_int = ta.dataframe(
            {
                "a": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 11.0],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.float32),
                ]
            ),
        )

        cls.base_df_float_float = ta.dataframe(
            {
                "a": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 11.0],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.float32),
                ]
            ),
        )

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    def test_bucketize_int_float(self):
        df = type(self).df_int_float

        buckets = [2.0, 5.0, 10.0]
        self.assertEqual(
            list(functional.bucketize(df["a"], buckets)), [0, 0, 1, 1, 2, 2, 3]
        )

    def test_bucketize_int_int(self):
        df = type(self).df_int_int
        buckets = [2, 5, 10]
        self.assertEqual(
            list(functional.bucketize(df["a"], [np.int32(val) for val in buckets])),
            [0, 0, 1, 1, 2, 2, 3],
        )
        self.assertEqual(
            list(functional.bucketize(df["a"], [np.int64(val) for val in buckets])),
            [0, 0, 1, 1, 2, 2, 3],
        )

    def test_bucketize_float_int(self):
        df = type(self).df_float_int
        buckets = [2, 5, 10]
        self.assertEqual(
            list(functional.bucketize(df["a"], [np.int32(val) for val in buckets])),
            [0, 0, 1, 1, 2, 2, 3],
        )
        self.assertEqual(
            list(functional.bucketize(df["a"], [np.int64(val) for val in buckets])),
            [0, 0, 1, 1, 2, 2, 3],
        )

    def test_bucketize_float_float(self):
        df = type(self).df_float_float
        buckets = [2.0, 5.0, 10.0]
        self.assertEqual(
            list(functional.bucketize(df["a"], buckets)), [0, 0, 1, 1, 2, 2, 3]
        )

    # TODO: add more of these
    def test_bucketize_float_id_list_input(self):
        df = ta.dataframe(
            {"a": [[1, 2, 3]]},
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.List(dt.float32)),
                ]
            ),
        )
        self.assertEqual(list(functional.bucketize(df["a"], [2.0])), [[0, 0, 1]])


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df_int_float = cls.base_df_int_float.copy()
        cls.df_int_int = cls.base_df_int_int.copy()
        cls.df_float_int = cls.base_df_float_int.copy()
        cls.df_float_float = cls.base_df_float_float.copy()


if __name__ == "__main__":
    unittest.main()
