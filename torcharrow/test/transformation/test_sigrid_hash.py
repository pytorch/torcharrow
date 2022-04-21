# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import functional


class _TestFunctionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_df_val = ta.dataframe(
            {
                "a": range(16),
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.int64),
                ]
            ),
        )

        cls.base_df_list = ta.dataframe(
            {
                "a": [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("a", dt.List(dt.int64)),
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

    def test_sigrid_hash(self):
        df = type(self).df_val
        salt = 0
        max_value = 100

        self.assertEqual(
            list(functional.sigrid_hash(df["a"], salt, max_value)),
            [6, 60, 54, 54, 9, 4, 91, 11, 67, 79, 2, 25, 92, 98, 83, 66],
        )

    def test_sigrid_hash_list(self):
        df = type(self).df_list
        salt = 0
        max_value = 100

        self.assertEqual(
            list(functional.sigrid_hash(df["a"], salt, max_value)),
            [[6, 60, 54, 54], [9, 4, 91], [11, 67, 79]],
        )


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df_val = cls.base_df_val.copy()
        cls.df_list = cls.base_df_list.copy()


if __name__ == "__main__":
    unittest.main()
