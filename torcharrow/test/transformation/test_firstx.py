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

    def test_firstx(self):
        df = type(self).df_list

        self.assertEqual(
            list(functional.firstx(df["a"], 1)),
            [[0], [4], [7]],
        )

        self.assertEqual(
            list(functional.firstx(df["a"], 2)),
            [[0, 1], [4, 5], [7, 8]],
        )

        self.assertEqual(
            list(functional.firstx(df["a"], 4)),
            [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df_list = cls.base_df_list.copy()


if __name__ == "__main__":
    unittest.main()
