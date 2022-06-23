# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import functional


class _TestClampBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_df_list = ta.dataframe(
            {
                "int64": [[0, 1, 2, 3], [-100, 100, 10], [0, -1, -2, -3]],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("int64", dt.List(dt.int64)),
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

    def test_clamp_list(self):
        df = type(self).df_list

        self.assertEqual(
            list(functional.clamp_list(df["int64"], 0, 20)),
            [[0, 1, 2, 3], [0, 20, 10], [0, 0, 0, 0]],
        )


class TestClampCpu(_TestClampBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df_list = cls.base_df_list.copy()


if __name__ == "__main__":
    unittest.main()
