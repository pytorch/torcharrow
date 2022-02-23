# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import numpy.testing
import torcharrow as ta
import torcharrow.dtypes as dt


# TODO: add/migrate more numeric tests
class _TestNumericOpsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare input data as CPU dataframe
        cls.base_add_df = ta.dataframe(
            {"c": [0, 1, 3], "d": [5, 5, 6], "e": [1.0, 1, 7]}
        )
        cls.base_log_df = ta.dataframe(
            {
                "int32": ta.column([1, 0, 4, None], dtype=dt.Int32(nullable=True)),
                "int64": ta.column([1, 0, 4, None], dtype=dt.Int64(nullable=True)),
                "float32": ta.column(
                    [1.0, 0.0, 4.0, None], dtype=dt.Float32(nullable=True)
                ),
                "float64": ta.column(
                    [1.0, 0.0, 4.0, None], dtype=dt.Float64(nullable=True)
                ),
            }
        )

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    def test_add(self):
        c = type(self).add_df["c"]
        d = type(self).add_df["d"]

        self.assertEqual(list(c + 1), [1, 2, 4])
        self.assertEqual(list(1 + c), [1, 2, 4])
        self.assertEqual(list(c + d), [5, 6, 9])

    def test_log(self):
        log_int32_col = type(self).log_df["int32"].log()
        log_int64_col = type(self).log_df["int64"].log()
        log_float32_col = type(self).log_df["float32"].log()
        log_float64_col = type(self).log_df["float64"].log()
        log_whole_df = type(self).log_df.log()

        for col in [
            log_int32_col,
            log_int64_col,
            log_float32_col,
            log_whole_df["int32"],
            log_whole_df["int64"],
            log_whole_df["float32"],
        ]:
            numpy.testing.assert_almost_equal(
                list(col)[:-1], [0.0, -float("inf"), math.log(4)]
            )
            self.assertEqual(col.dtype, dt.Float32(nullable=True))
            self.assertEqual(list(col)[-1], None)

        for col in [log_float64_col, log_whole_df["float64"]]:
            numpy.testing.assert_almost_equal(
                list(col)[:-1], [0.0, -float("inf"), math.log(4)]
            )
            self.assertEqual(col.dtype, dt.Float64(nullable=True))
            self.assertEqual(list(col)[-1], None)


class TestNumericOpsCpu(_TestNumericOpsBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.add_df = cls.base_add_df.copy()
        cls.log_df = cls.base_log_df.copy()


if __name__ == "__main__":
    unittest.main()
