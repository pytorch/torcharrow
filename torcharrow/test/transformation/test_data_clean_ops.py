# Copyright (c) Meta Platforms, Inc. and affiliates.
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt


# TODO: add/migrate more data clean ops, such as fill_null, drop_null, drop_duplicates
class _TestDataCleanOpsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_df1 = ta.DataFrame(
            {
                "int32": ta.Column([None, 2, 3, 4], dtype=dt.Int32(nullable=True)),
                "int64": ta.Column([1, None, 3, 4], dtype=dt.Int64(nullable=True)),
                "float32": ta.Column(
                    [1.0, 2.0, None, 4.0], dtype=dt.Float32(nullable=True)
                ),
                "float64": ta.Column(
                    [1.0, 2.0, 3.0, None], dtype=dt.Float64(nullable=True)
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

    def test_fill_null(self):
        fill_null_int32 = type(self).df1["int32"].fill_null(42)
        fill_null_int64 = type(self).df1["int64"].fill_null(42)
        fill_null_float32 = type(self).df1["float32"].fill_null(42.0)
        fill_null_float64 = type(self).df1["float64"].fill_null(42.0)
        fill_null_df = type(self).df1.fill_null(42)

        for col in [fill_null_int32, fill_null_df["int32"]]:
            self.assertEqual(list(col), [42, 2, 3, 4])

        for col in [fill_null_int64, fill_null_df["int64"]]:
            self.assertEqual(list(col), [1, 42, 3, 4])

        for col in [fill_null_float32, fill_null_df["float32"]]:
            self.assertEqual(list(col), [1, 2, 42, 4])

        for col in [fill_null_float64, fill_null_df["float64"]]:
            self.assertEqual(list(col), [1, 2, 3, 42])


class TestDataCleanOpsCpu(_TestDataCleanOpsBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df1 = cls.base_df1.copy()


if __name__ == "__main__":
    unittest.main()
