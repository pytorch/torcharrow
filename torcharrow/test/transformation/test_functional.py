# Copyright (c) Meta Platforms, Inc. and affiliates.
import unittest

import torcharrow as ta
from torcharrow import functional


class _TestFunctionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare input data as CPU dataframe
        cls.base_df1 = ta.DataFrame(
            {"a": [[11, 12, 13], [21, 22, 23, 24, 25, 26], [31, 32]]}
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
            list(functional.slice(type(self).df1["a"], 2, 3)),
            [[12, 13], [22, 23, 24], [32]],
        )


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df1 = cls.base_df1.copy()


if __name__ == "__main__":
    unittest.main()
