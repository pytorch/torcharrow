# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torcharrow as ta
import torcharrow._torcharrow
import torcharrow.dtypes as dt
from torcharrow import Scope, IStringColumn
from torcharrow.functional import functional
from torcharrow.velox_rt.functional import velox_functional


class TestStringColumnCpu(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_velox_functional(self):
        str_col = ta.Column(
            ["", "abc", "XYZ", "123", "xyz123", None], device=self.device
        )

        self.assertEqual(
            list(velox_functional.torcharrow_isalpha(str_col)),
            [False, True, True, False, False, None],
        )

        self.assertEqual(
            list(velox_functional.upper(str_col)),
            ["", "ABC", "XYZ", "123", "XYZ123", None],
        )

    def test_functional_dispatch(self):
        str_col = ta.Column(
            ["", "abc", "XYZ", "123", "xyz123", None], device=self.device
        )

        # Test dispatch
        self.assertEqual(
            list(functional.torcharrow_isalpha(str_col)),
            [False, True, True, False, False, None],
        )

        self.assertEqual(
            list(functional.upper(str_col)), ["", "ABC", "XYZ", "123", "XYZ123", None]
        )

    def test_factory_dispatch(self):
        rand_col = functional.rand(size=42)

        self.assertEqual(rand_col.dtype, dt.Float64(nullable=True))
        self.assertEqual(42, len(rand_col))
        for i in range(42):
            self.assertLessEqual(0.0, rand_col[i])
            self.assertLess(rand_col[i], 1.0)


if __name__ == "__main__":
    unittest.main()
