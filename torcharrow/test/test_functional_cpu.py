# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torcharrow as ta
import torcharrow._torcharrow
import torcharrow.dtypes as dt
from numpy.testing import assert_array_almost_equal
from torcharrow import functional
from torcharrow.velox_rt.functional import velox_functional


class TestFunctionalCpu(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_velox_functional(self):
        str_col = ta.column(
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
        str_col = ta.column(
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

        # Validate that invoking unknown UDFs errors nicely.
        with self.assertRaises(RuntimeError) as ex:
            assert functional.idontexist(str_col)
        self.assertTrue(
            str(ex.exception).startswith("Request for unknown Velox UDF: idontexist")
        )

        # Validate that invoking unknown UDFs with unsupported signatures errors nicely too.
        with self.assertRaises(RuntimeError) as ex:
            assert functional.firstx(str_col, 1)
        msg = str(ex.exception)
        print(msg)
        self.assertTrue(
            msg.startswith("Velox UDF signature is not supported: (varchar,bigint)")
        )

        supported_sig = msg[msg.find("Supported signatures:") : :]
        self.assertTrue(
            "(array(bigint),bigint) -> array(bigint), (array(integer),bigint) -> array(integer), (array(bigint),integer) -> array(bigint), (array(integer),integer) -> array(integer)"
            in supported_sig
        )

    def test_factory_dispatch(self):
        rand_col = functional.rand(size=42)

        self.assertEqual(rand_col.dtype, dt.Float64(nullable=True))
        self.assertEqual(42, len(rand_col))
        for i in range(42):
            self.assertLessEqual(0.0, rand_col[i])
            self.assertLess(rand_col[i], 1.0)

    def test_scale_to_0_1(self):
        c = ta.column([1, 2, 3, None, 4, 5], device=self.device)
        self.assertEqual(
            list(functional.scale_to_0_1(c)), [0.0, 0.25, 0.5, None, 0.75, 1.0]
        )

        c = ta.column([2, 2, 2], device=self.device)
        assert_array_almost_equal(
            [0.11920291930437088, 0.11920291930437088, 0.11920291930437088],
            list(functional.scale_to_0_1(c)),
            decimal=15,
        )

        c = ta.column(["foo", "bar"])
        with self.assertRaises(AssertionError):
            functional.scale_to_0_1(c)


if __name__ == "__main__":
    unittest.main()
