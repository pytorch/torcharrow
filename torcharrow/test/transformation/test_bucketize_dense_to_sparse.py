# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torcharrow as ta
from torcharrow import functional


class _TestBucketizeToSparseBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_df = ta.dataframe(
            {
                "float": [3.0, 4.0, 5.0, 6.0, 5.0, 8.0],
                "float_null": [1.0, 2.0, 3.0, None, 5.0, None],
            }
        )

        cls.setUpTestCaseData()

    @classmethod
    def setUpTestCaseData(cls):
        # Override in subclass
        # Python doesn't have native "abstract base test" support.
        # So use unittest.SkipTest to skip in base class: https://stackoverflow.com/a/59561905.
        raise unittest.SkipTest("abstract base test")

    def test_basic_bucketize_dense_to_sparse(self):
        df = type(self).df

        buckets = [1, 2, 3, 4, 5, 6, 7]

        self.assertEqual(
            list(
                functional.bucketize_dense_to_sparse(
                    df["float"], [np.float32(val) for val in buckets]
                )
            ),
            [[2], [3], [4], [5], [4], [7]],
        )

    def test_missing_bucketize_dense_to_sparse(self):
        df = type(self).df

        buckets = [1, 2, 3, 4, 5, 6, 7]

        self.assertEqual(
            list(
                functional.bucketize_dense_to_sparse(
                    df["float_null"], [np.float32(val) for val in buckets]
                )
            ),
            [[0], [1], [2], [], [4], []],
        )


class TestBucketizeToSparseCpu(_TestBucketizeToSparseBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df = cls.base_df.copy()


if __name__ == "__main__":
    unittest.main()
