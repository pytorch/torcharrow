# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
from typing import List

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import functional


class _TestFunctionalBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_df = ta.dataframe(
            {
                "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 10, 11, 12]],
                "input_id_scores": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 10.0, 11.0, 12.0],
                ],
                "matching_ids": [[1, 2, 3], [0, 10], [7, 10, 10], [10, 10, 11, 13]],
                "matching_id_scores": [
                    [1.0, 2.0, 3.0],
                    [0.0, 10.0],
                    [7.0, 10.0, 10.0],
                    [10.0, 10.0, 11.0, 13.0],
                ],
            },
            dtype=dt.Struct(
                fields=[
                    dt.Field("input_ids", dt.List(dt.int64)),
                    dt.Field("input_id_scores", dt.List(dt.float32)),
                    dt.Field("matching_ids", dt.List(dt.int64)),
                    dt.Field("matching_id_scores", dt.List(dt.float32)),
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

    def _is_almost_equal(self, actual: List[float], expected: List[float]):
        self.assertEqual(len(actual), len(expected))
        for actual_val, expected_val in zip(actual, expected):
            self.assertAlmostEqual(actual_val, expected_val)

    def test_has_id_overlap(self):
        df = type(self).df

        self._is_almost_equal(
            list(functional.has_id_overlap(df["input_ids"], df["matching_ids"])),
            [1.0, 0.0, 1.0, 1.0],
        )

    def test_id_overlap_count(self):
        df = type(self).df

        self._is_almost_equal(
            list(functional.id_overlap_count(df["input_ids"], df["matching_ids"])),
            [3.0, 0.0, 1.0, 3.0],
        )

    def test_get_max_count(self):
        df = type(self).df

        self._is_almost_equal(
            list(functional.get_max_count(df["input_ids"], df["matching_ids"])),
            [3.0, 0.0, 1.0, 3.0],
        )

    def test_get_jaccard_similarity(self):
        df = type(self).df

        self._is_almost_equal(
            list(
                functional.get_jaccard_similarity(df["input_ids"], df["matching_ids"])
            ),
            [3.0 / 3.0, 0 / 5.0, 1.0 / 5.0, 3.0 / 5.0],
        )

    def test_get_cosine_similarity(self):
        df = type(self).df

        self._is_almost_equal(
            list(
                functional.get_cosine_similarity(
                    df["input_ids"],
                    df["input_id_scores"],
                    df["matching_ids"],
                    df["matching_id_scores"],
                )
            ),
            [
                (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0)
                / (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0)
                / (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0),
                0.0,
                7.0
                * 7.0
                / math.sqrt(7.0 * 7.0 + 8.0 * 8.0 + 9.0 * 9.0)
                / math.sqrt(7.0 * 7.0 + 20.0 * 20.0),
                (20.0 * 20.0 + 11 * 11)
                / math.sqrt(20 * 20 + 11 * 11 + 12 * 12)
                / math.sqrt(20 * 20 + 11 * 11 + 13 * 13),
            ],
        )

    def test_get_score_sum(self):
        df = type(self).df

        self._is_almost_equal(
            list(
                functional.get_score_sum(
                    df["input_ids"],
                    df["input_id_scores"],
                    df["matching_ids"],
                    df["matching_id_scores"],
                )
            ),
            [(1.0 + 2.0 + 3.0), (0.0), (7.0 + 7.0), (10.0 + 10.0 + 11.0)],
        )

    def test_get_score_min(self):
        df = type(self).df

        self._is_almost_equal(
            list(
                functional.get_score_min(
                    df["input_ids"], df["matching_ids"], df["matching_id_scores"]
                )
            ),
            [1.0, 0.0, 7.0, 10.0],
        )

    def test_get_score_max(self):
        df = type(self).df

        self._is_almost_equal(
            list(
                functional.get_score_max(
                    df["input_ids"], df["matching_ids"], df["matching_id_scores"]
                )
            ),
            [3.0, 0.0, 7.0, 11.0],
        )


class TestFunctionalCpu(_TestFunctionalBase):
    @classmethod
    def setUpTestCaseData(cls):
        cls.df = cls.base_df.copy()


if __name__ == "__main__":
    unittest.main()
