# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow.imap_column import MapColumn


class TestMapColumn(unittest.TestCase):
    def base_test_map(self):
        c = ta.column(dt.Map(dt.string, dt.int64), device=self.device)
        self.assertIsInstance(c, MapColumn)

        c = c.append([{"abc": 123}])
        self.assertDictEqual(c[0], {"abc": 123})

        c = c.append([{"de": 45, "fg": 67}])
        self.assertDictEqual(c[1], {"de": 45, "fg": 67})

        c = c.append([None])
        self.assertIsNone(c[2])

        c2 = ta.column(
            [None, None, {"foo": 123}],
            dt.Map(dt.string, dt.int64, nullable=True),
            device=self.device,
        )
        self.assertIsNone(c2[0])
        self.assertIsNone(c2[1])
        self.assertDictEqual(c2[2], {"foo": 123})

    def base_test_infer(self):
        c = ta.column(
            [
                {"helsinki": [-1.0, 21.0], "moscow": [-4.0, 24.0]},
                {},
                {"nowhere": [], "algiers": [11.0, 25, 2], "kinshasa": [22.0, 26.0]},
            ],
            device=self.device,
        )
        self.assertIsInstance(c, MapColumn)
        self.assertEqual(len(c), 3)
        self.assertEqual(c.dtype, dt.Map(dt.string, dt.List(dt.float32)))

        self.assertEqual(
            list(c),
            [
                {"helsinki": [-1.0, 21.0], "moscow": [-4.0, 24.0]},
                {},
                {"nowhere": [], "algiers": [11.0, 25, 2], "kinshasa": [22.0, 26.0]},
            ],
        )

    def base_test_keys_values_get(self):
        c = ta.column([{"abc": 123}, {"de": 45, "fg": 67}, None], device=self.device)

        self.assertEqual(list(c.maps.keys()), [["abc"], ["de", "fg"], None])
        self.assertEqual(list(c.maps.values()), [[123], [45, 67], None])
        self.assertEqual(list(c.maps.get("de", 0)), [0, 45, None])

    def base_test_get_operator(self):
        col_rep = [
            {"helsinki": [-1.0, 21.0], "moscow": [-4.0, 24.0]},
            {},
            {"nowhere": [], "algiers": [11.0, 25, 2], "kinshasa": [22.0, 26.0]},
        ]
        c = ta.column(
            col_rep,
            device=self.device,
        )
        indicies = [0, 2]
        expected = [col_rep[i] for i in indicies]
        result = [c[i] for i in indicies]
        self.assertEqual(expected, result)

    def base_test_slice_operation(self):
        col_rep = [
            {"helsinki": [-1.0, 21.0], "moscow": [-4.0, 24.0]},
            {},
            {"nowhere": [], "algiers": [11.0, 25, 2], "kinshasa": [22.0, 26.0]},
            {"london": [], "new york": [500]},
        ]
        c = ta.column(
            col_rep,
            device=self.device,
        )
        expected_slice_every_other = col_rep[0:4:2]
        result_every_other = c[0:4:2]
        self.assertEqual(expected_slice_every_other, list(result_every_other))

        expected_slice_most = col_rep[1:]
        result_most = c[1:4:1]
        self.assertEqual(expected_slice_most, list(result_most))

    def base_test_equality_operators(self):
        col_rep = [
            {"helsinki": [-1.0, 21.0], "moscow": [-4.0, 24.0]},
            {"boston": [-4.0]},
            {"nowhere": [], "algiers": [11.0, 25, 2], "kinshasa": [22.0, 26.0]},
            {"london": [], "new york": [500]},
        ]
        c = ta.column(
            col_rep,
            device=self.device,
        )
        c2 = ta.column(
            col_rep,
            device=self.device,
        )
        self.assertTrue(all(c == c2))
        self.assertFalse(any(c != c2))


if __name__ == "__main__":
    unittest.main()
