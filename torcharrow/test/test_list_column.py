# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torcharrow as ta
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
from torcharrow.ilist_column import ListColumn


class TestListColumn(unittest.TestCase):
    def base_test_empty(self):
        c = ta.column(dt.List(dt.int64), device=self.device)

        self.assertTrue(isinstance(c, ListColumn))
        self.assertEqual(c.dtype, dt.List(dt.int64))

        self.assertEqual(c.length, 0)
        self.assertEqual(c.null_count, 0)

    def base_test_nonempty(self):
        c = ta.column(dt.List(dt.int64), device=self.device)
        for i in range(4):
            c = c.append([list(range(i))])
        c = c.append([None])

        verdict = [list(range(i)) for i in range(4)]
        for i, lst in zip(range(4), verdict):
            self.assertEqual(c[i], lst)
        self.assertIsNone(c[4])

        c2 = ta.column(
            [None, None, [1, 2, 3]],
            dt.List(dt.int64, nullable=True),
            device=self.device,
        )
        self.assertIsNone(c2[0])
        self.assertIsNone(c2[1])
        self.assertEqual(c2[2], [1, 2, 3])

    def base_test_list_with_none(self):
        with self.assertRaises(ValueError) as ex:
            ta.column(
                [None, None, [1, 2, 3]],
                dt.List(dt.int64),
                device=self.device,
            )
        self.assertTrue(
            "None found in the list for non-nullable type: List(int64)"
            in str(ex.exception),
            f"Exception message is not as expected: {str(ex.exception)}",
        )

    def base_test_append_concat(self):
        base_list = [["hello", "world"], ["how", "are", "you"]]
        sf1 = ta.column(base_list, dtype=dt.List(dt.string), device=self.device)
        self.assertEqual(list(sf1), base_list)

        append1 = [["I", "am", "fine", "and", "you"]]
        sf2 = sf1.append(append1)
        self.assertEqual(list(sf2), base_list + append1)

        append2 = [["I", "am", "fine", "too"]]
        sf3 = ta.concat([sf2, ta.column(append2, device=self.device)])
        self.assertEqual(list(sf3), base_list + append1 + append2)

        # concat everything
        sf_all = ta.concat([sf1, sf2, sf3])
        self.assertEqual(list(sf_all), list(sf1) + list(sf2) + list(sf3))

    def base_test_nested_numerical_twice(self):
        c = ta.column(
            dt.List(dt.List(dt.Int64(nullable=False), nullable=True), nullable=False),
            device=self.device,
        )
        vals = [[[1, 2], None, [3, 4]], [[4], [5]]]
        c = c.append(vals)
        self.assertEqual(list(c), vals)

        d = ta.column(
            dt.List(
                dt.List(dt.Int64(nullable=False), nullable=True),
                nullable=False,
            ),
            device=self.device,
        )
        for val in vals:
            d = d.append([val])
        self.assertEqual(list(d), vals)

    def base_test_nested_string_once(self):
        c = ta.column(dt.List(dt.string), device=self.device)
        c = c.append([[]])
        c = c.append([["a"]])
        c = c.append([["b", "c"]])
        self.assertEqual(list([[], ["a"], ["b", "c"]]), list(c))

    def base_test_nested_string_twice(self):
        c = ta.column(dt.List(dt.List(dt.string)), device=self.device)
        c = c.append([[]])
        c = c.append([[[]]])
        c = c.append([[["a"]]])
        c = c.append([[["b", "c"], ["d", "e", "f"]]])
        self.assertEqual(list(c), [[], [[]], [["a"]], [["b", "c"], ["d", "e", "f"]]])

    def base_test_get_count_join(self):
        c = ta.column(dt.List(dt.string), device=self.device)
        c = c.append([["The", "fox"], ["jumps"], ["over", "the", "river"]])

        self.assertEqual(list(c.list.get(0)), ["The", "jumps", "over"])
        self.assertEqual(list(c.list.join(" ")), ["The fox", "jumps", "over the river"])

    def base_test_slice(self):
        c = ta.column(
            [list(range(5)), list(range(5, 10)), list(range(3))], device=self.device
        )
        self.assertEqual(
            list(c.list.slice(0, 4)),
            [list(range(4)), list(range(5, 9)), list(range(3))],
        )

    def base_test_map_reduce_etc(self):
        c = ta.column(dt.List(dt.string), device=self.device)
        c = c.append([["The", "fox"], ["jumps"], ["over", "the", "river"]])
        self.assertEqual(
            list(c.list.map(str.upper)),
            [["THE", "FOX"], ["JUMPS"], ["OVER", "THE", "RIVER"]],
        )
        self.assertEqual(
            list(c.list.filter(lambda x: x.endswith("fox"))), [["fox"], [], []]
        )

        c = ta.column(dt.List(dt.int64), device=self.device)
        c = c.append([list(range(1, i)) for i in range(1, 6)])
        self.assertEqual(list(c.list.reduce(operator.mul, 1)), [1, 1, 2, 6, 24])

        c = ta.column(
            [["what", "a", "wonderful", "world!"], ["really?"]], device=self.device
        )
        self.assertEqual(
            list(c.list.map(len, dtype=dt.List(dt.int64))), [[4, 1, 9, 6], [7]]
        )

        # flat map on original columns (not on list)
        fst = ["what", "a", "wonderful", "world!"]
        snd = ["really?"]
        c = ta.column([fst, snd], device=self.device)
        self.assertEqual(list(c.flatmap(lambda xs: [xs, xs])), [fst, fst, snd, snd])

        ta.column([1, 2, 3, 4], device=self.device).map(str, dtype=dt.string)

    def base_test_fixed_size_list(self):
        # Creation
        d = ta.column(
            [[1, 2], [3, 4]], dtype=dt.List(item_dtype=dt.int64, fixed_size=2)
        )
        self.assertEqual(d.dtype.fixed_size, 2)
        self.assertEqual(type(d._data.type()), velox.VeloxFixedArrayType)
        self.assertEqual(d._data.type().fixed_width(), 2)

        # Unequal length cells are disallowed
        with self.assertRaises(Exception) as ex:
            ta.column([[1, 2], [4]], dtype=dt.List(item_dtype=dt.int64, fixed_size=2))
        self.assertTrue(
            "Exception: VeloxRuntimeError" in str(ex.exception),
            f"Wanted VeloxRuntimeError: {str(ex.exception)}",
        )
        self.assertTrue(
            "Reason: Invalid length element at index 1, got length 1, want length 2"
            in str(ex.exception),
            f"Wanted detailed reason {str(ex.exception)}",
        )

        # Appending another fixed list of same size
        e = d.append(d)
        self.assertEqual(list(e), [[1, 2], [3, 4], [1, 2], [3, 4]])
        self.assertEqual(e.dtype.fixed_size, 2)
        self.assertEqual(type(e._data.type()), velox.VeloxFixedArrayType)
        self.assertEqual(e._data.type().fixed_width(), 2)

        # Appending a non-fixed list of same size
        f = d.append(ta.column([[4, 5], [5, 6]]))
        self.assertEqual(list(f), [[1, 2], [3, 4], [4, 5], [5, 6]])
        self.assertEqual(f.dtype.fixed_size, 2)
        self.assertEqual(type(f._data.type()), velox.VeloxFixedArrayType)
        self.assertEqual(f._data.type().fixed_width(), 2)

        # Appending a fixed list of different size
        with self.assertRaises(ValueError) as ex:
            d.append(
                ta.column([[1, 2, 3]], dtype=dt.List(item_dtype=dt.int64, fixed_size=3))
            )
        self.assertTrue(
            "value incompatible with list fixed_size" in str(ex.exception),
            f"Unexpected failure reason: {str(ex.exception)}",
        )

        # Appending a non-fixed list of different size
        with self.assertRaises(ValueError) as ex:
            d.append(ta.column([[1]]))
        self.assertTrue(
            "value incompatible with list fixed_size" in str(ex.exception),
            f"Unexpected failure reason: {str(ex.exception)}",
        )

    def base_test_cast(self):
        list_dtype = dt.List(item_dtype=dt.int64, fixed_size=2)
        c_list = ta.column(
            [[1, 2], [3, 4]],
            dtype=list_dtype,
            device=self.device,
        )

        int_dtype = dt.int64
        # TODO: Nested cast should be supported in the future
        for arg in (int_dtype, list_dtype):
            with self.assertRaisesRegexp(
                expected_exception=TypeError,
                expected_regex=r"List\(int64, fixed_size=2\) for.*is not supported",
            ):
                c_list.cast(arg)


if __name__ == "__main__":
    unittest.main()
