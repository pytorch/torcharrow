# Copyright (c) Facebook, Inc. and its affiliates.
import operator
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import IListColumn, INumericalColumn, Scope


class TestListColumn(unittest.TestCase):
    def base_test_empty(self):
        c = ta.Column(dt.List(dt.int64), device=self.device)

        self.assertTrue(isinstance(c, IListColumn))
        self.assertEqual(c.dtype, dt.List(dt.int64))

        self.assertEqual(c.length, 0)
        self.assertEqual(c.null_count, 0)

    def base_test_nonempty(self):
        c = ta.Column(dt.List(dt.int64), device=self.device)
        for i in range(4):
            c = c.append([list(range(i))])
        c = c.append([None])

        verdict = [list(range(i)) for i in range(4)]
        for i, lst in zip(range(4), verdict):
            self.assertEqual(c[i], lst)
        self.assertIsNone(c[4])

        c2 = ta.Column([None, None, [1, 2, 3]], dt.List(dt.int64), device=self.device)
        self.assertIsNone(c2[0])
        self.assertIsNone(c2[1])
        self.assertEqual(c2[2], [1, 2, 3])

    def base_test_append_concat(self):
        base_list = [["hello", "world"], ["how", "are", "you"]]
        sf1 = ta.Column(base_list, dtype=dt.List(dt.string), device=self.device)
        self.assertEqual(list(sf1), base_list)

        append1 = [["I", "am", "fine", "and", "you"]]
        sf2 = sf1.append(append1)
        self.assertEqual(list(sf2), base_list + append1)

        append2 = [["I", "am", "fine", "too"]]
        sf3 = ta.concat([sf2, ta.Column(append2, device=self.device)])
        self.assertEqual(list(sf3), base_list + append1 + append2)

        # concat everything
        sf_all = ta.concat([sf1, sf2, sf3])
        self.assertEqual(list(sf_all), list(sf1) + list(sf2) + list(sf3))

    def base_test_nested_numerical_twice(self):
        c = ta.Column(
            dt.List(dt.List(dt.Int64(nullable=False), nullable=True), nullable=False),
            device=self.device,
        )
        vals = [[[1, 2], None, [3, 4]], [[4], [5]]]
        c = c.append(vals)
        self.assertEqual(list(c), vals)

        d = ta.Column(
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
        c = ta.Column(dt.List(dt.string), device=self.device)
        c = c.append([[]])
        c = c.append([["a"]])
        c = c.append([["b", "c"]])
        self.assertEqual(list([[], ["a"], ["b", "c"]]), list(c))

    def base_test_nested_string_twice(self):
        c = ta.Column(dt.List(dt.List(dt.string)), device=self.device)
        c = c.append([[]])
        c = c.append([[[]]])
        c = c.append([[["a"]]])
        c = c.append([[["b", "c"], ["d", "e", "f"]]])
        self.assertEqual(list(c), [[], [[]], [["a"]], [["b", "c"], ["d", "e", "f"]]])

    def base_test_get_count_join(self):
        c = ta.Column(dt.List(dt.string), device=self.device)
        c = c.append([["The", "fox"], ["jumps"], ["over", "the", "river"]])

        self.assertEqual(list(c.list.get(0)), ["The", "jumps", "over"])
        self.assertEqual(list(c.list.count("The")), [1, 0, 0])
        self.assertEqual(list(c.list.join(" ")), ["The fox", "jumps", "over the river"])

    def base_test_slice(self):
        c = ta.Column(
            [list(range(5)), list(range(5, 10)), list(range(3))], device=self.device
        )
        self.assertEqual(
            list(c.list.slice(0, 4)),
            [list(range(4)), list(range(5, 9)), list(range(3))],
        )

    def base_test_map_reduce_etc(self):
        c = ta.Column(dt.List(dt.string), device=self.device)
        c = c.append([["The", "fox"], ["jumps"], ["over", "the", "river"]])
        self.assertEqual(
            list(c.list.map(str.upper)),
            [["THE", "FOX"], ["JUMPS"], ["OVER", "THE", "RIVER"]],
        )
        self.assertEqual(
            list(c.list.filter(lambda x: x.endswith("fox"))), [["fox"], [], []]
        )

        c = ta.Column(dt.List(dt.int64), device=self.device)
        c = c.append([list(range(1, i)) for i in range(1, 6)])
        self.assertEqual(list(c.list.reduce(operator.mul, 1)), [1, 1, 2, 6, 24])

        c = ta.Column(
            [["what", "a", "wonderful", "world!"], ["really?"]], device=self.device
        )
        self.assertEqual(
            list(c.list.map(len, dtype=dt.List(dt.int64))), [[4, 1, 9, 6], [7]]
        )

        # flat map on original columns (not on list)
        fst = ["what", "a", "wonderful", "world!"]
        snd = ["really?"]
        c = ta.Column([fst, snd], device=self.device)
        self.assertEqual(list(c.flatmap(lambda xs: [xs, xs])), [fst, fst, snd, snd])

        ta.Column([1, 2, 3, 4], device=self.device).map(str, dtype=dt.string)


if __name__ == "__main__":
    unittest.main()
