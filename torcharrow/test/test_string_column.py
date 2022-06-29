# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as ty
import unittest
from unittest import mock

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow.istring_column import StringColumn


class TestStringColumn(unittest.TestCase):
    # This create_column function is created so that child classes
    # can override to generate column on other runtimes
    def create_column(
        self,
        data: ty.Union[ty.Iterable, dt.DType, None] = None,
        dtype: ty.Optional[dt.DType] = None,
    ):
        raise NotImplementedError

    def base_test_empty(self):
        empty = ta.column(dt.string, device=self.device)
        self.assertTrue(isinstance(empty, StringColumn))
        self.assertEqual(empty.dtype, dt.string)
        self.assertEqual(empty.length, 0)
        self.assertEqual(empty.null_count, 0)
        # self.assertEqual(empty._offsets[0], 0)

    def base_test_indexing(self):
        col = ta.column(
            [None] * 3 + ["3", "4", "5"],
            dtype=dt.String(nullable=True),
            device=self.device,
        )

        # index
        self.assertEqual(col[0], None)
        self.assertEqual(col[-1], "5")

        # slice

        # continuous slice
        # document this is broken, will fix in the next diff.
        c = col[3 : len(col)]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(c), 3)

        # non continuous slice
        # document this is broken, will fix in the next diff.
        d = col[::2]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(d), 3)

        # slice has Python not Pandas semantics
        # document this is broken, will fix in the next diff.
        e = col[: len(col) - 1]
        self.assertEqual(len(e), len(col) - 1)

        # indexing via lists
        # document this is broken, will fix in the next diff.
        f = col[[0, 1, 2]]
        self.assertEqual(list(f), list(col[:3]))

        # head/tail are special slices
        # document this is broken, will fix in the next diff.
        self.assertEqual(list(col.head(2)), [None, None])
        self.assertEqual(list(col.tail(2)), ["4", "5"])

    def base_test_append_offsets(self):
        c = ta.column(dt.string, device=self.device)
        c = c.append(["abc", "de", "", "f"])
        # self.assertEqual(list(c._offsets), [0, 3, 5, 5, 6])
        self.assertEqual(list(c), ["abc", "de", "", "f"])
        # TODO : check that error is thrown!
        # with self.assertRaises(TypeError):
        #     # TypeError: a dt.string is required (got type NoneType)
        #     c.append(None)

        c = ta.column(["abc", "de", "", "f", None])
        # self.assertEqual(list(c._offsets), [0, 3, 5, 5, 6, 6])
        self.assertEqual(list(c), ["abc", "de", "", "f", None])

    def base_test_string_column_from_tuple(self):
        data_str = ("one", "two", "three", "four", "five", "six")
        col_str = ta.column(data_str, device=self.device)
        self.assertEqual(tuple(col_str), data_str)

    def base_test_string_split_methods(self):
        s = ["a b c", "1,2,3", "d e f g h", "hello.this.is.very.very.very.very.long"]
        c = ta.column(s, device=self.device)
        self.assertEqual(list(c.str.split(".")), [v.split(".") for v in s])
        self.assertEqual(list(c.str.split()), [v.split() for v in s])
        self.assertEqual(list(c.str.split(",")), [v.split(",") for v in s])
        # with max splits
        self.assertEqual(list(c.str.split(".", -1)), [v.split(".") for v in s])
        self.assertEqual(list(c.str.split(".", 0)), [v.split(".") for v in s])
        self.assertEqual(list(c.str.split(".", 2)), [v.split(".", 2) for v in s])
        self.assertEqual(list(c.str.split(".", 10)), [v.split(".", 10) for v in s])

    def base_test_string_categorization_methods(self):
        # isalpha/isnumeric/isalnum/isdigit/isdecimal/isspace/islower/isupper/istitle
        self.assertEqual(
            list(
                self.create_column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", None],
                    dt.String(True),
                ).str.isalpha()
            ),
            [False, True, True, False, False, True, False, None],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["+3.e12", "abc", "0"],
                ).str.isnumeric()
            ),
            [False, False, True],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", None],
                    dt.String(True),
                ).str.isalnum()
            ),
            [False, True, True, True, True, True, False, None],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", "\u00B2", None],
                    dt.String(True),
                ).str.isdigit()
            ),
            [False, False, False, True, False, False, False, True, None],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", "\u00B2", None],
                    dt.String(True),
                ).str.isdecimal()
            ),
            [False, False, False, True, False, False, False, False, None],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["\n", "\t", " ", "", "a"],
                ).str.isspace()
            ),
            [True, True, True, False, False],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["UPPER", "lower", ".abc", ".ABC", "123"],
                ).str.islower()
            ),
            [False, True, True, False, False],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["UPPER", "lower", ".abc", ".ABC", "123"],
                ).str.isupper()
            ),
            [True, False, False, True, False],
        )
        self.assertEqual(
            list(
                self.create_column(
                    ["A B C", "ABc", "Abc", "abc", " ", ""],
                ).str.istitle()
            ),
            [True, False, True, False, False, False],
        )

    def base_test_concat(self):
        s1 = ["abc", "de", "", "f", None]
        s2 = ["12", "567", "77", None, "55"]
        c1 = ta.column(s1, device=self.device)
        c2 = ta.column(s2, device=self.device)
        concat1 = [x + y for (x, y) in zip(s1[:-2], s2[:-2])] + [None, None]
        self.assertEqual(list(c1 + c2), concat1)

        concat2 = [x + "_suffix" for x in s1[:-1]] + [None]
        self.assertEqual(list(c1 + "_suffix"), concat2)

        concat3 = ["prefix_" + x for x in s1[:-1]] + [None]
        self.assertEqual(list("prefix_" + c1), concat3)

    def base_test_comparison(self):
        c = ta.column(["abc", "de", "", "f", None], device=self.device)
        d = ta.column(["abc", "77", "", None, "55"], device=self.device)
        self.assertEqual(list(c == c), [True, True, True, True, None])
        self.assertEqual(list(c == d), [True, False, True, None, None])
        self.assertEqual(list(c == "de"), [False, True, False, False, None])

        self.assertEqual(list(c != c), [False, False, False, False, None])
        self.assertEqual(list(c != d), [False, True, False, None, None])
        self.assertEqual(list(c != "de"), [True, False, True, True, None])

        self.assertEqual(list(c < c), [False, False, False, False, None])
        self.assertEqual(list(c < d), [False, False, False, None, None])
        self.assertEqual(list(c < "de"), [True, False, True, False, None])

        self.assertEqual(list(c <= c), [True, True, True, True, None])
        self.assertEqual(list(c <= d), [True, False, True, None, None])
        self.assertEqual(list(c <= "de"), [True, True, True, False, None])

        self.assertEqual(list(c > c), [False, False, False, False, None])
        self.assertEqual(list(c > d), [False, True, False, None, None])
        self.assertEqual(list(c > "de"), [False, False, False, True, None])

        self.assertEqual(list(c >= c), [True, True, True, True, None])
        self.assertEqual(list(c >= d), [True, True, True, None, None])
        self.assertEqual(list(c >= "de"), [False, True, False, True, None])

        # validate comparing non-equal length columns fails
        with self.assertRaises(TypeError):
            assert c == c.append([None])

    def base_test_string_lifted_methods(self):
        s = ["abc", "de", "", "f"]
        c = ta.column(s, device=self.device)
        self.assertEqual(list(c.str.length()), [len(i) for i in s])
        self.assertEqual(list(c.str.slice(stop=2)), [i[:2] for i in s])
        self.assertEqual(list(c.str.slice(1, 2)), [i[1:2] for i in s])
        self.assertEqual(list(c.str.slice(1)), [i[1:] for i in s])

        self.assertEqual(
            list(ta.column(["UPPER", "lower"], device=self.device).str.lower()),
            ["upper", "lower"],
        )
        self.assertEqual(
            list(ta.column(["UPPER", "lower"], device=self.device).str.upper()),
            ["UPPER", "LOWER"],
        )

        # strip
        self.assertEqual(
            list(ta.column(["  ab", " cde\n  "], device=self.device).str.strip()),
            ["ab", "cde"],
        )

    def base_test_string_pattern_matching_methods(self):
        s = ["hello.this", "is.interesting.", "this.is_24", "paradise", "h", ""]

        self.assertEqual(
            list(ta.column(s, device=self.device).str.startswith("h")),
            [True, False, False, False, True, False],
        )
        self.assertEqual(
            list(ta.column(s, device=self.device).str.endswith("this")),
            [True, False, False, False, False, False],
        )
        self.assertEqual(
            list(ta.column(s, device=self.device).str.find("this")),
            [6, -1, 0, -1, -1, -1],
        )

        self.assertEqual(
            list(
                ta.column(s, device=self.device).str.replace(
                    "this", "that", regex=False
                )
            ),
            [v.replace("this", "that") for v in s],
        )

    def base_test_regular_expressions(self):
        S = ta.column(
            [
                "Finland",
                "Colombia",
                "Florida",
                "Japan",
                "Puerto Rico",
                "Russia",
                "france",
            ],
            device=self.device,
        )
        # count
        self.assertEqual(list(S[S.str.count(r"(^F.*)") == 1]), ["Finland", "Florida"])
        self.assertEqual(S.str.count(r"(^F.*)").sum(), 2)
        # match
        self.assertEqual(list(S[S.str.match(r"(^P.*)") == True]), ["Puerto Rico"])

        # replace
        # TODO: support replace with regex
        # self.assertEqual(
        #    list(S.str.replace("(-d)", "")),
        #    [
        #        "Finland",
        #        "Colombia",
        #        "Florida",
        #        "Japan",
        #        "Puerto Rico",
        #        "Russia",
        #        "france",
        #    ],
        # )

        # contains
        self.assertEqual(
            list(S.str.contains("^F.*")),
            [True, False, True, False, False, False, False],
        )

        # findall (creates a list), we select the non empty ones.

        l = S.str.findall("^[Ff].*")
        self.assertEqual(
            list(l[l.list.length() > 0]),
            [
                ["Finland"],
                ["Florida"],
                ["france"],
            ],
        )

        # extract all
        l = S.str.findall("^[Ff](.*)")
        self.assertEqual(
            list(l[l.list.length() > 0]),
            [
                ["Finland"],
                ["Florida"],
                ["france"],
            ],
        )

        l = S.str.findall("[tCc]o")
        self.assertEqual(
            list(l[l.list.length() > 0]),
            [
                ["Co"],
                ["to", "co"],
            ],
        )

    def base_test_is_unique(self):
        unique_column = ta.column(
            [f"test{x}" for x in range(3)],
            device=self.device,
        )

        self.assertTrue(unique_column.is_unique)

        non_unique_column = ta.column(
            [
                "test",
                "test",
            ],
            device=self.device,
        )

        self.assertFalse(non_unique_column.is_unique)

    def base_test_is_monotonic_increasing(self):
        c = ta.column([f"test{x}" for x in range(5)], device=self.device)
        self.assertTrue(c.is_monotonic_increasing)
        self.assertFalse(c.is_monotonic_decreasing)

    def base_test_is_monotonic_decreasing(self):
        c = ta.column([f"test{x}" for x in range(5, 0, -1)], device=self.device)
        self.assertFalse(c.is_monotonic_increasing)
        self.assertTrue(c.is_monotonic_decreasing)

    def base_test_if_else(self):
        left_repr = ["a1", "a2", "a3", "a4"]
        right_repr = ["b1", "b2", "b3", "b4"]
        cond_repr = [True, False, True, False]
        cond = ta.column(cond_repr, device=self.device)
        left = ta.column(left_repr, device=self.device)
        right = ta.column(right_repr, device=self.device)

        # Ensure py-iterables work as intended
        expected = [left_repr[0], right_repr[1], left_repr[2], right_repr[3]]
        result = ta.if_else(cond, left_repr, right_repr)
        self.assertEqual(expected, list(result))

        # Non common d type
        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="then and else branches must have compatible types, got.*and.*, respectively",
        ), mock.patch("torcharrow.icolumn.dt.common_dtype") as mock_common_dtype:
            mock_common_dtype.return_value = None

            ta.if_else(cond, left, right)

            mock_common_dtype.assert_called_once_with(
                left.dtype,
                right.dtype,
            )

        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="then and else branches must have compatible types, got.*and.*, respectively",
        ), mock.patch(
            "torcharrow.icolumn.dt.common_dtype"
        ) as mock_common_dtype, mock.patch(
            "torcharrow.icolumn.dt.is_void"
        ) as mock_is_void:
            mock_is_void.return_value = True
            mock_common_dtype.return_value = dt.int64

            ta.if_else(cond, left, right)
            mock_common_dtype.assert_called_once_with(
                left.dtype,
                right.dtype,
            )
            mock_is_void.assert_called_once_with(mock_common_dtype.return_value)

        # Invalid condition input
        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="condition must be a boolean vector",
        ):
            ta.if_else(
                cond=left,
                left=left,
                right=right,
            )

    def base_test_str(self):
        c = ta.column([f"test{x}" for x in range(5)], device=self.device)
        c.id = 321

        expected = "Column(['test0', 'test1', 'test2', 'test3', 'test4'], id = 321)"
        self.assertEqual(expected, str(c))

    def base_test_repr(self):
        c = ta.column([f"test{x}" for x in range(5)], device=self.device)

        expected = (
            "0  'test0'\n"
            "1  'test1'\n"
            "2  'test2'\n"
            "3  'test3'\n"
            "4  'test4'\n"
            f"dtype: string, length: 5, null_count: 0, device: {self.device}"
        )
        self.assertEqual(expected, repr(c))

    def base_test_is_valid_at(self):
        c = ta.column([f"test{x}" for x in range(5)], device=self.device)

        # Normal access
        self.assertTrue(all(c.is_valid_at(x) for x in range(5)))

        # Negative access
        self.assertTrue(c.is_valid_at(-1))

        # Out of bounds access
        with self.assertRaises(expected_exception=RuntimeError):
            self.assertFalse(c.is_valid_at(10))

    def base_test_cast(self):
        c_repr = ["0", "1", "2", "3", "4", None]
        c_repr_after_cast = [0, 1, 2, 3, 4, None]
        c = ta.column(c_repr, device=self.device)

        result = c.cast(dt.int64)
        self.assertEqual(c_repr_after_cast, list(result))

    def base_test_drop_null(self):
        c_repr = ["0", "1", "2", "3", "4", None]
        c = ta.column(c_repr, device=self.device)

        result = c.drop_null()

        self.assertEqual(c_repr[:-1], list(result))

        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="how parameter for flat columns not supported",
        ):
            c.drop_null(how="any")

    def base_test_drop_duplicates(self):
        c_repr = ["test", "test2", "test3", "test"]
        c = ta.column(c_repr, device=self.device)

        result = c.drop_duplicates()

        self.assertEqual(c_repr[:-1], list(result))

        # TODO: Add functionality for last
        with self.assertRaises(expected_exception=AssertionError):
            c.drop_duplicates(keep="last")

        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="subset parameter for flat columns not supported",
        ):
            c.drop_duplicates(subset=c_repr[:2])

    def base_test_fill_null(self):
        c_repr = ["0", "1", None, "3", "4", None]
        expected_fill = "TEST"
        expected_repr = ["0", "1", expected_fill, "3", "4", expected_fill]
        c = ta.column(c_repr, device=self.device)

        result = c.fill_null(expected_fill)

        self.assertEqual(expected_repr, list(result))

        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="fill_null with bytes is not supported",
        ):
            c.fill_null(expected_fill.encode())

    def base_test_isin(self):
        c_repr = [f"test{x}" for x in range(5)]
        c = ta.column(c_repr, device=self.device)
        self.assertTrue(all(c.isin(values=c_repr + ["test_123"])))
        self.assertFalse(any(c.isin(values=["test5", "test6", "test7"])))

    def base_test_bool(self):
        c = ta.column([f"test{x}" for x in range(5)], device=self.device)
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex=r"The truth value of a.*is ambiguous. Use a.any\(\) or a.all\(\).",
        ):
            bool(c)

    def base_test_flatmap(self):
        c = ta.column(["test1", "test2", None, None, "test3"], device=self.device)
        expected_result = [
            "test1",
            "test1",
            "test2",
            "test2",
            None,
            None,
            None,
            None,
            "test3",
            "test3",
        ]
        result = c.flatmap(lambda xs: [xs, xs])
        self.assertEqual(expected_result, list(result))

    def base_test_any(self):
        c_some = ta.column(["test1", "test2", None, None, "test3"], device=self.device)
        c_none = ta.column([], dtype=dt.string, device=self.device)
        c_none = c_none.append([None])
        self.assertTrue(any(c_some))
        self.assertFalse(any(c_none))

    def base_test_all(self):
        c_all = ta.column(["test", "test2", "test3"], device=self.device)
        c_partial = ta.column(["test", "test2", None, None], device=self.device)
        c_none = ta.column([], dtype=dt.string, device=self.device)
        c_none = c_none.append([None])
        self.assertTrue(all(c_all))
        self.assertFalse(all(c_partial))
        self.assertFalse(all(c_none))


if __name__ == "__main__":
    unittest.main()
