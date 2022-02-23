# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as ty
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow.istring_column import IStringColumn


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
        self.assertTrue(isinstance(empty, IStringColumn))
        self.assertEqual(empty.dtype, dt.string)
        self.assertEqual(empty.length, 0)
        self.assertEqual(empty.null_count, 0)
        # self.assertEqual(empty._offsets[0], 0)

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


if __name__ == "__main__":
    unittest.main()
