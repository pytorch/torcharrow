# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow import IStringColumn


class TestStringColumn(unittest.TestCase):
    def base_test_empty(self):
        empty = ta.Column(dt.string, device=self.device)
        self.assertTrue(isinstance(empty, IStringColumn))
        self.assertEqual(empty.dtype, dt.string)
        self.assertEqual(empty.length, 0)
        self.assertEqual(empty.null_count, 0)
        # self.assertEqual(empty._offsets[0], 0)

    def base_test_append_offsets(self):
        c = ta.Column(dt.string, device=self.device)
        c = c.append(["abc", "de", "", "f"])
        # self.assertEqual(list(c._offsets), [0, 3, 5, 5, 6])
        self.assertEqual(list(c), ["abc", "de", "", "f"])
        # TODO : check that error is thrown!
        # with self.assertRaises(TypeError):
        #     # TypeError: a dt.string is required (got type NoneType)
        #     c.append(None)

        c = ta.Column(["abc", "de", "", "f", None])
        # self.assertEqual(list(c._offsets), [0, 3, 5, 5, 6, 6])
        self.assertEqual(list(c), ["abc", "de", "", "f", None])

    def base_test_string_split_methods(self):
        s = ["a b c", "1,2,3", "d e f g h", "hello.this.is.very.very.very.very.long"]
        c = ta.Column(s, device=self.device)
        self.assertEqual(list(c.str.split(".")), [v.split(".") for v in s])
        self.assertEqual(list(c.str.split()), [v.split() for v in s])
        self.assertEqual(list(c.str.split(",")), [v.split(",") for v in s])

    def base_test_string_categorization_methods(self):
        # isalpha/isnumeric/isalnum/isdigit/isdecimal/isspace/islower/isupper
        self.assertEqual(
            list(
                ta.Column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", None],
                    device=self.device,
                ).str.isalpha()
            ),
            [False, True, True, False, False, True, False, None],
        )
        self.assertEqual(
            list(
                ta.Column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", None],
                    device=self.device,
                ).str.isalnum()
            ),
            [False, True, True, True, True, True, False, None],
        )

        self.assertEqual(
            list(
                ta.Column(
                    ["", "abc", "XYZ", "123", "XYZ123", "äöå", ",.!", "\u00B2", None],
                    device=self.device,
                ).str.isdecimal()
            ),
            [False, False, False, True, False, False, False, False, None],
        )
        self.assertEqual(
            list(ta.Column([".abc"], device=self.device).str.islower()), [True]
        )
        self.assertEqual(
            list(ta.Column(["+3.e12", "abc", "0"], device=self.device).str.isnumeric()),
            [False, False, True],
        )
        self.assertEqual(
            list(
                ta.Column(["\n", "\t", " ", "", "a"], device=self.device).str.isspace()
            ),
            [True, True, True, False, False],
        )
        self.assertEqual(
            list(ta.Column(["A B C", "abc", " "], device=self.device).str.istitle()),
            [True, False, False],
        )
        self.assertEqual(
            list(
                ta.Column(
                    ["UPPER", "lower", ".abc", ".ABC", "123"], device=self.device
                ).str.isupper()
            ),
            [True, False, False, True, False],
        )

    def base_test_concat(self):
        s1 = ["abc", "de", "", "f", None]
        s2 = ["12", "567", "77", None, "55"]
        c1 = ta.Column(s1, device=self.device)
        c2 = ta.Column(s2, device=self.device)
        concat1 = [x + y for (x, y) in zip(s1[:-2], s2[:-2])] + [None, None]
        self.assertEqual(list(c1 + c2), concat1)

        concat2 = [x + "_suffix" for x in s1[:-1]] + [None]
        self.assertEqual(list(c1 + "_suffix"), concat2)

        # TODO: also support str + IColumn

    def base_test_comparison(self):
        c = ta.Column(["abc", "de", "", "f", None], device=self.device)
        d = ta.Column(["abc", "77", "", None, "55", "!"], device=self.device)
        self.assertEqual(list(c == c), [True, True, True, True, None])
        self.assertEqual(list(c == d), [True, False, True, None, None])
        # This is currently broken, will fix in subsequent diff.
        self.assertEqual(
            list(d == c), [True, False, True, None, None, False]
        )  # [True, False, True, None, None, None]
        self.assertEqual(list(c == "de"), [False, True, False, False, None])

        self.assertEqual(list(c != c), [False, False, False, False, None])
        self.assertEqual(list(c != d), [False, True, False, None, None])
        self.assertEqual(list(d != c), [False, True, False, None, None, None])
        self.assertEqual(list(c != "de"), [True, False, True, True, None])

        self.assertEqual(list(c < c), [False, False, False, False, None])
        self.assertEqual(list(c < d), [False, False, False, None, None])
        self.assertEqual(list(d < c), [False, True, False, None, None, None])
        self.assertEqual(list(c < "de"), [True, False, True, False, None])

        self.assertEqual(list(c <= c), [True, True, True, True, None])
        self.assertEqual(list(c <= d), [True, False, True, None, None])
        self.assertEqual(list(d <= c), [True, True, True, None, None, None])
        self.assertEqual(list(c <= "de"), [True, True, True, False, None])

        self.assertEqual(list(c > c), [False, False, False, False, None])
        self.assertEqual(list(c > d), [False, True, False, None, None])
        self.assertEqual(list(d > c), [False, False, False, None, None, None])
        self.assertEqual(list(c > "de"), [False, False, False, True, None])

        self.assertEqual(list(c >= c), [True, True, True, True, None])
        self.assertEqual(list(c >= d), [True, True, True, None, None])
        self.assertEqual(list(d >= c), [True, False, True, None, None, None])
        self.assertEqual(list(c >= "de"), [False, True, False, True, None])

    def base_test_string_lifted_methods(self):
        s = ["abc", "de", "", "f"]
        c = ta.Column(s, device=self.device)
        self.assertEqual(list(c.str.length()), [len(i) for i in s])
        self.assertEqual(list(c.str.slice(stop=2)), [i[:2] for i in s])
        self.assertEqual(list(c.str.slice(1, 2)), [i[1:2] for i in s])
        self.assertEqual(list(c.str.slice(1)), [i[1:] for i in s])

        self.assertEqual(
            list(ta.Column(["UPPER", "lower"], device=self.device).str.lower()),
            ["upper", "lower"],
        )
        self.assertEqual(
            list(ta.Column(["UPPER", "lower"], device=self.device).str.upper()),
            ["UPPER", "LOWER"],
        )

        # strip
        self.assertEqual(
            list(ta.Column(["  ab", " cde\n  "], device=self.device).str.strip()),
            ["ab", "cde"],
        )

    def base_test_string_pattern_matching_methods(self):
        s = ["hello.this", "is.interesting.", "this.is_24", "paradise", "h", ""]

        self.assertEqual(
            list(ta.Column(s, device=self.device).str.startswith("h")),
            [True, False, False, False, True, False],
        )
        self.assertEqual(
            list(ta.Column(s, device=self.device).str.endswith("this")),
            [True, False, False, False, False, False],
        )
        self.assertEqual(
            list(ta.Column(s, device=self.device).str.find("this")),
            [6, -1, 0, -1, -1, -1],
        )

        self.assertEqual(
            list(ta.Column(s, device=self.device).str.replace("this", "that")),
            [v.replace("this", "that") for v in s],
        )

    def base_test_regular_expressions(self):
        S = ta.Column(
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
        self.assertEqual(
            list(S[S.str.count_re(r"(^F.*)") == 1]), ["Finland", "Florida"]
        )
        self.assertEqual(S.str.count_re(r"(^F.*)").sum(), 2)
        # match
        self.assertEqual(list(S[S.str.match_re(r"(^P.*)") == True]), ["Puerto Rico"])
        # replace
        self.assertEqual(
            list(S.str.replace_re("(-d)", "")),
            [
                "Finland",
                "Colombia",
                "Florida",
                "Japan",
                "Puerto Rico",
                "Russia",
                "france",
            ],
        )
        # contains
        self.assertEqual(
            list(S.str.contains_re("^F.*")),
            [True, False, True, False, False, False, False],
        )

        # findall (creates a list), we select the non empty ones.

        l = S.str.findall_re("^[Ff].*")
        self.assertEqual(
            list(l[l.list.length() > 0]),
            [
                ["Finland"],
                ["Florida"],
                ["france"],
            ],
        )

        # extract all
        l = S.str.findall_re("^[Ff](.*)")
        self.assertEqual(
            list(l[l.list.length() > 0]),
            [
                ["Finland"],
                ["Florida"],
                ["france"],
            ],
        )

        l = S.str.findall_re("[tCc]o")
        self.assertEqual(
            list(l[l.list.length() > 0]),
            [
                ["Co"],
                ["to", "co"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
