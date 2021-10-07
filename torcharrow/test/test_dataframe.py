# Copyright (c) Facebook, Inc. and its affiliates.
import unittest
from typing import List, Optional, NamedTuple

import torcharrow.dtypes as dt
from torcharrow import IDataFrame, Scope, me

# run python3 -m unittest outside this directory to run all tests


class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.ts = Scope()

    def base_test_internals_empty(self):
        empty = self.ts.DataFrame()

        # testing internals...
        self.assertTrue(isinstance(empty, IDataFrame))

        self.assertEqual(empty.length(), 0)
        self.assertEqual(empty.null_count(), 0)
        self.assertEqual(empty.columns, [])

    def base_test_internals_full(self):
        df = self.ts.DataFrame(dt.Struct([dt.Field("a", dt.int64)]))
        for i in range(4):
            df = df.append([(i,)])

        for i in range(4):
            self.assertEqual(df[i], (i,))

        self.assertEqual(df.length(), 4)
        self.assertEqual(df.null_count(), 0)
        self.assertEqual(list(df), list((i,) for i in range(4)))
        m = df[0 : len(df)]
        self.assertEqual(list(df[0 : len(df)]), list((i,) for i in range(4)))
        # TODO enforce runtime type check!
        # with self.assertRaises(TypeError):
        #     # TypeError: a tuple of type dt.Struct([dt.Field(a, dt.int64)]) is required, got None
        #     df=df.append([None])
        #     self.assertEqual(df.length(), 5)
        #     self.assertEqual(df.null_count(), 1)

    def base_test_internals_full_nullable(self):
        with self.assertRaises(TypeError):
            #  TypeError: nullable structs require each field (like a) to be nullable as well.
            df = self.ts.DataFrame(
                dt.Struct(
                    [dt.Field("a", dt.int64), dt.Field("b", dt.int64)], nullable=True
                )
            )
        df = self.ts.DataFrame(
            dt.Struct(
                [dt.Field("a", dt.int64.with_null()), dt.Field("b", dt.Int64(True))],
                nullable=True,
            )
        )

        for i in [0, 1, 2]:
            df = df.append([None])
            # but all public APIs report this back as None

            self.assertEqual(df[i], None)
            self.assertEqual(df.valid(i), False)
            self.assertEqual(df.null_count(), i + 1)
        for i in [3]:
            df = df.append([(i, i * i)])
            self.assertEqual(df[i], (i, i * i))
            self.assertEqual(df.valid(i), True)

        self.assertEqual(df.length(), 4)
        self.assertEqual(df.null_count(), 3)
        self.assertEqual(len(df["a"]), 4)
        self.assertEqual(len(df["b"]), 4)
        self.assertEqual(len(df._mask), 4)

        self.assertEqual(list(df), [None, None, None, (3, 9)])

        df = df.append([(4, 4 * 4), (5, 5 * 5)])
        self.assertEqual(list(df), [None, None, None, (3, 9), (4, 16), (5, 25)])

        # len
        self.assertEqual(len(df), 6)

    def base_test_internals_column_indexing(self):
        df = self.ts.DataFrame()
        df["a"] = self.ts.Column([None] * 3, dtype=dt.Int64(nullable=True))
        df["b"] = self.ts.Column([1, 2, 3])
        df["c"] = self.ts.Column([1.1, 2.2, 3.3])

        # index
        self.assertEqual(list(df["a"]), [None] * 3)
        # pick & column -- note: creates a view
        self.assertEqual(df[["a", "c"]].columns, ["a", "c"])
        # pick and index
        self.assertEqual(list(df[["a", "c"]]["a"]), [None] * 3)
        self.assertEqual(list(df[["a", "c"]]["c"]), [1.1, 2.2, 3.3])

        # slice

        self.assertEqual(df[:"b"].columns, ["a"])
        self.assertEqual(df["b":].columns, ["b", "c"])
        self.assertEqual(df["a":"c"].columns, ["a", "b"])

    def base_test_construction(self):
        # Column type is List of Struct
        # When struct is represented by Python tuples (not NamedTuple)
        # explicit dtype is required since tuple doesn't contain name information
        data1 = [
            [(1, "a"), (2, "b")],
            [(3, "c"), (4, "d"), (5, "e")],
        ]
        # only data will fail
        with self.assertRaises(TypeError) as ex:
            a = self.ts.Column(data1)
        self.assertTrue(
            "Cannot infer type from Python tuple" in str(ex.exception),
            f"Exception message is not as expected: {str(ex.exception)}",
        )
        # data + dtype
        a = self.ts.Column(
            data1,
            dtype=dt.List(
                dt.Struct([dt.Field("col1", dt.int64), dt.Field("col2", dt.string)])
            ),
        )
        self.assertEqual(list(a), data1)

        # Basic test of DataFrame construction from Dict
        data2 = {"a": list(range(10)), "b": list(range(10, 20))}
        dtype2 = dt.Struct([dt.Field("a", dt.int32), dt.Field("b", dt.int16)])
        expected2 = list(zip(*data2.values()))
        # only data, inferred as int64
        df = self.ts.DataFrame(data2)
        self.assertEqual(list(df), expected2)
        self.assertEqual(
            df.dtype, dt.Struct([dt.Field("a", dt.int64), dt.Field("b", dt.int64)])
        )
        # data + dtype, use specified dtype (int32 and int16)
        df = self.ts.DataFrame(data2, dtype2)
        self.assertEqual(list(df), expected2)
        self.assertEqual(df.dtype, dtype2)

        # DataFrame construction from dict with Struct represented as tuple
        # dtype is required since otherwise the dtype cannot be inferred
        data3 = {
            "a": [1, 2, 3],
            "b": [(1, "a"), (2, "b"), (3, "c")],
        }
        dtype3 = dt.Struct(
            [
                dt.Field("a", dt.int64),
                dt.Field(
                    "b",
                    dt.Struct([dt.Field("b1", dt.int64), dt.Field("b2", dt.string)]),
                ),
            ]
        )
        # only data will fail
        with self.assertRaises(TypeError) as ex:
            df = self.ts.DataFrame(data3)
        self.assertTrue(
            "Cannot infer type from Python tuple" in str(ex.exception),
            f"Excpeion message is not as expected: {str(ex.exception)}",
        )
        # data + dtype
        df = self.ts.DataFrame(data3, dtype3)
        self.assertEqual(list(df), list(zip(*data3.values())))
        self.assertEqual(df.dtype, dtype3)

    def base_test_infer(self):
        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3]})
        self.assertEqual(df.columns, ["a", "b"])
        self.assertEqual(
            df.dtype,
            dt.Struct(
                [dt.Field("a", dt.int64), dt.Field("b", dt.Float64(nullable=True))]
            ),
        )

        self.assertEqual(df.dtype.get("a"), dt.int64)
        self.assertEqual(list(df), list(zip([1, 2, 3], [1.0, None, 3])))

        df = self.ts.DataFrame()
        self.assertEqual(len(df), 0)

        df["a"] = self.ts.Column([1, 2, 3], dtype=dt.int32)
        self.assertEqual(df._dtype.get("a"), dt.int32)
        self.assertEqual(len(df), 3)

        df["b"] = [1.0, None, 3]
        self.assertEqual(len(df), 3)

        df = self.ts.DataFrame([(1, 2), (2, 3), (4, 5)], columns=["a", "b"])
        self.assertEqual(list(df), [(1, 2), (2, 3), (4, 5)])

        B = dt.Struct([dt.Field("b1", dt.int64), dt.Field("b2", dt.int64)])
        A = dt.Struct([dt.Field("a", dt.int64), dt.Field("b", B)])
        df = self.ts.DataFrame([(1, (2, 22)), (2, (3, 33)), (4, (5, 55))], dtype=A)

        self.assertEqual(list(df), [(1, (2, 22)), (2, (3, 33)), (4, (5, 55))])

    @staticmethod
    def _add(a, b):
        return a + b

    def base_test_map_where_filter(self):
        # TODO have to decide on whether to follow Pandas, map, filter or our own.

        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = ["a", "b", "C"]
        df["d"] = [100, 200, None]

        # keep None
        self.assertEqual(
            list(df.map({100: 1000}, columns=["d"], dtype=dt.Int64(nullable=True))),
            [1000, None, None],
        )

        # maps None
        self.assertEqual(
            list(
                df.map(
                    {None: 1, 100: 1000}, columns=["d"], dtype=dt.Int64(nullable=True)
                )
            ),
            [1000, None, 1],
        )

        # maps as function
        self.assertEqual(
            list(df.map(TestDataFrame._add, columns=["a", "a"], dtype=dt.int64)),
            [2, 4, 6],
        )

        # filter
        self.assertEqual(
            list(df.filter(str.islower, columns=["c"])),
            [(1, 11, "a", 100), (2, 22, "b", 200)],
        )

    def base_test_transform(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]

        # column level without type hints
        self.assertEqual(
            list(df["a"].transform(lambda l: [x + 1 for x in l])), [2, 3, 4]
        )

        with self.assertRaises(ValueError):
            # wrong number of rows
            df["a"].transform(lambda l: [-1] + [x + 1 for x in l])

        def batch_str(a):
            return list(map(str, a))

        # TODO: add some basic type check for from_list
        # with self.assertRaises(TypeError):
        #     # forgot the output type annotation
        #     df["a"].transform(batch_str)

        self.assertEqual(
            list(df["a"].transform(batch_str, dtype=dt.string)), ["1", "2", "3"]
        )

        # columns level with type hints
        def batch_str_ann(a) -> List[List[str]]:
            assert isinstance(a, list)  # verify the python format
            return [[str(x)] * x for x in a]

        self.assertEqual(
            list(df["a"].transform(batch_str_ann, format="python")),
            [["1"], ["2", "2"], ["3", "3", "3"]],
        )

        with self.assertRaises(AssertionError):
            # forgot the format arg, column instead of list is passed
            df["a"].transform(batch_str_ann)

        # df-level without type hints
        def myadd(a, b):
            return [x + y for x, y in zip(a, b)]

        self.assertEqual(
            list(df.transform(myadd, columns=["a", "b"], dtype=dt.int64)), [12, 24, 36]
        )

        # df-level with type hints
        def myadd_hint(a, b) -> List[int]:
            return [x + y for x, y in zip(a, b)]

        self.assertEqual(
            list(df.transform(myadd_hint, columns=["a", "b"])), [12, 24, 36]
        )

    def base_test_sort_stuff(self):
        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3]})
        self.assertEqual(
            list(df.sort(by="a", ascending=False)),
            list(zip([3, 2, 1], [3, None, 1.0])),
        )
        # Not allowing None in comparison might be too harsh...
        # TODO CLARIFY THIS
        # with self.assertRaises(TypeError):
        #     # TypeError: '<' not supported between instances of 'NoneType' and 'float'
        #     self.assertEqual(
        #         list(df.sort(by="b", ascending=False)),
        #         list(zip([3, 2, 1], [3, None, 1.0])),
        #     )

        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [1.0, None, 3], "c": [4, 4, 1]})
        self.assertEqual(
            list(df.sort(by=["c", "a"], ascending=False)),
            list([(2, None, 4), (1, 1.0, 4), (3, 3.0, 1)]),
        )

        self.assertEqual(
            list(df.nlargest(n=2, columns=["c", "a"], keep="first")),
            [(2, None, 4), (1, 1.0, 4)],
        )
        self.assertEqual(
            list(df.nsmallest(n=2, columns=["c", "a"], keep="first")),
            [(3, 3.0, 1), (1, 1.0, 4)],
        )
        self.assertEqual(list(df.reverse()), [(3, 3.0, 1), (2, None, 4), (1, 1.0, 4)])

    def base_test_operators(self):
        # without None
        c = self.ts.DataFrame({"a": [0, 1, 3]})

        d = self.ts.DataFrame({"a": [5, 5, 6]})
        e = self.ts.DataFrame({"a": [1.0, 1, 7]})

        self.assertEqual(list(c == c), [(True,)] * 3)
        self.assertEqual(list(c == d), [(False,)] * 3)

        # NOTE: Yoo cannot compare Columns with assertEqual,
        #       since torcharrow overrode __eq__
        #       this always compare with list(), etc
        #       or write (a==b).all()

        self.assertEqual(list(c == 1), [(i,) for i in [False, True, False]])
        self.assertTrue(
            ((c == 1) == self.ts.DataFrame({"a": [False, True, False]})).all()
        )

        # <, <=, >=, >

        self.assertEqual(list(c <= 2), [(i,) for i in [True, True, False]])
        self.assertEqual(list(c < d), [(i,) for i in [True, True, True]])
        self.assertEqual(list(c >= d), [(i,) for i in [False, False, False]])
        self.assertEqual(list(c > 2), [(i,) for i in [False, False, True]])

        # +,-,*,/,//,**

        self.assertEqual(list(-c), [(i,) for i in [0, -1, -3]])
        self.assertEqual(list(+-c), [(i,) for i in [0, -1, -3]])

        self.assertEqual(list(c + 1), [(i,) for i in [1, 2, 4]])
        # self.assertEqual(list(c.add(1)), [(i,) for i in [1, 2, 4]])

        self.assertEqual(list(1 + c), [(i,) for i in [1, 2, 4]])
        # self.assertEqual(list(c.radd(1)), [(i,) for i in [1, 2, 4]])

        self.assertEqual(list(c + d), [(i,) for i in [5, 6, 9]])
        # self.assertEqual(list(c.add(d)), [(i,) for i in [5, 6, 9]])

        self.assertEqual(list(c + 1), [(i,) for i in [1, 2, 4]])
        self.assertEqual(list(1 + c), [(i,) for i in [1, 2, 4]])
        self.assertEqual(list(c + d), [(i,) for i in [5, 6, 9]])

        self.assertEqual(list(c - 1), [(i,) for i in [-1, 0, 2]])
        self.assertEqual(list(1 - c), [(i,) for i in [1, 0, -2]])
        self.assertEqual(list(d - c), [(i,) for i in [5, 4, 3]])

        self.assertEqual(list(c * 2), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(2 * c), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(c * d), [(i,) for i in [0, 5, 18]])

        self.assertEqual(list(c * 2), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(2 * c), [(i,) for i in [0, 2, 6]])
        self.assertEqual(list(c * d), [(i,) for i in [0, 5, 18]])

        self.assertEqual(list(c / 2), [(i,) for i in [0.0, 0.5, 1.5]])
        # #TODO check numpy devision issue
        # with self.assertRaises(ZeroDivisionError):
        #     self.assertEqual(list(2 / c), [(i,) for i in [0.0, 0.5, 1.5]])
        self.assertEqual(list(c / d), [(i,) for i in [0.0, 0.2, 0.5]])

        self.assertEqual(list(d // 2), [(i,) for i in [2, 2, 3]])
        self.assertEqual(list(2 // d), [(i,) for i in [0, 0, 0]])
        self.assertEqual(list(c // d), [(i,) for i in [0, 0, 0]])
        self.assertEqual(list(e // d), [(i,) for i in [0.0, 0.0, 1.0]])
        # THIS ASSERTION SHOULD NOT HAPPEN, FIX derive_dtype
        # TypeError: integer argument expected, got float
        # TODO check numpy devision issue
        # with self.assertRaises(TypeError):
        #     self.assertEqual(list(d // e), [(i,) for i in [0.0, 0.0, 1.0]])

        self.assertEqual(list(c ** 2), [(i,) for i in [0, 1, 9]])
        self.assertEqual(list(2 ** c), [(i,) for i in [1, 2, 8]])
        self.assertEqual(list(c ** d), [(i,) for i in [0, 1, 729]])

        #     # # null handling

        c = self.ts.DataFrame({"a": [0, 1, 3, None]})
        self.assertEqual(list(c + 1), [(i,) for i in [1, 2, 4, None]])

        # # TODO decideo on special handling with fill_values, maybe just drop functionality?
        # self.assertEqual(list(c.add(1, fill_value=17)), [(i,) for i in [1, 2, 4, 18]])
        # self.assertEqual(list(c.radd(1, fill_value=-1)), [(i,) for i in [1, 2, 4, 0]])
        f = self.ts.Column([None, 1, 3, None])
        # self.assertEqual(
        #     list(c.radd(f, fill_value=100)), [(i,) for i in [100, 2, 6, 200]]
        # )
        self.assertEqual(list((c + f).fillna(100)), [(i,) for i in [100, 2, 6, 100]])

        # &, |, ^, ~
        g = self.ts.Column([True, False, True, False])
        h = self.ts.Column([False, False, True, True])
        self.assertEqual(g & h, [False, False, True, False])
        self.assertEqual(g | h, [True, False, True, True])
        self.assertEqual(g ^ h, [True, False, False, True])
        self.assertEqual(True & g, [True, False, True, False])
        self.assertEqual(True | g, [True, True, True, True])
        self.assertEqual(True ^ g, [False, True, False, True])
        self.assertEqual(~g, [True, True, True, True])

        i = self.ts.Column([1, 2, 0])
        j = self.ts.Column([3, 2, 3])
        self.assertEqual(i & j, [1, 2, 0])
        self.assertEqual(i | j, [3, 2, 3])
        self.assertEqual(i ^ j, [2, 0, 3])
        self.assertEqual(2 & i, [0, 2, 0])
        self.assertEqual(2 | i, [3, 2, 2])
        self.assertEqual(2 ^ i, [3, 0, 2])
        self.assertEqual(~i, [-2, -3, -4])

        # Expressions should throw if anything is wrong
        try:
            u = self.ts.Column(list(range(5)))
            v = -u
            uv = self.ts.DataFrame({"a": u, "b": v})
            uu = self.ts.DataFrame({"a": u, "b": u})
            x = uv == 1
            y = uu["a"] == uv["a"]
            z = uv == uu
            z["a"]
            (z | (x["a"]))
        except:
            self.assertTrue(False)

    def base_test_na_handling(self):
        c = self.ts.DataFrame({"a": [None, 2, 17.0]})

        self.assertEqual(list(c.fillna(99.0)), [(i,) for i in [99.0, 2, 17.0]])
        self.assertEqual(list(c.dropna()), [(i,) for i in [2, 17.0]])

        c = c.append([(2,)])
        self.assertEqual(list(c.drop_duplicates()), [(i,) for i in [None, 2, 17.0]])

        # duplicates with subset
        d = self.ts.DataFrame({"a": [None, 2, 17.0, 7, 2], "b": [1, 2, 17.0, 2, 1]})
        self.assertEqual(
            list(d.drop_duplicates(subset="a")),
            [(None, 1.0), (2.0, 2.0), (17.0, 17.0), (7.0, 2.0)],
        )
        self.assertEqual(
            list(d.drop_duplicates(subset="b")), [(None, 1.0), (2.0, 2.0), (17.0, 17.0)]
        )
        self.assertEqual(
            list(d.drop_duplicates(subset=["b", "a"])),
            [(None, 1.0), (2.0, 2.0), (17.0, 17.0), (7.0, 2.0), (2.0, 1.0)],
        )
        self.assertEqual(
            list(d.drop_duplicates()),
            [(None, 1.0), (2.0, 2.0), (17.0, 17.0), (7.0, 2.0), (2.0, 1.0)],
        )

    def base_test_agg_handling(self):
        import functools
        import operator

        c = [1, 4, 2, 7, 9, 0]
        C = self.ts.DataFrame({"a": [1, 4, 2, 7, 9, 0, None]})

        self.assertEqual(C.min()["a"], min(c))
        self.assertEqual(C.max()["a"], max(c))
        self.assertEqual(C.sum()["a"], sum(c))
        self.assertEqual(C.prod()["a"], functools.reduce(operator.mul, c, 1))
        # TODO check for mode in numpy
        # self.assertEqual(C.mode()["a"], statistics.mode(c))
        # TODO wolfram: support int->float
        # self.assertEqual(C.std()["a"], statistics.stdev(c))
        # self.assertEqual(C.mean()["a"], statistics.mean(c))
        # self.assertEqual(C.median()["a"], statistics.median(c))

        self.assertEqual(
            list(C.cummin()),
            [(i,) for i in [min(c[:i]) for i in range(1, len(c) + 1)] + [None]],
        )
        self.assertEqual(
            list(C.cummax()),
            [(i,) for i in [max(c[:i]) for i in range(1, len(c) + 1)] + [None]],
        )
        self.assertEqual(
            list(C.cumsum()),
            [(i,) for i in [sum(c[:i]) for i in range(1, len(c) + 1)] + [None]],
        )
        self.assertEqual(
            list(C.cumprod()),
            [
                (i,)
                for i in [
                    functools.reduce(operator.mul, c[:i], 1)
                    for i in range(1, len(c) + 1)
                ]
                + [None]
            ],
        )
        self.assertEqual((C % 2 == 0)[:-1].all(), all(i % 2 == 0 for i in c))
        self.assertEqual((C % 2 == 0)[:-1].any(), any(i % 2 == 0 for i in c))

    def base_test_isin(self):
        c = [1, 4, 2, 7]
        C = self.ts.DataFrame({"a": c + [None]})
        self.assertEqual(
            list(C.isin([1, 2, 3])), [(i,) for i in [True, False, True, False, False]]
        )

    def base_test_isin2(self):
        df = self.ts.DataFrame({"A": [1, 2, 3], "B": [1, 1, 1]})
        self.assertEqual(list(df.nunique()), [("A", 3), ("B", 1)])

    def base_test_describe_dataframe(self):
        # TODO introduces cyclic dependency between Column and Dataframe, need diff design...
        c = self.ts.DataFrame({"a": self.ts.Column([1, 2, 3])})
        self.assertEqual(
            list(c.describe()),
            [
                ("count", 3.0),
                ("mean", 2.0),
                ("std", 1.0),
                ("min", 1.0),
                ("25%", 1.5),
                ("50%", 2.0),
                ("75%", 2.5),
                ("max", 3.0),
            ],
        )

    def base_test_drop_keep_rename_reorder_pipe(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = [111, 222, 333]
        self.assertEqual(list(df.drop([])), [(1, 11, 111), (2, 22, 222), (3, 33, 333)])
        self.assertEqual(list(df.drop(["c", "a"])), [(11,), (22,), (33,)])

        self.assertEqual(list(df.keep([])), [])
        self.assertEqual(list(df.keep(["c", "a"])), [(1, 111), (2, 222), (3, 333)])

        self.assertEqual(
            list(df.rename({"a": "c", "c": "a"})),
            [(1, 11, 111), (2, 22, 222), (3, 33, 333)],
        )
        self.assertEqual(
            list(df.reorder(list(reversed(df.columns)))),
            [(111, 11, 1), (222, 22, 2), (333, 33, 3)],
        )

        def f(df):
            return df

        self.assertEqual(list(df), list(df.pipe(f)))

        def g(df, num):
            return df + num

        self.assertEqual(list(df + 13), list(df.pipe(g, 13)))

    def base_test_me_on_str(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]
        df["c"] = ["a", "b", "C"]

        self.assertEqual(
            list(df.where(me["c"].str.capitalize() == me["c"])), [(3, 33, "C")]
        )

    def base_test_locals_and_me_equivalence(self):
        df = self.ts.DataFrame()
        df["a"] = [1, 2, 3]
        df["b"] = [11, 22, 33]

        self.assertEqual(
            list(df.where((me["a"] > 1) & (me["b"] == 33))),
            list(df[(df["a"] > 1) & (df["b"] == 33)]),
        )

        self.assertEqual(list(df.select("*")), list(df))

        self.assertEqual(list(df.select("a")), list(df.keep(["a"])))
        self.assertEqual(list(df.select("*", "-a")), list(df.drop(["a"])))

        gf = self.ts.DataFrame({"a": df["a"], "b": df["b"], "c": df["a"] + df["b"]})
        self.assertEqual(list(df.select("*", d=me["a"] + me["b"])), list(gf))

    def base_test_groupby_size_pipe(self):
        df = self.ts.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [2, 2, 1]})
        self.assertEqual(list(df.groupby("a").size), [(1, 2), (2, 1)])

        df = self.ts.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4]})

        # TODO have to add type inference here
        # self.assertEqual(list(df.groupby('A').pipe({'B': lambda x: x.max() - x.min()})),
        #                  [('a',  2), ('b', 2)])

        # self.assertEqual(list(df.groupby('A').select(B=me['B'].max() - me['B'].min())),
        #                  [('a',  2), ('b', 2)])

    def base_test_groupby_agg(self):
        df = self.ts.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4]})

        self.assertEqual(list(df.groupby("A").agg("sum")), [("a", 4), ("b", 6)])

        df = self.ts.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [2, 2, 1]})

        self.assertEqual(list(df.groupby("a").agg("sum")), [(1, 3, 4), (2, 3, 1)])

        self.assertEqual(
            list(df.groupby("a").agg(["sum", "min"])),
            [(1, 3, 4, 1, 2), (2, 3, 1, 3, 1)],
        )

        self.assertEqual(
            list(df.groupby("a").agg({"c": "max", "b": ["min", "mean"]})),
            [(1, 2, 1, 1.5), (2, 1, 3, 3.0)],
        )

    def base_test_groupby_iter_get_item_ops(self):
        df = self.ts.DataFrame({"A": ["a", "b", "a", "b"], "B": [1, 2, 3, 4]})
        for g, gf in df.groupby("A"):
            if g == ("a",):
                self.assertEqual(list(gf), [(1,), (3,)])
            elif g == ("b",):
                self.assertEqual(list(gf), [(2,), (4,)])
            else:
                self.assertTrue(False)

        self.assertEqual(list(df.groupby("A").sum()), [("a", 4), ("b", 6)])
        self.assertEqual(list(df.groupby("A")["B"].sum()), [4, 6])

    def base_test_column_overriden(self):
        df = self.ts.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        self.assertEqual(list(df), [(1, "a"), (2, "b"), (3, "c")])
        self.assertEqual(
            df.dtype, dt.Struct([dt.Field("a", dt.int64), dt.Field("b", dt.string)])
        )

        df["a"] = df["a"].map(lambda x: "str_" + str(x), dtype=dt.string)
        self.assertEqual(list(df["a"]), ["str_1", "str_2", "str_3"])
        self.assertEqual(list(df), [("str_1", "a"), ("str_2", "b"), ("str_3", "c")])
        self.assertEqual(
            df.dtype, dt.Struct([dt.Field("a", dt.string), dt.Field("b", dt.string)])
        )

    def test_infer_func_output_dtype(self):
        df = self.ts.DataFrame({"a": [1, 2, 3], "b": [11, 22, 33]})

        def myadd(a: int, b: int) -> str:
            return f"{a}_{b}"

        self.assertEqual(
            list(df.map(myadd, columns=["a", "b"])), ["1_11", "2_22", "3_33"]
        )

        def mynullable(a: int) -> Optional[int]:
            return a if a % 2 == 1 else None

        r = df["a"].map(mynullable)
        self.assertEqual(df["a"].dtype, dt.int64)
        self.assertEqual(r.dtype, dt.int64.with_null())
        self.assertEqual(list(r), [1, None, 3])

        class Ret(NamedTuple):
            plus: int
            minus: int

        def mymultiret(a: int, b: int) -> Ret:
            return Ret(a + b, a - b)

        r = df.map(mymultiret, columns=["a", "b"])
        self.assertEqual(
            r.dtype,
            dt.Struct([dt.Field("plus", dt.int64), dt.Field("minus", dt.int64)]),
        )
        self.assertEqual(list(r), [(12, -10), (24, -20), (36, -30)])

        # test regular dict without dtype works
        r = df.map({None: 1, 1: 1000}, columns=["a"])
        self.assertEqual(list(r), [1000, None, None])
        self.assertEqual(r.dtype, dt.int64)


if __name__ == "__main__":
    unittest.main()
