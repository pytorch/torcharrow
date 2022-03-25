# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import statistics
import typing as ty
import unittest
from math import ceil, floor, log, isnan

import numpy as np
import numpy.testing
import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow.icolumn import Column
from torcharrow.inumerical_column import NumericalColumn
from torcharrow.scope import Scope


class TestNumericalColumn(unittest.TestCase):
    def base_test_empty(self):
        empty_i64_column = ta.column(dtype=dt.int64, device=self.device)

        # testing internals...
        self.assertTrue(isinstance(empty_i64_column, NumericalColumn))
        self.assertEqual(empty_i64_column.dtype, dt.int64)
        self.assertEqual(len(empty_i64_column), 0)
        self.assertEqual(empty_i64_column.null_count, 0)
        self.assertEqual(len(empty_i64_column), 0)

        return empty_i64_column

    def base_test_full(self):
        col = ta.column([i for i in range(4)], dtype=dt.int64, device=self.device)

        # self.assertEqual(col._offset, 0)
        self.assertEqual(len(col), 4)
        self.assertEqual(col.null_count, 0)
        self.assertEqual(list(col), list(range(4)))
        m = col[0 : len(col)]
        self.assertEqual(list(m), list(range(4)))
        return col

    def base_test_is_immutable(self):
        col = ta.column([i for i in range(4)], dtype=dt.int64, device=self.device)
        with self.assertRaises(AttributeError):
            # AssertionError: can't append a finalized list
            col._append(None)

    def base_test_full_nullable(self):
        col = ta.column(dtype=dt.Int64(nullable=True), device=self.device)

        col = col.append([None, None, None])
        self.assertEqual(col[-1], None)

        col = col.append([3])
        self.assertEqual(col[-1], 3)

        self.assertEqual(col.length, 4)
        self.assertEqual(col.null_count, 3)

        self.assertEqual(col[0], None)
        self.assertEqual(col[3], 3)

        self.assertEqual(list(col), [None, None, None, 3])

    def base_test_indexing(self):
        col = ta.column(
            [None] * 3 + [3, 4, 5], dtype=dt.Int64(nullable=True), device=self.device
        )

        # index
        self.assertEqual(col[0], None)
        self.assertEqual(col[-1], 5)

        # slice

        # continuous slice
        c = col[3 : len(col)]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(c), 3)

        # non continuous slice
        d = col[::2]
        self.assertEqual(len(col), 6)
        self.assertEqual(len(d), 3)

        # slice has Python not Panda semantics
        e = col[: len(col) - 1]
        self.assertEqual(len(e), len(col) - 1)

        # indexing via lists
        f = col[[0, 1, 2]]
        self.assertEqual(list(f), list(col[:3]))

        # head/tail are special slices
        self.assertEqual(list(col.head(2)), [None, None])
        self.assertEqual(list(col.tail(2)), [4, 5])

    def base_test_boolean_column(self):

        col = ta.column(dt.boolean, device=self.device)
        self.assertIsInstance(col, NumericalColumn)

        col = col.append([True, False, False])
        self.assertEqual(list(col), [True, False, False])

        # numerics can be converted to booleans...
        col = col.append([1])
        self.assertEqual(list(col), [True, False, False, True])

    def base_test_infer(self):
        # not enough info
        with self.assertRaises(ValueError):
            ta.column([], device=self.device)

        # int
        c = ta.column([1], device=self.device)
        self.assertEqual(c.dtype, dt.int64)
        self.assertEqual(list(c), [1])
        c = ta.column([np.int64(2)], device=self.device)
        self.assertEqual(c.dtype, dt.int64)
        self.assertEqual(list(c), [2])
        c = ta.column([np.int32(3)], device=self.device)
        self.assertEqual(c.dtype, dt.int32)
        self.assertEqual(list(c), [3])
        c = ta.column([np.int16(4)], device=self.device)
        self.assertEqual(c.dtype, dt.int16)
        self.assertEqual(list(c), [4])
        c = ta.column([np.int8(5)], device=self.device)
        self.assertEqual(c.dtype, dt.int8)
        self.assertEqual(list(c), [5])

        # bool
        c = ta.column([True, None], device=self.device)
        self.assertEqual(c.dtype, dt.Boolean(nullable=True))
        self.assertEqual(list(c), [True, None])

        # note implicit promotion of bool to int
        c = ta.column([True, 1], device=self.device)
        self.assertEqual(c.dtype, dt.int64)
        self.assertEqual(list(c), [1, 1])

        # float
        c = ta.column([1.0, 2.0], device=self.device)
        self.assertEqual(c.dtype, dt.float32)
        self.assertEqual(list(c), [1.0, 2.0])
        c = ta.column([1, 2.0], device=self.device)
        self.assertEqual(c.dtype, dt.float32)
        self.assertEqual(list(c), [1.0, 2.0])
        c = ta.column([np.float64(1.0), 2.0], device=self.device)
        self.assertEqual(c.dtype, dt.float64)
        self.assertEqual(list(c), [1.0, 2.0])
        c = ta.column([np.float64(1.0), np.float32(2.0)], device=self.device)
        self.assertEqual(c.dtype, dt.float64)
        self.assertEqual(list(c), [1.0, 2.0])

    def base_test_map_where_filter(self):
        col = ta.column(
            [None] * 3 + [3, 4, 5], dtype=dt.Int64(nullable=True), device=self.device
        )

        # Values that are not found in the dict are converted to None
        self.assertEqual(list(col.map({3: 33})), [None, None, None, 33, None, None])

        # maps None
        self.assertEqual(
            list(col.map({None: 1, 3: 33})),
            [1, 1, 1, 33, None, None],
        )

        # propagates None
        self.assertEqual(
            list(col.map({None: 1, 3: 33}, na_action="ignore")),
            [None, None, None, 33, None, None],
        )

        # maps as function
        self.assertEqual(
            list(
                col.map(
                    lambda x: 1 if x is None else 33 if x == 3 else x,
                )
            ),
            [1, 1, 1, 33, 4, 5],
        )

        left = ta.column([0] * 6, device=self.device)
        right = ta.column([99] * 6, device=self.device)
        self.assertEqual(
            list(ta.if_else(col > 3, left, right)),
            [None, None, None, 99, 0, 0],
        )

        # filter
        self.assertEqual(list(col.filter([True, False] * 3)), [None, None, 4])

    @staticmethod
    def _accumulate(col, val):
        if len(col) == 0:
            col._append(val)
        else:
            col._append(col[-1] + val)
        return col

    @staticmethod
    def _finalize(col):
        return col._finalize()

    def base_test_reduce(self):
        c = ta.column([1, 2, 3], device=self.device)
        d = c.reduce(
            fun=TestNumericalColumn._accumulate,
            initializer=Scope._EmptyColumn(dt.int64, device=self.device),
            finalizer=TestNumericalColumn._finalize,
        )
        self.assertEqual(list(d), [1, 3, 6])

    def base_test_sort_stuff(self):
        col = ta.column([2, 1, 3], device=self.device)

        self.assertEqual(list(col.sort()), [1, 2, 3])
        self.assertEqual(list(col.sort(ascending=False)), [3, 2, 1])
        self.assertEqual(
            list(ta.column([None, 1, 5, 2], device=self.device).sort()), [1, 2, 5, None]
        )
        self.assertEqual(
            list(
                ta.column([None, 1, 5, 2], device=self.device).sort(na_position="first")
            ),
            [None, 1, 2, 5],
        )
        self.assertEqual(
            list(
                ta.column([None, 1, 5, 2], device=self.device).sort(na_position="last")
            ),
            [1, 2, 5, None],
        )

        self.assertEqual(
            list(
                ta.column([None, 1, 5, 2], device=self.device).sort(na_position="last")
            ),
            [1, 2, 5, None],
        )

        # self.assertEqual(
        #     list(ta.column([None, 1, 5, 2]).nlargest(n=2, keep="first")), [5, 2] # TODO zhongxu
        # )
        """
        self.assertEqual(
            list(
                ta.column([None, 1, 5, 2], device=self.device).nsmallest(
                    n=2, keep="last"
                )
            ),
            [1, 2],
        )
        """

    def base_test_operators(self):
        # without None
        c = ta.column([0, 1, 3], device=self.device)
        d = ta.column([5, 5, 6], device=self.device)
        e = ta.column([1.0, 1, 7], device=self.device)

        # ==, !=

        self.assertEqual(list(c == c), [True] * 3)
        self.assertEqual(list(c == d), [False] * 3)
        self.assertEqual(list(c != c), [False] * 3)
        self.assertEqual(list(c != d), [True] * 3)

        self.assertEqual(list(c == 1), [False, True, False])
        self.assertEqual(list(1 == c), [False, True, False])
        self.assertTrue(
            ((c == 1) == ta.column([False, True, False], device=self.device)).all()
        )
        self.assertTrue(
            ((1 == c) == ta.column([False, True, False], device=self.device)).all()
        )

        # validate comparing non-equal length columns fails
        with self.assertRaises(TypeError):
            assert c == c.append([None])

        # <, <=, >=, >

        self.assertEqual(list(c <= 2), [True, True, False])
        self.assertEqual(list(c <= e), [True, True, True])
        self.assertEqual(list(c < 1), [True, False, False])
        self.assertEqual(list(c < d), [True, True, True])
        self.assertEqual(list(c >= 1), [False, True, True])
        self.assertEqual(list(c >= d), [False, False, False])
        self.assertEqual(list(c > 2), [False, False, True])
        self.assertEqual(list(c > d), [False, False, False])

        # +,-,*,/,//,**,%

        self.assertEqual(list(-c), [0, -1, -3])
        self.assertEqual(list(+-c), [0, -1, -3])

        self.assertEqual(list(c + 1), [1, 2, 4])
        self.assertEqual(list(e + 1), [2.0, 2.0, 8.0])

        # self.assertEqual(list(c.add(1)), [1, 2, 4])

        self.assertEqual(list(1 + c), [1, 2, 4])
        # self.assertEqual(list(c.radd(1)), [1, 2, 4])

        self.assertEqual(list(c + d), [5, 6, 9])
        # self.assertEqual(list(c.add(d)), [5, 6, 9])

        self.assertEqual(list(c + 1), [1, 2, 4])
        self.assertEqual(list(1 + c), [1, 2, 4])
        self.assertEqual(list(c + d), [5, 6, 9])

        self.assertEqual(list(c - 1), [-1, 0, 2])
        self.assertEqual(list(1 - c), [1, 0, -2])
        self.assertEqual(list(d - c), [5, 4, 3])

        self.assertEqual(list(c * 2), [0, 2, 6])
        self.assertEqual(list(2 * c), [0, 2, 6])
        self.assertEqual(list(c * d), [0, 5, 18])

        self.assertEqual(list(c / 2), [0.0, 0.5, 1.5])
        res = list(c / 0)
        self.assertTrue(isnan(res[0]))
        self.assertEqual(res[1:], [float("inf"), float("inf")])
        res = list(-c / 0)
        self.assertTrue(isnan(res[0]))
        self.assertEqual(res[1:], [float("-inf"), float("-inf")])
        self.assertEqual(list(2 / c), [float("inf"), 2.0, 0.6666666865348816])
        self.assertEqual(list(c / d), [0.0, 0.20000000298023224, 0.5])

        self.assertEqual(list(d // 2), [2, 2, 3])
        self.assertEqual(list(2 // d), [0, 0, 0])
        self.assertEqual(list(c // d), [0, 0, 0])
        self.assertEqual(list(e // d), [0.0, 0.0, 1.0])
        self.assertEqual(list(d // e), [5.0, 5.0, 0.0])
        # integer floordiv 0 -> exception
        with self.assertRaises(ZeroDivisionError):
            list(c // 0)
        # float floordiv 0 -> inf
        e = ta.column([1.0, -1.0], device=self.device)
        self.assertTrue(list(e // 0), [float("inf"), float("-inf")])
        # float 0.0 floordiv 0 -> nan
        e = ta.column([0.0], device=self.device)
        self.assertTrue(isnan(list(e // 0)[0]))

        self.assertEqual(list(c ** 2), [0, 1.0, 9.0])
        self.assertEqual(list(c ** 2.0), [0, 1.0, 9.0])
        self.assertEqual(list(c ** -2.0), [float("inf"), 1.0, 0.1111111119389534])
        self.assertEqual(list(c ** 0), [1.0, 1.0, 1.0])
        self.assertEqual(list(2 ** c), [1, 2, 8])
        self.assertEqual(list(c ** d), [0, 1, 729])
        with self.assertRaises(Exception) as ex:
            list(c ** -2)
        self.assertTrue(
            "Integers to negative integer powers are not allowed" in str(ex.exception)
        )
        e = ta.column([999999], device=self.device)
        with self.assertRaises(Exception) as ex:
            list(c ** e)
        self.assertTrue(
            "Inf is outside the range of representable values of type int64",
            str(ex.exception),
        )

        self.assertEqual(list(d % 2), [1, 1, 0])
        self.assertEqual(list(2 % d), [2, 2, 2])
        self.assertEqual(list(c % d), [0, 1, 3])
        e = ta.column([13, -13, 13, -13], device=self.device)
        f = ta.column([3, 3, -3, -3], device=self.device)
        self.assertEqual(list(e % f), [1, 2, -2, -1])
        # integer mod 0 -> exception
        with self.assertRaises(ZeroDivisionError):
            list(c % 0)
        # float mod 0 -> nan
        e = ta.column([1.0], device=self.device)
        self.assertTrue(isnan(list(e % 0)[0]))

        # TODO: Decide ...null handling.., bring back or ignore

        # c = ta.column([0, 1, 3, None])
        # self.assertEqual(list(c.add(1)), [1, 2, 4, None])

        # self.assertEqual(list(c.add(1, fill_value=17)), [1, 2, 4, 18])
        # self.assertEqual(list(c.radd(1, fill_value=-1)), [1, 2, 4, 0])
        # f = ta.column([None, 1, 3, None])
        # self.assertEqual(list(c.radd(f, fill_value=100)), [100, 2, 6, 200])

        # &, |, ^, ~
        g = ta.column([True, False, True, False], device=self.device)
        h = ta.column([False, False, True, True], device=self.device)
        self.assertEqual(list(g & h), [False, False, True, False])
        self.assertEqual(list(g | h), [True, False, True, True])
        self.assertEqual(list(g ^ h), [True, False, False, True])
        self.assertEqual(list(True & g), [True, False, True, False])
        self.assertEqual(list(True | g), [True, True, True, True])
        self.assertEqual(list(True ^ g), [False, True, False, True])
        self.assertEqual(list(~g), [False, True, False, True])

        i = ta.column([1, 2, 0], device=self.device)
        j = ta.column([3, 2, 3], device=self.device)
        self.assertEqual(list(i & j), [1, 2, 0])
        self.assertEqual(list(i | j), [3, 2, 3])
        self.assertEqual(list(i ^ j), [2, 0, 3])
        self.assertEqual(list(2 & i), [0, 2, 0])
        self.assertEqual(list(2 | i), [3, 2, 2])
        self.assertEqual(list(2 ^ i), [3, 0, 2])
        self.assertEqual(list(~i), [-2, -3, -1])

    # TODO Test type promotion rules

    def base_test_na_handling(self):
        c = ta.column([None, 2, 17.0], device=self.device)

        self.assertEqual(list(c.fill_null(99.0)), [99.0, 2, 17.0])
        self.assertEqual(c.fill_null(99.0).dtype, dt.float32)
        self.assertEqual(list(c.drop_null()), [2.0, 17.0])

        c = c.append([2])
        self.assertEqual(set(c.drop_duplicates()), {None, 2, 17.0})

    def base_test_agg_handling(self):
        import functools
        import operator

        c = [1, 4, 2, 7, 9, 1]
        D = ta.column(c, device=self.device)
        C = ta.column(c + [None], device=self.device)

        self.assertEqual(C.dtype, dt.Int64(nullable=True))
        self.assertEqual(C.min(), min(c))
        self.assertEqual(C.max(), max(c))
        self.assertEqual(C.sum(), sum(c))
        self.assertEqual(C.mode(), statistics.mode(c))

        self.assertEqual(D.std(), (statistics.stdev((float(i) for i in c))))

        self.assertEqual(C.std(), (statistics.stdev((float(i) for i in c))))
        self.assertEqual(C.mean(), statistics.mean(c))
        self.assertEqual(C.median(), statistics.median(c))

        self.assertEqual(
            list(C._cummin()), [min(c[:i]) for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C._cummax()), [max(c[:i]) for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C.cumsum()), [sum(c[:i]) for i in range(1, len(c) + 1)] + [None]
        )
        self.assertEqual(
            list(C._cumprod()),
            [functools.reduce(operator.mul, c[:i], 1) for i in range(1, len(c) + 1)]
            + [None],
        )
        self.assertEqual((C % 2 == 0)[:-1].all(), all(i % 2 == 0 for i in c))
        self.assertEqual((C % 2 == 0)[:-1].any(), any(i % 2 == 0 for i in c))

    def base_test_in_nunique(self):
        c = [1, 4, 2, 7]
        C = ta.column(c + [None])
        self.assertEqual(list(C.isin([1, 2, 3])), [True, False, True, False, False])
        C = C.append(c)
        d = set(c)
        d.add(None)
        # self.assertEqual(C.nunique(), len(set(C) - {None}))
        # self.assertEqual(C.nunique(drop_null=False), len(set(C)))

        self.assertEqual(C.is_unique, False)
        self.assertEqual(ta.column([1, 2, 3], device=self.device).is_unique, True)

        self.assertEqual(
            ta.column([1, 2, 3], device=self.device).is_monotonic_increasing, True
        )
        self.assertEqual(ta.column(dtype=dt.int64).is_monotonic_decreasing, True)

    def base_test_math_ops(self):
        c = [1.0, 4.2, 2, 7, -9, -2.5]
        C = ta.column(c + [None], device=self.device)
        self.assertEqual(C.dtype, dt.Float32(nullable=True))

        numpy.testing.assert_almost_equal(list(C.abs())[:-1], [abs(i) for i in c], 6)
        self.assertEqual(list(C.abs())[-1], None)

        self.assertEqual(list(C.ceil())[:-1], [ceil(i) for i in c])
        self.assertEqual(list(C.ceil())[-1], None)

        self.assertEqual(list(C.floor())[:-1], [floor(i) for i in c])
        self.assertEqual(list(C.floor())[-1], None)

        self.assertEqual(list(C.round()), list(np.array(c).round()) + [None])
        numpy.testing.assert_almost_equal(
            list(C.round(2))[:-1], list(np.array(c).round(2)), 6
        )
        self.assertEqual(list(C.round(2))[-1], None)
        c_round_list = [
            1.1,
            1.5,
            1.8,
            2.5,
            -1.1,
            -1.5,
            -1.8,
            -2.5,
            1.12,
            1.15,  # note: round(1.15, 1) is 1.1 in python, but 1.2 in Pandas/Numpy
            1.25,
            11.1,
            11.5,
            11.9,
        ]
        c_round = ta.column(c_round_list, device=self.device)
        self.assertEqual(list(c_round.round()), list(np.array(c_round_list).round()))
        for decimals in [-1, 1, 2]:
            numpy.testing.assert_almost_equal(
                list(c_round.round(decimals)),
                list(np.array(c_round_list).round(decimals)),
                6,
            )

        # self.assertEqual(list(C.hash_values()), [hash(i) for i in c] + [None])

        c1 = ta.column(
            [1, 0, 4, None], device=self.device, dtype=dt.Int32(nullable=True)
        )
        c2 = ta.column(
            [1, 0, 4, None], device=self.device, dtype=dt.Float32(nullable=True)
        )
        for col in [c1, c2]:
            numpy.testing.assert_almost_equal(
                list(col.log())[:-1], [0.0, -float("inf"), log(4)]
            )
            self.assertEqual(col.log().dtype, dt.Float32(nullable=True))
            self.assertEqual(list(col.log())[-1], None)

        c3 = ta.column(
            [1.0, 0.0, 4.0, None], device=self.device, dtype=dt.Float64(nullable=True)
        )
        numpy.testing.assert_almost_equal(
            list(c3.log())[:-1], [0.0, -float("inf"), log(4)]
        )
        self.assertEqual(c3.log().dtype, dt.Float64(nullable=True))
        self.assertEqual(list(c2.log())[-1], None)

    def base_test_describe(self):
        # requires 'implicitly' torcharrow.dataframe import dataframe
        c = ta.column([1, 2, 3], device=self.device)
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

    def helper_test_cast(
        self,
        from_type: dt.DType,
        to_type: dt.DType,
        data: ty.Iterable[ty.Union[int, float]],
        validation: ty.Callable = lambda x: x,
    ):
        # pyre-fixme[16]: `TestNumericalColumn` has no attribute `device`.
        col_from = ta.column(data, device=self.device, dtype=from_type)
        col_casted = col_from.cast(to_type)
        self.assertEqual(list(col_casted), [validation(d) for d in data])
        self.assertEqual(col_casted.dtype, to_type)

    def base_test_cast(self):
        data_bool = [True, False, True, False, False, False, True, True]
        data_int8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
        data_int8_nulls = [0, 1, 2, None, 4, 5, None, None, 8, None, 10]
        data_int16 = data_int8 + [200, 300, 20000]
        data_int16_nulls = [None, None] + data_int16
        data_int32 = data_int16 + [33000, 5000000]

        # note: Velox and Python do not agree on how to round a floating point
        #       number that is halfway between two integers. That is, given 4.5,
        #       Velox will convert that to the integer 5 during a cast. Python,
        #       using the round() function, will round that to 4. We need to be
        #       careful to not use such values in correctness tests.
        data_float = [0.0, 1.2, 2.3, 3.4, 4.51, 5.6, 6.7, 8.9, 9.0, 10.1]

        self.helper_test_cast(from_type=dt.boolean, to_type=dt.int8, data=data_bool)
        self.helper_test_cast(from_type=dt.boolean, to_type=dt.float64, data=data_bool)
        self.helper_test_cast(from_type=dt.int64, to_type=dt.int8, data=data_int8)
        self.helper_test_cast(from_type=dt.int64, to_type=dt.int16, data=data_int16)
        self.helper_test_cast(from_type=dt.int64, to_type=dt.int64, data=data_int32)
        self.helper_test_cast(from_type=dt.int64, to_type=dt.int32, data=data_int32)
        self.helper_test_cast(from_type=dt.int32, to_type=dt.int64, data=data_int32)
        self.helper_test_cast(from_type=dt.int8, to_type=dt.int64, data=data_int8)
        self.helper_test_cast(from_type=dt.int8, to_type=dt.int8, data=data_int8)
        self.helper_test_cast(
            from_type=dt.int64, to_type=dt.float64, data=data_int32, validation=float
        )
        self.helper_test_cast(
            from_type=dt.int32, to_type=dt.float32, data=data_int32, validation=float
        )

        # FIXME: why does this test fail in Velox? all casts TO booleans seem to fail,
        #        even when the original integer values are just 0 and 1.
        # self.helper_test_cast(from_type=dt.boolean, to_type=dt.boolean, data=data_bool)

        # note: round() instead of int(); simple conversion to ints does a floor
        self.helper_test_cast(
            from_type=dt.float32, to_type=dt.int32, data=data_float, validation=round
        )

        # non-nullable version of a type should be able to covert to a nullable version of the same type
        self.helper_test_cast(
            from_type=dt.int8, to_type=dt.Int8(nullable=True), data=data_int8
        )
        self.helper_test_cast(
            from_type=dt.int16, to_type=dt.Int16(nullable=True), data=data_int16
        )
        self.helper_test_cast(
            from_type=dt.int32, to_type=dt.Int32(nullable=True), data=data_int32
        )
        self.helper_test_cast(
            from_type=dt.int64, to_type=dt.Int64(nullable=True), data=data_int32
        )
        self.helper_test_cast(
            from_type=dt.float32, to_type=dt.Float32(nullable=True), data=data_int32
        )
        self.helper_test_cast(
            from_type=dt.float64, to_type=dt.Float64(nullable=True), data=data_int32
        )

        # int32 with nulls -> float64 with nulls
        def int_to_float_optional(x: ty.Optional[int]) -> ty.Optional[float]:
            if x is None:
                return None
            return float(x)

        self.helper_test_cast(
            from_type=dt.Int64(nullable=True),
            to_type=dt.Float64(nullable=True),
            data=data_int8_nulls,
            validation=int_to_float_optional,
        )

        with self.assertRaises(ValueError):
            # ValueError: Cannot cast a column with nulls to a non-nullable type
            self.helper_test_cast(
                from_type=dt.Int8(nullable=True), to_type=dt.int32, data=data_int8_nulls
            )

        with self.assertRaises(ValueError):
            # ValueError: Cannot cast a column with nulls to a non-nullable type
            self.helper_test_cast(
                from_type=dt.Int8(nullable=True), to_type=dt.int64, data=data_int8_nulls
            )

        with self.assertRaises(ValueError):
            # ValueError: Cannot cast a column with nulls to a non-nullable type
            self.helper_test_cast(
                from_type=dt.Int16(nullable=True),
                to_type=dt.int32,
                data=data_int16_nulls,
            )

        with self.assertRaises(ValueError):
            # ValueError: Cannot cast a column with nulls to a non-nullable type
            self.helper_test_cast(
                from_type=dt.Int16(nullable=True),
                to_type=dt.int64,
                data=data_int16_nulls,
            )

    def base_test_column_from_tuple(self):
        data_int = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        col_int = ta.column(data_int, device=self.device)
        self.assertEqual(tuple(col_int), data_int)

        data_float = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        col_float = ta.column(data_float, device=self.device)
        self.assertEqual(tuple(col_float), data_float)

    def base_test_column_from_numpy_array(self):
        seq_float = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        seq_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        seq_bool = [True, False, True, False, False, False, True, True, False]

        array_float32 = np.array(seq_float, dtype=np.float32)
        column_float32 = ta.column(array_float32, device=self.device)
        self.assertEqual(list(column_float32), seq_float)

        array_float64 = np.array(seq_float, dtype=np.float64)
        column_float64 = ta.column(array_float64, device=self.device)
        self.assertEqual(list(column_float64), seq_float)

        array_int32 = np.array(seq_int, dtype=np.int32)
        column_int32 = ta.column(array_int32, device=self.device)
        self.assertEqual(list(column_int32), seq_int)

        array_int64 = np.array(seq_int, dtype=np.int64)
        column_int64 = ta.column(array_int64, device=self.device)
        self.assertEqual(list(column_int64), seq_int)

        array_bool = np.array(seq_bool, dtype=np.bool)
        column_bool = ta.column(array_bool, device=self.device)
        self.assertEquals(list(column_bool), seq_bool)

    def base_test_append_automatic_conversions(self):
        # ints ARE converted to floats
        seq_float = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        column_float = ta.column(seq_float, device=self.device)
        column_float = column_float.append([11, 12])
        self.assertEqual(list(column_float), seq_float + [11.0, 12.0])

        # floats ARE NOT converted to ints
        seq_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        column_int = ta.column(seq_int, device=self.device)
        with self.assertRaises(TypeError):
            # TypeError: append(): incompatible function arguments.
            column_int.append([11.0, 12.0])

        # ints ARE converted to bools
        seq_bool = [True, False, True, False, False, True, True]
        column_bool = ta.column(seq_bool, device=self.device)
        column_bool = column_bool.append([1, 1, 0, 0])
        self.assertEqual(list(column_bool), seq_bool + [True, True, False, False])

        # floats ARE NOT converted to bools
        column_bool = ta.column(seq_bool, device=self.device)
        with self.assertRaises(TypeError):
            # TypeError: append(): incompatible function arguments.
            column_bool = column_bool.append([1.0, 1.0, 0.0, 0.0])

    # experimental
    def base_test_batch_collate(self):
        c = ta.column([1, 2, 3, 4, 5, 6, 7], device=self.device)
        # test iter
        it = c.batch(2)
        res = []
        for i in it:
            res.append(list(i))
        self.assertEqual(res, [[1, 2], [3, 4], [5, 6], [7]])
        # test collate
        it = c.batch(2)
        self.assertEqual(list(Column.unbatch(it)), [1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":

    unittest.main()
