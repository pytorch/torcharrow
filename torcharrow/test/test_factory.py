# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt

# Test torcharrow.DataFrame and torcharrow.Column
# TODO: Add tests for CPU <-> GPU column interop once GPU device is supported
class TestFactory(unittest.TestCase):
    def test_column_cast(self):
        data = [1, 2, 3]
        col_int64 = ta.Column(data, device="cpu")
        self.assertEqual(list(col_int64), data)
        self.assertEqual(col_int64.dtype, dt.int64)

        col_int32 = ta.Column(col_int64, dtype=dt.int32, device="cpu")
        self.assertEqual(list(col_int32), data)
        self.assertEqual(col_int32.dtype, dt.int32)

        col_int16 = ta.Column(col_int64, dtype=dt.int16, device="cpu")
        self.assertEqual(list(col_int16), data)
        self.assertEqual(col_int16.dtype, dt.int16)

        col_int8 = ta.Column(col_int64, dtype=dt.int8, device="cpu")
        self.assertEqual(list(col_int8), data)
        self.assertEqual(col_int8.dtype, dt.int8)

        col_float32 = ta.Column(col_int64, dtype=dt.float32, device="cpu")
        self.assertEqual(list(col_float32), [1.0, 2.0, 3.0])
        self.assertEqual(col_float32.dtype, dt.float32)

        # TODO: Support more cast, such as int <-> string, float -> int

    def test_dataframe_cast(self):
        data = {
            "list_null": [[1, 2], [3, None], [4, 5], [6]],
            "ids": [[1, 2], [3], [1, 4], [5]],
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
            "c": ["a", "b", "c", "d"],
            # TODO support nested map in torcharrow.DataFrame factory
            # "nest": {"n1": [1, 2, 3, None], "n2": [4, 5, 6, None]},
        }
        dtype = dt.Struct(
            [
                dt.Field("list_null", dt.List(dt.Int64(nullable=True))),
                dt.Field("ids", dt.List(dt.int64)),
                dt.Field("a", dt.int64),
                dt.Field("b", dt.int64),
                dt.Field("c", dt.string),
            ],
        )

        df = ta.DataFrame(data, device="cpu")
        self.assertEqual(df.dtype, dtype)
        self.assertEqual(
            list(df),
            [
                ([1, 2], [1, 2], 1, 10, "a"),
                ([3, None], [3], 2, 20, "b"),
                ([4, 5], [1, 4], 3, 30, "c"),
                ([6], [5], 4, 40, "d"),
            ],
        )

        # call torcharrow.DataFrame with type cast
        casted_fields = list(dtype.fields)
        casted_fields[2] = dt.Field("a", dt.float64)
        casted_fields[3] = dt.Field("b", dt.int32)
        casted_dtype = dt.Struct(casted_fields)

        casted_eager_df = ta.DataFrame(df, dtype=casted_dtype, device="cpu")
        self.assertEqual(casted_eager_df.dtype, casted_eager_df.dtype)
        self.assertEqual(
            list(casted_eager_df),
            [
                ([1, 2], [1, 2], 1.0, 10, "a"),
                ([3, None], [3], 2.0, 20, "b"),
                ([4, 5], [1, 4], 3.0, 30, "c"),
                ([6], [5], 4.0, 40, "d"),
            ],
        )
