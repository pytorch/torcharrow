# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest
from typing import List, Tuple

import pyarrow as pa
import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow._interop import _arrowtype_to_dtype, _dtype_to_arrowtype
from torcharrow.idataframe import IDataFrame


class TestArrowInterop(unittest.TestCase):
    supported_types: Tuple[Tuple[pa.DataType, dt.DType], ...] = (
        (pa.bool_(), dt.Boolean(True)),
        (pa.int8(), dt.Int8(True)),
        (pa.int16(), dt.Int16(True)),
        (pa.int32(), dt.Int32(True)),
        (pa.int64(), dt.Int64(True)),
        (pa.float32(), dt.Float32(True)),
        (pa.float64(), dt.Float64(True)),
        (pa.string(), dt.String(True)),
        (pa.large_string(), dt.String(True)),
    )

    unsupported_types: Tuple[pa.DataType, ...] = (
        pa.null(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.time32("s"),
        pa.time64("us"),
        pa.timestamp("s"),
        pa.date32(),
        pa.date64(),
        pa.duration("s"),
        pa.float16(),
        pa.binary(),
        pa.large_binary(),
        pa.decimal128(38),
        pa.list_(pa.int64()),
        pa.large_list(pa.int64()),
        pa.map_(pa.int64(), pa.int64()),
        pa.dictionary(pa.int64(), pa.int64()),
        # Union type needs to be tested differently
    )

    def _test_from_arrow_array_numeric(
        self, pydata: List, arrow_type: pa.DataType, expected_dtype: dt.DType
    ):
        s = pa.array(pydata, type=arrow_type)
        t = ta.from_arrow(s, device=self.device)
        self.assertFalse(isinstance(t, IDataFrame))
        expected_dtype = expected_dtype.with_null(nullable=s.null_count > 0)
        self.assertEqual(t.dtype, expected_dtype)
        for pa_val, ta_val in zip(s.to_pylist(), list(t)):
            if pa_val and math.isnan(pa_val) and ta_val and math.isnan(ta_val):
                pass
            else:
                self.assertEqual(pa_val, ta_val)

    def base_test_from_arrow_array_no_null_value_but_null_buffer(self):
        # Construct an arrow array that:
        #   * Doesn't contain null value
        #   * But contains null buffer (all marked as not null)
        data = pa.py_buffer(b"\x01\x02\x03\x04\x05\x06\x07\x08")
        valid = pa.py_buffer(b"\xff")
        arrow_array = pa.Array.from_buffers(pa.int8(), 8, [valid, data])

        self.assertEqual(arrow_array.to_pylist(), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(arrow_array.null_count, 0)
        self.assertIsNotNone(arrow_array.buffers()[0])

        column = ta.from_arrow(arrow_array)
        self.assertEqual(list(column), [1, 2, 3, 4, 5, 6, 7, 8])

    def base_test_from_arrow_array_boolean(self):
        pydata = [True, True, False, None, False]
        for (arrow_type, expected_dtype) in TestArrowInterop.supported_types:
            if pa.types.is_boolean(arrow_type):
                s = pa.array(pydata, type=arrow_type)
                t = ta.from_arrow(s, device=self.device)
                self.assertFalse(isinstance(t, IDataFrame))
                expected_dtype = expected_dtype.with_null(nullable=s.null_count > 0)
                self.assertEqual(t.dtype, expected_dtype)
                self.assertEqual(s.to_pylist(), list(t))

    def base_test_from_arrow_array_integer(self):
        pydata = [1, 2, 3, None, 5, None]
        for (arrow_type, expected_dtype) in TestArrowInterop.supported_types:
            if pa.types.is_integer(arrow_type):
                self._test_from_arrow_array_numeric(pydata, arrow_type, expected_dtype)

    def base_test_from_arrow_array_float(self):
        pydata = [1.0, math.nan, 3, None, 5.0, None]
        for (arrow_type, expected_dtype) in TestArrowInterop.supported_types:
            if pa.types.is_floating(arrow_type):
                self._test_from_arrow_array_numeric(pydata, arrow_type, expected_dtype)

    def base_test_from_arrow_array_string(self):
        self.assertEqual(pa.utf8(), pa.string())
        self.assertEqual(pa.large_utf8(), pa.large_string())

        pydata = ["a", "b", None, "d", None, "f", "g"]
        for (arrow_type, expected_dtype) in TestArrowInterop.supported_types:
            if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                s = pa.array(pydata, type=arrow_type)
                t = ta.from_arrow(s, device=self.device)
                self.assertFalse(isinstance(t, IDataFrame))
                expected_dtype = expected_dtype.with_null(nullable=s.null_count > 0)
                self.assertEqual(t.dtype, expected_dtype)
                self.assertEqual(s.to_pylist(), list(t))

    def base_test_from_arrow_table(self):
        pt = pa.table(
            {
                "f1": pa.array([1, 2, 3], type=pa.int64()),
                "f2": pa.array(["foo", "bar", None], type=pa.string()),
                "f3": pa.array([3.0, 1, 2.4], type=pa.float32()),
            }
        )
        df = ta.from_arrow(pt, device=self.device)
        self.assertTrue(isinstance(df, IDataFrame))
        self.assertTrue(dt.is_struct(df.dtype))
        self.assertEqual(len(df), len(pt))
        for (i, ta_field) in enumerate(df.dtype.fields):
            pa_field = pt.schema.field(i)
            self.assertEqual(ta_field.name, pa_field.name)
            self.assertEqual(
                ta_field.dtype, _arrowtype_to_dtype(pa_field.type, pa_field.nullable)
            )
            self.assertEqual(list(df[ta_field.name]), pt[i].to_pylist())

    def base_test_from_arrow_table_with_struct(self):
        label = pa.array([1, 0, 2], type=pa.int8())
        int_1 = pa.array([1, 2, 3])
        int_2 = pa.array([10, 20, None])
        dense_features = pa.StructArray.from_arrays(
            [int_1, int_2],
            fields=[
                pa.field("int_1", int_1.type, nullable=False),
                pa.field("int_2", int_2.type, nullable=True),
            ],
        )

        table = pa.Table.from_arrays(
            [label, dense_features],
            schema=pa.schema(
                [
                    pa.field("label", label.type, nullable=False),
                    pa.field("dense_features", dense_features.type, nullable=False),
                ]
            ),
        )
        df = ta.from_arrow(table)
        self.assertTrue(isinstance(df, IDataFrame))
        self.assertEqual(df.columns, ["label", "dense_features"])
        self.assertEqual(df["dense_features"].columns, ["int_1", "int_2"])
        self.assertEquals(df["dense_features"]["int_1"].dtype, dt.Int64(nullable=False))
        self.assertEquals(df["dense_features"]["int_2"].dtype, dt.Int64(nullable=True))

        self.assertEqual(list(df), [(1, (1, 10)), (0, (2, 20)), (2, (3, None))])

    def base_test_from_arrow_table_with_chunked_arrays(self):
        chunked = pa.chunked_array([[1, 2, 3], [4, 5, 6]])
        pt = pa.table({"f1": chunked})
        df = ta.from_arrow(pt, device=self.device)
        self.assertTrue(isinstance(df, IDataFrame))
        self.assertTrue(dt.is_struct(df.dtype))
        self.assertEqual(len(df), len(pt))
        for (i, ta_field) in enumerate(df.dtype.fields):
            pa_field = pt.schema.field(i)
            self.assertEqual(ta_field.name, pa_field.name)
            self.assertEqual(
                ta_field.dtype, _arrowtype_to_dtype(pa_field.type, pa_field.nullable)
            )
            self.assertEqual(list(df[ta_field.name]), pt[i].to_pylist())

    def _test_to_arrow_array_numeric(
        self, pydata: List, dtype: dt.DType, expected_arrowtype: pa.DataType
    ):
        t = ta.Column(pydata, dtype=dtype, device=self.device)
        s = t.to_arrow()
        self.assertTrue(isinstance(s, type(pa.array([], type=expected_arrowtype))))
        self.assertEqual(s.type, expected_arrowtype)
        for pa_val, ta_val in zip(s.to_pylist(), list(t)):
            if pa_val and math.isnan(pa_val) and ta_val and math.isnan(ta_val):
                pass
            else:
                self.assertEqual(pa_val, ta_val)

    def base_test_to_arrow_array_boolean(self):
        pydata = [True, True, False, None, False]
        for (expected_arrowtype, dtype) in TestArrowInterop.supported_types:
            if dt.is_boolean(dtype):
                t = ta.Column(pydata, dtype=dtype, device=self.device)
                s = t.to_arrow()
                self.assertTrue(
                    isinstance(s, type(pa.array([], type=expected_arrowtype)))
                )
                self.assertEqual(s.type, expected_arrowtype)
                self.assertEqual(s.to_pylist(), list(t))

    def base_test_to_arrow_array_integer(self):
        pydata = [1, 2, 3, None, 5, None]
        for (expected_arrowtype, dtype) in TestArrowInterop.supported_types:
            if dt.is_integer(dtype):
                self._test_to_arrow_array_numeric(pydata, dtype, expected_arrowtype)

    def base_test_to_arrow_array_float(self):
        pydata = [1.0, math.nan, 3, None, 5.0, None]
        for (expected_arrowtype, dtype) in TestArrowInterop.supported_types:
            if dt.is_floating(dtype):
                self._test_to_arrow_array_numeric(pydata, dtype, expected_arrowtype)

    def base_test_to_arrow_array_string(self):
        pydata = ["a", "b", None, "d", None, "f", "g"]
        for (expected_arrowtype, dtype) in TestArrowInterop.supported_types:
            if dt.is_string(dtype):
                t = ta.Column(pydata, dtype=dtype, device=self.device)
                s = t.to_arrow()
                pa.types.is_string(s.type)
                self.assertEqual(s.to_pylist(), list(t))

    def base_test_to_arrow_array_slice(self):
        # Only export the slice part but not the entire buffer when it's a slice
        pydata = [1, 2, 3, None, 5, None]
        t = ta.Column(pydata, device=self.device)
        t_slice = t[1:4]
        s = t_slice.to_arrow()
        self.assertEqual(len(s), len(t_slice))
        self.assertEqual(s.type, _dtype_to_arrowtype(t.dtype))
        self.assertEqual(s.to_pylist(), list(t_slice))
        self.assertEqual(s.is_valid().to_pylist(), [True, True, False])

    def base_test_to_arrow_table(self):
        df = ta.DataFrame(
            {
                "f1": [1, 2, 3],
                "f2": ["foo", "bar", None],
                "f3": [3.0, 1, 2.4],
            },
            device=self.device,
        )
        pt = df.to_arrow()
        self.assertTrue(isinstance(pt, pa.Table))
        self.assertTrue(dt.is_struct(df.dtype))
        self.assertEqual(len(df), len(pt))
        for (i, ta_field) in enumerate(df.dtype.fields):
            pa_field = pt.schema.field(i)
            self.assertEqual(ta_field.name, pa_field.name)
            self.assertEqual(
                ta_field.dtype, _arrowtype_to_dtype(pa_field.type, pa_field.nullable)
            )
            self.assertEqual(list(df[ta_field.name]), pt[i].to_pylist())

    def base_test_to_arrow_table_with_struct(self):
        df = ta.DataFrame(
            [
                (1, (10, 11)),
                (2, (20, 21)),
                (3, (30, None)),
            ],
            dtype=dt.Struct(
                [
                    dt.Field("labels", dt.int8),
                    dt.Field(
                        "dense_features",
                        dt.Struct(
                            [
                                dt.Field("int_1", dt.int32),
                                dt.Field("int_2", dt.Int32(nullable=True)),
                            ]
                        ),
                    ),
                ]
            ),
            device=self.device,
        )

        pt = df.to_arrow()
        self.assertTrue(isinstance(pt, pa.Table))
        self.assertEqual(len(df), len(pt))
        for (i, ta_field) in enumerate(df.dtype.fields):
            pa_field = pt.schema.field(i)
            self.assertEqual(ta_field.name, pa_field.name)
            self.assertEqual(
                ta_field.dtype, _arrowtype_to_dtype(pa_field.type, pa_field.nullable)
            )

        self.assertEqual(pt[0].to_pylist(), [1, 2, 3])
        self.assertEqual(
            pt[1].to_pylist(),
            [
                {"int_1": 10, "int_2": 11},
                {"int_1": 20, "int_2": 21},
                {"int_1": 30, "int_2": None},
            ],
        )

    def base_test_array_ownership_transferred(self):
        pydata = [1, 2, 3]
        s = pa.array(pydata)
        t = ta.from_arrow(s, device=self.device)
        del s
        # Check that the data are still around
        self.assertEqual(pydata, list(t))

    def base_test_array_memory_reclaimed(self):
        initial_memory = pa.total_allocated_bytes()

        s = pa.array([1, 2, 3])
        memory_checkpoint_1 = pa.total_allocated_bytes()
        self.assertGreater(memory_checkpoint_1, initial_memory)

        # Extra memory are allocated when exporting Arrow data, for the new
        # ArrowArray and ArrowSchema objects and the private_data inside the
        # objects
        t = ta.from_arrow(s, device=self.device)
        memory_checkpoint_2 = pa.total_allocated_bytes()
        self.assertGreater(memory_checkpoint_2, memory_checkpoint_1)

        # Deleting the pyarrow array should not release any memory since the
        # buffers are now owned by the TA column
        del s
        self.assertEqual(pa.total_allocated_bytes(), memory_checkpoint_2)

        del t
        self.assertLessEqual(pa.total_allocated_bytes(), initial_memory)

    def base_test_array_unsupported_types(self):
        for arrow_type in TestArrowInterop.unsupported_types:
            s = pa.array([], type=arrow_type)
            with self.assertRaises(RuntimeError) as ex:
                t = ta.from_arrow(s, device=self.device)
            self.assertTrue(
                f"Unsupported Arrow type: {str(arrow_type)}" in str(ex.exception)
            )

        union_array = pa.UnionArray.from_sparse(
            types=pa.array([0], type=pa.int8()), children=[pa.array([1])]
        )
        with self.assertRaises(RuntimeError) as ex:
            t = ta.from_arrow(union_array, device=self.device)
        self.assertTrue(
            f"Unsupported Arrow type: {str(union_array.type)}" in str(ex.exception)
        )

    def base_test_table_ownership_transferred(self):
        f1_pydata = [1, 2, 3]
        f2_pydata = ["foo", "bar", None]
        f3_pydata = [3.0, None, 2.0]
        pt = pa.table(
            {
                "f1": pa.array(f1_pydata, type=pa.int64()),
                "f2": pa.array(f2_pydata, type=pa.string()),
                "f3": pa.array(f3_pydata, type=pa.float32()),
            }
        )
        df = ta.from_arrow(pt, device=self.device)
        del pt
        # Check that the data are still around
        self.assertEqual(list(df["f1"]), f1_pydata)
        self.assertEqual(list(df["f2"]), f2_pydata)
        self.assertEqual(list(df["f3"]), f3_pydata)

    def base_test_table_memory_reclaimed(self):
        initial_memory = pa.total_allocated_bytes()

        pt = pa.table(
            {
                "f1": pa.array([1, 2, 3], type=pa.int64()),
                "f2": pa.array(["foo", "bar", None], type=pa.string()),
                "f3": pa.array([3.0, None, 2.4], type=pa.float64()),
            }
        )
        memory_checkpoint_1 = pa.total_allocated_bytes()
        self.assertGreater(memory_checkpoint_1, initial_memory)

        # Extra memory are allocated when exporting Arrow data, for the new
        # ArrowArray and ArrowSchema objects and the private_data inside the
        # objects
        df = ta.from_arrow(pt, device=self.device)
        memory_checkpoint_2 = pa.total_allocated_bytes()
        self.assertGreater(memory_checkpoint_2, memory_checkpoint_1)

        del pt
        del df
        self.assertEqual(pa.total_allocated_bytes(), initial_memory)

    def base_test_table_unsupported_types(self):
        pt = pa.table(
            {
                "f1": pa.array([1, 2, 3], type=pa.int64()),
                "f2": pa.array(["foo", "bar", None], type=pa.string()),
                "f3": pa.array([[1, 2], [3, 4, 5], [6]], type=pa.list_(pa.int8())),
            }
        )
        with self.assertRaises(RuntimeError) as ex:
            df = ta.from_arrow(pt, device=self.device)
        self.assertTrue(
            f"Unsupported Arrow type: {str(pt.field(2).type)}" in str(ex.exception)
        )

    def base_test_nullability(self):
        pydata = [1, 2, 3]
        s = pa.array(pydata)
        t = ta.from_arrow(s, device=self.device)
        self.assertFalse(t.dtype.nullable)

        pydata = [1, 2, 3]
        s = pa.array(pydata)
        t = ta.from_arrow(
            s, dtype=_arrowtype_to_dtype(s.type, nullable=True), device=self.device
        )
        self.assertTrue(t.dtype.nullable)

        pydata = [1, 2, 3, None]
        s = pa.array(pydata)
        t = ta.from_arrow(s, device=self.device)
        self.assertTrue(t.dtype.nullable)

        pydata = [1, 2, 3, None]
        s = pa.array(pydata)
        with self.assertRaises(ValueError) as ex:
            t = ta.from_arrow(
                s, dtype=_arrowtype_to_dtype(s.type, nullable=False), device=self.device
            )
        self.assertTrue(
            "Cannot store nulls in a non-nullable column" in str(ex.exception)
        )

        pt = pa.table(
            {"f1": [1, 2, 3, None], "f2": [4, 5, 6, 7]},
            schema=pa.schema(
                [
                    pa.field("f1", pa.int64(), nullable=True),
                    pa.field("f2", pa.float32(), nullable=False),
                ]
            ),
        )
        df = ta.from_arrow(pt, device=self.device)
        self.assertEqual(df["f1"].dtype.nullable, pt.field("f1").nullable)
        self.assertEqual(df["f2"].dtype.nullable, pt.field("f2").nullable)

        # Test that the `nullable` set in the `dtype` parameter of `from_arrow`
        # is respected over the nullable inferred from pa.Table
        pt = pa.table(
            {"f1": [1, 2, 3, None], "f2": [4, 5, 6, 7]},
            schema=pa.schema(
                [
                    pa.field("f1", pa.int64(), nullable=True),
                    pa.field("f2", pa.float32(), nullable=True),
                ]
            ),
        )
        f1_arrowtype = pt.field("f1").type
        f2_arrowtype = pt.field("f2").type
        df = ta.from_arrow(
            pt,
            dtype=dt.Struct(
                fields=[
                    dt.Field(name="f1", dtype=_arrowtype_to_dtype(f1_arrowtype, True)),
                    dt.Field(name="f2", dtype=_arrowtype_to_dtype(f2_arrowtype, False)),
                ]
            ),
            device=self.device,
        )
        self.assertTrue(df["f1"].dtype.nullable)
        self.assertFalse(df["f2"].dtype.nullable)

        pt = pa.table(
            {"f1": [1, 2, 3, None]},
            schema=pa.schema(
                [
                    pa.field("f1", pa.int64(), nullable=False),
                ]
            ),
        )
        with self.assertRaises(ValueError) as ex:
            df = ta.from_arrow(pt, device=self.device)
        self.assertTrue(
            "Cannot store nulls in a non-nullable column" in str(ex.exception)
        )
