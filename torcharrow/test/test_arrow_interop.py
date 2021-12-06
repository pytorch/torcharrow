# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest
from typing import List, Tuple

import pyarrow as pa
import torcharrow as ta
import torcharrow.dtypes as dt


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
        pa.struct([pa.field("f1", pa.int64())]),
        pa.dictionary(pa.int64(), pa.int64()),
        # Union type needs to be tested differently
    )

    def _test_construction_numeric(
        self, pydata: List, arrow_type: pa.DataType, expected_dtype: dt.DType
    ):
        s = pa.array(pydata, type=arrow_type)
        t = ta.from_arrow(s, device=self.device)
        expected_dtype = expected_dtype.with_null(nullable=s.null_count > 0)
        self.assertEqual(t.dtype, expected_dtype)
        for pa_val, ta_val in zip([i.as_py() for i in s], list(t)):
            if pa_val and math.isnan(pa_val) and ta_val and math.isnan(ta_val):
                pass
            else:
                self.assertEqual(pa_val, ta_val)

    def base_test_arrow_array(self):
        s = pa.array([True, True, False, None, False])
        t = ta.from_arrow(s, device=self.device)
        self.assertEqual(t.dtype, dt.Boolean(True))
        self.assertEqual([i.as_py() for i in s], list(t))

        self.assertEqual(pa.utf8(), pa.string())
        self.assertEqual(pa.large_utf8(), pa.large_string())

        pydata_int = [1, 2, 3, None, 5, None]
        pydata_float = [1.0, math.nan, 3, None, 5.0, None]
        pydata_string = ["a", "b", None, "d", None, "f", "g"]
        for (arrow_type, expected_dtype) in TestArrowInterop.supported_types:
            if pa.types.is_integer(arrow_type):
                self._test_construction_numeric(pydata_int, arrow_type, expected_dtype)
            elif pa.types.is_floating(arrow_type):
                self._test_construction_numeric(
                    pydata_float, arrow_type, expected_dtype
                )
            elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                s = pa.array(pydata_string, type=arrow_type)
                t = ta.from_arrow(s, device=self.device)
                dt.replace(expected_dtype, nullable=False)
                self.assertEqual(t.dtype, expected_dtype)
                self.assertEqual([i.as_py() for i in s], list(t))

    def base_test_ownership_transferred(self):
        pydata = [1, 2, 3]
        s = pa.array(pydata)
        t = ta.from_arrow(s)
        del s
        # Check that the data are still around
        self.assertEqual(pydata, list(t))

    def base_test_memory_reclaimed(self):
        initial_memory = pa.total_allocated_bytes()

        s = pa.array([1, 2, 3])
        memory_checkpoint_1 = pa.total_allocated_bytes()
        self.assertGreater(memory_checkpoint_1, initial_memory)

        # Extra memory are allocated when exporting Arrow data, for the new
        # ArrowArray and ArrowSchema objects and the private_data inside the
        # objects
        t = ta.from_arrow(s)
        memory_checkpoint_2 = pa.total_allocated_bytes()
        self.assertGreater(memory_checkpoint_2, memory_checkpoint_1)

        # Deleting the pyarrow array should not release any memory since the
        # buffers are now owned by the TA column
        del s
        self.assertEqual(pa.total_allocated_bytes(), memory_checkpoint_2)

        del t
        self.assertEqual(pa.total_allocated_bytes(), initial_memory)

    def base_test_unsupported_types(self):
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
