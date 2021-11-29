# Copyright (c) Facebook, Inc. and its affiliates.
import math
import unittest

import pyarrow as pa
import torcharrow as ta
import torcharrow.dtypes as dt


class TestArrowInterop(unittest.TestCase):
    def base_test_arrow_array(self):
        s = pa.array([1, 2, 3])
        t = ta.from_arrow(s, device=self.device)
        self.assertEqual(t.dtype, dt.Int64(True))
        self.assertEqual([i.as_py() for i in s], list(t))

        s = pa.array([1.0, math.nan, 3])
        t = ta.from_arrow(s, device=self.device)
        self.assertEqual(t.dtype, dt.Float64(True))
        for i, j in zip([i.as_py() for i in s], list(t)):
            if math.isnan(i) and math.isnan(j):
                pass
            else:
                self.assertEqual(i, j)

        s = pa.array([1.0, None, 3])
        t = ta.from_arrow(s, device=self.device)
        self.assertEqual(t.dtype, dt.Float64(True))
        self.assertEqual([i.as_py() for i in s], list(t))

        s = pa.array([1, 2, 3], type=pa.int16())
        t = ta.from_arrow(s, dtype=dt.Int16(False), device=self.device)
        self.assertEqual(t.dtype, dt.Int16(False))
        self.assertEqual(
            [i.as_py() for i in s],
            list(t),
        )

        s = pa.array([True, False, True])
        t = ta.from_arrow(s, device=self.device)
        self.assertEqual(t.dtype, dt.Boolean(True))
        self.assertEqual(
            [i.as_py() for i in s], list(ta.from_arrow(s, device=self.device))
        )

        s = pa.array(["a", "b", "c", "d", "e", "f", "g"])
        t = ta.from_arrow(s, device=self.device)
        self.assertEqual(t.dtype, dt.String(True))
        self.assertEqual([i.as_py() for i in s], list(t))

        # TODO Test that nested types and other unsupported types are error-ed out

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
