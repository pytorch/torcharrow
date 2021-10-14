# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import numpy as np
import pyarrow as pa
import torcharrow.dtypes as dt
from torcharrow.scope import Scope


class TestInterop(unittest.TestCase):
    def setUp(self):
        self.ts = Scope()

    def base_test_arrow_array(self):
        s = pa.array([1, 2, 3])
        t = self.ts.from_arrow(s)
        self.assertEqual([i.as_py() for i in s], list(t))

        s = pa.array([1.0, np.nan, 3])
        t = self.ts.from_arrow(s)
        self.assertEqual(t.dtype, dt.Float64(False))
        # can't compare nan, so

        for i, j in zip([i.as_py() for i in s], list(t)):
            if np.isnan(i) and np.isnan(j):
                pass
            else:
                self.assertEqual(i, j)

        s = pa.array([1.0, None, 3])
        t = self.ts.from_arrow(s)
        self.assertEqual(t.dtype, dt.Float64(True))
        self.assertEqual([i.as_py() for i in s], list(t))

        s = pa.array([1, 2, 3], type=pa.uint32())
        self.assertEqual(
            [i.as_py() for i in s], list(self.ts.from_arrow(s, dt.Int16(False)))
        )

        s = pa.array([1, 2, 3])
        t = self.ts.from_arrow(s)
        self.assertEqual(t.dtype, dt.Int64(False))
        self.assertEqual([i.as_py() for i in s], list(self.ts.from_arrow(s)))

        s = pa.array([True, False, True])
        t = self.ts.from_arrow(s)
        self.assertEqual(t.dtype, dt.Boolean(False))
        self.assertEqual([i.as_py() for i in s], list(self.ts.from_arrow(s)))

        s = pa.array(["a", "b", "c", "d", "e", "f", "g"])
        t = self.ts.from_arrow(s)
        self.assertEqual(t.dtype, dt.String(False))
        self.assertEqual([i.as_py() for i in s], list(t))

    # TODO: migrate other tests from test_legacy_interop.py
