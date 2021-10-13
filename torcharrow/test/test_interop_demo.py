# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import numpy as np
import pyarrow as pa
import torcharrow.dtypes as dt
from torcharrow.scope import Scope

from .test_interop import TestInterop


class TestInteropDemo(TestInterop):
    def setUp(self):
        self.ts = Scope({"device": "demo"})

    def test_arrow_array(self):
        return self.base_test_arrow_array()


if __name__ == "__main__":
    unittest.main()
