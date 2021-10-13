# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import numpy as np
import pyarrow as pa
import torcharrow.dtypes as dt
from torcharrow.scope import Scope

from .test_interop import TestInterop


class TestInteropCpu(TestInterop):
    def setUp(self):
        self.ts = Scope({"device": "cpu"})

    def test_arrow_array(self):
        # TODO: support arrow interop in CPU backend
        pass


if __name__ == "__main__":
    unittest.main()
