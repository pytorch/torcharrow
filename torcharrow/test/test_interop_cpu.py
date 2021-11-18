# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import numpy as np
import pyarrow as pa
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
from torcharrow.scope import Scope

from .test_interop import TestInterop


class TestInteropCpu(TestInterop):
    def setUp(self):
        self.device = "cpu"

    def test_arrow_array(self):
        # TODO: support arrow interop in CPU backend
        pass

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_to_pytorch(self):
        return self.base_test_to_pytorch()

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_pytorch_transform(self):
        return self.base_test_pytorch_transform()


if __name__ == "__main__":
    unittest.main()
