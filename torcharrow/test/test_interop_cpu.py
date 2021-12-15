# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torcharrow.pytorch as tap

from .test_interop import TestInterop


class TestInteropCpu(TestInterop):
    def setUp(self):
        self.device = "cpu"

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_to_pytorch(self):
        return self.base_test_to_pytorch()

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_pad_sequence(self):
        return self.base_test_pad_sequence()

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_pytorch_transform(self):
        return self.base_test_pytorch_transform()


if __name__ == "__main__":
    unittest.main()
