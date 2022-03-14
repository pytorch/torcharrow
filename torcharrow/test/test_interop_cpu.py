# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

    def test_from_pysequence(self):
        return self.base_test_from_pysequence()

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_pytorch_transform(self):
        return self.base_test_pytorch_transform()

    @unittest.skipUnless(tap.available, "Requires PyTorch")
    def test_dense_features_no_mask(self):
        return self.base_test_dense_features_no_mask()


if __name__ == "__main__":
    unittest.main()
