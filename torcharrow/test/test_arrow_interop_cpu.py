# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from .test_arrow_interop import TestArrowInterop


class TestArrowInteropCpu(TestArrowInterop):
    def setUp(self):
        self.device = "cpu"

    def test_arrow_array(self):
        return self.base_test_arrow_array()

    def test_ownership_transferred(self):
        return self.base_test_ownership_transferred()

    def test_memory_reclaimed(self):
        return self.base_test_memory_reclaimed()


if __name__ == "__main__":
    unittest.main()
