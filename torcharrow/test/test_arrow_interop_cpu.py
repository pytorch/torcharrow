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

    def test_unsupported_types(self):
        return self.base_test_unsupported_types()

    def test_nullability(self):
        return self.base_test_nullability()


if __name__ == "__main__":
    unittest.main()
