# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from .test_arrow_interop import TestArrowInterop


class TestArrowInteropCpu(TestArrowInterop):
    def setUp(self):
        self.device = "cpu"

    def test_arrow_array(self):
        return self.base_test_arrow_array()

    def test_arrow_table(self):
        return self.base_test_arrow_table()

    def test_array_ownership_transferred(self):
        return self.base_test_array_ownership_transferred()

    def test_array_memory_reclaimed(self):
        return self.base_test_array_memory_reclaimed()

    def test_array_unsupported_types(self):
        return self.base_test_array_unsupported_types()

    def test_table_ownership_transferred(self):
        return self.base_test_table_ownership_transferred()

    def test_table_memory_reclaimed(self):
        return self.base_test_table_memory_reclaimed()

    def test_table_unsupported_types(self):
        return self.base_test_table_unsupported_types()

    def test_nullability(self):
        return self.base_test_nullability()


if __name__ == "__main__":
    unittest.main()
