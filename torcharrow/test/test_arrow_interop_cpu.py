# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from .test_arrow_interop import TestArrowInterop


class TestArrowInteropCpu(TestArrowInterop):
    def setUp(self):
        self.device = "cpu"

    def test_from_arrow_array_boolean(self):
        return self.base_test_from_arrow_array_boolean()

    def test_from_arrow_array_integer(self):
        return self.base_test_from_arrow_array_integer()

    def test_from_arrow_array_float(self):
        return self.base_test_from_arrow_array_float()

    def test_from_arrow_array_string(self):
        return self.base_test_from_arrow_array_string()

    def test_from_arrow_table(self):
        return self.base_test_from_arrow_table()

    def test_from_arrow_table_with_chunked_arrays(self):
        return self.base_test_from_arrow_table_with_chunked_arrays()

    def test_to_arrow_array_boolean(self):
        return self.base_test_to_arrow_array_boolean

    def test_to_arrow_array_integer(self):
        return self.base_test_to_arrow_array_integer

    def test_to_arrow_array_float(self):
        return self.base_test_to_arrow_array_float

    def test_to_arrow_array_string(self):
        return self.base_test_to_arrow_array_string

    def test_to_arrow_array_slice(self):
        return self.base_test_to_arrow_array_slice

    def test_to_arrow_table(self):
        return self.base_test_to_arrow_table

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
