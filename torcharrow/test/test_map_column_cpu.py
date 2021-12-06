# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

from .test_map_column import TestMapColumn


class TestMapColumnCpu(TestMapColumn):
    def setUp(self):
        self.device = "cpu"

    def test_map(self):
        self.base_test_map()

    def test_infer(self):
        self.base_test_infer()

    def test_keys_values_get(self):
        self.base_test_keys_values_get()


if __name__ == "__main__":
    unittest.main()
