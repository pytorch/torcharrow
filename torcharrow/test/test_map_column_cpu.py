# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

    def test_get_operator(self):
        self.base_test_get_operator()

    def test_slice_operation(self):
        self.base_test_slice_operation()

    def test_equality_operators(self):
        self.base_test_equality_operators()


if __name__ == "__main__":
    unittest.main()
