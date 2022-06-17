# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from .test_list_column import TestListColumn


class TestListColumnCpu(TestListColumn):
    def setUp(self):
        self.device = "cpu"

    def test_empty(self):
        self.base_test_empty()

    def test_nonempty(self):
        self.base_test_nonempty()

    def test_list_with_none(self):
        self.base_test_list_with_none()

    def test_append_concat(self):
        return self.base_test_append_concat()

    def test_nested_numerical_twice(self):
        self.base_test_nested_numerical_twice()

    def test_nested_string_once(self):
        self.base_test_nested_string_once()

    def test_nested_string_twice(self):
        self.base_test_nested_string_twice()

    def test_get_count_join(self):
        self.base_test_get_count_join()

    def test_slice(self):
        self.base_test_slice()

    def test_map_reduce_etc(self):
        self.base_test_map_reduce_etc()

    def test_fixed_size_list(self):
        self.base_test_fixed_size_list()

    def test_column_from_dataframe_list(self):
        self.base_test_column_from_dataframe_list()


if __name__ == "__main__":
    unittest.main()
