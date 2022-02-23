# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as ty
import unittest

import torcharrow as ta
import torcharrow.dtypes as dt

from .test_string_column import TestStringColumn


class TestStringColumnCpu(TestStringColumn):
    def setUp(self):
        self.device = "cpu"

    def create_column(
        self,
        data: ty.Union[ty.Iterable, dt.DType, None] = None,
        dtype: ty.Optional[dt.DType] = None,
    ):
        return ta.column(data, dtype, self.device)

    def test_empty(self):
        self.base_test_empty()

    def test_append_offsets(self):
        self.base_test_append_offsets()

    def test_string_column_from_tuple(self):
        self.base_test_string_column_from_tuple()

    def test_comparison(self):
        return self.base_test_comparison()

    def test_concat(self):
        return self.base_test_concat()

    def test_string_split_methods(self):
        self.base_test_string_split_methods()

    def test_string_categorization_methods(self):
        return self.base_test_string_categorization_methods()

    def test_string_lifted_methods(self):
        self.base_test_string_lifted_methods()

    def test_string_pattern_matching_methods(self):
        return self.base_test_string_pattern_matching_methods()

    def test_regular_expressions(self):
        self.base_test_regular_expressions()


if __name__ == "__main__":
    unittest.main()
