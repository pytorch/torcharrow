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

    def test_indexing(self):
        self.base_test_indexing()

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

    def test_is_unique(self):
        self.base_test_is_unique()

    def test_is_monotonic_increasing(self):
        self.base_test_is_monotonic_increasing()

    def test_is_monotonic_decreasing(self):
        self.base_test_is_monotonic_decreasing()

    def test_if_else(self):
        self.base_test_if_else()

    def test_repr(self):
        self.base_test_repr()

    def test_str(self):
        self.base_test_str()

    def test_is_valid_at(self):
        self.base_test_is_valid_at()

    def test_cast(self):
        self.base_test_cast()

    def test_drop_null(self):
        self.base_test_drop_null()

    def test_drop_duplicates(self):
        self.base_test_drop_duplicates()

    def test_fill_null(self):
        self.base_test_fill_null()

    def test_isin(self):
        self.base_test_isin()

    def test_bool(self):
        self.base_test_bool()

    def test_flatmap(self):
        self.base_test_flatmap()

    def test_any(self):
        self.base_test_any()

    def test_all(self):
        self.base_test_all()


if __name__ == "__main__":
    unittest.main()
