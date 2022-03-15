# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing as ty
from abc import ABC

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
import torcharrow._torcharrow as velox
import torcharrow.interop
from torcharrow.dispatcher import Device
from torcharrow.dtypes import DType
from torcharrow.icolumn import Column
from torcharrow.scope import Scope


class ColumnCpuMixin(ABC):
    # pyre-fixme[11]: Annotation `BaseColumn` is not defined as a type.
    _data: velox.BaseColumn
    _finalized: bool

    @staticmethod
    def _from_velox(
        device: Device,
        dtype: DType,
        data: velox.BaseColumn,
        finalized: bool,
    ) -> Column:
        col = Scope._Column(dtype=dtype, device=device)
        col._data = data
        col._finalized = finalized
        return col

    def _gets(self, indices):
        arrow_array = self.to_arrow()
        project = arrow_array.take(indices)
        return torcharrow.interop.from_arrow(project, self.dtype, self.device)

    def _slice(self, start, stop, step):
        arrow_array = self.to_arrow()
        project = arrow_array[start:stop:step]
        # Velox library Bridge.cpp:importFromArrayImpl() fails:
        # "Offsets are not supported during arrow conversion yet."
        #
        # TODO: when Velox supports importing arrow data with offsets,
        # we can use the following zero copy line.
        #   return torcharrow.interop.from_arrow(z, self.dtype, self.device)
        # For now we flatten the data instead.
        flat = project.to_pylist()
        return Scope._Column(data=flat, dtype=self.dtype, device=self.device)

    # Velox column returned from generic dispatch always assumes returned column is nullable
    # This help method allows to alter it based on context (e.g. methods in StringMethods can have better inference)
    def _with_null(self, nullable: bool):
        return self._from_velox(
            # pyre-fixme[16]: `ColumnCpuMixin` has no attribute `device`.
            # pyre-fixme[16]: `ColumnCpuMixin` has no attribute `dtype`.
            self.device,
            # pyre-fixme[16]: `ColumnCpuMixin` has no attribute `dtype`.
            self.dtype.with_null(nullable),
            self._data,
            True,
        )

    def _concat_with(self, columns: ty.List[Column]):
        # pyre-fixme[16]: `ColumnCpuMixin` has no attribute `to_pylist`.
        concat_list = self.to_pylist()
        for column in columns:
            concat_list += column.to_pylist()
        # pyre-fixme[16]: `ColumnCpuMixin` has no attribute `dtype`.
        return Scope._FromPySequence(concat_list, self.dtype)
