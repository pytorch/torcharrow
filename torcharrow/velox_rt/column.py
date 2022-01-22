# Copyright (c) Facebook, Inc. and its affiliates.
import typing as ty

import torcharrow._torcharrow as velox
from torcharrow.dispatcher import Device
from torcharrow.dtypes import DType
from torcharrow.icolumn import IColumn
from torcharrow.scope import Scope

# TODO: Rename this class to IColumnVelox or IColumnCpu
class ColumnFromVelox:
    _data: velox.BaseColumn
    _finalized: bool

    @staticmethod
    def _from_velox(
        device: Device,
        dtype: DType,
        data: velox.BaseColumn,
        finalized: bool,
    ) -> IColumn:
        col = Scope._Column(dtype=dtype, device=device)
        col._data = data
        col._finalized = finalized
        return col

    # Velox column returned from generic dispatch always assumes returned column is nullable
    # This help method allows to alter it based on context (e.g. methods in IStringMethods can have better inference)
    def _with_null(self, nullable: bool):
        return self._from_velox(
            self.device, self.dtype.with_null(nullable), self._data, True
        )

    def _concat_with(self, columns: ty.List[IColumn]):
        concat_list = self.to_pylist()
        for column in columns:
            concat_list += column.to_pylist()
        return Scope._FromPyList(concat_list, self.dtype)
