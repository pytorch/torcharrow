# Copyright (c) Facebook, Inc. and its affiliates.
import torcharrow._torcharrow as velox
from torcharrow import Scope
from torcharrow.dispatcher import Device
from torcharrow.dtypes import DType
from torcharrow.icolumn import IColumn


class ColumnFromVelox:
    _data: velox.BaseColumn
    _finialized: bool

    @staticmethod
    def from_velox(
        device: Device,
        dtype: DType,
        data: velox.BaseColumn,
        finialized: bool,
    ) -> IColumn:
        col = Scope._Column(dtype=dtype, device=device)
        col._data = data
        col._finialized = finialized
        return col

    # Velox column returned from generic dispatch always assumes returned column is nullable
    # This help method allows to alter it based on context (e.g. methods in IStringMethods can have better inference)
    def with_null(self, nullable: bool):
        return self.from_velox(
            self.device, self.dtype.with_null(nullable), self._data, True
        )
