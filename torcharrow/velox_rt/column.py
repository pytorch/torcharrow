# Copyright (c) Facebook, Inc. and its affiliates.
import typing as ty

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
import torcharrow._torcharrow as velox
from torcharrow.dispatcher import Device
from torcharrow.dtypes import DType
from torcharrow.icolumn import IColumn
from torcharrow.scope import Scope

# TODO: Rename this class to IColumnVelox or IColumnCpu
# pyre-fixme[13]: Attribute `_finalized` is never initialized.
class ColumnFromVelox:
    # pyre-fixme[11]: Annotation `BaseColumn` is not defined as a type.
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
            # pyre-fixme[16]: `ColumnFromVelox` has no attribute `device`.
            # pyre-fixme[16]: `ColumnFromVelox` has no attribute `dtype`.
            self.device,
            # pyre-fixme[16]: `ColumnFromVelox` has no attribute `dtype`.
            self.dtype.with_null(nullable),
            self._data,
            True,
        )

    def _concat_with(self, columns: ty.List[IColumn]):
        # pyre-fixme[16]: `ColumnFromVelox` has no attribute `to_pylist`.
        concat_list = self.to_pylist()
        for column in columns:
            concat_list += column.to_pylist()
        # pyre-fixme[16]: `ColumnFromVelox` has no attribute `dtype`.
        return Scope._FromPySequence(concat_list, self.dtype)
