# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
import warnings
from typing import List

import numpy as np
import torcharrow as ta
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow import Scope
from torcharrow.dispatcher import Dispatcher
from torcharrow.ilist_column import IListColumn, IListMethods

from .column import ColumnFromVelox
from .typing import get_velox_type

# -----------------------------------------------------------------------------
# IListColumn


class ListColumnCpu(ColumnFromVelox, IListColumn):

    # private constructor
    def __init__(self, device, dtype, data, offsets, mask):
        assert dt.is_list(dtype)
        IListColumn.__init__(self, device, dtype)

        self._data = velox.Column(
            velox.VeloxArrayType(get_velox_type(dtype.item_dtype))
        )
        if len(data) > 0:
            self.append(data)
        self._finialized = False

        self.list = ListMethodsCpu(self)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time
    @staticmethod
    def _empty(device, dtype: dt.List):
        return ListColumnCpu(
            device,
            dtype,
            Scope._EmptyColumn(dtype.item_dtype, device),
            ar.array("I", [0]),
            ar.array("b"),
        )

    @staticmethod
    def _fromlist(device: str, data: List[List], dtype: dt.List):
        if dt.is_primitive(dtype.item_dtype):
            velox_column = velox.Column(get_velox_type(dtype), data)
            return ColumnFromVelox.from_velox(
                device,
                dtype,
                velox_column,
                True,
            )
        else:
            warnings.warn(
                "Complex types are not supported (properly) for "
                "ListColumnCpu._fromlist yet. Falling back to the default "
                "(inefficient) implementation"
            )
            assert len(data) <= 100000, (
                "The default _fromlist implementation "
                f"will be too slow for {len(data)} elements"
            )
            col = ListColumnCpu._empty(device, dtype)
            for i in data:
                col._append(i)
            return col._finalize()

    def _append_null(self):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        elif value is None:
            self._data.append_null()
        else:
            new_element_column = ta.Column(self._dtype.item_dtype)
            new_element_column = new_element_column.append(value)
            self._data.append(new_element_column._data)

    def _finalize(self):
        self._finialized = True
        return self

    def __len__(self):
        return len(self._data)

    @property
    def null_count(self):
        return self._data.get_null_count()

    def _getmask(self, i):
        if i < 0:
            i += len(self._data)
        return self._data.is_null_at(i)

    def _getdata(self, i):
        if i < 0:
            i += len(self._data)
        if self._data.is_null_at(i):
            return self.dtype.default_value()
        else:
            return list(
                ColumnFromVelox.from_velox(
                    self.device,
                    self._dtype.item_dtype,
                    self._data[i],
                    False,
                )
            )

    # printing ----------------------------------------------------------------

    def __str__(self):
        return f"Column([{', '.join('None' if i is None else str(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else str(i)] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self._dtype}, length: {self.length}, null_count: {self.null_count}"
        return tab + dt.NL + typ

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(len(self)):
            item = self._get(i)
            if item is None:
                yield item
            else:
                yield list(item)

    # inerop
    def to_torch(self, _propagate_py_list=True):
        pytorch.ensure_available()
        import torch

        # TODO: more efficient/straightfowrad interop
        arrow_array = self.to_arrow()

        elements = ColumnFromVelox.from_velox(
            self.device, self._dtype.item_dtype, self._data.elements(), True
        ).to_torch()
        # special case: if the nested type is List (which happens for List[str] that can't be represented as tensor)
        # then we fallback to string types
        if isinstance(elements, list) and _propagate_py_list:
            return [
                (
                    elements[
                        arrow_array.offsets[i]
                        .as_py() : arrow_array.offsets[i + 1]
                        .as_py()
                    ]
                    if self[i] is not None
                    else None
                )
                for i in range(len(self))
            ]
        # TODO: clarify int32 vs int64
        offsets = torch.tensor(
            arrow_array.offsets.to_numpy(),
            dtype=torch.int32,
        )
        res = pytorch.PackedList(values=elements, offsets=offsets)
        if not self._dtype.nullable:
            return res

        presence = torch.tensor(
            arrow_array.is_valid().to_numpy(zero_copy_only=False), dtype=torch.bool
        )
        return pytorch.WithPresence(values=res, presence=presence)


# ------------------------------------------------------------------------------
# ListMethodsCpu


class ListMethodsCpu(IListMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: ListColumnCpu):
        super().__init__(parent)


# ------------------------------------------------------------------------------
# registering the factory
Dispatcher.register((dt.List.typecode + "_empty", "cpu"), ListColumnCpu._empty)
Dispatcher.register((dt.List.typecode + "_fromlist", "cpu"), ListColumnCpu._fromlist)
