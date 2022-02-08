# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
import warnings
from typing import List, Callable

import torcharrow as ta
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow.dispatcher import Dispatcher
from torcharrow.icolumn import IColumn
from torcharrow.ilist_column import IListColumn, IListMethods
from torcharrow.scope import Scope

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
        self._finalized = False

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
    def _from_pysequence(device: str, data: List[List], dtype: dt.List):
        if dt.is_primitive(dtype.item_dtype):
            velox_column = velox.Column(get_velox_type(dtype), data)
            return ColumnFromVelox._from_velox(
                device,
                dtype,
                velox_column,
                True,
            )
        else:
            warnings.warn(
                "Complex types are not supported (properly) for "
                "ListColumnCpu._from_pysequence yet. Falling back to the default "
                "(inefficient) implementation"
            )
            assert len(data) <= 100000, (
                "The default _from_pysequence implementation "
                f"will be too slow for {len(data)} elements"
            )
            col = ListColumnCpu._empty(device, dtype)
            for i in data:
                col._append(i)
            return col._finalize()

    def _append_null(self):
        if self._finalized:
            raise AttributeError("It is already finalized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finalized:
            raise AttributeError("It is already finalized.")
        elif value is None:
            self._data.append_null()
        else:
            new_element_column = ta.Column(self._dtype.item_dtype)
            new_element_column = new_element_column.append(value)
            self._data.append(new_element_column._data)

    def _finalize(self):
        self._finalized = True
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
                ColumnFromVelox._from_velox(
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
    def _to_tensor_default(self, _propagate_py_list=True):
        pytorch.ensure_available()
        import torch

        # TODO: more efficient/straightfowrad interop
        arrow_array = self.to_arrow()

        elements = ColumnFromVelox._from_velox(
            self.device, self._dtype.item_dtype, self._data.elements(), True
        )._to_tensor_default()
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

    def _to_tensor_pad_sequence(self, batch_first: bool, padding_value):
        pytorch.ensure_available()

        # TODO: pad_sequence also works for nest numeric list
        assert dt.is_numerical(self.dtype.item_dtype)
        assert not self.dtype.nullable

        import torch
        from torch.nn.utils.rnn import pad_sequence

        packed_list: pytorch.PackedList = self._to_tensor_default()

        unpad_tensors: List[torch.tensor] = [
            packed_list.values[packed_list.offsets[i] : packed_list.offsets[i + 1]]
            for i in range(len(self))
        ]

        pad_token_ids = pad_sequence(
            unpad_tensors,
            batch_first=batch_first,
            padding_value=float(padding_value),
        )

        return pad_token_ids


# ------------------------------------------------------------------------------
# ListMethodsCpu


class ListMethodsCpu(IListMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: ListColumnCpu):
        super().__init__(parent)

    def vmap(self, fun: Callable[[IColumn], IColumn]):
        elements = ColumnFromVelox._from_velox(
            self._parent.device,
            self._parent._dtype.item_dtype,
            self._parent._data.elements(),
            True,
        )
        new_elements = fun(elements)

        new_data = self._parent._data.withElements(new_elements._data)
        return ColumnFromVelox._from_velox(
            self._parent.device, new_data.dtype(), new_data, True
        )


# ------------------------------------------------------------------------------
# registering the factory
Dispatcher.register((dt.List.typecode + "_empty", "cpu"), ListColumnCpu._empty)
Dispatcher.register(
    (dt.List.typecode + "_from_pysequence", "cpu"), ListColumnCpu._from_pysequence
)
