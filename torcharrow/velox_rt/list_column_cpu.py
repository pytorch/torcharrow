# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import array as ar
import warnings
from typing import Callable, List, Optional, Union

import torcharrow as ta
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow import functional
from torcharrow.dispatcher import Dispatcher
from torcharrow.icolumn import Column
from torcharrow.ilist_column import ListColumn, ListMethods
from torcharrow.scope import Scope

from .column import ColumnCpuMixin
from .typing import get_velox_type

# -----------------------------------------------------------------------------
# ListColumn


class ListColumnCpu(ColumnCpuMixin, ListColumn):

    # private constructor
    def __init__(self, device, dtype, data, offsets, mask):
        assert dt.is_list(dtype)
        ListColumn.__init__(self, device, dtype)

        self._data = velox.Column(
            velox.VeloxArrayType(get_velox_type(dtype.item_dtype))
            if dtype.fixed_size == -1
            else velox.VeloxFixedArrayType(
                dtype.fixed_size, get_velox_type(dtype.item_dtype)
            )
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
            return ColumnCpuMixin._from_velox(
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
            new_element_column = ta.column(self._dtype.item_dtype)
            new_element_column = new_element_column.append(value)
            if self._dtype.fixed_size != -1 and self.dtype.fixed_size != len(
                new_element_column
            ):
                raise ValueError("value incompatible with list fixed_size")
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
                ColumnCpuMixin._from_velox(
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

        elements = ColumnCpuMixin._from_velox(
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
        # pyre-fixme[16]: `DType` has no attribute `item_dtype`.
        assert dt.is_numerical(self.dtype.item_dtype)
        assert self.null_count == 0

        import torch
        from torch.nn.utils.rnn import pad_sequence

        packed_list: Union[
            pytorch.WithPresence, pytorch.PackedList
        ] = self._to_tensor_default()

        if isinstance(packed_list, pytorch.WithPresence):
            # presence tensor will be provided if dtype is nullable.
            # However, as long as there is no null value, the collation can still be done, and we just need to discard the presence tensor
            assert torch.all(packed_list.presence)
            packed_list = packed_list.values

        flattened_values = packed_list.values
        if isinstance(flattened_values, pytorch.WithPresence):
            # presence tensor will be provided if item_type is nullable.
            # However, as long as there is no null value, the collation can still be done, and we just need to discard the presence tensor
            assert torch.all(flattened_values.presence)
            flattened_values = flattened_values.values

        # pyre-fixme[11]: Annotation `tensor` is not defined as a type.
        unpad_tensors: List[torch.tensor] = [
            flattened_values[packed_list.offsets[i] : packed_list.offsets[i + 1]]
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


class ListMethodsCpu(ListMethods):
    """Vectorized string functions for StringColumn"""

    def __init__(self, parent: ListColumnCpu):
        super().__init__(parent)

    def length(self):
        return functional.cardinality(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def slice(self, start: int = 0, stop: Optional[int] = None) -> ListColumn:
        if start < 0:
            raise NotImplementedError("Negative start position is not supported yet")

        if stop is None:
            return functional.slice(self._parent, start + 1, 2**31 - 1)._with_null(
                self._parent.dtype.nullable
            )

        if stop < 0:
            raise NotImplementedError("Negative start position is not supported yet")

        return functional.slice(self._parent, start + 1, stop - start)._with_null(
            self._parent.dtype.nullable
        )

    def vmap(self, fun: Callable[[Column], Column]):
        elements = ColumnCpuMixin._from_velox(
            self._parent.device,
            # pyre-fixme[16]: `DType` has no attribute `item_dtype`.
            self._parent._dtype.item_dtype,
            # pyre-fixme[16]: `ListColumn` has no attribute `_data`.
            self._parent._data.elements(),
            True,
        )
        new_elements = fun(elements)

        # pyre-fixme[16]: `Column` has no attribute `_data`.
        new_data = self._parent._data.withElements(new_elements._data)
        return ColumnCpuMixin._from_velox(
            self._parent.device, new_data.dtype(), new_data, True
        )


# ------------------------------------------------------------------------------
# registering the factory
Dispatcher.register((dt.List.typecode + "_empty", "cpu"), ListColumnCpu._empty)
Dispatcher.register(
    (dt.List.typecode + "_from_pysequence", "cpu"), ListColumnCpu._from_pysequence
)
