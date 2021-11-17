# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np
import torcharrow as ta
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow import Scope
from torcharrow.dispatcher import Dispatcher
from torcharrow.dispatcher import Dispatcher
from torcharrow.icolumn import IColumn
from torcharrow.imap_column import IMapColumn, IMapMethods

from .column import ColumnFromVelox
from .typing import get_velox_type

# -----------------------------------------------------------------------------
# IMapColumn


class MapColumnCpu(ColumnFromVelox, IMapColumn):
    def __init__(self, device, dtype, key_data, item_data, mask):
        assert dt.is_map(dtype)
        IMapColumn.__init__(self, device, dtype)

        self._data = velox.Column(
            velox.VeloxMapType(
                get_velox_type(dtype.key_dtype), get_velox_type(dtype.item_dtype)
            )
        )

        self._finialized = False

        self.maps = MapMethodsCpu(self)

    # Lifecycle: _empty -> _append* -> _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(device, dtype):
        key_data = Scope._EmptyColumn(
            dt.List(dtype.key_dtype).with_null(dtype.nullable)
        )
        item_data = Scope._EmptyColumn(
            dt.List(dtype.item_dtype).with_null(dtype.nullable)
        )
        return MapColumnCpu(device, dtype, key_data, item_data, ar.array("b"))

    @staticmethod
    def _full(device, data, dtype=None, mask=None):
        assert isinstance(data, tuple) and len(data) == 2
        key_data, item_data = data
        assert isinstance(key_data, IColumn)
        assert isinstance(item_data, IColumn)
        assert len(item_data) == len(key_data)

        if dtype is None:
            dtype = dt.Map(
                dt.typeof_np_ndarray(key_data.dtype),
                dt.typeof_np_ndarray(item_data.dtype),
            )
        # else:
        #     if dtype != dt.typeof_np_dtype(data.dtype):
        #         # TODO fix nullability
        #         # raise TypeError(f'type of data {data.dtype} and given type {dtype} must be the same')
        #         pass
        if not dt.is_map(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = IMapColumn._valid_mask(len(key_data))
        elif len(key_data) != len(mask):
            raise ValueError(
                f"data length {len(key_data)} must be the same as mask length {len(mask)}"
            )
        # TODO check that all non-masked items are legal numbers (i.e not nan)
        return MapColumnCpu(device, dtype, key_data, item_data, mask)

    @staticmethod
    def _fromlist(device, data: List, dtype):
        # default implementation
        col = MapColumnCpu._empty(device, dtype)
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
            new_key = ta.Column(self._dtype.key_dtype)
            new_value = ta.Column(self._dtype.item_dtype)
            new_key = new_key.append(value.keys())
            new_value = new_value.append(value.values())
            self._data.append(new_key._data, new_value._data)

    def _finalize(self, mask=None):
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
            key_col = ColumnFromVelox.from_velox(
                self.device,
                self._dtype.key_dtype,
                self._data.keys()[i],
                True,
            )
            value_col = ColumnFromVelox.from_velox(
                self.device,
                self._dtype.item_dtype,
                self._data.values()[i],
                True,
            )

            return {key_col[j]: value_col[j] for j in range(len(key_col))}

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

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

    # interop
    def to_torch(self):
        pytorch.ensure_available()
        import torch

        # TODO: more efficient/straightfowrad interop

        # FIXME: https://github.com/facebookresearch/torcharrow/issues/62 to_arrow doesn't work as expected for map
        # arrow_array = self.to_arrow()

        keys = ColumnFromVelox.from_velox(
            self.device, dt.List(self._dtype.key_dtype), self._data.keys(), True
        ).to_torch(_propagate_py_list=False)
        values = ColumnFromVelox.from_velox(
            self.device, dt.List(self._dtype.item_dtype), self._data.values(), True
        ).to_torch(_propagate_py_list=False)

        # TODO: should we propagate python list if both keys and vals are lists of strings?
        assert isinstance(keys, pytorch.PackedList)
        assert isinstance(values, pytorch.PackedList)
        assert torch.all(keys.offsets == values.offsets)
        res = pytorch.PackedMap(
            keys=keys.values, values=values.values, offsets=keys.offsets
        )
        if not self._dtype.nullable:
            return res

        presence = torch.tensor(
            # FIXME: https://github.com/facebookresearch/torcharrow/issues/62
            # arrow_array.is_valid().to_numpy(zero_copy_only=False), dtype=torch.bool
            [self[i] is not None for i in range(len(self))],
            dtype=torch.bool,
        )
        return pytorch.WithPresence(values=res, presence=presence)


# ------------------------------------------------------------------------------
# registering the factory
Dispatcher.register((dt.Map.typecode + "_empty", "cpu"), MapColumnCpu._empty)
Dispatcher.register((dt.Map.typecode + "_full", "cpu"), MapColumnCpu._full)
Dispatcher.register((dt.Map.typecode + "_fromlist", "cpu"), MapColumnCpu._fromlist)
# -----------------------------------------------------------------------------
# MapMethods


@dataclass
class MapMethodsCpu(IMapMethods):
    """Vectorized list functions for IListColumn"""

    def __init__(self, parent: MapColumnCpu):
        super().__init__(parent)

    def keys(self):
        me = self._parent
        return ColumnFromVelox.from_velox(
            me.device, dt.List(me._dtype.key_dtype), me._data.keys(), True
        )

    def values(self):
        me = self._parent
        return ColumnFromVelox.from_velox(
            me.device, dt.List(me._dtype.item_dtype), me._data.values(), True
        )
