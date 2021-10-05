import array as ar
from typing import List

import numpy as np
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow.ilist_column import IListColumn, IListMethods
from torcharrow.scope import ColumnFactory


# -----------------------------------------------------------------------------
# IListColumn


class ListColumnDemo(IListColumn):

    # private constructor
    def __init__(self, scope, device, dtype, data, offsets, mask):
        assert dt.is_list(dtype)
        super().__init__(scope, device, dtype)

        self._data = data
        self._offsets = offsets
        self._mask = mask
        self.list = ListMethodsStd(self)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time
    @staticmethod
    def _empty(scope, device, dtype):
        return ListColumnDemo(
            scope,
            device,
            dtype,
            scope._EmptyColumn(dtype.item_dtype, device),
            ar.array("I", [0]),
            ar.array("b"),
        )

    @staticmethod
    def _fromlist(scope, device, data: List, dtype):
        # default (inefficient) implementation
        col = ListColumnDemo._empty(scope, device, dtype)
        for i in data:
            col._append(i)
        return col._finalize()

    def _append_null(self):
        self._mask.append(True)
        self._offsets.append(self._offsets[-1])
        self._data._extend([])

    def _append_value(self, value):
        self._mask.append(False)
        self._offsets.append(self._offsets[-1] + len(value))
        self._data._extend(value)

    def _finalize(self):
        self._data = self._data._finalize()
        self._offsets = np.array(self._offsets, dtype=np.int32, copy=False)
        if not isinstance(self._mask, np.ndarray):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        return self

    def __len__(self):
        return len(self._offsets) - 1

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return list(self._data[self._offsets[i] : self._offsets[i + 1]])

    def append(self, values):
        """Returns column/dataframe with values appended."""
        # tmp = self.scope.Column(values, dtype=self.dtype, device = self.device)
        # res= IListColumn(*self._meta(),
        #     np.append(self._data,tmp._data),
        #     np.append(self._offsets,tmp._offsets[1:] + self._offsets[-1]),
        #     np.append(self._mask,tmp._mask))

        # TODO replace this with vectorized code like the one above, except that is buggy
        res = self._EmptyColumn(self.dtype)
        for v in self:
            res._append(v)
        for v in values:
            res._append(v)
        return res._finalize()

    def concat(self, values):
        """Returns column/dataframe with values appended."""
        # tmp = self.scope.Column(values, dtype=self.dtype, device = self.device)
        # res= IListColumn(*self._meta(),
        #     np.append(self._data,tmp._data),
        #     np.append(self._offsets,tmp._offsets[1:] + self._offsets[-1]),
        #     np.append(self._mask,tmp._mask))

        # TODO replace this with vectorized code like the one above, except that is buggy
        res = self._EmptyColumn(self.dtype)
        for v in self:
            res._append(v)
        for v in values:
            res._append(v)
        return res._finalize()

    def to_torch(self, _propagate_py_list=True):
        pytorch.ensure_available()
        import torch

        assert len(self._data) == self._offsets[-1]
        elems = self._data.to_torch()
        # special case: if the nested type is List (which happens for List[str] that can't be represented as tensor)
        # then we fallback to string types
        if isinstance(elems, list) and _propagate_py_list:
            return [
                (
                    elems[self._offsets[i] : self._offsets[i + 1]]
                    if not self._mask[i]
                    else None
                )
                for i in range(len(self))
            ]
        # TODO: clarify int32 vs int64
        offsets = torch.tensor(
            self._offsets,
            dtype=torch.int32,
        )
        res = pytorch.PackedList(values=elems, offsets=offsets)
        if not self._dtype.nullable:
            return res

        assert len(self._mask) == len(self)
        presence = torch.tensor(self._mask, dtype=torch.bool).bitwise_not()
        return pytorch.WithPresence(values=res, presence=presence)

    # printing ----------------------------------------------------------------

    def __str__(self):
        return f"Column([{', '.join('None' if i is None else str(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else str(i)] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self._dtype}, length: {self.length()}, null_count: {self.null_count()}"
        return tab + dt.NL + typ


# ------------------------------------------------------------------------------
# ListMethodsStd


class ListMethodsStd(IListMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: ListColumnDemo):
        super().__init__(parent)


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.List.typecode + "_empty", "demo"), ListColumnDemo._empty)
ColumnFactory.register(
    (dt.List.typecode + "_fromlist", "demo"), ListColumnDemo._fromlist
)
