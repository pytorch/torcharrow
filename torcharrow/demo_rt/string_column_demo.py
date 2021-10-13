# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
from dataclasses import dataclass
from typing import cast, List

import numpy as np
import numpy.ma as ma
import torcharrow.dtypes as dt
import torcharrow.pytorch as pytorch
from tabulate import tabulate
from torcharrow.expression import expression
from torcharrow.istring_column import IStringColumn, IStringMethods
from torcharrow.scope import ColumnFactory


# ------------------------------------------------------------------------------
# StringColumnDemo


class StringColumnDemo(IStringColumn):

    # private constructor
    def __init__(self, scope, device, dtype, data, mask):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(scope, device, dtype)

        self._data = data
        self._mask = mask
        self.str = StringMethodsStd(self)

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(scope, device, dtype):
        return StringColumnDemo(scope, device, dtype, [], ar.array("b"))

    @staticmethod
    def _full(scope, device, data, dtype=None, mask=None):
        assert isinstance(data, np.ndarray) and data.ndim == 1
        if dtype is None:
            dtype = dt.typeof_np_ndarray(data.dtype)
            if dtype is None:  # could be an object array
                mask = np.vectorize(_is_not_str)(data)
                dtype = dt.string
        else:
            pass
            # if dtype != typeof_np_ndarray:
            # pass
            # TODO refine this test
            # raise TypeError(f'type of data {data.dtype} and given type {dtype }must be the same')
        if not dt.is_string(dtype):
            raise TypeError(f"construction of columns of type {dtype} not supported")
        if mask is None:
            mask = np.vectorize(_is_not_str)(data)
        elif len(data) != len(mask):
            raise ValueError(f"data length {len(data)} must be mask length {len(mask)}")
        # TODO check that all non-masked items are strings
        return StringColumnDemo(scope, device, dtype, data, mask)

    @staticmethod
    def _fromlist(scope, device, data: List, dtype):
        # default implementation
        col = StringColumnDemo._empty(scope, device, dtype)
        for i in data:
            col._append(i)
        return col._finalize()

    def _append_null(self):
        self._mask.append(True)
        self._data.append(dt.String.default)
        # REP: offsets.append(offsets[-1])

    def _append_value(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Expect str, got: {type(value)}")
        self._mask.append(False)
        self._data.append(value)
        # REP: offsets.append(offsets[-1] + len(i))

    def _finalize(self):
        self._data = np.array(self._data, dtype=object)
        if isinstance(self._mask, (bool, np.bool8)):
            self._mask = StringColumnDemo._valid_mask(len(self._data))
        elif isinstance(self._mask, ar.array):
            self._mask = np.array(self._mask, dtype=np.bool8, copy=False)
        else:
            assert isinstance(self._mask, np.ndarray)
        return self

    def __len__(self):
        return len(self._data)

    def null_count(self):
        return self._mask.sum()

    def getmask(self, i):
        return self._mask[i]

    def getdata(self, i):
        return self._data[i]
        # REP: return self._data[self._offsets[i]: self._offsets[i + 1]]

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def gets(self, indices):
        data = self._data[indices]
        mask = self._mask[indices]
        return self.scope._FullColumn(data, self.dtype, self.device, mask)

    def slice(self, start, stop, step):
        range = slice(start, stop, step)
        return self.scope._FullColumn(
            self._data[range], self.dtype, self.device, self._mask[range]
        )

    def append(self, values):
        """Returns column/dataframe with values appended."""
        tmp = self.scope.Column(values, dtype=self.dtype, device=self.device)
        return self.scope._FullColumn(
            np.append(self._data, tmp._data),
            self.dtype,
            self.device,
            np.append(self._mask, tmp._mask),
        )

    # operators ---------------------------------------------------------------
    def __add__(self, other):
        res = self.to_python()
        if isinstance(other, StringColumnDemo):
            other_str = other.to_python()
            assert len(res) == len(other_str)
            for i in range(len(res)):
                if res[i] is None or other_str[i] is None:
                    res[i] = None
                else:
                    res[i] += other_str[i]
            return self._FromPyList(
                res, dt.String(self.dtype.nullable or other.dtype.nullable)
            )
        else:
            assert isinstance(other, str)
            for i in range(len(res)):
                if res[i] is not None:
                    res[i] += other
            return self._FromPyList(res, self.dtype)

    @expression
    def __eq__(self, other):
        if isinstance(other, StringColumnDemo):
            res = self._EmptyColumn(
                dt.Boolean(self.dtype.nullable or other.dtype.nullable)
            )
            for (m, i), (n, j) in zip(self.items(), other.items()):
                if m or n:
                    res._append_null()
                else:
                    res._append_value(i == j)
            return res._finalize()
        else:
            res = self._EmptyColumn(dt.Boolean(self.dtype.nullable))
            for (m, i) in self.items():
                if m:
                    res._append_null()
                else:
                    res._append_value(i == other)
            return res._finalize()

    # printing ----------------------------------------------------------------

    def __str__(self):
        def quote(x):
            return f"'{x}'"

        return f"Column([{', '.join('None' if i is None else quote(i) for i in self)}])"

    def __repr__(self):
        tab = tabulate(
            [["None" if i is None else f"'{i}'"] for i in self],
            tablefmt="plain",
            showindex=True,
        )
        typ = f"dtype: {self.dtype}, length: {self.length()}, null_count: {self.null_count()}"
        return tab + dt.NL + typ

    def to_torch(self):
        # there are no string tensors, so we're using regular python conversion
        return list(self)


# ------------------------------------------------------------------------------
# StringMethodsStd


class StringMethodsStd(IStringMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: StringColumnDemo):
        super().__init__(parent)


# ------------------------------------------------------------------------------
# registering the factory
ColumnFactory.register((dt.String.typecode + "_empty", "demo"), StringColumnDemo._empty)
ColumnFactory.register((dt.String.typecode + "_full", "demo"), StringColumnDemo._full)
ColumnFactory.register(
    (dt.String.typecode + "_fromlist", "demo"), StringColumnDemo._fromlist
)


def _is_not_str(s):
    return not isinstance(s, str)
