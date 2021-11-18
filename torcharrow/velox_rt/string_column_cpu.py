# Copyright (c) Facebook, Inc. and its affiliates.
import array as ar
from dataclasses import dataclass
from typing import cast, List

import numpy as np
import numpy.ma as ma
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow.dispatcher import Dispatcher
from torcharrow.expression import expression
from torcharrow.functional import functional
from torcharrow.istring_column import IStringColumn, IStringMethods
from torcharrow.scope import Scope, Device

from .column import ColumnFromVelox
from .typing import get_velox_type

# ------------------------------------------------------------------------------
# StringColumnCpu


class StringColumnCpu(ColumnFromVelox, IStringColumn):

    # private constructor
    def __init__(self, device, dtype, data, mask):  # REP offsets
        assert dt.is_string(dtype)
        IStringColumn.__init__(self, device, dtype)

        self._data = velox.Column(get_velox_type(dtype))
        for m, d in zip(mask.tolist(), data):
            if m:
                self._data.append_null()
            else:
                self._data.append(d)
        self._finialized = False

        self.str = StringMethodsCpu(self)
        # REP: self._offsets = offsets

    # Any _empty must be followed by a _finalize; no other ops are allowed during this time

    @staticmethod
    def _empty(device, dtype):
        # REP  ar.array("I", [0])
        return StringColumnCpu(device, dtype, [], ar.array("b"))

    @staticmethod
    def _full(device, data, dtype=None, mask=None):
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
        return StringColumnCpu(device, dtype, data, mask)

    @staticmethod
    def _fromlist(device: str, data: List[str], dtype: dt.DType):
        velox_column = velox.Column(get_velox_type(dtype), data)
        return ColumnFromVelox.from_velox(
            device,
            dtype,
            velox_column,
            True,
        )

    def _append_null(self):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finialized:
            raise AttributeError("It is already finialized.")
        else:
            self._data.append(value)

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
            return self.dtype.default
        else:
            return self._data[i]

    @staticmethod
    def _valid_mask(ct):
        raise np.full((ct,), False, dtype=np.bool8)

    def _gets(self, indices):
        data = self._data[indices]
        mask = self._mask[indices]
        return self._scope._FullColumn(data, self.dtype, self.device, mask)

    def _slice(self, start, stop, step):
        range = slice(start, stop, step)
        return self._scope._FullColumn(
            self._data[range], self.dtype, self.device, self._mask[range]
        )

    # operators ---------------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, StringColumnCpu):
            return functional.concat(self, other).with_null(
                self.dtype.nullable or other.dtype.nullable
            )
        else:
            assert isinstance(other, str)
            return functional.concat(self, other).with_null(self.dtype.nullable)

    @expression
    def __eq__(self, other):
        if isinstance(other, StringColumnCpu):
            return functional.eq(self, other).with_null(
                self.dtype.nullable or other.dtype.nullable
            )
        else:
            assert isinstance(other, str)
            return functional.eq(self, other).with_null(self.dtype.nullable)

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
        typ = f"dtype: {self.dtype}, length: {self.length}, null_count: {self.null_count}, device: cpu"
        return tab + dt.NL + typ

    # interop
    def to_torch(self):
        # there are no string tensors, so we're using regular python list conversion
        return self.to_pylist()


# ------------------------------------------------------------------------------
# StringMethodsCpu


class StringMethodsCpu(IStringMethods):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent: StringColumnCpu):
        super().__init__(parent)

    def length(self):
        return functional.length(self._parent).with_null(self._parent.dtype.nullable)

    def slice(self, start: int = None, stop: int = None) -> IStringColumn:
        start = start or 0
        if stop is None:
            return functional.substr(self._parent, start + 1).with_null(
                self._parent.dtype.nullable
            )
        else:
            return functional.substr(self._parent, start + 1, stop - start).with_null(
                self._parent.dtype.nullable
            )

    def split(self, sep=None):
        sep = sep or " "
        return functional.split(self._parent, sep).with_null(
            self._parent.dtype.nullable
        )

    def strip(self):
        return functional.trim(self._parent).with_null(self._parent.dtype.nullable)

    def lower(self) -> IStringColumn:
        return functional.lower(self._parent).with_null(self._parent.dtype.nullable)

    def upper(self) -> IStringColumn:
        return functional.upper(self._parent).with_null(self._parent.dtype.nullable)

    # Check whether all characters in each string are  -----------------------------------------------------
    # alphabetic/numeric/digits/decimal...

    def isalpha(self) -> IStringColumn:
        return functional.torcharrow_isalpha(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isalnum(self) -> IStringColumn:
        return functional.torcharrow_isalnum(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isdecimal(self) -> IStringColumn:
        return functional.isdecimal(self._parent).with_null(self._parent.dtype.nullable)

    def islower(self) -> IStringColumn:
        return functional.torcharrow_islower(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isupper(self) -> IStringColumn:
        return functional.isupper(self._parent).with_null(self._parent.dtype.nullable)

    def isspace(self) -> IStringColumn:
        return functional.torcharrow_isspace(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def istitle(self) -> IStringColumn:
        return functional.torcharrow_istitle(self._parent).with_null(
            self._parent.dtype.nullable
        )

    def isnumeric(self) -> IStringColumn:
        return functional.isnumeric(self._parent).with_null(self._parent.dtype.nullable)

    # Pattern matching related methods  -----------------------------------------------------

    def startswith(self, pat):
        return (
            functional.substr(self._parent, 1, len(pat)).with_null(
                self._parent.dtype.nullable
            )
            == pat
        )

    def endswith(self, pat):
        return (
            functional.substr(
                self._parent, self._parent.str.length() - len(pat) + 1
            ).with_null(self._parent.dtype.nullable)
            == pat
        )

    def find(self, sub):
        return (
            functional.strpos(self._parent, sub).with_null(self._parent.dtype.nullable)
            - 1
        )

    def replace(self, old, new):
        return functional.replace(self._parent, old, new).with_null(
            self._parent.dtype.nullable
        )

    # Regular expressions -----------------------------------------------------

    def match_re(self, pattern: str):
        return functional.match_re(self._parent, pattern).with_null(
            self._parent.dtype.nullable
        )

    def contains_re(
        self,
        pattern: str,
    ):
        return functional.regexp_like(self._parent, pattern).with_null(
            self._parent.dtype.nullable
        )

    def findall_re(self, pattern: str):
        return functional.regexp_extract_all(self._parent, pattern).with_null(
            self._parent.dtype.nullable
        )


# ------------------------------------------------------------------------------
# registering the factory
Dispatcher.register((dt.String.typecode + "_empty", "cpu"), StringColumnCpu._empty)
Dispatcher.register((dt.String.typecode + "_full", "cpu"), StringColumnCpu._full)
Dispatcher.register(
    (dt.String.typecode + "_fromlist", "cpu"), StringColumnCpu._fromlist
)


def _is_not_str(s):
    return not isinstance(s, str)
