# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import array as ar
from typing import Optional, Sequence

import numpy as np

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
import torcharrow._torcharrow as velox
import torcharrow.dtypes as dt
from tabulate import tabulate
from torcharrow._functional import functional
from torcharrow.dispatcher import Dispatcher
from torcharrow.expression import expression
from torcharrow.istring_column import StringColumn, StringMethods
from torcharrow.trace import trace

from .column import ColumnFromVelox
from .typing import get_velox_type

# ------------------------------------------------------------------------------
# StringColumnCpu


class StringColumnCpu(ColumnFromVelox, StringColumn):

    # private constructor
    def __init__(self, device, dtype, data, mask):  # REP offsets
        assert dt.is_string(dtype)
        StringColumn.__init__(self, device, dtype)

        self._data = velox.Column(get_velox_type(dtype))
        for m, d in zip(mask.tolist(), data):
            if m:
                self._data.append_null()
            else:
                self._data.append(d)
        self._finalized = False

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
    def _from_pysequence(device: str, data: Sequence[str], dtype: dt.DType):
        # pyre-fixme[16]: Module `torcharrow` has no attribute `_torcharrow`.
        velox_column = velox.Column(get_velox_type(dtype), data)
        return ColumnFromVelox._from_velox(
            device,
            dtype,
            velox_column,
            True,
        )

    # TODO Add native kernel support
    @staticmethod
    def _from_arrow(device: str, array, dtype: dt.DType):
        import pyarrow as pa

        assert isinstance(array, pa.Array)

        pydata = [i.as_py() for i in array]
        return StringColumnCpu._from_pysequence(device, pydata, dtype)

    def _append_null(self):
        if self._finalized:
            raise AttributeError("It is already finalized.")
        self._data.append_null()

    def _append_value(self, value):
        if self._finalized:
            raise AttributeError("It is already finalized.")
        else:
            self._data.append(value)

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
            return functional.concat(self, other)._with_null(
                self.dtype.nullable or other.dtype.nullable
            )
        else:
            assert isinstance(other, str)
            return functional.concat(self, other)._with_null(self.dtype.nullable)

    def __radd__(self, other):
        """Vectorized b + a."""
        assert isinstance(other, str)
        return functional.concat(other, self)._with_null(self.dtype.nullable)

    def _checked_binary_op_call(self, other, op_name):
        f = functional.__getattr__(op_name)
        nullable = self.dtype.nullable
        if isinstance(other, StringColumnCpu):
            if len(other) != len(self):
                raise TypeError("columns must have equal length")
            nullable = nullable or other.dtype.nullable
        else:
            assert isinstance(other, str)
        return f(self, other)._with_null(nullable)

    @trace
    @expression
    def __eq__(self, other):
        return self._checked_binary_op_call(other, "eq")

    @trace
    @expression
    def __ne__(self, other):
        return self._checked_binary_op_call(other, "neq")

    @trace
    @expression
    def __lt__(self, other):
        return self._checked_binary_op_call(other, "lt")

    @trace
    @expression
    def __le__(self, other):
        return self._checked_binary_op_call(other, "lte")

    @trace
    @expression
    def __gt__(self, other):
        return self._checked_binary_op_call(other, "gt")

    @trace
    @expression
    def __ge__(self, other):
        return self._checked_binary_op_call(other, "gte")

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
    def _to_tensor_default(self):
        # there are no string tensors, so we're using regular python list conversion
        return self.to_pylist()


# ------------------------------------------------------------------------------
# StringMethodsCpu


class StringMethodsCpu(StringMethods):
    """Vectorized string functions for StringColumn"""

    def __init__(self, parent: StringColumnCpu):
        super().__init__(parent)

    def length(self):
        return functional.length(self._parent)._with_null(self._parent.dtype.nullable)

    def slice(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> StringColumn:
        start = start or 0
        if stop is None:
            return functional.substr(self._parent, start + 1)._with_null(
                self._parent.dtype.nullable
            )
        else:
            return functional.substr(self._parent, start + 1, stop - start)._with_null(
                self._parent.dtype.nullable
            )

    def split(self, pat=None, n=-1):
        pat = pat or " "
        if n <= 0:
            split_result = functional.split(self._parent, pat)
        else:
            split_result = functional.split(self._parent, pat, n + 1)
        return split_result._with_null(self._parent.dtype.nullable)

    def strip(self):
        return functional.trim(self._parent)._with_null(self._parent.dtype.nullable)

    def lower(self) -> StringColumn:
        return functional.lower(self._parent)._with_null(self._parent.dtype.nullable)

    def upper(self) -> StringColumn:
        return functional.upper(self._parent)._with_null(self._parent.dtype.nullable)

    # Check whether all characters in each string are  -----------------------------------------------------
    # alphabetic/numeric/digits/decimal...

    def isalpha(self) -> StringColumn:
        return functional.torcharrow_isalpha(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def isalnum(self) -> StringColumn:
        return functional.torcharrow_isalnum(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def isdigit(self) -> StringColumn:
        return functional.torcharrow_isdigit(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def isdecimal(self) -> StringColumn:
        return functional.torcharrow_isdecimal(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def islower(self) -> StringColumn:
        return functional.torcharrow_islower(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def isupper(self) -> StringColumn:
        return functional.torcharrow_isupper(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def isspace(self) -> StringColumn:
        return functional.torcharrow_isspace(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def istitle(self) -> StringColumn:
        return functional.torcharrow_istitle(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    def isnumeric(self) -> StringColumn:
        return functional.torcharrow_isnumeric(self._parent)._with_null(
            self._parent.dtype.nullable
        )

    # Pattern matching related methods  -----------------------------------------------------

    def startswith(self, pat):
        return (
            functional.substr(self._parent, 1, len(pat))._with_null(
                self._parent.dtype.nullable
            )
            == pat
        )

    def endswith(self, pat):
        return (
            functional.substr(
                self._parent, self._parent.str.length() - len(pat) + 1
            )._with_null(self._parent.dtype.nullable)
            == pat
        )

    def count(self, pat: str):
        # TODO: calculate without materializing all the occurrences
        return self.findall(pat).list.length()

    def find(self, sub):
        return (
            functional.strpos(self._parent, sub)._with_null(self._parent.dtype.nullable)
            - 1
        )

    def replace(self, pat: str, repl: str, regex: bool = True):
        if regex:
            raise TypeError(f"replace with regex is not implemented yet")
        else:
            return functional.replace(self._parent, pat, repl)._with_null(
                self._parent.dtype.nullable
            )

    def match(self, pat: str):
        return functional.match_re(self._parent, pat)._with_null(
            self._parent.dtype.nullable
        )

    def contains(self, pat: str, regex=True):
        if regex:
            return functional.regexp_like(self._parent, pat)._with_null(
                self._parent.dtype.nullable
            )
        else:
            raise TypeError(f"contains with regex is not implemented yet")

    def findall(self, pat: str):
        return functional.regexp_extract_all(self._parent, pat)._with_null(
            self._parent.dtype.nullable
        )

    def cat(self, col):
        return functional.concat(self._parent, col)._with_null(
            self._parent.dtype.nullable
        )


# ------------------------------------------------------------------------------
# registering the factory
Dispatcher.register((dt.String.typecode + "_empty", "cpu"), StringColumnCpu._empty)
Dispatcher.register((dt.String.typecode + "_full", "cpu"), StringColumnCpu._full)
Dispatcher.register(
    (dt.String.typecode + "_from_pysequence", "cpu"), StringColumnCpu._from_pysequence
)
Dispatcher.register(
    (dt.String.typecode + "_from_arrow", "cpu"), StringColumnCpu._from_arrow
)


def _is_not_str(s) -> bool:
    return not isinstance(s, str)
