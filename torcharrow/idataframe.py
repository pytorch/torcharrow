#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from typing import (
    Any,
    Callable,
    Dict,
    get_type_hints,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow.dispatcher import Device

from .expression import eval_expression, expression, Var
from .icolumn import Column
from .scope import Scope
from .trace import trace, traceproperty

# assumes that these have been imported already:
# from .inumerical_column import NumericalColumn
# from .istring_column import StringColumn
# from .imap_column import MapColumn
# from .ilist_column import MapColumn

# ------------------------------------------------------------------------------
# DataFrame Factory with default scope and device


def dataframe(
    data: Optional[Union[Iterable, dt.DType]] = None,
    dtype: Optional[dt.DType] = None,
    columns: Optional[List[str]] = None,
    device: Device = "",
):
    """Creates a TorchArrow DataFrame.

    Parameters
    ----------
    data : dict or list of tuples
        Defines the contents of the DataFrame.  Dict keys are used for column
        names, and values for columns.  Use dtype to force a particular column
        ordering.  When Data is a list of tuples, dtype has to be provided to
        infer field names.

    dtype : dtype, default None
        Data type to force.  If None the type will be automatically inferred
        where possible.  Should be a dt.Struct() providing a list of
        dt.Fields.

    columns: list of strings, default None
        The name of columns. Used when data is a list of tuples without
        a custom dtype provided. This should be left to be None when data
        and dtype are both None (the semantic is constructing a default
        empty DataFrame without any columns).

    device: Device, default ""
        Device selects which runtime to use from scope.  TorchArrow supports
        multiple runtimes (CPU and GPU).  If not supplied, uses the Velox
        vectorized runtime.  Valid values are "cpu" (Velox), "gpu" (coming
        soon).

    Examples
    --------

    A Dataframe is just a set of named and strongly typed columns of equal
    length:

    >>> import torcharrow as ta
    >>> df = ta.dataframe({'a': list(range(7)),
    >>>                    'b': list(reversed(range(7))),
    >>>                    'c': list(range(7))
    >>>                   })
    >>> df
      index    a    b    c
    -------  ---  ---  ---
          0    0    6    0
          1    1    5    1
          2    2    4    2
          3    3    3    3
          4    4    2    4
          5    5    1    5
          6    6    0    6
    dtype: Struct([Field('a', int64), Field('b', int64), Field('c', int64)]), count: 7, null_count: 0

    DataFrames are immutable, except you can always add a new column, provided
    its name hasn't been used. The column is appended to the set of existing
    columns at the end:

    >>> df['d'] = ta.column(list(range(99, 99+7)))
    >>> df
      index    a    b    c    d
    -------  ---  ---  ---  ---
          0    0    6    0   99
          1    1    5    1  100
          2    2    4    2  101
          3    3    3    3  102
          4    4    2    4  103
          5    5    1    5  104
          6    6    0    6  105
    dtype: Struct([Field('a', int64), Field('b', int64), Field('c', int64), Field('d', int64)]), count: 7, null_count: 0

    Building a nested Dataframe:

    >>> df_inner = ta.dataframe({'b1': [11, 22, 33], 'b2':[111,222,333]})
    >>> df_outer = ta.dataframe({'a': [1, 2, 3], 'b':df_inner})
    >>> df_outer
      index    a  b
    -------  ---  ---------
          0    1  (11, 111)
          1    2  (22, 222)
          2    3  (33, 333)
    dtype: Struct([Field('a', int64), Field('b', Struct([Field('b1', int64), Field('b2', int64)]))]), count: 3, null_count: 0

    Build a Dataframe from a list of tuples:

    >>> import torcharrow.dtypes as dt
    >>> l = [(1, 'a'), (2, 'b'), (3, 'c')]
    >>> ta.dataframe(l, dtype = dt.Struct([dt.Field('t1', dt.int64), dt.Field('t2', dt.string)]))
      index    t1  t2
    -------  ----  ----
          0     1  a
          1     2  b
          2     3  c
    dtype: Struct([Field('t1', int64), Field('t2', string)]), count: 3, null_count: 0

    or

    >>> ta.dataframe(l, columns=['t1', 't2'])
      index    t1  t2
    -------  ----  ----
          0     1  a
          1     2  b
          2     3  c
    dtype: Struct([Field('t1', int64), Field('t2', string)]), count: 3, null_count: 0

    """

    device = device or Scope.default.device
    return Scope._DataFrame(data, dtype=dtype, columns=columns, device=device)


# -----------------------------------------------------------------------------
# DataFrames aka (StructColumns, can be nested as StructColumns:-)

DataOrDTypeOrNone = Optional[Union[Mapping, Sequence, dt.DType]]


class DataFrame(Column):
    """Dataframe, ordered dict of typed columns of the same length"""

    def __init__(self, device, dtype):
        assert dt.is_struct(dtype)
        super().__init__(device, dtype)

    @property  # type: ignore
    def columns(self):
        """The column labels of the DataFrame."""
        return [f.name for f in self.dtype.fields]

    @abc.abstractmethod
    def _set_field_data(self, name: str, col: Column, empty_df: bool):
        """
        PRIVATE _set field data, append if field doesn't exist
        self._dtype is already updated upon invocation
        """
        raise self._not_supported("_set_field_data")

    def __contains__(self, key: str) -> bool:
        for f in self.dtype.fields:
            if key == f.name:
                return True
        return False

    @trace
    def __setitem__(self, name: str, value: Any) -> None:
        if isinstance(value, Column):
            assert self.device == value.device
            col = value
        else:
            col = ta.column(value)

        empty_df = len(self.dtype.fields) == 0

        # Update dtype
        # pyre-fixme[16]: `DType` has no attribute `get_index`.
        idx = self.dtype.get_index(name)
        if idx is None:
            # append column
            new_fields = self.dtype.fields + [dt.Field(name, col.dtype)]
        else:
            # override column
            new_fields = list(self.dtype.fields)
            new_fields[idx] = dt.Field(name, col.dtype)
        self._dtype = dt.Struct(fields=new_fields)

        # Update field data
        self._set_field_data(name, col, empty_df)

    @trace
    def copy(self):
        raise self._not_supported("copy")

    @trace
    @expression
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ):
        raise self._not_supported("describe")

    @trace
    @expression
    def isin(self, values: Union[list, dict, Column]):
        """
        Check whether each element in the dataframe is contained in values.

        Parameters
        ----------
        values - array-like, column or dict
            Which values to check the presence of.

        Returns
        -------
        DataFrame of booleans showing whether each element is contained in values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.dataframe({"a": [1, 2, 3],
                              "b": [4, 5, 6]
                              })
        >>> df.isin([1, 2, 5])
          index  a      b
        -------  -----  -----
            0  True   False
            1  True   True
            2  False  False
        dtype: Struct([Field('a', boolean), Field('b', boolean)]), count: 3, null_count: 0
        """
        raise self._not_supported("isin")

    def log(self) -> DataFrame:
        raise self._not_supported("log")

    # aggregation

    @trace
    @expression
    def min(self):
        """
        Return the minimum of the non-null values for each column.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.dataframe({"a": [1,2,None,4],
                                "b": [5, 6, None, 8]
                                })
        >>> df.min()
        index    a    b
        -------  ---  ---
            0    1    5
        dtype: Struct([Field('a', Int64(nullable=True)), Field('b', Int64(nullable=True))]), count: 1, null_count: 0
        """
        raise self._not_supported("min")

    @trace
    @expression
    def max(self):
        """
        Return the maximal of the non-null values for each column.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.dataframe({"a": [1,2,None,4],
                                "b": [5, 6, None, 8]
                                })
        >>> df.max()
        index    a    b
        -------  ---  ---
            0    4    8
        dtype: Struct([Field('a', Int64(nullable=True)), Field('b', Int64(nullable=True))]), count: 1, null_count: 0
        """
        raise self._not_supported("max")

    @trace
    @expression
    def sum(self):
        """
        Return the sum of the non-null values for each column.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.dataframe({"a": [1,2,None,4],
                                "b": [5, 6, None, 8]
                                })
        >>> df.sum()
        index    a    b
        -------  ---  ---
            0    7    19
        dtype: Struct([Field('a', Int64(nullable=True)), Field('b', Int64(nullable=True))]), count: 1, null_count: 0
        """
        raise self._not_supported("sum")

    @trace
    @expression
    def mean(self):
        """
        Return the mean of the non-null values for each column.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.dataframe({"a": [1.0,2.0,None,6.3],
                                "b": [5.0, 6.0, None, 10.6]
                                })
        >>> df.mean()
        index    a    b
        -------  ---  ---
            0  3.1  7.2
        dtype: Struct([Field('a', Float32(nullable=True)), Field('b', Float32(nullable=True))]), count: 1, null_count: 0
        """
        raise self._not_supported("mean")

    @trace
    @expression
    def std(self):
        raise self._not_supported("std")

    @trace
    @expression
    def median(self):
        raise self._not_supported("median")

    @trace
    @expression
    def mode(self):
        raise self._not_supported("mode")

    @trace
    @expression
    def all(self):
        raise self._not_supported("all")

    @trace
    @expression
    def any(self):
        raise self._not_supported("any")

    # column alnternating
    @trace
    @expression
    def drop(self, columns: List[str]):
        """
        Returns DataFrame without the removed columns.
        """
        raise self._not_supported("drop")

    @trace
    @expression
    def rename(self, mapper: Dict[str, str]):
        """
        Returns DataFrame with column names remapped.
        """
        raise self._not_supported("rename")

    @trace
    @expression
    def reorder(self, columns: List[str]):
        """
        EXPERIMENTAL API

        Returns DataFrame with the columns in the prescribed order.
        """
        raise self._not_supported("rename")

    # functional API

    @trace
    @expression
    def transform(
        self,
        func: Callable,
        dtype: Optional[dt.DType] = None,
        format: str = "column",
        columns: Optional[List[str]] = None,
    ):
        """
        Like map() but invokes the callable on mini-batches of rows at a time.
        The column is passed to the callable as TorchArrow column by default.
        If `format='python'` the input is converted to python types instead.
        If `format='torch'` the input is converted to PyTorch types
        dtype required if result type != item type and the type hint is missing on the callable.
        """
        if columns is None:
            return super().map(func, dtype=dtype, format=format)

        for i in columns:
            if i not in self.columns:
                raise KeyError("column {i} not in dataframe")

        if dtype is None:
            signature = get_type_hints(func)
            if "return" in signature:
                dtype = dt.dtype_from_batch_pytype(signature["return"])
            else:
                assert self._dtype is not None
                dtype = self._dtype
            # TODO: check type annotations of inputs too in order to infer the input format

        if len(columns) == 1:
            raw_res = func(self._format_transform_column(self[columns[0]], format))
        else:
            raw_res = func(
                *(self._format_transform_column(self[c], format) for c in columns)
            )
        return self._format_transform_result(raw_res, format, dtype, len(self))

    # relational tools
    @trace
    @expression
    def select(self, *args, **kwargs):
        """
        Analogous to SQL's `SELECT`.

        Transform a dataframe by selecting old columns and new (computed)
        columns.

        The special symbol `me` can be used to refer to self.

        Parameters
        -----------
        args : positional string arguments
            Column names to keep in the projection. A column name of "*" is a
            shortcut to denote all columns. A column name beginning with "-"
            means remove this column.

        kwargs : named value arguments
            New column name expressions to add to the projection


        Examples
        --------
        >>> from torcharrow import ta
        >>> xf = ta.dataframe({
        >>>    'A': ['a', 'b', 'a', 'b'],
        >>>    'B': [1, 2, 3, 4],
        >>>    'C': [10,11,12,13]})
        >>> xf.select(*xf.columns,D=me['B']+me['C'])
          index  A      B    C    D
        -------  ---  ---  ---  ---
              0  a      1   10   11
              1  b      2   11   13
              2  a      3   12   15
              3  b      4   13   17
        dtype: Struct([Field('A', string), Field('B', int64), Field('C', int64), Field('D', int64)]), count: 4, null_count: 0

        Using '*' and '-colname':

        >>> xf.select('*','-B',D=me['B']+me['C'])
          index  A      C    D
        -------  ---  ---  ---
              0  a     10   11
              1  b     11   13
              2  a     12   15
              3  b     13   17
        dtype: Struct([Field('A', string), Field('C', int64), Field('D', int64)]), count: 4, null_count: 0
        """
        raise self._not_supported("select")

    @trace
    @expression
    def where(self, *conditions):
        """
        Analogous to SQL's where (NOT Pandas where)

        Filter a dataframe to only include rows satisfying a given set
        of conditions. df.where(p) is equivalent to writing df[p].

        Examples
        --------

        >>> from torcharrow import ta
        >>> xf = ta.dataframe({
        >>>    'A':['a', 'b', 'a', 'b'],
        >>>    'B': [1, 2, 3, 4],
        >>>    'C': [10,11,12,13]})
        >>> xf.where(xf['B']>2)
          index  A      B    C
        -------  ---  ---  ---
              0  a      3   12
              1  b      4   13
        dtype: Struct([Field('A', string), Field('B', int64), Field('C', int64)]), count: 2, null_count: 0

        When referring to self in an expression, the special value `me` can be
        used.

        >>> from torcharrow import me
        >>> xf.where(me['B']>2)
          index  A      B    C
        -------  ---  ---  ---
              0  a      3   12
              1  b      4   13
        dtype: Struct([Field('A', string), Field('B', int64), Field('C', int64)]), count: 2, null_count: 0
        """
        raise self._not_supported("select")

    # interop

    @trace
    def to_pandas(self):
        """Convert self to Pandas DataFrame"""
        raise self._not_supported("to_pandas")

    @trace
    def to_arrow(self):
        """Convert self to arrow table"""
        raise self._not_supported("to_arrow")

    @trace
    def to_pylist(self):
        tup_type = self._dtype.py_type
        return [
            tup_type(*v)
            for v in zip(*(self[f.name].to_pylist() for f in self._dtype.fields))
        ]

    @trace
    def to_tensor(self, conversion=None):
        raise self._not_supported("to_tensor")

    def _get_column(self, column):
        """Return the named column"""
        raise self._not_supported("get_column")

    def _get_columns(self, columns):
        """Return a new dataframe referencing the columns[s1],..,column[sm]"""
        raise self._not_supported("get_columns")

    def _slice_columns(self, start, stop):
        """Return a new dataframe with the slice rows[start:stop]"""
        raise self._not_supported("slice_columns")


# TODO Make this abstract and add all the abstract methods here ...
# TODO Current short cut has 'everything', excpet for columns as a  DataFrameDemo
# TODO Make GroupedDatFrame also an IGroupedDataframe to make it truly compositional


# -----------------------------------------------------------------------------
# DataFrameVariable me


class DataFrameVar(Var, DataFrame):
    # A dataframe variable is purely symbolic,
    # It should only appear as part of a relational expression

    def __init__(self, name: str, qualname: str = ""):
        super().__init__(name, qualname)

    def _append_null(self):
        return self._not_supported("_append_null")

    def _append_value(self, value):
        return self._not_supported("_append_value")

    def _finalize(self, mask=None):
        return self._not_supported("_finalize")

    def __len__(self):
        return self._not_supported("len")

    def null_count(self):
        return self._not_supported("null_count")

    def _getmask(self, i):
        return self._not_supported("_getmask")

    def _getdata(self, i):
        return self._not_supported("getdata")

    def _set_field_data(self, name: str, col: Column, empty_df: bool):
        raise self._not_supported("_set_field_data")

    def _concat_with(self, columns: List[Column]):
        """Returns concatenated columns."""
        raise self._not_supported("_concat_with")

    @property  # type: ignore
    def columns(self):
        return self._not_supported("columns")


# The super variable...
me = DataFrameVar("me", "torcharrow.idataframe.me")
