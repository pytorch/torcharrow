#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import annotations

import abc
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    get_type_hints,
)

import torcharrow as ta
import torcharrow.dtypes as dt
from torcharrow.dispatcher import Device

from .expression import Var, eval_expression, expression
from .icolumn import IColumn
from .scope import Scope
from .trace import trace, traceproperty

# assumes that these have been imported already:
# from .inumerical_column import INumericalColumn
# from .istring_column import IStringColumn
# from .imap_column import IMapColumn
# from .ilist_column import IMapColumn

# ------------------------------------------------------------------------------
# DataFrame Factory with default scope and device


def DataFrame(
    data: Union[Iterable, dt.DType, Literal[None]] = None,
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
    >>> df = ta.DataFrame({'a': list(range(7)),
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

    >>> df['d'] = ta.Column(list(range(99, 99+7)))
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

    >>> df_inner = ta.DataFrame({'b1': [11, 22, 33], 'b2':[111,222,333]})
    >>> df_outer = ta.DataFrame({'a': [1, 2, 3], 'b':df_inner})
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
    >>> ta.DataFrame(l, dtype = dt.Struct([dt.Field('t1', dt.int64), dt.Field('t2', dt.string)]))
      index    t1  t2
    -------  ----  ----
          0     1  a
          1     2  b
          2     3  c
    dtype: Struct([Field('t1', int64), Field('t2', string)]), count: 3, null_count: 0

    or

    >>> ta.DataFrame(l, columns=['t1', 't2'])
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

DataOrDTypeOrNone = Union[Mapping, Sequence, dt.DType, Literal[None]]


class IDataFrame(IColumn):
    """Dataframe, ordered dict of typed columns of the same length"""

    def __init__(self, device, dtype):
        assert dt.is_struct(dtype)
        super().__init__(device, dtype)

    @property  # type: ignore
    def columns(self):
        """The column labels of the DataFrame."""
        return [f.name for f in self.dtype.fields]

    @abc.abstractmethod
    def _set_field_data(self, name: str, col: IColumn, empty_df: bool):
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
        if isinstance(value, IColumn):
            assert self.device == value.device
            col = value
        else:
            col = ta.Column(value)

        empty_df = len(self.dtype.fields) == 0

        # Update dtype
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
    def transform(
        self,
        func: Callable,
        /,
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

    def get_column(self, column):
        """Return the named column"""
        raise self._not_supported("get_column")

    def get_columns(self, columns):
        """Return a new dataframe referencing the columns[s1],..,column[sm]"""
        raise self._not_supported("get_columns")

    def slice_columns(self, start, stop):
        """Return a new dataframe with the slice rows[start:stop]"""
        raise self._not_supported("slice_columns")

    @trace
    def to_pylist(self):
        tup_type = self._dtype.py_type
        return [
            tup_type(*v)
            for v in zip(*(self[f.name].to_pylist() for f in self._dtype.fields))
        ]


# TODO Make this abstract and add all the abstract methods here ...
# TODO Current short cut has 'everything', excpet for columns as a  DataFrameDemo
# TODO Make GroupedDatFrame also an IGroupedDataframe to make it truly compositional


# -----------------------------------------------------------------------------
# DataFrameVariable me


class IDataFrameVar(Var, IDataFrame):
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

    def _set_field_data(self, name: str, col: IColumn, empty_df: bool):
        raise self._not_supported("_set_field_data")

    def _concat_with(self, columns: List[IColumn]):
        """Returns concatenated columns."""
        raise self._not_supported("_concat_with")

    @property  # type: ignore
    def columns(self):
        return self._not_supported("columns")


# The super variable...
me = IDataFrameVar("me", "torcharrow.idataframe.me")
