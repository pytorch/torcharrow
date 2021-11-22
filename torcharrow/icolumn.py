#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import annotations

import abc
import itertools
import math
import operator
import statistics
import typing as ty
from collections import OrderedDict, defaultdict
from functools import partial, reduce

import numpy as np
import torcharrow as ta
import torcharrow.dtypes as dt
from tabulate import tabulate

from .dispatcher import Device
from .expression import expression
from .scope import Scope
from .trace import trace, traceproperty

# ------------------------------------------------------------------------------
# Column Factory with default device


def Column(
    data: ty.Union[ty.Iterable, dt.DType, ty.Literal[None]] = None,
    dtype: ty.Optional[dt.DType] = None,
    device: Device = "",
):
    """
    Creates a TorchArrow Column.  Allocates memory on device or default device.

    Parameters
    ----------
    data: array-like or Iterable
        Defines the contents of the column.

    dtype: dtype, default None
        Data type to force.  If None the type will be automatically
        inferred where possible.

    device: Device, default ""
        Device selects which runtime to use from scope.  TorchArrow supports
        multiple runtimes (CPU and GPU).  If not supplied, uses the Velox
        vectorized runtime.  Valid values are "cpu" (Velox), "gpu" (coming
        soon).

    Examples
    --------

    Creating a column using auto-inferred type:

    >>> import torcharrow as ta
    >>> s = ta.Column([1,2,None,4])
    >>> s
    0  1
    1  2
    2  None
    3  4
    dtype: Int64(nullable=True), length: 4, null_count: 1

    Create a column with arbitrarily data types, here a non-nullable
    column of a list of non-nullable strings of arbitrary length:

    >>> sf = ta.Column([ ["hello", "world"], ["how", "are", "you"] ], dtype =dt.List(dt.string))
    >>> sf.dtype
    List(item_dtype=String(nullable=False), nullable=False, fixed_size=-1)

    Create a column of average climate data, one map per continent,
    with city as key and yearly average min and max temperature:

    >>> mf = ta.Column([
    >>>     {'helsinki': [-1.3, 21.5], 'moscow': [-4.0,24.3]},
    >>>     {'algiers':[11.2, 25.2], 'kinshasa':[22.2,26.8]}
    >>>     ])
    >>> mf
    0  {'helsinki': [-1.3, 21.5], 'moscow': [-4.0, 24.3]}
    1  {'algiers': [11.2, 25.2], 'kinshasa': [22.2, 26.8]}
    dtype: Map(string, List(float64)), length: 2, null_count: 0

    """
    device = device or Scope.default.device
    return Scope._Column(data, dtype=dtype, device=device)


def concat(columns: ty.List[IColumn]):
    """Returns concatenated columns."""
    return columns[0]._concat_with(columns[1:])


def if_else(cond: IColumn, left: IColumn, right: IColumn):
    """
    Return a column of elements where each of them is selected from either
    `left` column or `righ` column, depending on the value in the corresponding
    position in `cond` column.

    Examples
    --------
    >>> import torcharrow as ta
    >>> cond = ta.Column([True, False, True, False])
    >>> left = ta.Column(["a1", "a2", "a3", "a4"])
    >>> right = ta.Column(["b1", "b2", "b3", "b4"])
    >>> ta.if_else(cond, left, right)
    0  'a1'
    1  'b2'
    2  'a3'
    3  'b4'
    dtype: string, length: 4, null_count: 0, device: cpu
    """
    return cond._if_else(left, right)


# ------------------------------------------------------------------------------
# IColumn


class IColumn(ty.Sized, ty.Iterable, abc.ABC):
    """Interface for Column are n vectors (n>=1) of columns"""

    def __init__(self, device, dtype: dt.DType):
        self._device = device
        self._dtype: dt.DType = dtype

        # id handling, used for tracing...
        self.id = f"c{Scope.default.ct.next()}"

    # getters ---------------------------------------------------------------

    @property
    def device(self):
        return self._device

    @property  # type: ignore
    @traceproperty
    def dtype(self) -> dt.DType:
        """dtype of the colum/frame"""
        return self._dtype

    @property  # type: ignore
    @traceproperty
    def is_nullable(self):
        """
        EXPERIMENTAL API

        A boolean indicating whether column/frame can have nulls
        """
        return self.dtype.nullable

    @property  # type: ignore
    @traceproperty
    def length(self):
        """Return number of rows including null values"""
        return len(self)

    @property  # type: ignore
    @traceproperty
    def null_count(self):
        """Return number of null values"""
        raise self._not_supported("null_count")

    @property  # type: ignore
    @traceproperty
    def is_unique(self):
        """
        EXPERIMENTAL API

        Return boolean if data values are unique.
        """
        seen = set()
        return not any(i in seen or seen.add(i) for i in self)

    @property  # type: ignore
    @traceproperty
    def is_monotonic_increasing(self):
        """
        EXPERIMENTAL API

        Return boolean if values in the object are monotonic increasing
        """
        return self._compare(operator.lt, initial=True)

    @property  # type: ignore
    @traceproperty
    def is_monotonic_decreasing(self):
        """
        EXPERIMENTAL API

        Return boolean if values in the object are monotonic decreasing
        """
        return self._compare(operator.gt, initial=True)

    # public append/copy/cast------------------------------------------------

    @trace
    def append(self, values):
        """
        Returns column/dataframe with values appended.

        Parameters
        ----------
        values: list of values or dataframe

        Examples
        --------
        >>> import torcharrow as ta
        >>> sf = ta.Column([ ["hello", "world"], ["how", "are", "you"] ], dtype =dt.List(dt.string))
        >>> sf = sf.append([["I", "am", "fine"]])
        >>> sf
        0  ['hello', 'world']
        1  ['how', 'are', 'you']
        2  ['I', 'am', 'fine']
        dtype: List(string), length: 3, null_count: 0
        """
        # TODO use _column_copy, but for now this works...
        res = Scope._EmptyColumn(self.dtype)
        for (m, d) in self._items():
            if m:
                res._append_null()
            else:
                res._append_value(d)
        for i in values:
            res._append(i)
        return res._finalize()

    def cast(self, dtype):
        """Cast the Column to the given dtype"""
        # TODO: support non-primitive types
        if dt.is_primitive(self.dtype):
            if dt.is_primitive(dtype):
                fun = dt.cast_as(dtype)
                res = Scope._EmptyColumn(dtype)
                for m, i in self.item():
                    if m:
                        res._append_null()
                    else:
                        res.append_value(fun(i))
                return res._finalize()
            else:
                raise TypeError('f"{astype}({dtype}) is not supported")')
        raise TypeError('f"{astype} for {type(self).__name__} is not supported")')

    # public simple observers -------------------------------------------------

    @trace
    @expression
    @abc.abstractmethod
    def __len__(self):
        """Return number of rows including null values"""
        raise self._not_supported("__len__")

    # printing ----------------------------------------------------------------

    def __str__(self):
        return f"Column([{', '.join(str(i) for i in self)}], id = {self.id})"

    def __repr__(self):
        rows = [[l if l is not None else "None"] for l in self]
        tab = tabulate(
            rows,
            tablefmt="plain",
            showindex=True,
        )
        typ = (
            f"dtype: {self._dtype}, length: {len(self)}, null_count: {self.null_count}"
        )
        return tab + dt.NL + typ

    # selectors/getters -------------------------------------------------------

    def is_valid_at(self, index):
        """
        EXPERIMENTAL API

        Return whether data at index i is valid, i.e., non-null
        """
        return not self._getmask(index)

    @trace
    @expression
    def __getitem__(self, arg):
        """
        If *arg* is a

        `n`, a number, return the row with index n
        `[n1,..,nm]` return a new column with the rows[n1],..,rows[nm]
        `[n1:n2:n3]`, return a new column slice with rows[n1:n2:n3]

        `s`, a string, return the column named s
        `[s1,..,sm]` return  dataframe having column[s1],..,column[sm]
        `[s1:s2]` return dataframe having columns[s1:s2]

        `[b1,..,bn]`, where bi are booleans, return all rows that are true
        `Column([b1..bn])` return all rows that are true
        """

        if isinstance(arg, int):
            return self._get(arg)
        elif isinstance(arg, str):
            return self.get_column(arg)
        elif isinstance(arg, slice):
            args = []
            for i in [arg.start, arg.stop, arg.step]:
                if isinstance(i, np.integer):
                    args.append(int(i))
                else:
                    args.append(i)
            if all(a is None or isinstance(a, int) for a in args):
                return self._slice(*args)
            elif all(a is None or isinstance(a, str) for a in args):
                if arg.step is not None:
                    raise TypeError(f"column slice can't have step argument {arg.step}")
                return self.slice_columns(arg.start, arg.stop)
            else:
                raise TypeError(
                    f"slice arguments {[type(a) for a in args]} should all be int or string"
                )
        elif isinstance(arg, (tuple, list)):
            if len(arg) == 0:
                return self
            if all(isinstance(a, bool) for a in arg):
                return self.filter(arg)
            if all(isinstance(a, int) for a in arg):
                return self._gets(arg)
            if all(isinstance(a, str) for a in arg):
                return self.get_columns(arg)
            else:
                raise TypeError("index should be list of int or list of str")
        elif isinstance(arg, IColumn) and dt.is_boolean(arg.dtype):
            return self.filter(arg)
        else:
            raise self._not_supported("__getitem__")

    @trace
    @expression
    def head(self, n=5):
        """
        Return the first `n` rows.

        Parameters
        ----------
        n : int
            Number of rows to return.

        See Also
        --------
        icolumn.tail : Return the last `n` rows.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.DataFrame({'a': list(range(7)),
        >>>             'b': list(reversed(range(7))),
        >>>             'c': list(range(7))
        >>>            })
        >>> df.head(2)
          index    a    b    c    d
        -------  ---  ---  ---  ---
              0    0    6    0   99
              1    1    5    1  100
        dtype: Struct([Field('a', int64), Field('b', int64), Field('c', int64), Field('d', int64)]), count: 2, null_count: 0
        """
        return self[:n]

    @trace
    @expression
    def tail(self, n=5):
        """
        EXPERIMENTAL API

        Return the last `n` rows.

        Parameters
        ----------
        n : int
            Number of rows to return.

        See Also
        --------
        icolumn.head : Return the first `n` rows.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.DataFrame({'a': list(range(7)),
        >>>             'b': list(reversed(range(7))),
        >>>             'c': list(range(7))
        >>>            })
        >>> df.tail(1)
          index    a    b    c    d
        -------  ---  ---  ---  ---
              0    6    0    6  105
        dtype: Struct([Field('a', int64), Field('b', int64), Field('c', int64), Field('d', int64)]), count: 2, null_count: 0
        """
        return self[-n:]

    # iterators

    def __iter__(self):
        """Return the iterator object itself."""
        for i in range(len(self)):
            yield self._get(i)

    # functools map/filter/flatmap/transform

    @trace
    @expression
    def map(
        self,
        arg: ty.Union[ty.Dict, ty.Callable],
        na_action: ty.Literal["ignore", None] = None,
        dtype: ty.Optional[dt.DType] = None,
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Maps rows according to input correspondence.

        Parameters
        ----------
        arg - dict or callable
            If arg is a dict then input is mapped using this dict and
            non-mapped values become null.  If arg is a callable, this
            is treated as a user-defined function (UDF) which is
            invoked on each element of the input.  Callables must be
            global functions or methods on class instances, lambdas
            are not supported.
        na_action - "ignore" or None, default None
            If your UDF returns null for null input, selecting
            "ignore" is an efficiency improvement where map will avoid
            calling your UDF on null values.  If None, aways calls the
            UDF.
        dtype - DType, default None
            DType is used to force the output type.  DType is required
            if result type != item type.
        columns - list of column names, default None
            Determines which columns to provide to the mapping dict or UDF.

        See Also
        --------
        flatmap, filter, reduce

        Examples
        --------
        >>> import torcharrow as ta
        >>> ta.Column([1,2,None,4]).map({1:111})
        0  111
        1  None
        2  None
        3  None
        dtype: Int64(nullable=True), length: 4, null_count: 3

        Using a defaultdict to provide a missing value:

        >>> from collections import defaultdict
        >>> ta.Column([1,2,None,4]).map(defaultdict(lambda: -1, {1:111}))
        0  111
        1   -1
        2   -1
        3   -1
        dtype: Int64(nullable=True), length: 4, null_count: 0

        Using user-supplied python function:

        >>> def add_ten(num):
        >>>     return num + 10
        >>>
        >>> ta.Column([1,2,None,4]).map(add_ten, na_action='ignore')
        0  11
        1  12
        2  None
        3  14
        dtype: Int64(nullable=True), length: 4, null_count: 1

        Note that .map(add_ten, na_action=None) in the example above
        would fail with a type error since addten is not defined for
        None/null.  To pass nulls to a UDF, the UDF needs to prepare
        for it:

        >>> def add_ten_or_0(num):
        >>>     return 0 if num is None else num + 10
        >>>
        >>> ta.Column([1,2,None,4]).map(add_ten_or_0, na_action=None)
        0  11
        1  12
        2   0
        3  14
        dtype: Int64(nullable=True), length: 4, null_count: 0

        Mapping to different types requires a dtype parameter:

        >>> ta.Column([1,2,None,4]).map(str, dtype=dt.string)
        0  '1'
        1  '2'
        2  'None'
        3  '4'
        dtype: string, length: 4, null_count: 0

        Mapping over a DataFrame, the UDF gets the whole row as a tuple:

        >>> def add_unary(tup):
        >>>     return tup[0]+tup[1]
        >>>
        >>> ta.DataFrame({'a': [1,2,3], 'b': [1,2,3]}).map(add_unary , dtype = dt.int64)
        0  2
        1  4
        2  6
        dtype: int64, length: 3, null_count: 0

        Multi-parameter UDFs:

        >>> def add_binary(a,b):
        >>>     return a + b
        >>>
        >>> ta.DataFrame({'a': [1,2,3], 'b': ['a', 'b', 'c'], 'c':[1,2,3]}).map(add_binary, columns = ['a','c'], dtype = dt.int64)
        0  2
        1  4
        2  6
        dtype: int64, length: 3, null_count: 0

        Multi-return UDFs - functions that return more than one column
        can be specified by returning a DataFrame (also known as a
        struct column); providing the return dtype is mandatory.

        ta.DataFrame({'a': [17, 29, 30], 'b': [3,5,11]}).map(divmod, columns= ['a','b'], dtype = dt.Struct([dt.Field('quotient', dt.int64), dt.Field('remainder', dt.int64)]))
          index    quotient    remainder
        -------  ----------  -----------
              0           5            2
              1           5            4
              2           2            8
        dtype: Struct([Field('quotient', int64), Field('remainder', int64)]), count: 3, null_count: 0

        UDFs with state can be written by capturing the state in a
        (data)class and use a method as a delegate:

        >>> def fib(n):
        >>>     if n == 0:
        >>>         return 0
        >>>     elif n == 1 or n == 2:
        >>>         return 1
        >>>     else:
        >>>         return fib(n-1) + fib(n-2)
        >>>
        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class State:
        >>>     state: int
        >>>     def __post_init__(self):
        >>>         self.state = fib(self.state)
        >>>     def add_fib(self, x):
        >>>         return self.state+x
        >>>
        >>> m = State(10)
        >>> ta.Column([1,2,3]).map(m.add_fib)
        0  56
        1  57
        2  58
        dtype: int64, length: 3, null_count: 0
        """
        # to avoid applying the function to missing values, use
        #   na_action == 'ignore'
        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")
        # Using the faster fromlist construction path by collecting the results
        # in a Python list and then passing the list to the constructor
        res = []
        if isinstance(arg, defaultdict):
            for masked, i in self._items():
                if not masked:
                    res.append(arg[i])
                elif na_action is None:
                    res.append(arg[None])
                else:
                    res.append(None)
        elif isinstance(arg, dict):
            for masked, i in self._items():
                if not masked:
                    if i in arg:
                        res.append(arg[i])
                    else:
                        res.append(None)
                elif None in arg and na_action is None:
                    res.append(arg[None])
                else:
                    res.append(None)
        else:  # arg must be a function
            assert isinstance(arg, ty.Callable)
            if dtype is None:
                (dtype, _) = dt.infer_dype_from_callable_hint(arg)

            for masked, i in self._items():
                if not masked:
                    res.append(arg(i))
                elif na_action is None:
                    res.append(arg(None))
                else:  # na_action == 'ignore'
                    res.append(None)

        dtype = dtype or self._dtype
        return Scope._FromPyList(res, dtype)

    @trace
    @expression
    def transform(
        self,
        func: ty.Callable,
        /,
        dtype: ty.Optional[dt.DType] = None,
        format: str = "column",
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Like map() but invokes the callable on mini-batches of rows at a time.
        The column is passed to the callable as TorchArrow column by default.
        If `format='python'` the input is converted to python types instead.
        If `format='torch'` the input is converted to PyTorch types
        dtype required if result type != item type and the type hint is missing on the callable.
        """
        if columns is not None:
            raise TypeError("columns parameter for flat columns not supported")

        if dtype is None:
            signature = ty.get_type_hints(func)
            if "return" in signature:
                dtype = dt.dtype_from_batch_pytype(signature["return"])
            else:
                # assume it's an identity mapping
                assert self._dtype is not None
                dtype = self._dtype
            # TODO: check type annotations of inputs too in order to infer the input format

        # TODO: if func is annotated, check whether its input parameter is IColumn when format="column"
        raw_res = func(self._format_transform_column(self, format))
        return self._format_transform_result(raw_res, format, dtype, len(self))

    @trace
    @expression
    def flatmap(
        self,
        arg: ty.Union[ty.Dict, ty.Callable],
        na_action: ty.Literal["ignore", None] = None,
        dtype: ty.Optional[dt.DType] = None,
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Maps rows to list of rows according to input correspondence
        dtype required if result type != item type.
        """

        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")

        def func(x):
            return arg.get(x, None) if isinstance(arg, dict) else arg(x)

        dtype = dtype or self.dtype
        res = Scope._EmptyColumn(dtype)
        for masked, i in self._items():
            if not masked:
                res._extend(func(i))
            elif na_action is None:
                res._extend(func(None))
            else:
                res._append_null()
        return res._finalize()

    @trace
    @expression
    def filter(
        self,
        predicate: ty.Union[ty.Callable, ty.Iterable],
        columns: ty.Optional[ty.List[str]] = None,
    ):
        """
        Select rows where predicate is True.
        Different from Pandas. Use keep for Pandas filter.

        Parameters
        ----------
        predicate - callable or iterable
            A predicate function or iterable of booleans the same
            length as the column.  If an n-ary predicate, use the
            columns parameter to provide arguments.
        columns - list of string names, default None
            Which columns to invoke the filter with.  If None, apply to
            all columns.

        See Also
        --------
        map, reduce, flatmap

        Examples
        --------
        >>> ta.Column([1,2,3,4]).filter([True, False, True, False]) == ta.Column([1,2,3,4]).filter(lambda x: x%2==1)
        0  1
        1  1
        dtype: boolean, length: 2, null_count: 0
        """
        if columns is not None:
            raise TypeError(f"columns parameter for flat columns not supported")

        if not isinstance(predicate, ty.Iterable) and not callable(predicate):
            raise TypeError(
                "predicate must be a unary boolean predicate or iterable of booleans"
            )
        res = Scope._EmptyColumn(self._dtype)
        if callable(predicate):
            for x in self:
                if predicate(x):
                    res._append(x)
        elif isinstance(predicate, ty.Iterable):
            for x, p in zip(self, predicate):
                if p:
                    res._append(x)
        else:
            pass
        return res._finalize()

    @trace
    @expression
    def reduce(self, fun, initializer=None, finalizer=None):
        """
        Apply binary function cumulatively to the rows[0:],
        so as to reduce the column/dataframe to a single value

        Parameters
        ----------
        fun - callable
            Binary function to invoke via reduce.
        initializer - element, or None
            The initial value used for reduce.  If None, uses the
            first element of the column.
        finalizer - callable, or None
            Function to call on the final value.  If None the last result
            of invoking fun is returned

        Examples
        --------
        >>> import operator
        >>> import torcharrow
        >>> ta.Column([1,2,3,4]).reduce(operator.mul)
        24
        """
        if len(self) == 0:
            if initializer is not None:
                return initializer
            else:
                raise TypeError("reduce of empty sequence with no initial value")
        start = 0
        if initializer is None:
            value = self[0]
            start = 1
        else:
            value = initializer
        for i in range(start, len(self)):
            value = fun(value, self[i])
        if finalizer is not None:
            return finalizer(value)
        else:
            return value

    # sorting -------------------------------------------------------

    @trace
    @expression
    def sort(
        self,
        by: ty.Optional[ty.List[str]] = None,
        ascending=True,
        na_position: ty.Literal["last", "first"] = "last",
    ):
        """
        Sort a column/a dataframe in ascending or descending order.

        Parameters
        ----------
        by : array-like, default None
            Columns to sort by, uses all columns for comparison if None.
        ascending : bool, default True
            If true, sort in ascending order, else descending.
        na_position : {{'last', 'first'}}, default "last"
            If 'last' order nulls after non-null values, if 'first' orders
            nulls before non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.DataFrame({'a': list(range(7)),
        >>>             'b': list(reversed(range(7))),
        >>>             'c': list(range(7))
        >>>            })
        >>> df.sort(by=['c', 'b']).head(2)
          index    a    b    c    d
        -------  ---  ---  ---  ---
              0    0    6    0   99
              1    1    5    1  100
        dtype: Struct([Field('a', int64), Field('b', int64), Field('c', int64), Field('d', int64)]), count: 2, null_count: 0
        """
        if by is not None:
            raise TypeError("sorting a non-structured column can't have 'by' parameter")
        res = Scope._EmptyColumn(self.dtype)
        if na_position == "first":
            res._extend([None] * self.null_count)
        res._extend(sorted((i for i in self if i is not None), reverse=not ascending))
        if na_position == "last":
            res._extend([None] * self.null_count)
        return res._finalize()

    # operators ---------------------------------------------------------------
    @trace
    @expression
    def __add__(self, other):
        """Vectorized a + b."""
        return self._py_arithmetic_op(other, operator.add)

    @trace
    @expression
    def __radd__(self, other):
        """Vectorized b + a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.add))

    @trace
    @expression
    def __sub__(self, other):
        """Vectorized a - b."""
        return self._py_arithmetic_op(other, operator.sub)

    @trace
    @expression
    def __rsub__(self, other):
        """Vectorized b - a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.sub))

    @trace
    @expression
    def __mul__(self, other):
        """Vectorized a * b."""
        return self._py_arithmetic_op(other, operator.mul)

    @trace
    @expression
    def __rmul__(self, other):
        """Vectorized b * a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.mul))

    @trace
    @expression
    def __floordiv__(self, other):
        """Vectorized a // b."""
        return self._py_arithmetic_op(other, operator.floordiv)

    @trace
    @expression
    def __rfloordiv__(self, other):
        """Vectorized b // a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.floordiv))

    @trace
    @expression
    def __truediv__(self, other):
        """Vectorized a / b."""
        return self._py_arithmetic_op(other, operator.truediv, div="__truediv__")

    @trace
    @expression
    def __rtruediv__(self, other):
        """Vectorized b / a."""
        return self._py_arithmetic_op(
            other, IColumn._swap(operator.truediv), div="__rtruediv__"
        )

    @trace
    @expression
    def __mod__(self, other):
        """Vectorized a % b."""
        return self._py_arithmetic_op(other, operator.mod)

    @trace
    @expression
    def __rmod__(self, other):
        """Vectorized b % a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.mod))

    @trace
    @expression
    def __pow__(self, other):
        """Vectorized a ** b."""
        return self._py_arithmetic_op(other, operator.pow)

    @trace
    @expression
    def __rpow__(self, other):
        """Vectorized b ** a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.pow))

    @trace
    @expression
    def __eq__(self, other):
        """Vectorized a == b."""
        return self._py_comparison_op(other, operator.eq)

    @trace
    @expression
    def __ne__(self, other):
        """Vectorized a != b."""
        return self._py_comparison_op(other, operator.ne)

    @trace
    @expression
    def __lt__(self, other):
        """Vectorized a < b."""
        return self._py_comparison_op(other, operator.lt)

    @trace
    @expression
    def __gt__(self, other):
        """Vectorized a > b."""
        return self._py_comparison_op(other, operator.gt)

    @trace
    @expression
    def __le__(self, other):
        """Vectorized a < b."""
        return self._py_comparison_op(other, operator.le)

    @trace
    @expression
    def __ge__(self, other):
        """Vectorized a < b."""
        return self._py_comparison_op(other, operator.ge)

    @trace
    @expression
    def __or__(self, other):
        """Vectorized bitwise or operation: a | b."""
        return self._py_arithmetic_op(other, operator.__or__)

    @trace
    @expression
    def __ror__(self, other):
        """Vectorized reverse bitwise or operation: b | a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.__or__))

    @trace
    @expression
    def __xor__(self, other):
        """Vectorized bitwise exclusive or operation: a ^ b."""
        return self._py_arithmetic_op(other, operator.__xor__)

    @trace
    @expression
    def __rxor__(self, other):
        """Vectorized reverse bitwise exclusive or operation: b ^ a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.__xor__))

    @trace
    @expression
    def __and__(self, other):
        """Vectorized bitwise and operation: a & b."""
        return self._py_arithmetic_op(other, operator.__and__)

    @trace
    @expression
    def __rand__(self, other):
        """Vectorized reverse bitwise and operation: b & a."""
        return self._py_arithmetic_op(other, IColumn._swap(operator.__and__))

    @trace
    @expression
    def __invert__(self):
        """Vectorized bitwise inverse operation: ~a."""
        if dt.is_boolean(self.dtype):
            return self._vectorize(lambda a: not a, self.dtype)
        return self._vectorize(operator.__invert__, self.dtype)

    @trace
    @expression
    def __neg__(self):
        """Vectorized: -a."""
        return self._vectorize(operator.neg, self.dtype)

    @trace
    @expression
    def __pos__(self):
        """Vectorized: +a."""
        return self._vectorize(operator.pos, self.dtype)

    @trace
    @expression
    def isin(self, values: ty.Union[list, dict]):
        """
        Check whether values are contained in column.

        Parameters
        ----------
        values - array-like, column or dict
            Which values to check the presence of.

        Returns
        -------
        Boolean column of the same length as self where item x denotes if
        member x has a value contained in values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> df = ta.DataFrame({'a': list(range(7)),
        >>>             'b': list(reversed(range(7))),
        >>>             'c': list(range(7))
        >>>            })
        >>> df[df['a'].isin([5])]
          index    a    b    c    d
        -------  ---  ---  ---  ---
              0    5    1    5  104
        dtype: Struct([Field('a', int64), Field('b', int64), Field('c', int64), Field('d', int64)]), count: 1, null_count: 0

        """
        # note mask is True
        res = Scope._EmptyColumn(dt.boolean)
        for m, i in self._items():
            if m:
                res._append_value(False)
            else:
                res._append_value(i in values)
        return res._finalize()

    @trace
    @expression
    def abs(self):
        """Absolute value of each element of the series."""
        return self._vectorize(abs, self.dtype)

    @trace
    @expression
    def ceil(self):
        """Rounds each value upward to the smallest integral"""
        return self._vectorize(math.ceil, self.dtype)

    @trace
    @expression
    def floor(self):
        """Rounds each value downward to the largest integral value"""
        return self._vectorize(math.floor, self.dtype)

    @trace
    @expression
    def round(self, decimals=0):
        """Round each value in a data to the given number of decimals."""
        return self._vectorize(partial(round, ndigits=decimals), self.dtype)

    # data cleaning -----------------------------------------------------------

    @trace
    @expression
    def fill_null(self, fill_value: ty.Union[dt.ScalarTypes, ty.Dict]):
        """
        Fill null values using the specified method.

        Parameters
        ----------
        fill_value : int, float, bool, or str

        See Also
        --------
        icolumn.drop_null : Return a column/frame with rows removed where a
        row has any or all nulls.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column([1,2,None,4])
        >>> s.fill_null(999)
        0    1
        1    2
        2  999
        3    4
        dtype: int64, length: 4, null_count: 0

        """
        if not isinstance(fill_value, IColumn._scalar_types):
            raise TypeError(f"fill_null with {type(fill_value)} is not supported")
        if isinstance(fill_value, IColumn._scalar_types):
            res = Scope._EmptyColumn(self.dtype.constructor(nullable=False))
            for m, i in self._items():
                if not m:
                    res._append_value(i)
                else:
                    res._append_value(fill_value)
            return res._finalize()
        else:
            raise TypeError(f"fill_null with {type(fill_value)} is not supported")

    @trace
    @expression
    def drop_null(self, how: ty.Literal["any", "all", None] = None):
        """
        Return a column/frame with rows removed where a row has any or all
        nulls.

        Parameters
        ----------
        how : {{'any','all', None}}, default None
            If 'any' drop row if any column is null.  If 'all' drop row if
            all columns are null.

        See Also
        --------
        icolumn.fill_null : Fill NA/NaN values using the specified method.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column([1,2,None,4])
        >>> s.drop_null()
        0    1
        1    2
        2    4
        dtype: int64, length: 3, null_count: 0
        """
        if how is not None:
            # "any or "all" is only used for DataFrame
            raise TypeError(f"how parameter for flat columns not supported")

        if dt.is_primitive(self.dtype):
            res = Scope._EmptyColumn(self.dtype.constructor(nullable=False))
            for m, i in self._items():
                if not m:
                    res._append_value(i)
            return res._finalize()
        else:
            raise TypeError(f"drop_null for type {self.dtype} is not supported")

    @trace
    @expression
    def drop_duplicates(
        self,
        subset: ty.Union[str, ty.List[str], ty.Literal[None]] = None,
        keep: ty.Literal["first", "last", False] = "first",
    ):
        """
        EXPERIMENTAL API

        Remove duplicate values from row/frame but keep the first, last, none
        """
        # TODO Add functionality for first and last
        assert keep == "first"
        if subset is not None:
            raise TypeError(f"subset parameter for flat columns not supported")
        res = Scope._EmptyColumn(self._dtype)
        res._extend(list(OrderedDict.fromkeys(self)))
        return res._finalize()

    # aggregation

    @trace
    @expression
    def min(self):
        """
        Return the minimum of the non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column([1,2,None,4])
        >>> s.min()
        1
        """
        import pyarrow.compute as pc

        # TODO: use pc.min once upgrade to later version of PyArrow
        return pc.min_max(self.to_arrow())[0].as_py()

    @trace
    @expression
    def max(self):
        """
        Return the maximum of the non-null values.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column([1,2,None,4])
        >>> s.max()
        4
        """
        import pyarrow.compute as pc

        # TODO: use pc.max once upgrade to later version of PyArrow
        return pc.min_max(self.to_arrow())[1].as_py()

    @trace
    @expression
    def sum(self):
        """
        Return sum of all non-null elements.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column([1,2,None,4])
        >>> s.sum()
        7
        """
        self._check(dt.is_numerical, "sum")
        return sum(self._data_iter())

    @trace
    @expression
    def mean(self):
        self._check(dt.is_numerical, "mean")
        """
        Return the mean of the non-null values in the series.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column([1,2,None,4])
        >>> s.mean(fill_value=999)
        251.5
        """
        m = statistics.mean((float(i) for i in list(self._data_iter())))
        return m

    @trace
    @expression
    def std(self):
        """Return the stddev(s) of the data."""
        self._check(dt.is_numerical, "std")
        return statistics.stdev((float(i) for i in list(self._data_iter())))

    @trace
    @expression
    def median(self):
        """Return the median of the values in the data."""
        self._check(dt.is_numerical, "median")
        return statistics.median((float(i) for i in list(self._data_iter())))

    @trace
    @expression
    def quantile(self, q, interpolation="midpoint"):
        """Compute the q-th percentile of non-null data."""
        if interpolation != "midpoint":
            raise TypeError(
                f"quantile for '{type(self).__name__}' with parameter other than 'midpoint' not supported "
            )
        if len(self) == 0 or len(q) == 0:
            return []
        out = []
        s = sorted(self)
        for percent in q:
            k = (len(self) - 1) * (percent / 100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                out.append(s[int(k)])
                continue
            d0 = s[int(f)] * (c - k)
            d1 = s[int(c)] * (k - f)
            out.append(d0 + d1)
        return out

    @trace
    @expression
    def mode(self):
        """Return the mode(s) of the data."""
        self._check(dt.is_numerical, "mode")
        return statistics.mode(self._data_iter())

    @trace
    @expression
    def all(self):
        """Return whether all non-null elements are True"""
        return all(self._data_iter())

    @trace
    @expression
    def any(self):
        """Return whether any non-null element is True in Column"""
        return any(self._data_iter())

    # cummin/cummax/cumsum/cumprod
    @trace
    @expression
    def cummin(self):
        """Return cumulative minimum of the data."""
        return self._accumulate(min)

    @trace
    @expression
    def cummax(self):
        """Return cumulative maximum of the data."""
        return self._accumulate(max)

    @trace
    @expression
    def cumsum(self):
        """Return cumulative sum of the data."""
        self._check(dt.is_numerical, "cumsum")
        return self._accumulate(operator.add)

    @trace
    @expression
    def cumprod(self):
        """Return cumulative product of the data."""
        self._check(dt.is_numerical, "cumprod")
        return self._accumulate(operator.mul)

    # describe ----------------------------------------------------------------
    @trace
    @expression
    def describe(
        self,
        percentiles_=None,
        include_columns: ty.Union[ty.List[dt.DType], ty.Literal[None]] = None,
        exclude_columns: ty.Union[ty.List[dt.DType], ty.Literal[None]] = None,
    ):
        """
        Generate descriptive statistics.

        Parameters
        ----------
        percentiles_ - array-like, default None
            Defines which percentiles to calculate.  If None, uses [25,50,75].

        Examples
        --------
        >>> import torcharrow
        >>> t = ta.Column([1,2,999,4])
        >>> t.describe()
          index  statistic      value
        -------  -----------  -------
              0  count          4
              1  mean         251.5
              2  std          498.335
              3  min            1
              4  25%            1.5
              5  50%            3
              6  75%          501.5
              7  max          999
        """
        import torcharrow.idataframe

        # Not supported: datetime_is_numeric=False,
        if include_columns is not None or exclude_columns is not None:
            raise TypeError(
                f"'include/exclude columns' parameter for '{type(self).__name__}' not supported "
            )
        if percentiles_ is None:
            percentiles_ = [25, 50, 75]
        percentiles_ = sorted(set(percentiles_))
        if len(percentiles_) > 0:
            if percentiles_[0] < 0 or percentiles_[-1] > 100:
                raise ValueError("percentiles must be betwen 0 and 100")

        if dt.is_numerical(self.dtype):
            res = Scope._EmptyColumn(
                dt.Struct(
                    [dt.Field("statistic", dt.string), dt.Field("value", dt.float64)]
                )
            )
            res._append(("count", self._count()))
            res._append(("mean", self.mean()))
            res._append(("std", self.std()))
            res._append(("min", self.min()))
            values = self.quantile(percentiles_, "midpoint")
            for p, v in zip(percentiles_, values):
                res._append((f"{p}%", v))
            res._append(("max", self.max()))
            return res._finalize()
        else:
            raise ValueError(f"describe undefined for {type(self).__name__}.")

    # interop

    @trace
    def to_pandas(self):
        """Convert self to pandas dataframe"""
        import pandas as pd  # type: ignore

        # default implementation, normally this should be zero copy...
        return pd.Series(self)

    @trace
    def to_arrow(self):
        """Convert self to pandas dataframe"""
        import pyarrow as pa  # type: ignore

        # default implementation, normally this should be zero copy...
        return pa.array(self)

    @trace
    def to_pylist(self):
        """Convert to plain Python container (list of scalars or containers)"""
        return list(self)

    @trace
    def to_torch(self):
        """Convert to PyTorch containers (Tensor, PackedList, PackedMap, etc)"""
        raise NotImplementedError()

    # batching/unbatching
    # NOTE experimental
    def batch(self, n):
        """
        EXPERIMENTAL API
        """
        assert n > 0
        i = 0
        while i < len(self):
            h = i
            i = i + n
            yield self[h:i]

    @staticmethod
    def unbatch(iter: ty.Iterable[IColumn]):
        """
        EXPERIMENTAL API
        """
        res = []
        for i in iter:
            res.append(i)
        if len(res) == 0:
            raise ValueError("can't determine column type")
        return ta.concat(res)

    # private helpers

    def _not_supported(self, name):
        raise TypeError(f"{name} for type {type(self).__name__} is not supported")

    _scalar_types = (int, float, bool, str)

    @trace
    @abc.abstractmethod
    def _append_null(self):
        """PRIVATE _append null value with updateing mask"""
        raise self._not_supported("_append_null")

    @trace
    @abc.abstractmethod
    def _append_value(self, value):
        """PRIVATE _append non-null value with updateing mask"""
        raise self._not_supported("_append_value")

    @trace
    def _append(self, value):
        """PRIVATE _append value"""
        if value is None:
            self._append_null()
        else:
            self._append_value(value)

    def _extend(self, values):
        """PRIVATE _extend values"""
        for value in values:
            self._append(value)

    def _vectorize(self, fun, dtype: dt.DType):
        return self.map(fun, "ignore", dtype)

    @staticmethod
    def _swap(op):
        return lambda a, b: op(b, a)

    def _check(self, pred, name):
        if not pred(self.dtype):
            raise ValueError(f"{name} undefined for {type(self).__name__}.")

    @staticmethod
    def _isin(values):
        return lambda value: value in values

    def _py_arithmetic_op(self, other, fun, div=""):
        others = None
        other_dtype = None
        if isinstance(other, IColumn):
            others = other._items()
            other_dtype = other.dtype
        else:
            others = itertools.repeat((False, other))
            other_dtype = dt.infer_dtype_from_value(other)

        if not dt.is_boolean_or_numerical(self.dtype) or not dt.is_boolean_or_numerical(
            other_dtype
        ):
            raise TypeError(f"{type(self).__name__}.{fun.__name__} is not supported")

        res = []
        if div != "":
            res_dtype = dt.Float64(self.dtype.nullable or other_dtype.nullable)
            for (m, i), (n, j) in zip(self._items(), others):
                # TODO Use error handling to mke this more efficient..
                if m or n:
                    res.append(None)
                elif div == "__truediv__" and j == 0:
                    res.append(None)
                elif div == "__rtruediv__" and i == 0:
                    res.append(None)
                else:
                    res.append(fun(i, j))
        else:
            res_dtype = dt.promote(self.dtype, other_dtype)
            if res_dtype is None:
                raise TypeError(f"{self.dtype} and {other_dtype} are incompatible")
            for (m, i), (n, j) in zip(self._items(), others):
                if m or n:
                    res.append(None)
                else:
                    res.append(fun(i, j))
        return Scope._FromPyList(res, res_dtype)

    def _py_comparison_op(self, other, pred):
        others = None
        other_dtype = None
        if isinstance(other, IColumn):
            others = itertools.chain(other._items(), itertools.repeat((True, None)))
            other_dtype = other.dtype
        else:
            others = itertools.repeat((False, other))
            other_dtype = dt.infer_dtype_from_value(other)
        res_dtype = dt.Boolean(self.dtype.nullable or other_dtype.nullable)
        res = []
        for (m, i), (n, j) in zip(self._items(), others):
            if m or n:
                res.append(None)
            else:
                res.append(pred(i, j))
        return Scope._FromPyList(res, res_dtype)

    def _compare(self, op, initial):
        assert initial in [True, False]
        if len(self) == 0:
            return initial
        it = iter(self)
        start = next(it)
        for step in it:
            if step is None:
                continue
            if op(start, step):
                start = step
                continue
            else:
                return False
        return True

    def _accumulate(self, func):
        total = None
        res = Scope._EmptyColumn(self.dtype)
        for m, i in self._items():
            if m:
                res._append_null()
            elif total is None:
                res._append_value(i)
                total = i
            else:
                total = func(total, i)
                res._append_value(total)
        m = res._finalize()
        return m

    @staticmethod
    def _format_transform_column(c: IColumn, format: str):
        if format == "column":
            return c
        if format == "python":
            return c.to_pylist()
        if format == "torch":
            return c.to_torch()
        raise ValueError(f"Invalid value for `format` argument: {format}")

    def _format_transform_result(
        self, raw: ty.Any, format: str, dtype: dt.DType, length: int
    ):
        if format == "torch":
            from . import pytorch

            pytorch.ensure_available()
            ret = pytorch.from_torch(raw, dtype=dtype)
        elif format == "python" or format == "column":
            ret = ta.Column(raw, dtype=dtype)
        else:
            raise ValueError(f"Invalid value for `format` argument: {format}")

        if len(ret) != length:
            raise ValueError(
                f"Output of transform must return the same number of rows: got {len(ret)} instead of {length}"
            )
        return ret

    @abc.abstractmethod
    def _getmask(self, i):
        """Return mask at index i"""
        raise self._not_supported("_getmask")

    @abc.abstractmethod
    def _getdata(self, i):
        """Return data at index i"""
        raise self._not_supported("getdata")

    def _get(self, index, fill_value=None):
        """Return data[index] or fill_value if data[i] not valid"""
        if self._getmask(index):
            return fill_value
        else:
            return self._getdata(index)

    def _gets(self, indices):
        """Return a new column with the rows[indices[0]],..,rows[indices[-1]]"""
        res = Scope._EmptyColumn(self.dtype)
        for i in indices:
            (m, d) = (self._getmask(i), self._getdata(i))
            if m:
                res._append_null()
            else:
                res._append_value(d)
        return res._finalize()

    def _slice(self, start, stop, step):
        """Return a new column with the slice rows[start:stop:step]"""
        res = Scope._EmptyColumn(self.dtype)
        for i in list(range(len(self)))[start:stop:step]:
            m = self._getmask(i)
            if m:
                res._append_null()
            else:
                res._append_value(self._getdata(i))
        return res._finalize()

    def _items(self):
        """Iterator returning mask,data pairs for all items of a column"""
        for i in range(len(self)):
            yield (self._getmask(i), self._getdata(i))

    def _data_iter(self, fill_value=None):
        """Iterator returning non-null or fill_value data of a column"""
        for m, i in self._items():
            if m:
                if fill_value is not None:
                    yield fill_value
            else:
                yield i

    @abc.abstractmethod
    def _concat_with(self, columns: ty.List[IColumn]):
        """Returns concatenated columns."""
        raise self._not_supported("_concat_with")

    def _if_else(self, then_, else_):
        """Vectorized if-then-else"""
        if not dt.is_boolean(self.dtype):
            raise TypeError("condition must be a boolean vector")
        if not isinstance(then_, IColumn):
            then_ = self._Column(then_)
        if not isinstance(else_, IColumn):
            else_ = self._Column(else_)
        lub = dt.common_dtype(then_.dtype, else_.dtype)
        if lub is None or dt.is_void(lub):
            raise TypeError(
                "then and else branches must have compatible types, got {then_.dtype} and {else_.dtype}, respectively"
            )
        res = Scope._EmptyColumn(lub)
        for (m, b), t, e in zip(self._items(), then_, else_):
            if m:
                res._append_null()
            elif b:
                res._append(t)
            else:
                res._append(e)
        return res._finalize()

    # private aggregation/topK functions -- names are to be discussed

    def _count(self):
        """Return number of non-NA/null observations pgf the column/frame"""
        return len(self) - self.null_count

    @trace
    @expression
    def _nlargest(
        self,
        n=5,
        columns: ty.Optional[ty.List[str]] = None,
        keep: ty.Literal["last", "first"] = "first",
    ):
        """Returns a new data of the *n* largest element."""
        # keep="all" not supported
        if columns is not None:
            raise TypeError(
                "computing n-largest on non-structured column can't have 'columns' parameter"
            )
        return self.sort(ascending=False).head(n)

    @trace
    @expression
    def _nsmallest(self, n=5, columns: ty.Optional[ty.List[str]] = None, keep="first"):
        """Returns a new data of the *n* smallest element."""
        # keep="all" not supported
        if columns is not None:
            raise TypeError(
                "computing n-smallest on non-structured column can't have 'columns' parameter"
            )

        return self.sort(ascending=True).head(n)

    @trace
    @expression
    def _nunique(self, drop_null=True):
        """Returns the number of unique values of the column"""
        if not drop_null:
            return len(set(self))
        else:
            return len(set(i for i in self if i is not None))
