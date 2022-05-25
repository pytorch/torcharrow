# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import array as ar
import builtins
import functools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torcharrow.dtypes as dt

from .icolumn import Column

# -----------------------------------------------------------------------------
# ListColumn


class ListColumn(Column):

    # private constructor
    def __init__(self, device, dtype):
        assert dt.is_list(dtype)
        super().__init__(device, dtype)
        self.list = ListMethods(self)


# -----------------------------------------------------------------------------
# ListMethods


class ListMethods(abc.ABC):
    """Vectorized list functions for ListColumn"""

    def __init__(self, parent):
        self._parent: ListColumn = parent

    def length(self):
        """
        Compute the length of each element in the Column.
        """
        me = self._parent
        return me._vectorize(len, dt.Int64(me.dtype.nullable))

    def join(self, sep):
        """
        Join lists contained as elements with passed delimiter.

        Parameters
        ----------
        sep - string
            Separator string placed between elements in the result

        See Also
        --------
        str.split - Split strings around given separator/delimiter.

        Examples
        >>> import torcharrow as ta
        >>> s = ta.column(['what a wonderful world!', 'really?'])
        >>> s.str.split(sep=' ').list.join(sep='-')
        0  'what-a-wonderful-world!'
        1  'really?'
        dtype: string, length: 2, null_count: 0
        """
        self._parent._prototype_support_warning("list.join")

        me = self._parent
        assert dt.is_string(me.dtype.item_dtype)

        def fun(i):
            return sep.join(i)

        return me._vectorize(
            fun, dt.String(me.dtype.item_dtype.nullable or me.dtype.nullable)
        )

    def get(self, i):
        self._parent._prototype_support_warning("list.get")

        me = self._parent

        def fun(xs):
            return xs[i]

        return me._vectorize(fun, me.dtype.item_dtype.with_null(me.dtype.nullable))

    def slice(self, start: int = 0, stop: Optional[int] = None) -> ListColumn:
        """
        Slice sublist from each element in the column

        Parameters
        ----------
        start - int, default 0
            Start position for slice operation. Negative starting position is not supported yet.
        start - int, optional
            Stop position for slice operation. Negative stop position is not supported yet.
        """
        raise NotImplementedError

    def vmap(self, fun: Callable[[Column], Column]):
        """
        (EXPERIMENTAL API) Vectorizing map. Expects a callable that working on a batch (represents by a Column).

        Examples
        --------
        >>> import torcharrow as ta
        >>> a = ta.column([[1, 2, None, 3], [4, None, 5]])

        >>> a
        0  [1, 2, None, 3]
        1  [4, None, 5]
        dtype: List(Int64(nullable=True)), length: 2, null_count: 0

        >>> a.list.vmap(lambda col: col + 1)
        0  [2, 3, None, 4]
        1  [5, None, 6]
        dtype: List(Int64(nullable=True), nullable=True), length: 2, null_count: 0

        >>> import torcharrow.dtypes as dt
        >>> b = ta.column([[(1, "a"), (2, "b")], [(3, "c")]],
            dtype=dt.List(
                dt.Struct([dt.Field("f1", dt.int64), dt.Field("f2", dt.string)])
            ))

        >>> b
        0  [(1, 'a'), (2, 'b')]
        1  [(3, 'c')]
        dtype: List(Struct([Field('f1', int64), Field('f2', string)])), length: 2, null_count: 0

        >>> b.list.vmap(lambda df: df["f2"])
        0  ['a', 'b']
        1  ['c']
        dtype: List(String(nullable=True), nullable=True), length: 2, null_count: 0
        """
        raise NotImplementedError()

    # functional tools
    def map(self, fun, dtype: Optional[dt.DType] = None):
        me = self._parent

        def func(xs):
            return list(builtins.map(fun, xs))

        if dtype is None:
            dtype = me.dtype
        return me._vectorize(func, dtype)

    def filter(self, pred):
        print()
        me = self._parent

        def func(xs):
            return list(builtins.filter(pred, xs))

        return me._vectorize(func, me.dtype)

    def reduce(self, fun, initializer=None, dtype: Optional[dt.DType] = None):
        me = self._parent

        def func(xs):
            return functools.reduce(fun, xs, initializer)

        # pyre-fixme[16]: `DType` has no attribute `item_dtype`.
        dtype = me.dtype.item_dtype if dtype is None else dtype
        return me._vectorize(func, dtype)

    def flatmap(self, fun, dtype: Optional[dt.DType] = None):
        # dtype must be given, if result is different from argument column
        me = self._parent

        def func(xs):
            ys = []
            for x in xs:
                ys._extend(fun(x))
            return ys

        return me._vectorize(func, me.dtype)


# ops on list  --------------------------------------------------------------
#  'count',
#  'extend',
#  'index',
#  'insert',
#  'pop',
#  'remove',
#  'reverse',
