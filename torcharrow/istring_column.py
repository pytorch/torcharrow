# Copyright (c) Facebook, Inc. and its affiliates.
import abc
import array as ar

# TODO: use re2
import re
import typing as ty
from dataclasses import dataclass

# import re2 as re  # type: ignore

import numpy as np
import numpy.ma as ma
import torcharrow.dtypes as dt
from torcharrow.expression import Call

from .expression import expression
from .icolumn import IColumn
from .scope import ColumnFactory

# ------------------------------------------------------------------------------
# IStringColumn


class IStringColumn(IColumn):

    # private constructor
    def __init__(self, scope, device, dtype):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(scope, device, dtype)
        # must be set by subclass
        self.str: IStringMethods = None


# ------------------------------------------------------------------------------
# IStringMethods


class IStringMethods(abc.ABC):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent):
        self._parent: IStringColumn = parent

    def length(self):
        return self._vectorize_int64(len)

    @abc.abstractmethod
    def cat(self, others=None, sep: str = "", fill_value: str = None) -> IStringColumn:
        """
        Concatenate strings with given separator and n/a substitition.
        """
        raise self._not_supported("cat")

    def slice(
        self, start: int = None, stop: int = None, step: int = None
    ) -> IStringColumn:
        """Slice substrings from each element in the Column."""

        def func(i):
            return i[start:stop:step]

        return self._vectorize_string(func)

    def split(self, sep=None, maxsplit=-1, expand=False):
        """Split strings around given separator/delimiter."""
        me = self._parent

        if not expand:

            def fun(i):
                return i.split(sep, maxsplit)

            return self._vectorize_list_string(fun)
        else:
            if maxsplit < 1:
                raise ValueError("maxsplit must be >0")

            def fun(i):
                ws = i.split(sep, maxsplit)
                return tuple(ws + ([None] * (maxsplit + 1 - len(ws))))

            dtype = dt.Struct(
                [
                    dt.Field(str(i), dt.String(nullable=True))
                    for i in range(maxsplit + 1)
                ],
                nullable=me.dtype.nullable,
            )

            return me._vectorize(fun, dtype=dtype)

    def rsplit(self, sep=None, maxsplit=-1, expand=False):
        """Split strings around given separator/delimiter."""
        me = self._parent

        if not expand:

            def fun(i):
                return i.rsplit(sep, maxsplit)

            return self._vectorize_list_string(fun)
        else:
            if maxsplit < 1:
                raise ValueError("maxsplit must be >0")

            def fun(i):
                ws = i.rsplit(sep, maxsplit)
                return tuple(
                    ([None] * (maxsplit + 1 - len(ws))) + i.rsplit(sep, maxsplit)
                )

            dtype = dt.Struct(
                [
                    dt.Field(str(i), dt.String(nullable=True))
                    for i in range(maxsplit + 1)
                ],
                nullable=me.dtype.nullable,
            )

            return me._vectorize(fun, dtype=dtype)

    def repeat(self, repeats):
        """
        Repeat elements of a Column.
        """
        return self._vectorize_string(lambda s: s * repeats)

    def translate(self, table):
        """
        Map all characters in the string through the given mapping table.
        Equivalent to standard :meth:`str.translate`.
        """

        def fun(i):
            return i.translate(table)

        return self._vectorize_string(fun)

    def strip(self, chars=None):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the Column from left and right sides.
        Equivalent to :meth:`str.strip`.
        """
        return self._vectorize_string(lambda s: s.strip(chars))

    # Check whether all characters in each string are  -----------------------------------------------------
    # alphabetic/numeric/digits/decimal...

    def isalpha(self):
        return self._vectorize_boolean(str.isalpha)

    def isnumeric(self):
        return self._vectorize_boolean(str.isnumeric)

    def isalnum(self):
        return self._vectorize_boolean(str.isalnum)

    def isdigit(self):
        return self._vectorize_boolean(str.isdigit)

    def isdecimal(self):
        return self._vectorize_boolean(str.isdecimal)

    def isspace(self):
        return self._vectorize_boolean(str.isspace)

    def islower(self):
        return self._vectorize_boolean(str.islower)

    def isupper(self):
        return self._vectorize_boolean(str.isupper)

    def istitle(self):
        return self._vectorize_boolean(str.istitle)

    # Convert strings in the Column -----------------------------------------------------

    def lower(self) -> IStringColumn:
        """
        Convert strings in the Column to lowercase.
        Equivalent to :meth:`str.lower`.
        """
        return self._vectorize_string(str.lower)

    def upper(self) -> IStringColumn:
        """
        Convert strings in the Column to uppercase.
        Equivalent to :meth:`str.upper`.
        """
        return self._vectorize_string(str.upper)

    def title(self) -> IStringColumn:
        """
        Convert strings in the Column to titlecase.
        Equivalent to :meth:`str.title`.
        """
        return self._vectorize_string(str.title)

    def capitalize(self):
        """
        Convert strings in the Column to be capitalized.
        Equivalent to :meth:`str.capitalize`.
        """
        return self._vectorize_string(str.capitalize)

    def swapcase(self) -> IStringColumn:
        """
        Convert strings in the Column to be swapcased.
        Equivalent to :meth:`str.swapcase`.
        """
        return self._vectorize_string(str.swapcase)

    def casefold(self) -> IStringColumn:
        """
        Convert strings in the Column to be casefolded.
        Equivalent to :meth:`str.casefold`.
        """
        return self._vectorize_string(str.casefold)

    # Pad strings in the Column -----------------------------------------------------
    def pad(self, width, side="left", fillchar=" "):
        fun = None
        if side == "left":

            def fun(i):
                return i.ljust(width, fillchar)

        if side == "right":

            def fun(i):
                return i.rjust(width, fillchar)

        if side == "center":

            def fun(i):
                return i.center(width, fillchar)

        return self._vectorize_string(fun)

    def ljust(self, width, fillchar=" "):
        def fun(i):
            return i.ljust(width, fillchar)

        return self._vectorize_string(fun)

    def rjust(self, width, fillchar=" "):
        def fun(i):
            return i.rjust(width, fillchar)

        return self._vectorize_string(fun)

    def center(self, width, fillchar=" "):
        def fun(i):
            return i.center(width, fillchar)

        return self._vectorize_string(fun)

    def zfill(self, width):
        def fun(i):
            return i.zfill(width)

        return self._vectorize_string(fun)

    # Pattern matching related methods  -----------------------------------------------------

    def count(self, pat):
        """Count occurrences of pattern in each string"""

        def fun(i):
            return i.count(pat)

        return self._vectorize_int64(fun)

    def startswith(self, pat):
        """Test if the beginning of each string element matches a pattern."""

        def pred(i):
            return i.startswith(pat)

        return self._vectorize_boolean(pred)

    def endswith(self, pat):
        """Test if the end of each string element matches a pattern."""

        def pred(i):
            return i.endswith(pat)

        return self._vectorize_boolean(pred)

    def find(self, sub, start=0, end=None):
        def fun(i):
            return i.find(sub, start, end)

        return self._vectorize_int64(fun)

    def rfind(self, sub, start=0, end=None):
        def fun(i):
            return i.rfind(sub, start, end)

        return self._vectorize_int64(fun)

    def replace(self, old, new, count=-1):
        """
        Replace each occurrence of pattern in the Column.
        """
        return self._vectorize_string(lambda s: s.replace(old, new, count))

    # helper -----------------------------------------------------

    def _vectorize_boolean(self, pred):
        return self._parent._vectorize(pred, dt.Boolean(self._parent.dtype.nullable))

    def _vectorize_string(self, func):
        return self._parent._vectorize(func, dt.String(self._parent.dtype.nullable))

    def _vectorize_list_string(self, func):
        return self._parent._vectorize(
            func, dt.List(dt.string, self._parent.dtype.nullable)
        )

    def _vectorize_int64(self, func):
        return self._parent._vectorize(func, dt.Int64(self._parent.dtype.nullable))

    def _not_supported(self, name):
        raise TypeError(f"{name} for type {type(self).__name__} is not supported")

    # Regular expressions -----------------------------------------------------
    #
    # Only allow string type for pattern input so it can be dispatch to other runtime (Velox, cuDF, etc)

    def count_re(self, pattern: str):
        """Count occurrences of pattern in each string"""
        return self.findall_re(pattern).list.length()

    def match_re(self, pattern: str):
        """Determine if each string matches a regular expression (see re.match())"""
        pattern = re.compile(pattern)

        def func(text):
            return True if pattern.match(text) else False

        return self._vectorize_boolean(func)

    def replace_re(self, pattern: str, repl: str, count=0):
        """Replace for each item the search string or pattern with the given value"""

        pattern = re.compile(pattern)

        def func(text):
            return re.sub(pattern, repl, text, count)

        return self._vectorize_string(func)

    def contains_re(
        self,
        pattern: str,
    ):
        """Test for each item if pattern is contained within a string; returns a boolean"""

        pattern = re.compile(pattern)

        def func(text):
            return pattern.search(text) is not None

        return self._vectorize_boolean(func)

    def findall_re(self, pattern: str):
        """
        Find for each item all occurrences of pattern (see re.findall())
        """
        pattern = re.compile(pattern)

        def func(text):
            return pattern.findall(text)

        return self._vectorize_list_string(func)
