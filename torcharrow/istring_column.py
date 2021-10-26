# Copyright (c) Facebook, Inc. and its affiliates.
import abc

# TODO: use re2
import re

import numpy.ma as ma
import torcharrow.dtypes as dt

from .expression import expression
from .icolumn import IColumn

# ------------------------------------------------------------------------------
# IStringColumn


class IStringColumn(IColumn):

    # private constructor
    def __init__(self, device, dtype):  # REP offsets
        assert dt.is_string(dtype)
        super().__init__(device, dtype)
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

    def slice(self, start: int = None, stop: int = None) -> IStringColumn:
        """Slice substrings from each element in the Column."""

        def func(i):
            return i[start:stop]

        return self._vectorize_string(func)

    def split(self, sep=None):
        """
        Split strings around given separator/delimiter.

        Parameters
        ----------
        sep - str, default None
            String literal to split on.  When None split according to whitespace.

        See Also
        --------
        list.join - Join lists contained as elements with passed delimiter.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column(['what a wonderful world!', 'really?'])
        >>> s.str.split(sep=' ')
        0  ['what', 'a', 'wonderful', 'world!']
        1  ['really?']
        dtype: List(string), length: 2, null_count: 0

        """
        me = self._parent

        def fun(i):
            return i.split(sep)

        return self._vectorize_list_string(fun)

    def strip(self):
        """
        Remove leading and trailing whitespaces.

        Strip whitespaces (including newlines) from each string in the Column
        from left and right sides.
        """
        return self._vectorize_string(lambda s: s.strip())

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

    # Pattern matching related methods  -----------------------------------------------------
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

    def find(self, sub):
        def fun(i):
            return i.find(sub)

        return self._vectorize_int64(fun)

    def replace(self, old, new):
        """
        Replace each occurrence of pattern in the Column.
        """
        return self._vectorize_string(lambda s: s.replace(old, new))

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
