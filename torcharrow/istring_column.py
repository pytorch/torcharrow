# Copyright (c) Facebook, Inc. and its affiliates.
import abc

import torcharrow.dtypes as dt

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

    def split(self, pat=None):
        """
        Split strings around given separator/delimiter.

        Parameters
        ----------
        pat - str, default None
            String literal to split on, does not yet support regular expressions.
            When None split according to whitespace.

        See Also
        --------
        list.join - Join lists contained as elements with passed delimiter.

        Examples
        --------
        >>> import torcharrow as ta
        >>> s = ta.Column(['what a wonderful world!', 'really?'])
        >>> s.str.split(pat=' ')
        0  ['what', 'a', 'wonderful', 'world!']
        1  ['really?']
        dtype: List(string), length: 2, null_count: 0

        """
        self._not_supported("split")

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
        self._parent._prototype_support_warning("str.isalpha")
        return self._vectorize_boolean(str.isalpha)

    def isnumeric(self):
        self._parent._prototype_support_warning("str.isnumeric")
        return self._vectorize_boolean(str.isnumeric)

    def isalnum(self):
        self._parent._prototype_support_warning("str.isalnum")
        return self._vectorize_boolean(str.isalnum)

    def isdigit(self):
        self._parent._prototype_support_warning("str.isdigit")
        return self._vectorize_boolean(str.isdigit)

    def isdecimal(self):
        self._parent._prototype_support_warning("str.isdecimal")
        return self._vectorize_boolean(str.isdecimal)

    def isspace(self):
        self._parent._prototype_support_warning("str.isspace")
        return self._vectorize_boolean(str.isspace)

    def islower(self):
        self._parent._prototype_support_warning("str.islower")
        return self._vectorize_boolean(str.islower)

    def isupper(self):
        self._parent._prototype_support_warning("str.isupper")
        return self._vectorize_boolean(str.isupper)

    def istitle(self):
        self._parent._prototype_support_warning("str.istitle")
        return self._vectorize_boolean(str.istitle)

    # Convert strings in the Column -----------------------------------------------------

    def lower(self) -> IStringColumn:
        """
        Convert strings in the Column to lowercase.
        Equivalent to :meth:`str.lower`.
        """
        self._parent._prototype_support_warning("str.lower")
        return self._vectorize_string(str.lower)

    def upper(self) -> IStringColumn:
        """
        Convert strings in the Column to uppercase.
        Equivalent to :meth:`str.upper`.
        """
        self._parent._prototype_support_warning("str.upper")
        return self._vectorize_string(str.upper)

    # Pattern matching related methods  -----------------------------------------------------
    def startswith(self, pat):
        """Test if the beginning of each string element matches a pattern."""
        self._parent._prototype_support_warning("str.startswith")

        def pred(i):
            return i.startswith(pat)

        return self._vectorize_boolean(pred)

    def endswith(self, pat):
        """Test if the end of each string element matches a pattern."""
        self._parent._prototype_support_warning("str.endswith")

        def pred(i):
            return i.endswith(pat)

        return self._vectorize_boolean(pred)

    def count(self, pat: str):
        """Count occurrences of pattern in each string of column"""
        # TODO: caculating the count without materializing all the occurance?
        return self.findall(pat).list.length()

    def find(self, sub):
        self._not_supported("find")

    def replace(self, pat: str, repl: str, regex: bool = True):
        """
        Replace each occurrence of pattern in the Column.
        """
        self._not_supported("replace")

    def match(self, pat: str):
        """Determine if each string matches a regular expression"""
        self._not_supported("match")

    def contains(self, pat: str, regex: bool = True):
        """Test for each item if pattern is contained within a string; returns a boolean"""
        self._not_supported("contains")

    def findall(self, pat: str):
        """
        Find for each item all occurrences of pattern (see re.findall())
        """
        self._not_supported("findall")

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
