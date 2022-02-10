# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import Optional

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


class IStringMethods(ABC):
    """Vectorized string functions for IStringColumn"""

    def __init__(self, parent):
        self._parent: IStringColumn = parent

    @abstractmethod
    def length(self):
        raise NotImplementedError

    @abstractmethod
    def slice(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> IStringColumn:
        """Slice substrings from each element in the Column."""
        raise NotImplementedError

    @abstractmethod
    def split(self, pat=None, n=-1):
        """
        Split strings around given separator/delimiter.

        Parameters
        ----------
        pat - str, default None
            String literal to split on, does not yet support regular expressions.
            When None split according to whitespace.
        n - int, default -1 means no limit
            Maximum number of splits to do. 0 will be interpreted as return all splits.

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
        >>> s.str.split(pat=' ', n=2)
        0  ['what', 'a', 'wonderful world!']
        1  ['really?']
        dtype: List(string), length: 2, null_count: 0

        """
        raise NotImplementedError

    @abstractmethod
    def strip(self):
        """
        Remove leading and trailing whitespaces.

        Strip whitespaces (including newlines) from each string in the Column
        from left and right sides.
        """
        raise NotImplementedError

    # Check whether all characters in each string are  -----------------------------------------------------
    # alphabetic/numeric/digits/decimal...

    @abstractmethod
    def isalpha(self):
        """
        Return True if the string is an alphabetic string, False otherwise.

        A string is alphabetic if all characters in the string are alphabetic
        and there is at least one character in the string.
        """
        raise NotImplementedError

    @abstractmethod
    def isnumeric(self):
        """
        Returns True if all the characters are numeric, otherwise False.

        A string is a numeric if each character of the string is numeric.
        """
        raise NotImplementedError

    @abstractmethod
    def isalnum(self):
        """
        Return True if all characters in the string are alphanumeric (either
        alphabets or numbers), False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def isdigit(self):
        """
        Return True if all characters in the string are numeric, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def isdecimal(self):
        """
        Return True if the string contains only decimal digit (from 0 to 9), False
        otherwise.

        A string is decimal if all characters in the string are decimal digits
        and there is at least one character in the string.
        """
        raise NotImplementedError

    @abstractmethod
    def isspace(self):
        """
        Return True all characters in the string are whitespace, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def islower(self):
        """
        Return True if the non-empty string is in lower case, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def isupper(self):
        """
        Return True if the non-empty string is in upper case, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def istitle(self):
        """
        Return True if each word of the string starts with an
        upper case letter, False otherwise.
        """
        raise NotImplementedError

    # Convert strings in the Column -----------------------------------------------------

    @abstractmethod
    def lower(self) -> IStringColumn:
        """
        Convert strings in the Column to lowercase.
        Equivalent to :meth:`str.lower`.
        """
        raise NotImplementedError

    @abstractmethod
    def upper(self) -> IStringColumn:
        """
        Convert strings in the Column to uppercase.
        Equivalent to :meth:`str.upper`.
        """
        raise NotImplementedError

    # Pattern matching related methods  -----------------------------------------------------

    @abstractmethod
    def startswith(self, pat):
        """Test if the beginning of each string element matches a pattern."""
        raise NotImplementedError

    @abstractmethod
    def endswith(self, pat):
        """Test if the end of each string element matches a pattern."""
        raise NotImplementedError

    @abstractmethod
    def count(self, pat: str):
        """Count occurrences of pattern in each string of column"""
        raise NotImplementedError

    @abstractmethod
    def find(self, sub):
        raise NotImplementedError

    @abstractmethod
    def replace(self, pat: str, repl: str, regex: bool = True):
        """
        Replace each occurrence of pattern in the Column.
        """
        raise NotImplementedError

    @abstractmethod
    def match(self, pat: str):
        """Determine if each string matches a regular expression"""
        raise NotImplementedError

    @abstractmethod
    def contains(self, pat: str, regex: bool = True):
        """Test for each item if pattern is contained within a string; returns a boolean"""
        raise NotImplementedError

    @abstractmethod
    def findall(self, pat: str):
        """
        Find for each item all occurrences of pattern (see re.findall())
        """
        raise NotImplementedError

    @abstractmethod
    def cat(self, col: IStringColumn):
        # TODO: docstring
        raise NotImplementedError
