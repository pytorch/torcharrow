# Copyright (c) Facebook, Inc. and its affiliates.

# For relative imports to work in Python 3.6
from .expression import *  # dependencies: None
from .trace import *  # dependencies: expression

# don't include
# from .dtypes import *
# since dtypes define Tuple and List which confuse mypy

from .scope import *  # dependencies: column_factory, dtypes

# following needs scope*
from .icolumn import Column, concat, if_else  # noqa
from .inumerical_column import *
from .istring_column import *
from .ilist_column import *
from .imap_column import *
from .idataframe import *

from .velox_rt import *

from . import pytorch
from .interop import from_pylist

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

# module level doc-string
__doc__ = """

torcharrow - a DataFrame library built on Arrow and Velox acceleration library
==============================================================================

**TorchArrow** is a Pandas inspired DataFrame library in Python built on the
Apache Arrow columnar memory format and leveraging the Velox acceleration
library for loading, filtering, mapping, joining, aggregating, and otherwise
manipulating tabular data on CPUs.

TorchArrow supports PyTorch's Tensors as first class citizens. It allows
mostly zero copy interop with Numpy, Pandas, PyArrow, CuDf and of course
integrates well with PyTorch data-wrangling workflows.

Examples
--------
>>> import torcharrow as ta
>>> df = ta.DataFrame({'a':[1,2,3], 'b':[4,None,6]})
>>> df['a'] + df['b']
0  5
1  None
2  9
dtype: Int64(nullable=True), length: 3, null_count: 1
"""
