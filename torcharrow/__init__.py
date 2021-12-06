# Copyright (c) Facebook, Inc. and its affiliates.

from . import pytorch  # noqa
from . import velox_rt  # noqa
from .icolumn import IColumn, Column, concat, if_else  # noqa
from .idataframe import IDataFrame, DataFrame, me  # noqa
from .interop import from_pylist, from_arrow  # noqa

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

__all__ = [
    "DataFrame",
    "Column",
    "concat",
    "if_else",
    "from_pylist",
    "me",
    "IDataFrame",
    "IColumn",
]

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
