# Copyright (c) Facebook, Inc. and its affiliates.

# For relative imports to work in Python 3.6
from .column_factory import *  # dependencies: None
from .expression import *  # dependencies: None
from .trace import *  # dependencies: expression

# don't include
# from .dtypes import *
# since dtypes define Tuple and List which confuse mypy

from .scope import *  # dependencies: column_factory, dtypes

# following needs scope*
from .icolumn import *  # dependencies: cyclic dependency to every other column
from .inumerical_column import *
from .istring_column import *
from .ilist_column import *
from .imap_column import *
from .idataframe import *

# velox_rt imports torcharrow._torcharrow which binds Velox RowType,
# which conflicts with koski_rt
# from .velox_rt import *

from .demo_rt import *

from .interop import *

from . import pytorch

# 0.1.0
# Arrow types and columns

# 0.2.0
# Pandas -- the Good parts --

# 0.3.0
# Multi-targetting, Numpy repr & ops, zero copy

__version__ = "0.3.0"
__author__ = "Facebook"
__credits__ = "Pandas, CuDF, Numpy, Arrow"

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
