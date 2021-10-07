# Copyright (c) Facebook, Inc. and its affiliates.
from .numerical_column_cpu import *
from .string_column_cpu import *
from .list_column_cpu import *
from .map_column_cpu import *
from .dataframe_cpu import *
import torcharrow
import torcharrow._torcharrow

# Initialize and register Velox functional
import torcharrow.velox_rt.functional

torcharrow._torcharrow.BaseColumn.dtype = (
    lambda self: torcharrow.dtypes.dtype_of_velox_type(self.type())
)
