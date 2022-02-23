# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .numerical_column_cpu import *
from .string_column_cpu import *
from .list_column_cpu import *
from .map_column_cpu import *
from .dataframe_cpu import *
import torcharrow

# pyre-fixme[21]: Could not find module `torcharrow._torcharrow`.
import torcharrow._torcharrow

# Initialize and register Velox functional
import torcharrow.velox_rt.functional

from .typing import dtype_of_velox_type

torcharrow._torcharrow.BaseColumn.dtype = lambda self: dtype_of_velox_type(self.type())
