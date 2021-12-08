# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional, List, Union

import pandas as pd  # type: ignore
import torcharrow.dtypes as dt
from torcharrow.scope import Scope

from .dispatcher import Dispatcher
from .icolumn import IColumn
from .idataframe import IDataFrame
from .interop_arrow import _from_arrow_array, _from_arrow_table


def from_arrow(
    data, dtype: Optional[dt.DType] = None, device: str = ""
) -> Union[IColumn, IDataFrame]:
    """
    Convert arrow array/table to a TorchArrow Column/DataFrame.
    """
    import pyarrow as pa

    assert isinstance(data, pa.Array) or isinstance(data, pa.Table)

    if dtype is not None:
        raise NotImplementedError

    device = device or Scope.default.device

    if isinstance(data, pa.Array):
        return _from_arrow_array(data, device=device)
    elif isinstance(data, pa.Table):
        return _from_arrow_table(data, device=device)
    else:
        raise ValueError


def from_pandas(data, dtype: Optional[dt.DType] = None, device: str = ""):
    """
    Convert Pandas series/dataframe to a TorchArrow Column/DataFrame.
    """
    raise NotImplementedError


def from_pylist(
    data: List, dtype: Optional[dt.DType] = None, device: str = ""
) -> IColumn:
    """
    Convert Python list of scalars or containers to a TorchArrow Column/DataFrame.
    """
    # TODO(https://github.com/facebookresearch/torcharrow/issues/80) Infer dtype
    device = device or Scope.default.device

    return Scope.default._FromPyList(data, dtype, device)
