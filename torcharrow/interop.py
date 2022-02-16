# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional, Sequence, Union

import torcharrow.dtypes as dt

from .icolumn import IColumn
from .idataframe import IDataFrame
from .interop_arrow import _from_arrow_array, _from_arrow_table
from .scope import Scope


def from_arrow(
    data, dtype: Optional[dt.DType] = None, device: str = ""
) -> Union[IColumn, IDataFrame]:
    """
    Convert arrow array/table to a TorchArrow Column/DataFrame.
    """
    import pyarrow as pa

    assert isinstance(data, pa.Array) or isinstance(data, pa.Table)

    device = device or Scope.default.device

    if isinstance(data, pa.Array):
        return _from_arrow_array(data, dtype, device=device)
    elif isinstance(data, pa.Table):
        return _from_arrow_table(data, dtype, device=device)
    else:
        raise ValueError


def from_pandas(data, dtype: Optional[dt.DType] = None, device: str = ""):
    """
    Convert Pandas series/dataframe to a TorchArrow Column/DataFrame.
    """
    raise NotImplementedError


def from_pysequence(
    data: Sequence, dtype: Optional[dt.DType] = None, device: str = ""
) -> IColumn:
    """
    Convert Python sequence of scalars or containers to a TorchArrow Column/DataFrame.
    """
    # TODO(https://github.com/facebookresearch/torcharrow/issues/80) Infer dtype
    device = device or Scope.default.device

    # pyre-fixme[6]: For 2nd param expected `DType` but got `Optional[DType]`.
    return Scope.default._FromPySequence(data, dtype, device)
