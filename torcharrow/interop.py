# Copyright (c) Facebook, Inc. and its affiliates.
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
import torcharrow.dtypes as dt
from torcharrow import Scope

from .dispatcher import Dispatcher


def from_arrow(data, dtype=None, device=""):
    """
    Convert arrow array/table to a TorchArrow Column/DataFrame.
    """
    device = device or Scope.default.device

    import pyarrow as pa
    from torcharrow._interop import _arrowtype_to_dtype

    assert isinstance(data, pa.Array) or isinstance(data, pa.Table)

    dtype = dtype or _arrowtype_to_dtype(data.type, data.null_count > 0)
    device = device or Scope.default.device

    call = Dispatcher.lookup((dtype.typecode + "_fromarrow", device))

    return call(device, data, dtype)


def from_pandas(data, dtype=None, device=""):
    """
    Convert Pandas series/dataframe to a TorchArrow Column/DataFrame.
    """
    raise NotImplementedError


def from_pylist(data, dtype=None, device=""):
    """
    Convert Python list of scalars or containers to a TorchArrow Column/DataFrame.
    """
    device = device or Scope.default.device

    return Scope.default._FromPyList(data, dtype, device)
