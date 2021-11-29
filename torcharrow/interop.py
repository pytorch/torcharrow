# Copyright (c) Facebook, Inc. and its affiliates.
import pandas as pd  # type: ignore
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

    # TODO Find out a way to propagate nullable property properly.
    # nullable = data.null_count > 0 is not quite right since it is legit for
    # either a nullable or non-nullable array to have null_count == 0. Also the
    # only backend we have right now, Velox, doesn't support nullability, so we
    # will need to make it support nullability or have some way around it to
    # carry over the nullable property to the exporting path (Velox -> Arrow)
    dtype = dtype or _arrowtype_to_dtype(data.type, True)
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
    # TODO(https://github.com/facebookresearch/torcharrow/issues/80) Infer dtype
    device = device or Scope.default.device

    return Scope.default._FromPyList(data, dtype, device)
