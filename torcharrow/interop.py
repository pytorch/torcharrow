# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Optional

from torcharrow import Scope

from .dispatcher import Dispatcher


def from_arrow(
    data,  # type: pa.Array
    device: str = "",
    nullable: Optional[bool] = None,
):
    """
    Convert arrow array/table to a TorchArrow Column/DataFrame.
    """
    device = device or Scope.default.device

    import pyarrow as pa
    from torcharrow._interop import _arrowtype_to_dtype

    assert isinstance(data, pa.Array) or isinstance(data, pa.Table)

    if nullable is None:
        # Using the most narrow type we can, we (i) don't restrict in any
        # way where it can be used (since we can pass a narrower typed
        # non-null column to a function expecting a nullable type, but not
        # vice versa), (ii) when we bring in a stricter type system in Velox
        # to allow functions to only be invokable on non-null types we
        # increase the amount of places we can use the from_arrow result
        nullable = data.null_count > 0
    if not nullable and data.null_count > 0:
        raise RuntimeError("Cannot store nulls in a non-nullable column")
    dtype = _arrowtype_to_dtype(data.type, nullable)

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
