import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
import torcharrow.dtypes as dt
from torcharrow import Scope


def from_arrow(data, dtype=None, scope=None, device=""):
    """
    Convert arrow array/table to a TorchArrow Column/DataFrame.
    """
    scope = scope or Scope.default
    device = device or scope.device

    return scope.from_arrow(data, dtype, device)


def from_pandas(data, dtype=None, scope=None, device=""):
    """
    Convert Pandas series/dataframe to a TorchArrow Column/DataFrame.
    """
    raise NotImplementedError


def from_torch(data, dtype=None, scope=None, device=""):
    raise NotImplementedError


def from_pylist(data, dtype=None, scope=None, device=""):
    """
    Convert Python list of scalars or containers to a TorchArrow Column/DataFrame.
    """
    scope = scope or Scope.default
    device = device or scope.device

    return scope._FromPyList(data, dtype, device)
