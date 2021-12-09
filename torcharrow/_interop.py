# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Optional, cast

# Skipping analyzing 'numpy': found module but no type hints or library stubs
import numpy as np  # type: ignore
import numpy.ma as ma  # type: ignore

# Skipping analyzing 'pandas': found module but no type hints or library stubs
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
import torcharrow.dtypes as dt
from torcharrow.scope import Scope


def from_pandas_dataframe(
    df,
    dtype: Optional[dt.DType] = None,
    columns: Optional[List[str]] = None,
    scope=None,
    device="",
):
    """
    Convert pandas dataframe to torcharrow dataframe (drops indices).

    Parameters
    ----------
    df : Pandas dataframe

    dtype : dtype, default None
    Data type to force, if None will automatically infer.

    columns : array-like
    List of column names to extract from df.

    scope : Scope or None
    Scope to use, or None for default scope.

    device : str or ""
    Device to use, or default if blank.

    Examples
    --------
    >>> import pandas as pd
    >>> import torcharrow as ta
    >>> pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
    >>> gdf = ta.from_pandas_dataframe(pdf)
    >>> gdf
      index    a    b
    -------  ---  ---
          0    0  0.1
          1    1  0.2
          2    2
          3    3  0.3
    dtype: Struct([Field('a', int64), Field('b', Float64(nullable=True))]), count: 4, null_count: 0

    """
    scope = scope or Scope.default
    device = device or scope.device

    if dtype is not None:
        assert dt.is_struct(dtype)
        dtype = cast(dt.Struct, dtype)
        res = {}
        for f in dtype.fields:
            # this shows that Column shoud also construct Dataframes!
            res[f.name] = from_pandas_series(
                pd.Series(df[f.name]), f.dtype, scope=scope
            )
        return scope.Frame(res, dtype=dtype, device=device)
    else:
        res = {}
        for n in df.columns:
            if columns is None or n in columns:
                res[n] = from_pandas_series(pd.Series(df[n]), scope=scope)
        return scope.Frame(res, device=device)


def from_pandas_series(series, dtype=None, scope=None, device=""):
    """ "
    Convert pandas series array to a torcharrow column (drops indices).
    """
    scope = scope or Scope.default
    device = device or scope.device

    return from_numpy(series.to_numpy(), dtype, scope, device)


def from_numpy(array, dtype, scope=None, device=""):
    """
    Convert 1dim numpy array to a torcharrow column (zero copy).
    """
    scope = scope or Scope.default
    device = device or scope.device

    if isinstance(array, ma.core.MaskedArray) and array.ndim == 1:
        return _from_numpy_ma(array.data, array.mask, dtype, scope, device)
    elif isinstance(array, np.ndarray) and array.ndim == 1:
        return _from_numpy_nd(array, dtype, scope, device)
    else:
        raise TypeError(f"cannot convert numpy array of type {array.dtype}")


def _is_not_str(s):
    return not isinstance(s, str)


def _from_numpy_ma(data, mask, dtype, scope=None, device=""):
    # adopt types
    if dtype is None:
        dtype = dt.typeof_np_dtype(data.dtype).with_null()
    else:
        assert dt.is_primitive_type(dtype)
        assert dtype == dt.typeof_np_dtype(data.dtype).with_null()
        # TODO if not, adopt the type or?
        # Something like ma.array
        # np.array([np.nan, np.nan,  3.]).astype(np.int64),
        # mask = np.isnan([np.nan, np.nan,  3.]))

    # create column, only zero copy supported
    if dt.is_boolean_or_numerical(dtype):
        assert not np.all(np.isnan(ma.array(data, mask).compressed()))

        return scope._FullColumn(data, dtype=dtype, mask=mask)
    elif dt.is_string(dtype) or dtype == "object":
        assert np.all(np.vectorize(_is_not_str)(ma.array(data, mask).compressed()))
        return scope._FullColumn(data, dtype=dtype, mask=mask)
    else:
        raise TypeError(f"cannot convert masked numpy array of type {data.dtype}")


def _from_numpy_nd(data, dtype, scope=None, device=""):
    # adopt types
    if dtype is None:
        dtype = dt.typeof_np_dtype(data.dtype)
        if dtype is None:
            dtype = dt.string
    else:
        assert dt.is_primitive(dtype)
        # TODO Check why teh following assert  isn't the case
        # assert dtype == dt.typeof_np_dtype(data.dtype)

    # create column, only zero copy supported
    if dt.is_boolean_or_numerical(dtype):
        mask = np.isnan(data)
        return scope._FullColumn(data, dtype=dtype, mask=mask)
    elif dt.is_string(dtype):
        mask = np.vectorize(_is_not_str)(data)
        if np.any(mask):
            dtype = dtype.with_null()
        return scope._FullColumn(data, dtype=dtype, mask=mask)
    else:
        raise TypeError("can not convert numpy array of type {data.dtype,}")


# def _column_without_nan(series, dtype):
#     if dtype is None or is_floating(dtype):
#         for i in series:
#             if isinstance(i, float) and np.isnan(i):
#                 yield None
#             else:
#                 yield i
#     else:
#         for i in series:
#             yield i


def _arrow_scalar_to_py(array):
    for i in array:
        yield i.as_py()


def _pandatype_to_dtype(t, nullable):
    return dt.typeof_nptype(t, nullable)


def _arrowtype_to_dtype(t, nullable):
    if pa.types.is_boolean(t):
        return dt.Boolean(nullable)
    if pa.types.is_int8(t):
        return dt.Int8(nullable)
    if pa.types.is_int16(t):
        return dt.Int16(nullable)
    if pa.types.is_int32(t):
        return dt.Int32(nullable)
    if pa.types.is_int64(t):
        return dt.Int64(nullable)
    if pa.types.is_float32(t):
        return dt.Float32(nullable)
    if pa.types.is_float64(t):
        return dt.Float64(nullable)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return dt.String(nullable)
    raise NotImplementedError(f"Unsupported Arrow type: {str(t)}")
